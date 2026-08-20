"""
Agente de IA para leitura de PDFs (RAG) + triagem, usando Gemini.

Este módulo é a mesma lógica do seu notebook do Colab (triagem estruturada +
RAG sobre PDFs + LangGraph), só que organizada como um módulo Python normal,
para poder ser chamada por uma API (main.py) em vez de rodar célula por célula.
"""

import time
import re
import os
import json
import shutil
import pathlib
from pathlib import Path
from collections import Counter
from typing import TypedDict, Optional, List, Dict, Literal

# --- Certificado corporativo (redes com proxy/firewall que inspeciona HTTPS) ---
# Precisa ser configurado ANTES de importar qualquer coisa relacionada a
# grpc/google/requests, senão a biblioteca já carrega sua lista de
# certificados padrão e ignora essa configuração.
#
# O gRPC (usado nas chamadas reais de embedding/chat do Gemini) se mostrou
# confiável usando o arquivo combined_cert.pem gerado manualmente (ver
# README). Priorizamos ele aqui; se não existir, caímos para o certifi.
_CERT_PATH = Path(__file__).resolve().parent / "combined_cert.pem"
if not _CERT_PATH.exists():
    try:
        import certifi
        _CERT_PATH = Path(certifi.where())
    except ImportError:
        _CERT_PATH = None

if _CERT_PATH:
    os.environ["SSL_CERT_FILE"] = str(_CERT_PATH)
    os.environ["REQUESTS_CA_BUNDLE"] = str(_CERT_PATH)
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = str(_CERT_PATH)

# A biblioteca `requests` (usada para ler links da web) costuma ser mais
# rígida que o gRPC na validação do certificado da empresa. O pacote abaixo
# faz o Python confiar diretamente no repositório de certificados do
# Windows (o mesmo que o navegador já usa) especificamente para `requests`,
# independente do arquivo de certificado usado acima.
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY não encontrada. Crie um arquivo .env na pasta backend/ "
        "com a linha: GEMINI_API_KEY=sua_chave_aqui"
    )

# ---------------------------------------------------------------------------
# 1. Conexão com o Gemini
# ---------------------------------------------------------------------------

llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# ---------------------------------------------------------------------------
# 2. Triagem estruturada
# ---------------------------------------------------------------------------

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nos documentos.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema.\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, aprovação especial ou algo fora do escopo dos documentos.\n'
)


class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


triagem_chain = llm_triagem.with_structured_output(TriagemOut)


def triagem(mensagem: str) -> dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem),
    ])
    return saida.model_dump()


# ---------------------------------------------------------------------------
# 3. RAG sobre PDFs — estado global do índice (carregado via /upload)
# ---------------------------------------------------------------------------

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente técnico especializado nos documentos fornecidos pela empresa. "
     "Responda SOMENTE com base no contexto fornecido, de forma objetiva, curta e direta "
     "(no máximo 3-4 frases). "
     "NÃO use markdown: não use asteriscos, negrito, títulos ou listas com marcadores — "
     "escreva em texto corrido, como se estivesse falando com a pessoa. "
     "Cite a seção ou o requisito relevante de forma natural dentro do texto, sem formatação especial. "
     "Use o histórico da conversa apenas para entender do que a pergunta atual está falando "
     "(ex: 'e o de rede?' referindo-se a um assunto mencionado antes) — mas responda sempre "
     "com base no contexto do documento, nunca inventando algo a partir do histórico. "
     "Se não houver base suficiente no contexto para responder com segurança, responda apenas 'Não sei'."),
    ("human",
     "{historico}Pergunta atual: {input}\n\nContexto do documento:\n{context}"),
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Guardamos o retriever atual aqui. Ele é (re)criado toda vez que novos PDFs
# são enviados via /upload. Para um uso com múltiplos usuários simultâneos
# em produção, o ideal é trocar isso por um índice por sessão/usuário.
_retriever = None
_indexed_files: List[str] = []

# Pasta onde o índice fica salvo em disco, para não precisar reindexar (e
# gastar cota da API de novo) toda vez que o servidor reiniciar.
_INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
_INDEX_META_PATH = _INDEX_DIR / "arquivos_indexados.json"


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    ini, fim = max(0, pos - janela // 2), min(len(txt), pos + janela // 2)
    return txt[ini:fim]


def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source", "")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({
            "documento": src,
            "pagina": page,
            "trecho": extrair_trecho(d.page_content, query),
        })
    return cites


def _salvar_indice(vectorstore, nomes_arquivos: List[str]) -> None:
    """Salva o índice em disco para poder ser recarregado sem gastar cota de novo."""
    try:
        _INDEX_DIR.mkdir(exist_ok=True)
        vectorstore.save_local(str(_INDEX_DIR))
        _INDEX_META_PATH.write_text(json.dumps(nomes_arquivos), encoding="utf-8")
        print(f"[DEBUG] Índice salvo em disco ({len(nomes_arquivos)} arquivo(s)).")
    except Exception as e:
        # Não é crítico se falhar ao salvar — o app continua funcionando,
        # só não vai persistir entre reinicializações.
        print(f"[DEBUG] Não foi possível salvar o índice em disco: {e}")


def _carregar_indice_salvo() -> None:
    """Tenta carregar um índice salvo anteriormente, ao iniciar o servidor."""
    global _retriever, _indexed_files
    if not _INDEX_DIR.exists():
        return
    try:
        vectorstore = FAISS.load_local(
            str(_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        _retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        if _INDEX_META_PATH.exists():
            _indexed_files = json.loads(_INDEX_META_PATH.read_text(encoding="utf-8"))
        print(f"[DEBUG] Índice carregado do disco: {_indexed_files}")
    except Exception as e:
        print(f"[DEBUG] Não havia índice salvo válido para carregar: {e}")


def limpar_indice() -> None:
    """Remove o índice salvo em disco e da memória (útil para começar do zero)."""
    global _retriever, _indexed_files
    _retriever = None
    _indexed_files = []
    if _INDEX_DIR.exists():
        shutil.rmtree(_INDEX_DIR, ignore_errors=True)


def _extrair_tempo_espera(erro: Exception) -> float:
    """Tenta extrair o 'retry_delay' sugerido pela API do Gemini na mensagem de erro."""
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(erro))
    if match:
        return float(match.group(1)) + 3  # margem de segurança
    return 30.0  # fallback se não conseguir extrair


# Estado do progresso da indexação em andamento, consultado pelo frontend
# via GET /upload-progress enquanto o POST /upload está rodando.
_progresso = {
    "em_andamento": False,
    "lote_atual": 0,
    "total_lotes": 0,
    "chunks_processados": 0,
    "total_chunks": 0,
    "aguardando_segundos": None,
    "mensagem": "",
}


def obter_progresso() -> dict:
    return dict(_progresso)


def _resetar_progresso():
    _progresso.update({
        "em_andamento": False,
        "lote_atual": 0,
        "total_lotes": 0,
        "chunks_processados": 0,
        "total_chunks": 0,
        "aguardando_segundos": None,
        "mensagem": "",
    })


# ---------------------------------------------------------------------------
# Leitura de diferentes tipos de arquivo e de links
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls"}


def _carregar_arquivo(caminho: Path) -> List[Document]:
    """Carrega um arquivo e retorna seus documentos, de acordo com a extensão."""
    ext = caminho.suffix.lower()

    if ext == ".pdf":
        return PyMuPDFLoader(str(caminho)).load()

    if ext == ".docx":
        return Docx2txtLoader(str(caminho)).load()

    if ext in (".txt", ".md"):
        texto = caminho.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=texto, metadata={"source": str(caminho), "page": 0})]

    if ext == ".csv":
        import csv
        with open(caminho, newline="", encoding="utf-8", errors="ignore") as f:
            linhas = [" | ".join(linha) for linha in csv.reader(f)]
        texto = "\n".join(linhas)
        return [Document(page_content=texto, metadata={"source": str(caminho), "page": 0})]

    if ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(str(caminho), data_only=True)
        docs = []
        for idx, planilha in enumerate(wb.worksheets):
            linhas = []
            for linha in planilha.iter_rows(values_only=True):
                linhas.append(" | ".join("" if c is None else str(c) for c in linha))
            texto = f"[Planilha: {planilha.title}]\n" + "\n".join(linhas)
            docs.append(Document(page_content=texto, metadata={"source": str(caminho), "page": idx}))
        return docs

    raise ValueError(f"Tipo de arquivo não suportado: {ext}")


def _carregar_url(url: str) -> List[Document]:
    """Baixa uma página web e extrai o texto principal dela."""
    import requests
    from bs4 import BeautifulSoup

    resposta = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resposta.raise_for_status()

    sopa = BeautifulSoup(resposta.text, "html.parser")
    for tag in sopa(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    texto = re.sub(r"\n{3,}", "\n\n", sopa.get_text(separator="\n")).strip()
    if not texto:
        raise ValueError("Não foi possível extrair texto dessa página.")

    return [Document(page_content=texto, metadata={"source": url, "page": 0})]


def _chunk_e_indexar(docs: List[Document], carregados: List[dict], nomes_indexados: List[str]) -> dict:
    """Recebe documentos já carregados, quebra em chunks e gera o índice FAISS
    em lotes (com retry automático em caso de limite de requisições)."""
    global _retriever, _indexed_files

    if not docs:
        _resetar_progresso()
        erros = [f"{c['arquivo']}: {c['erro']}" for c in carregados if c.get("erro")]
        detalhe = "Nenhum conteúdo pôde ser extraído."
        if erros:
            detalhe += " Detalhes: " + " | ".join(erros)
        return {"ok": False, "detalhe": detalhe, "arquivos": carregados}

    _progresso["mensagem"] = "Preparando o conteúdo..."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    chunks = [d for d in chunks if isinstance(d, Document)]

    TAMANHO_LOTE = 80
    MAX_TENTATIVAS_POR_LOTE = 6
    vectorstore = None
    total = len(chunks)
    total_lotes = (total + TAMANHO_LOTE - 1) // TAMANHO_LOTE

    _progresso["total_chunks"] = total
    _progresso["total_lotes"] = total_lotes
    _progresso["mensagem"] = f"Gerando embeddings de {total} trechos..."

    try:
        for i in range(0, total, TAMANHO_LOTE):
            lote = chunks[i : i + TAMANHO_LOTE]
            lote_num = i // TAMANHO_LOTE + 1
            print(f"[DEBUG] indexando lote {i}-{i+len(lote)} de {total} chunks...")

            _progresso["lote_atual"] = lote_num
            _progresso["aguardando_segundos"] = None
            _progresso["mensagem"] = f"Processando lote {lote_num} de {total_lotes}..."

            tentativa = 0
            while True:
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(lote, embeddings)
                    else:
                        vectorstore.add_documents(lote)
                    break
                except Exception as e:
                    tentativa += 1
                    limite_excedido = "429" in str(e) or "quota" in str(e).lower()
                    if not limite_excedido or tentativa > MAX_TENTATIVAS_POR_LOTE:
                        raise
                    espera = _extrair_tempo_espera(e)
                    print(
                        f"[DEBUG] Limite de requisições atingido. Aguardando "
                        f"{espera:.0f}s antes de tentar de novo (tentativa {tentativa})..."
                    )
                    _progresso["aguardando_segundos"] = round(espera)
                    _progresso["mensagem"] = (
                        f"Limite da API atingido, aguardando {round(espera)}s "
                        f"(lote {lote_num} de {total_lotes})..."
                    )
                    time.sleep(espera)

            _progresso["chunks_processados"] = min(i + TAMANHO_LOTE, total)

            if i + TAMANHO_LOTE < total:
                time.sleep(2)
    except Exception as e:
        _resetar_progresso()
        return {
            "ok": False,
            "detalhe": (
                f"Falha ao gerar embeddings ({e.__class__.__name__}: {e}). "
                f"Tente novamente em alguns instantes ou envie o conteúdo em partes menores."
            ),
            "arquivos": carregados,
        }

    _retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    _indexed_files = nomes_indexados
    _salvar_indice(vectorstore, _indexed_files)
    _resetar_progresso()

    contagem = Counter(pathlib.Path(d.metadata.get("source", "")).name for d in chunks)

    return {
        "ok": True,
        "arquivos": carregados,
        "total_chunks": len(chunks),
        "chunks_por_arquivo": dict(contagem),
    }


def indexar_fontes(caminhos: List[Path], urls: Optional[List[str]] = None) -> dict:
    """Carrega arquivos (PDF/DOCX/TXT/MD/CSV/XLSX) e/ou links da web informados,
    gera embeddings e (re)constrói o índice FAISS."""
    _resetar_progresso()
    _progresso["em_andamento"] = True
    _progresso["mensagem"] = "Lendo os arquivos..."

    docs: List[Document] = []
    carregados: List[dict] = []
    nomes: List[str] = []

    for caminho in caminhos:
        try:
            novos = _carregar_arquivo(caminho)
            docs.extend(novos)
            carregados.append({"arquivo": caminho.name, "paginas": len(novos)})
            nomes.append(caminho.name)
        except Exception as e:
            carregados.append({"arquivo": caminho.name, "erro": str(e)})

    for url in (urls or []):
        if not url or not url.strip():
            continue
        url = url.strip()
        try:
            novos = _carregar_url(url)
            docs.extend(novos)
            carregados.append({"arquivo": url, "paginas": len(novos)})
            nomes.append(url)
        except Exception as e:
            carregados.append({"arquivo": url, "erro": str(e)})

    return _chunk_e_indexar(docs, carregados, nomes)


def _formatar_historico(historico: Optional[List[Dict[str, str]]]) -> str:
    """Formata os últimos turnos da conversa para dar contexto em perguntas de
    acompanhamento (ex: 'e sobre isso?'). Limitado às últimas 3 trocas para
    não inflar o prompt (e a cota) sem necessidade."""
    if not historico:
        return ""
    trechos = []
    for turno in historico[-3:]:
        pergunta_ant = (turno.get("pergunta") or "").strip()
        resposta_ant = (turno.get("resposta") or "").strip()
        if pergunta_ant and resposta_ant:
            trechos.append(f"Usuário: {pergunta_ant}\nAssistente: {resposta_ant}")
    if not trechos:
        return ""
    return "Histórico recente da conversa:\n" + "\n\n".join(trechos) + "\n\n"


def perguntar_rag(pergunta: str, historico: Optional[List[Dict[str, str]]] = None) -> Dict:
    if _retriever is None:
        return {
            "answer": "Nenhum PDF foi indexado ainda. Envie um PDF em /upload antes de perguntar.",
            "citacoes": [],
            "contexto_encontrado": False,
        }

    # Para a busca no PDF, combina a pergunta atual com a última pergunta do
    # histórico (se houver) — ajuda a achar o trecho certo quando a pergunta
    # atual é vaga tipo "e sobre isso?".
    pergunta_busca = pergunta
    if historico:
        ultima = (historico[-1].get("pergunta") or "").strip()
        if ultima:
            pergunta_busca = f"{ultima} {pergunta}"

    docs_relacionados = _retriever.invoke(pergunta_busca)
    print(f"[DEBUG] docs encontrados: {len(docs_relacionados)}")

    if not docs_relacionados:
        print("[DEBUG] Nenhum doc encontrado, retornando Não sei.")
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    historico_formatado = _formatar_historico(historico)
    answer = document_chain.invoke({
        "input": pergunta,
        "context": docs_relacionados,
        "historico": historico_formatado,
    })
    txt = (answer or "").strip()
    print(f"[DEBUG] resposta da IA (bruta): {txt!r}")

    if txt.rstrip(".!?") == "Não sei":
        print("[DEBUG] Resposta bateu com 'Não sei', marcando como sem contexto.")
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    return {
        "answer": txt,
        "citacoes": formatar_citacoes(docs_relacionados, pergunta)[:2],
        "contexto_encontrado": True,
    }


def status_indice() -> dict:
    return {"indexado": _retriever is not None, "arquivos": _indexed_files}


# Tenta recarregar um índice salvo de uma execução anterior, assim o app não
# perde o PDF indexado toda vez que o servidor reinicia (economiza cota).
_carregar_indice_salvo()


# ---------------------------------------------------------------------------
# 4. Orquestração com LangGraph (triagem -> RAG -> pedir info / abrir chamado)
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    pergunta: str
    historico: List[Dict[str, str]]
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str


def node_auto_resolver(state: AgentState) -> AgentState:
    resposta_rag = perguntar_rag(state["pergunta"], state.get("historico"))
    if resposta_rag["contexto_encontrado"]:
        return {
            "resposta": resposta_rag["answer"],
            "citacoes": resposta_rag.get("citacoes", []),
            "rag_sucesso": True,
            "acao_final": "AUTO_RESOLVER",
        }
    return {
        "resposta": (
            "Não encontrei essa informação no documento indexado. "
            "Tente reformular a pergunta, ou confirme se o PDF certo foi carregado."
        ),
        "citacoes": [],
        "rag_sucesso": False,
        "acao_final": "SEM_CONTEXTO",
    }


_workflow = StateGraph(AgentState)
_workflow.add_node("auto_resolver", node_auto_resolver)

# Fluxo simples: sempre tenta responder com base no(s) PDF(s) indexado(s).
# Se não achar nada relevante, avisa claramente em vez de inventar ou pedir
# detalhes confusos (evita respostas estranhas em perguntas fora do escopo).
_workflow.add_edge(START, "auto_resolver")
_workflow.add_edge("auto_resolver", END)

grafo = _workflow.compile()


def perguntar_agente(pergunta: str, historico: Optional[List[Dict[str, str]]] = None) -> dict:
    """Ponto de entrada único usado pela API: roda o RAG e retorna a resposta.

    `historico` (opcional) é uma lista de trocas anteriores da conversa, no
    formato [{"pergunta": "...", "resposta": "..."}], usada para dar
    contexto a perguntas de acompanhamento.
    """
    resultado = grafo.invoke({"pergunta": pergunta, "historico": historico or []})
    return {
        "resposta": resultado.get("resposta"),
        "citacoes": resultado.get("citacoes", []),
        "acao_final": resultado.get("acao_final"),
    }
