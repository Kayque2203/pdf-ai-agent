"""
Agente de IA para leitura de PDFs (RAG) + triagem, usando Gemini.

Este módulo é a mesma lógica do seu notebook do Colab (triagem estruturada +
RAG sobre PDFs + LangGraph), só que organizada como um módulo Python normal,
para poder ser chamada por uma API (main.py) em vez de rodar célula por célula.
"""

import os
import re
import pathlib
from pathlib import Path
from collections import Counter
from typing import TypedDict, Optional, List, Dict, Literal

# --- Certificado corporativo (redes com proxy/firewall que inspeciona HTTPS) ---
# Precisa ser configurado ANTES de importar qualquer coisa relacionada a
# grpc/google, senão a biblioteca já carrega sua lista de certificados padrão
# e ignora essa configuração.
_CERT_PATH = Path(__file__).resolve().parent / "combined_cert.pem"
if _CERT_PATH.exists():
    os.environ["SSL_CERT_FILE"] = str(_CERT_PATH)
    os.environ["REQUESTS_CA_BUNDLE"] = str(_CERT_PATH)
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = str(_CERT_PATH)

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
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
     "Se não houver base suficiente no contexto para responder com segurança, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}"),
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Guardamos o retriever atual aqui. Ele é (re)criado toda vez que novos PDFs
# são enviados via /upload. Para um uso com múltiplos usuários simultâneos
# em produção, o ideal é trocar isso por um índice por sessão/usuário.
_retriever = None
_indexed_files: List[str] = []


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


def indexar_pdfs(caminhos: List[Path]) -> dict:
    """Carrega os PDFs informados, gera embeddings e (re)constrói o índice FAISS."""
    global _retriever, _indexed_files

    docs = []
    carregados = []
    for caminho in caminhos:
        try:
            loader = PyMuPDFLoader(str(caminho))
            paginas = loader.load()
            docs.extend(paginas)
            carregados.append({"arquivo": caminho.name, "paginas": len(paginas)})
        except Exception as e:
            carregados.append({"arquivo": caminho.name, "erro": str(e)})

    if not docs:
        return {"ok": False, "detalhe": "Nenhum PDF pôde ser lido.", "arquivos": carregados}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    chunks = [d for d in chunks if isinstance(d, Document)]

    vectorstore = FAISS.from_documents(chunks, embeddings)
    _retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    _indexed_files = [c.name for c in caminhos]

    contagem = Counter(pathlib.Path(d.metadata.get("source", "")).name for d in chunks)

    return {
        "ok": True,
        "arquivos": carregados,
        "total_chunks": len(chunks),
        "chunks_por_arquivo": dict(contagem),
    }


def perguntar_rag(pergunta: str) -> Dict:
    if _retriever is None:
        return {
            "answer": "Nenhum PDF foi indexado ainda. Envie um PDF em /upload antes de perguntar.",
            "citacoes": [],
            "contexto_encontrado": False,
        }

    docs_relacionados = _retriever.invoke(pergunta)
    print(f"[DEBUG] docs encontrados: {len(docs_relacionados)}")

    if not docs_relacionados:
        print("[DEBUG] Nenhum doc encontrado, retornando Não sei.")
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
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


# ---------------------------------------------------------------------------
# 4. Orquestração com LangGraph (triagem -> RAG -> pedir info / abrir chamado)
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str


def node_triagem(state: AgentState) -> AgentState:
    return {"triagem": triagem(state["pergunta"])}


def node_auto_resolver(state: AgentState) -> AgentState:
    resposta_rag = perguntar_rag(state["pergunta"])
    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update


def node_pedir_info(state: AgentState) -> AgentState:
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ", ".join(faltantes) if faltantes else "tema e contexto específico"
    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO",
    }


def node_abrir_chamado(state: AgentState) -> AgentState:
    t = state["triagem"]
    return {
        "resposta": f"Abrindo chamado com urgência {t['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO",
    }


def decidir_pos_auto_resolver(state: AgentState) -> str:
    if state.get("rag_sucesso"):
        return "ok"
    return "precisa_triagem"


def decidir_pos_triagem(state: AgentState) -> str:
    decisao = state["triagem"]["decisao"]
    if decisao == "ABRIR_CHAMADO":
        return "chamado"
    return "info"


_workflow = StateGraph(AgentState)
_workflow.add_node("triagem", node_triagem)
_workflow.add_node("auto_resolver", node_auto_resolver)
_workflow.add_node("pedir_info", node_pedir_info)
_workflow.add_node("abrir_chamado", node_abrir_chamado)

# Tenta responder com base no(s) PDF(s) indexado(s) primeiro. Só passa pela
# triagem (pedir mais informações / abrir chamado) se o RAG não encontrar
# nada relevante no documento.
_workflow.add_edge(START, "auto_resolver")
_workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "ok": END,
    "precisa_triagem": "triagem",
})
_workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
})
_workflow.add_edge("pedir_info", END)
_workflow.add_edge("abrir_chamado", END)

grafo = _workflow.compile()


def perguntar_agente(pergunta: str) -> dict:
    """Ponto de entrada único usado pela API: roda triagem + RAG + decisão."""
    resultado = grafo.invoke({"pergunta": pergunta})
    return {
        "resposta": resultado.get("resposta"),
        "citacoes": resultado.get("citacoes", []),
        "acao_final": resultado.get("acao_final"),
        "triagem": resultado.get("triagem", {}),
    }
