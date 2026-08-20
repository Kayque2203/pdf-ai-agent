"""
API do Agente de IA para leitura de PDFs.

Rodar localmente:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Depois é só abrir o frontend/index.html no navegador (ele chama esta API em
http://localhost:8000).
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_agent

app = FastAPI(title="Agente de IA - Leitor de Documentos")

# Libera o acesso a partir do arquivo HTML (rodando localmente ou em outro domínio).
# Em produção dentro da empresa, troque "*" pelo domínio real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pasta onde os arquivos enviados ficam salvos nesta sessão do servidor.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "agente_pdf_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class Pergunta(BaseModel):
    pergunta: str
    # Últimas trocas da conversa (opcional), no formato
    # [{"pergunta": "...", "resposta": "..."}], para permitir perguntas de
    # acompanhamento que fazem referência ao que já foi perguntado.
    historico: Optional[List[Dict[str, str]]] = None


@app.get("/health")
def health():
    return {"status": "ok", **rag_agent.status_indice()}


@app.post("/reset")
def reset():
    """Apaga o índice atual (memória + disco), para indexar outro conteúdo do zero."""
    rag_agent.limpar_indice()
    return {"ok": True}


@app.get("/upload-progress")
def upload_progress():
    """Consultado pelo frontend enquanto um /upload está em andamento, para
    mostrar uma barra/mensagem de progresso em vez de uma tela travada."""
    return rag_agent.obter_progresso()


@app.post("/upload")
def upload_pdfs(files: List[UploadFile] = File(default=[]), url: Optional[str] = Form(default=None)):
    """Recebe um ou mais arquivos (PDF/DOCX/TXT/MD/CSV/XLSX) e/ou um link,
    salva e reconstrói o índice de busca (RAG).

    Definido como função síncrona (não "async def") de propósito: o FastAPI
    roda funções síncronas numa thread separada automaticamente, o que
    libera o servidor para responder a outras requisições (como
    /upload-progress) enquanto a indexação — que pode levar minutos em
    arquivos grandes — está rodando.
    """
    if not files and not url:
        raise HTTPException(status_code=400, detail="Nenhum arquivo ou link enviado.")

    caminhos_salvos = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in rag_agent.SUPPORTED_EXTENSIONS:
            tipos = ", ".join(sorted(rag_agent.SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}': tipo de arquivo não suportado. Tipos aceitos: {tipos}",
            )
        destino = UPLOAD_DIR / f.filename
        with destino.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        caminhos_salvos.append(destino)

    try:
        resultado = rag_agent.indexar_fontes(caminhos_salvos, [url] if url else [])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao indexar: {e.__class__.__name__}: {e}",
        )

    if not resultado["ok"]:
        raise HTTPException(status_code=422, detail=resultado["detalhe"])
    return resultado


@app.post("/chat")
def chat(payload: Pergunta):
    """Recebe uma pergunta (e opcionalmente o histórico recente) e retorna a
    resposta do agente com base no que foi indexado."""
    if not payload.pergunta.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")
    return rag_agent.perguntar_agente(payload.pergunta, payload.historico)
