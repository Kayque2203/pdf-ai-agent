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
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_agent

app = FastAPI(title="Agente de IA - Leitor de PDFs")

# Libera o acesso a partir do arquivo HTML (rodando localmente ou em outro domínio).
# Em produção dentro da empresa, troque "*" pelo domínio real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pasta onde os PDFs enviados ficam salvos nesta sessão do servidor.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "agente_pdf_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class Pergunta(BaseModel):
    pergunta: str


@app.get("/health")
def health():
    return {"status": "ok", **rag_agent.status_indice()}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Recebe um ou mais PDFs, salva e reconstrói o índice de busca (RAG)."""
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    caminhos_salvos = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{f.filename}' não é um PDF.")
        destino = UPLOAD_DIR / f.filename
        with destino.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        caminhos_salvos.append(destino)

    resultado = rag_agent.indexar_pdfs(caminhos_salvos)
    if not resultado["ok"]:
        raise HTTPException(status_code=422, detail=resultado["detalhe"])
    return resultado


@app.post("/chat")
def chat(payload: Pergunta):
    """Recebe uma pergunta e retorna a resposta do agente (triagem + RAG)."""
    if not payload.pergunta.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")
    return rag_agent.perguntar_agente(payload.pergunta)
