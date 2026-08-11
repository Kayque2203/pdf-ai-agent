# Agente de Documentos (PDF + Gemini)

Mesma lógica do seu notebook do Colab (triagem + RAG sobre PDFs + LangGraph),
só que rodando como uma aplicação normal: **backend em Python (FastAPI)** +
**frontend em HTML/CSS/JS**.

```
agente-pdf/
├── backend/
│   ├── main.py          # API (endpoints /upload e /chat)
│   ├── rag_agent.py      # a lógica do seu notebook (Gemini + RAG + LangGraph)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## 1. Pré-requisitos

- Python 3.10+ instalado
- Uma chave de API do Gemini (a mesma que você usava no `userdata.get('GEMINI_API_KEY')`
  do Colab). Se ainda não tem: https://aistudio.google.com/app/apikey

> Se o seu notebook da Brainfarma usa uma chave corporativa/própria (Vertex AI,
> por exemplo, em vez da AI Studio API), me avise — a troca é só na parte de
> conexão do `rag_agent.py`, o resto continua igual.

## 2. Rodar o backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Crie o arquivo `.env` dentro de `backend/` (copie o `.env.example`) e cole sua chave:

```
GEMINI_API_KEY=sua_chave_aqui
```

Suba o servidor:

```bash
uvicorn main:app --reload --port 8000
```

Deixe esse terminal aberto — é ele que processa os PDFs e conversa com o Gemini.
Teste em http://localhost:8000/health (deve responder `{"status": "ok", ...}`).

## 3. Abrir o frontend

Basta abrir o arquivo `frontend/index.html` duas vezes no navegador (duplo clique,
ou clique direito → Abrir com → navegador). Não precisa de servidor web para isso,
já que ele conversa com o backend via `http://localhost:8000`.

## 4. Usar

1. Arraste um ou mais PDFs na área de upload (ou clique para escolher).
2. Clique em **Indexar documentos** — o backend lê o PDF, quebra em pedaços
   (chunks) e gera os embeddings (pode levar alguns segundos a minutos em
   arquivos grandes).
3. Pergunte no chat. O agente decide se responde direto (com base no PDF, com
   citação de página), pede mais informações, ou sinaliza abertura de chamado
   — exatamente como no seu notebook.

## Coisas para ajustar antes de usar na empresa

- **Rede da empresa**: se o notebook/computador da Brainfarma tiver proxy ou
  firewall corporativo, pode ser necessário liberar acesso a
  `generativelanguage.googleapis.com` (API do Gemini) e aos pacotes do `pip`.
  Se algo travar na instalação ou nas chamadas, isso é o primeiro lugar a checar
  com o time de TI.
- **Múltiplos usuários ao mesmo tempo**: hoje o índice (`_retriever` em
  `rag_agent.py`) é global — o último PDF indexado vale para todo mundo que
  usar o backend. Para uso individual (só você, no seu notebook) está ótimo.
  Se depois quiser que cada pessoa tenha seus próprios PDFs, dá pra evoluir
  para um índice por sessão — posso te ajudar quando chegar nessa etapa.
- **Hospedar de verdade** (em vez de só rodar local): o backend pode subir em
  qualquer servidor Linux da empresa ou em um serviço de nuvem (Render,
  Fly.io, Azure/GCP, conforme o que a Brainfarma já usa); o frontend, sendo
  puro HTML/CSS/JS, pode ser hospedado em qualquer lugar simples (até uma
  pasta compartilhada) — só ajustar `API_URL` no `script.js` para o endereço
  do backend hospedado.
- **Segurança**: como fica com dados internos da empresa, vale confirmar com
  o time de segurança/compliance se pode enviar esses PDFs para a API do
  Gemini (Google), dependendo da política de dados da Brainfarma.
