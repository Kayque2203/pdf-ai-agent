# Agente de Documentos (RAG + Gemini)

Agente de IA que lê PDFs e responde perguntas sobre o conteúdo, com citação de página. Feito com **FastAPI** (backend em Python) e **HTML/CSS/JS puro** (frontend), usando **LangChain + LangGraph + Gemini** para a parte de IA.

Este projeto nasceu de um notebook do Google Colab (curso *Imersão Agentes de IA — Alura/Google*) e foi reestruturado para rodar como uma aplicação local de verdade, sem depender do Colab.

## Como funciona

1. Você envia um ou mais PDFs pela interface web.
2. O backend quebra o(s) PDF(s) em pedaços (*chunks*), gera embeddings com o Gemini e indexa tudo num vetor de busca (FAISS).
3. Ao perguntar algo, o agente busca os trechos mais relevantes do PDF, monta uma resposta com o Gemini, e retorna a resposta junto com a página exata de onde veio a informação.
4. Se não encontrar nada relevante no documento, o agente pede mais detalhes em vez de inventar uma resposta.

## Stack

- **Backend:** FastAPI, LangChain, LangGraph, FAISS, PyMuPDF
- **IA:** Google Gemini (`gemini-3.5-flash` para chat, `gemini-embedding-001` para embeddings)
- **Frontend:** HTML + CSS + JavaScript puro (sem framework, sem build step)

## Estrutura do projeto

```
agente-pdf/
├── backend/
│   ├── main.py          # API FastAPI (endpoints /upload, /chat, /health)
│   ├── rag_agent.py      # Lógica de IA: triagem, RAG, orquestração com LangGraph
│   ├── requirements.txt
│   └── .env.example      # Modelo do arquivo de variáveis de ambiente
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Pré-requisitos

- Python 3.10 ou superior
- Uma chave de API do Gemini — crie a sua gratuitamente em [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Como rodar

### 1. Clonar e instalar as dependências

```bash
git clone <url-do-seu-repositorio>
cd agente-pdf/backend

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> No Windows, se o comando `pip` der erro de "Acesso negado", use `python -m pip install -r requirements.txt` no lugar.

### 2. Configurar a chave da API

Copie o arquivo de exemplo e cole sua chave:

```bash
cp .env.example .env
```

Edite o `.env` e preencha:

```
GEMINI_API_KEY=sua_chave_aqui
```

### 3. Subir o backend

```bash
uvicorn main:app --reload --port 8000
```

> No Windows, se `uvicorn` der erro de "Acesso negado", use `python -m uvicorn main:app --port 8000` (sem `--reload`, para evitar reinicializações desnecessárias).

O backend fica em `http://127.0.0.1:8000`. Teste em `http://127.0.0.1:8000/health` — deve responder `{"status": "ok", ...}`.

### 4. Abrir o frontend

Abra o arquivo `frontend/index.html` diretamente no navegador (duplo clique), ou use a extensão **Live Server** do VS Code para uma experiência mais estável.

> Se o frontend estiver hospedado em outro endereço (ex: Live Server usa `127.0.0.1:5500`), confira se a constante `API_URL` no topo de `frontend/script.js` aponta para o endereço correto do backend.

### 5. Usar

1. Envie um ou mais PDFs na barra lateral.
2. Clique em **Indexar documentos** e aguarde a confirmação.
3. Pergunte qualquer coisa sobre o conteúdo no chat.

> ⚠️ O índice fica apenas em memória. Toda vez que o backend for reiniciado, é necessário indexar o(s) PDF(s) novamente antes de perguntar.

## Rodando atrás de proxy corporativo (SSL/HTTPS interceptado)

Algumas redes de empresa usam um proxy (Zscaler, Palo Alto, Fortinet, etc.) que inspeciona o tráfego HTTPS e substitui os certificados por um certificado próprio da empresa. Se isso acontecer, você vai ver erros como:

```
SSL_ERROR_SSL: ... self signed certificate in certificate chain
```

Para resolver:

1. No navegador, acesse `https://generativelanguage.googleapis.com`, abra os detalhes do certificado (ícone de cadeado → certificado → caminho de certificação) e exporte o certificado **raiz** (o item mais no topo da hierarquia) no formato **Base-64 / PEM**. Salve como `backend/corporativo.cer`.
2. No terminal, dentro de `backend/` (com o ambiente virtual ativado), rode:

```bash
python -c "import certifi; print(certifi.where())"
```

3. Combine o certificado da empresa com a lista padrão do Python:

```bash
# Windows PowerShell
Get-Content .\venv\Lib\site-packages\certifi\cacert.pem, .\corporativo.cer | Set-Content combined_cert.pem

# macOS / Linux
cat venv/lib/python*/site-packages/certifi/cacert.pem corporativo.cer > combined_cert.pem
```

4. **Também** adicione o certificado direto dentro do arquivo do `certifi` (algumas bibliotecas do Google ignoram variáveis de ambiente e leem esse arquivo diretamente):

```bash
# Windows PowerShell
Get-Content .\corporativo.cer | Add-Content .\venv\Lib\site-packages\certifi\cacert.pem

# macOS / Linux
cat corporativo.cer >> venv/lib/python*/site-packages/certifi/cacert.pem
```

O `rag_agent.py` já detecta automaticamente o arquivo `combined_cert.pem` (se ele existir na pasta `backend/`) e configura as variáveis de ambiente necessárias antes de qualquer chamada à API — não precisa fazer mais nada além dos passos acima.

## Limitações conhecidas / próximos passos

- **Um índice por vez:** o índice de busca é global no processo do backend. Se duas pessoas usarem o mesmo backend ao mesmo tempo, o PDF de uma vai substituir o da outra. Para múltiplos usuários simultâneos, o ideal é implementar um índice por sessão (ex: usando um `session_id` gerado no frontend).
- **Índice não persiste:** reiniciar o backend apaga o índice da memória — é necessário reindexar os PDFs.
- **CORS aberto:** o backend libera `allow_origins=["*"]` para facilitar o desenvolvimento local. Antes de expor esse backend publicamente, restrinja para o domínio real do frontend.

## Licença

Uso interno / educacional. Adapte livremente conforme a necessidade do seu time.
