const API_URL = "http://127.0.0.1:8000"; // troque pela URL do backend quando hospedar

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const indexBtn = document.getElementById("index-btn");
const fileListEl = document.getElementById("file-list");
const indexStatusEl = document.getElementById("index-status");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messagesEl = document.getElementById("messages");

let selectedFiles = [];

// ---------- Seleção de arquivos ----------

fileInput.addEventListener("change", (e) => {
  addFiles(e.target.files);
});

["dragover", "dragenter"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  addFiles(e.dataTransfer.files);
});

function addFiles(fileListRaw) {
  const novos = Array.from(fileListRaw).filter(f => f.type === "application/pdf");
  selectedFiles = [...selectedFiles, ...novos];
  renderFileList();
  indexBtn.disabled = selectedFiles.length === 0;
}

function renderFileList() {
  fileListEl.innerHTML = "";
  selectedFiles.forEach(f => {
    const item = document.createElement("div");
    item.className = "file-item";
    item.innerHTML = `<span>${f.name}</span><span class="size">${formatBytes(f.size)}</span>`;
    fileListEl.appendChild(item);
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------- Upload / indexação ----------

indexBtn.addEventListener("click", async () => {
  if (selectedFiles.length === 0) return;

  indexBtn.disabled = true;
  indexBtn.textContent = "Indexando...";
  indexStatusEl.className = "index-status";
  indexStatusEl.textContent = "Lendo e indexando os PDFs. Isso pode levar alguns instantes em arquivos grandes...";

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append("files", f));

  try {
    const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Falha ao indexar os PDFs.");

    indexStatusEl.className = "index-status ok";
    indexStatusEl.textContent = `✓ ${data.arquivos.length} arquivo(s) indexado(s), ${data.total_chunks} trechos gerados.`;

    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();

    addMessage("system", "Documentos indexados. Pode perguntar!");
  } catch (err) {
    indexStatusEl.className = "index-status error";
    indexStatusEl.textContent = `Erro: ${err.message}`;
  } finally {
    indexBtn.disabled = false;
    indexBtn.textContent = "Indexar documentos";
  }
});

// ---------- Chat ----------

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const pergunta = chatInput.value.trim();
  if (!pergunta) return;

  addMessage("user", pergunta);
  chatInput.value = "";
  chatInput.disabled = true;
  sendBtn.disabled = true;

  const thinkingId = addMessage("agent", "Pensando...", null);

  try {
    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Erro ao consultar o agente.");

    updateMessage(thinkingId, data.resposta, data.acao_final, data.citacoes);
  } catch (err) {
    updateMessage(thinkingId, `Erro: ${err.message}`, null, []);
  } finally {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
});

let msgCounter = 0;

function addMessage(role, text, tag = undefined) {
  const id = `msg-${msgCounter++}`;
  const div = document.createElement("div");
  div.id = id;
  div.className = `message message-${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return id;
}

function updateMessage(id, text, acaoFinal, citacoes) {
  const div = document.getElementById(id);
  div.textContent = "";

  if (acaoFinal) {
    const tagEl = document.createElement("span");
    const tagClass = { AUTO_RESOLVER: "tag-auto", PEDIR_INFO: "tag-info", ABRIR_CHAMADO: "tag-chamado" }[acaoFinal] || "tag-info";
    tagEl.className = `tag ${tagClass}`;
    tagEl.textContent = acaoFinal.replace("_", " ");
    div.appendChild(tagEl);
    div.appendChild(document.createElement("br"));
  }

  div.appendChild(document.createTextNode(text));

  if (citacoes && citacoes.length > 0) {
    const cWrap = document.createElement("div");
    cWrap.className = "citations";
    citacoes.forEach(c => {
      const cEl = document.createElement("div");
      cEl.className = "citation";
      cEl.innerHTML = `<b>${c.documento}</b> · pág. ${c.pagina} — "${c.trecho}..."`;
      cWrap.appendChild(cEl);
    });
    div.appendChild(cWrap);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}
