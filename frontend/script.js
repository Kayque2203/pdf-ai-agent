const API_URL = "http://127.0.0.1:8000"; // troque pela URL do backend quando hospedar

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const indexBtn = document.getElementById("index-btn");
const fileListEl = document.getElementById("file-list");
const indexStatusEl = document.getElementById("index-status");
const resetBtn = document.getElementById("reset-btn");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messagesEl = document.getElementById("messages");

let selectedFiles = [];

// ---------- Checar se já existe um índice salvo (de uma sessão anterior) ----------

async function checarIndiceExistente() {
  try {
    const res = await fetch(`${API_URL}/health`);
    const data = await res.json();
    if (data.indexado) {
      indexStatusEl.className = "index-status ok";
      indexStatusEl.textContent = `✓ Já indexado: ${data.arquivos.join(", ")}`;
      if (resetBtn) resetBtn.style.display = "block";
      chatInput.disabled = false;
      sendBtn.disabled = false;
      addMessage("system", "Documento já indexado de uma sessão anterior. Pode perguntar direto!");
    }
  } catch (err) {
    // Backend ainda não está no ar — sem problema, o usuário vai indexar normalmente.
  }
}

resetBtn?.addEventListener("click", async () => {
  try {
    await fetch(`${API_URL}/reset`, { method: "POST" });
  } catch (err) {
    // segue o baile mesmo se der erro de rede
  }
  if (resetBtn) resetBtn.style.display = "none";
  indexStatusEl.className = "index-status";
  indexStatusEl.textContent = "";
  chatInput.disabled = true;
  sendBtn.disabled = true;
  selectedFiles = [];
  renderFileList();
  indexBtn.disabled = true;
  addMessage("system", "Índice apagado. Envie um novo PDF para indexar.");
});

checarIndiceExistente();

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
  const todos = Array.from(fileListRaw);
  const novos = todos.filter(f => f.type === "application/pdf");
  const rejeitados = todos.filter(f => f.type !== "application/pdf");

  selectedFiles = [...selectedFiles, ...novos];
  renderFileList();
  indexBtn.disabled = selectedFiles.length === 0;

  if (rejeitados.length > 0) {
    indexStatusEl.className = "index-status error";
    const nomes = rejeitados.map(f => f.name).join(", ");
    indexStatusEl.textContent = `Ignorado (não é PDF): ${nomes}`;
  }
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
    if (resetBtn) resetBtn.style.display = "block";

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
    const tagClass = { AUTO_RESOLVER: "tag-auto", SEM_CONTEXTO: "tag-info" }[acaoFinal] || "tag-info";
    tagEl.className = `tag ${tagClass}`;
    tagEl.textContent = acaoFinal.replace("_", " ");
    div.appendChild(tagEl);
    div.appendChild(document.createElement("br"));
  }

  div.appendChild(document.createTextNode(text));

  if (citacoes && citacoes.length > 0) {
    const cWrap = document.createElement("div");
    cWrap.className = "citations";

    // Agrupa as páginas por documento, para não atribuir uma citação de um
    // PDF ao nome de outro quando há mais de um documento indexado.
    const porDocumento = new Map();
    citacoes.forEach(c => {
      if (!porDocumento.has(c.documento)) porDocumento.set(c.documento, []);
      porDocumento.get(c.documento).push(c.pagina);
    });
    const partes = [...porDocumento.entries()].map(
      ([doc, paginas]) => `${doc} — pág. ${paginas.join(", ")}`
    );

    const resumo = document.createElement("div");
    resumo.className = "citations-summary";
    resumo.textContent = `📎 Fonte: ${partes.join(" · ")}`;
    cWrap.appendChild(resumo);

    const detalhes = document.createElement("details");
    detalhes.className = "citations-details";
    const sumario = document.createElement("summary");
    sumario.textContent = "Ver trecho original";
    detalhes.appendChild(sumario);

    citacoes.forEach(c => {
      const cEl = document.createElement("div");
      cEl.className = "citation";
      const paginaEl = document.createElement("b");
      paginaEl.textContent = `${c.documento} · pág. ${c.pagina}`;
      cEl.appendChild(paginaEl);
      cEl.appendChild(document.createTextNode(` — "${c.trecho}..."`));
      detalhes.appendChild(cEl);
    });
    cWrap.appendChild(detalhes);

    div.appendChild(cWrap);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}
