const statusEl = document.getElementById("status");
const messageEl = document.getElementById("message");
const folderEl = document.getElementById("folder");
const aulaEl = document.getElementById("aula");
const resumoEl = document.getElementById("resumo");
const lastEl = document.getElementById("last-text");
const pendingsEl = document.getElementById("pendings");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnPendings = document.getElementById("btn-pendings");
const saveAudio = document.getElementById("save-audio");
const micDevice = document.getElementById("mic-device");
const systemDevice = document.getElementById("system-device");

let defaultsApplied = false;

function source() {
  const el = document.querySelector('input[name="source"]:checked');
  return el ? el.value : "mic";
}

function fillSelect(el, items, allLabel) {
  const current = el.value;
  el.innerHTML = "";
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = allLabel;
  el.appendChild(all);
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item.name;
    const tag = item.kind === "analog" ? "entrada P2" : item.kind;
    opt.textContent = tag ? `${item.name} (${tag})` : item.name;
    el.appendChild(opt);
  }
  if ([...el.options].some((o) => o.value === current)) {
    el.value = current;
  }
}

async function loadDevices() {
  try {
    const res = await fetch("/api/devices");
    const data = await res.json();
    fillSelect(micDevice, data.mics || [], "Todos (entrada P2 + USB + array)");
    fillSelect(systemDevice, data.loopbacks || [], "Todos os alto-falantes");
  } catch (err) {
    /* lista de dispositivos é opcional; Gravar ainda funciona */
  }
}

function render(data) {
  statusEl.textContent = data.state || "ocioso";
  messageEl.textContent = data.message || "";
  folderEl.textContent = data.folder || "—";
  aulaEl.textContent = data.aula || "—";
  resumoEl.textContent = data.resumo || "—";
  lastEl.textContent = data.last_text || "";
  btnStart.disabled = !!data.recording;
  btnStop.disabled = !data.recording;
  if (!defaultsApplied && typeof data.save_audio_default === "boolean") {
    saveAudio.checked = data.save_audio_default;
    defaultsApplied = true;
  }
  pendingsEl.innerHTML = "";
  const list = data.pendings || [];
  if (!list.length) {
    const li = document.createElement("li");
    li.textContent = "Nenhuma pendência";
    pendingsEl.appendChild(li);
  } else {
    for (const item of list) {
      const li = document.createElement("li");
      li.textContent = `${item.stem} (${(item.need || []).join(", ")})`;
      pendingsEl.appendChild(li);
    }
  }
}

async function refresh() {
  const res = await fetch("/api/status");
  render(await res.json());
}

async function start() {
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source: source(),
      save_audio: saveAudio.checked,
      mic_device: micDevice.value,
      system_device: systemDevice.value,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    const text = (data.errors || [data.message || "erro"]).join(" · ");
    statusEl.textContent = "erro";
    messageEl.textContent = text;
    return;
  }
  render(data);
}

async function stop() {
  const res = await fetch("/api/stop", { method: "POST" });
  render(await res.json());
}

async function processPendings() {
  const res = await fetch("/api/process-pendings", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    messageEl.textContent = (data.errors || [data.message || "erro"]).join(" · ");
    return;
  }
  render(data);
}

btnStart.addEventListener("click", start);
btnStop.addEventListener("click", stop);
btnPendings.addEventListener("click", processPendings);
setInterval(refresh, 1000);
loadDevices();
refresh();
