const statusEl = document.getElementById("status");
const messageEl = document.getElementById("message");
const folderEl = document.getElementById("folder");
const aulaEl = document.getElementById("aula");
const resumoEl = document.getElementById("resumo");
const lastEl = document.getElementById("last-text");
const pendingsEl = document.getElementById("pendings");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnSkip = document.getElementById("btn-skip");
const btnPendings = document.getElementById("btn-pendings");
const saveAudio = document.getElementById("save-audio");
const micDevice = document.getElementById("mic-device");
const systemDevice = document.getElementById("system-device");
const alertsEl = document.getElementById("alerts");

let defaultsApplied = false;
const notified = new Set();

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

function showAlerts(alerts) {
  alertsEl.innerHTML = "";
  for (const alert of alerts) {
    const p = document.createElement("p");
    p.textContent = alert.text;
    alertsEl.appendChild(p);
  }
  alertsEl.hidden = !alerts.length;
  document.title = alerts.length ? `⚠ (${alerts.length}) STUDYARD` : "STUDYARD";
  const keys = new Set(alerts.map((a) => a.key));
  for (const key of [...notified]) {
    if (!keys.has(key)) notified.delete(key);
  }
  for (const alert of alerts) {
    if (notified.has(alert.key)) continue;
    notified.add(alert.key);
    /* notificação do Windows: o aviso chega mesmo com outra janela na frente */
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("STUDYARD", { body: alert.text, tag: alert.key });
    }
  }
}

function formatElapsed(total) {
  const s = Math.max(0, Math.floor(Number(total) || 0));
  const m = Math.floor(s / 60);
  return `${m} min ${s % 60} s`;
}

function render(data) {
  showAlerts(data.alerts || []);
  statusEl.textContent = data.state || "ocioso";
  if (data.state === "summarizing" && data.progress) {
    const p = data.progress;
    messageEl.textContent =
      `Resumindo parte ${p.part} de ${p.parts} — ${p.tokens} tokens — ${formatElapsed(p.elapsed_s)}`;
  } else {
    messageEl.textContent = data.message || "";
  }
  folderEl.textContent = data.folder || "—";
  aulaEl.textContent = data.aula || "—";
  resumoEl.textContent = data.resumo || "—";
  lastEl.textContent = data.last_text || "";
  const finishing = !!data.finishing;
  btnStart.disabled = !!data.recording || finishing;
  btnStop.disabled = !data.recording || finishing;
  btnSkip.hidden = data.state !== "summarizing";
  btnPendings.disabled = finishing || data.state === "summarizing";
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
  try {
    const res = await fetch("/api/status");
    render(await res.json());
  } catch (err) {
    showAlerts([
      {
        key: "app-down",
        text: "O STUDYARD parou de responder: a gravação pode ter sido interrompida. Abra o app de novo.",
      },
    ]);
  }
}

async function start() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
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
  btnStop.disabled = true;
  btnStart.disabled = true;
  statusEl.textContent = "stopping";
  messageEl.textContent = "Finalizando transcrição — o resumo começa em seguida.";
  const res = await fetch("/api/stop", { method: "POST" });
  const data = await res.json();
  if (res.status === 409) {
    await refresh();
    return;
  }
  render(data);
}

async function skipSummary() {
  btnSkip.disabled = true;
  const res = await fetch("/api/skip-summary", { method: "POST" });
  const data = await res.json();
  if (res.status === 409) {
    await refresh();
    return;
  }
  render(data);
}

async function processPendings() {
  btnPendings.disabled = true;
  const res = await fetch("/api/process-pendings", { method: "POST" });
  const data = await res.json();
  if (res.status === 409) {
    await refresh();
    return;
  }
  if (!res.ok) {
    messageEl.textContent = (data.errors || [data.message || "erro"]).join(" · ");
    btnPendings.disabled = false;
    return;
  }
  render(data);
}

btnStart.addEventListener("click", start);
btnStop.addEventListener("click", stop);
btnSkip.addEventListener("click", skipSummary);
btnPendings.addEventListener("click", processPendings);
setInterval(refresh, 1000);
loadDevices();
refresh();
