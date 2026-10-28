const $ = (id) => document.getElementById(id);
const state = { mode: "recorded", samples: [], currentFile: null, drive: null, response: null, channel: 0, running: false };
const DRIVE_CHANNELS = ["R", "G", "B", "ON_R", "ON_G", "ON_B", "OFF_R", "OFF_G", "OFF_B"];
const RESPONSE_CHANNELS = ["R", "G", "B", "ON-red", "ON-green", "ON-blue", "OFF-red", "OFF-green", "OFF-blue"];
const DRIVE_COLORS = ["#7ee8a2", "#9ff0b8", "#c6f7d4", "#ffc66d", "#ffd792", "#ffe8ba", "#89baff", "#a8cdff", "#c9e1ff"];
const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function notify(message, isError = false) {
  const el = $("notice");
  el.textContent = message || "";
  el.classList.toggle("show", Boolean(message));
  el.classList.toggle("error", Boolean(message) && isError);
}

function setPill(id, ready, text) {
  const el = $(id);
  el.classList.toggle("ready", Boolean(ready));
  el.classList.toggle("warn", !ready);
  el.innerHTML = `<i class="dot"></i>${text}`;
}

function shortClass(name) {
  return String(name || "").replaceAll("___", " / ").replaceAll("_", " ");
}

function fmtNumber(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

async function boot() {
  try {
    const status = await api("/api/status");
    const byName = Object.fromEntries(status.layers.map((layer) => [layer.name, layer]));
    setPill("pill-processing", byName.processing?.ready, "处理层 · 就绪");
    setPill("pill-hardware", byName.hardware?.ready,
      `硬件层 · ${status.hardware.mode === "http" ? "HTTP 实采" : status.hardware.mode === "recorded" ? "实测回放" : "未连接"}`);
    setPill("pill-decision", byName.decision?.ready,
      `决策层 · ${byName.decision?.ready ? "模型就绪" : "模型未安装"}`);
    $("footer-status").innerHTML = status.recorded_dataset.ready
      ? `录制样本 <code>${status.recorded_dataset.sample_count}</code> 条 · 模型 <code>${status.decision.model}</code>`
      : `录制数据未安装 · 模型 <code>${status.decision.model}</code>`;
    if (status.recorded_dataset.ready) {
      const data = await api("/api/samples");
      state.samples = data.samples;
      populateSamples();
    } else {
      $("sample-select").innerHTML = '<option value="">无录制样本</option>';
      notify(status.recorded_dataset.error || "未找到录制数据，可上传图片使用处理层。", true);
    }
  } catch (error) {
    notify(`无法连接本地 API：${error.message}`, true);
    $("footer-status").textContent = "API 离线";
  }
}

function populateSamples() {
  const select = $("sample-select");
  select.innerHTML = state.samples.map((sample) => {
    const index = String(sample.index).padStart(3, "0");
    return `<option value="${sample.index}">#${index} · ${shortClass(sample.class_name)}</option>`;
  }).join("");
  select.onchange = () => updateSampleMeta();
  updateSampleMeta();
}

function updateSampleMeta() {
  const sample = state.samples[Number($("sample-select").value)];
  $("sample-meta").innerHTML = sample
    ? `<span>ID · ${sample.sample_id.split("/").pop()}</span><span>真实类 · ${shortClass(sample.class_name)}</span>`
    : "<span>未选择样本</span>";
}

function switchMode(mode) {
  state.mode = mode;
  $("tab-recorded").classList.toggle("active", mode === "recorded");
  $("tab-upload").classList.toggle("active", mode === "upload");
  $("recorded-source").style.display = mode === "recorded" ? "" : "none";
  $("upload-source").style.display = mode === "upload" ? "" : "none";
  $("run-button").textContent = mode === "recorded" ? "运行完整链路" : "处理并请求硬件";
  notify("");
}

function bindSourceControls() {
  $("tab-recorded").onclick = () => switchMode("recorded");
  $("tab-upload").onclick = () => switchMode("upload");
  const zone = $("drop-zone"), input = $("file-input"), preview = $("upload-preview");
  zone.onclick = () => input.click();
  zone.ondragover = (event) => { event.preventDefault(); zone.classList.add("dragging"); };
  zone.ondragleave = () => zone.classList.remove("dragging");
  zone.ondrop = (event) => {
    event.preventDefault(); zone.classList.remove("dragging");
    const file = event.dataTransfer.files?.[0];
    if (file) loadFile(file);
  };
  input.onchange = () => { if (input.files?.[0]) loadFile(input.files[0]); };
  async function loadFile(file) {
    if (!file.type.startsWith("image/")) { notify("请选择 PNG / JPG / WEBP 图片。", true); return; }
    state.currentFile = file;
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    zone.querySelector("strong").textContent = file.name;
    zone.querySelector("small").textContent = `${(file.size / 1024).toFixed(0)} KB · 已载入处理层输入`;
    notify("");
  }
  $("run-button").onclick = runPipeline;
  $("reset-button").onclick = () => { if (state.mode === "recorded") runRecorded(); else notify("选择图片后点击运行。"); };
}

function base64FromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function runPipeline() {
  if (state.running) return;
  state.running = true;
  const button = $("run-button");
  const previous = button.textContent;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>运行中';
  notify("处理层计算中；随后按当前硬件后端获取响应。");
  try {
    let payload;
    if (state.mode === "recorded") {
      payload = await api("/api/run", { method: "POST", body: JSON.stringify({ source: "recorded", sample_index: Number($("sample-select").value) }) });
    } else {
      if (!state.currentFile) throw new Error("请先选择一张叶片图片。");
      const imageBase64 = await base64FromFile(state.currentFile);
      payload = await api("/api/run", { method: "POST", body: JSON.stringify({ source: "upload", image_base64: imageBase64, filename: state.currentFile.name }) });
    }
    renderResult(payload);
    notify(payload.source === "recorded"
      ? `完成：回放真实实测电流，决策耗时 ${fmtNumber(payload.timings_ms.decision_ms)} ms。`
      : "完成：处理、硬件采集与决策均已返回。");
  } catch (error) {
    notify(error.message, true);
  } finally {
    state.running = false;
    button.disabled = false;
    button.textContent = previous;
  }
}

function renderResult(payload) {
  state.drive = payload.drive;
  state.response = payload.response;
  renderDrive(payload.drive, payload.timings_ms || {});
  renderResponse(payload.response, payload.timings_ms || {});
  renderDecision(payload.decision);
}

function prepareCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(canvas.clientWidth, 280);
  const height = canvas.clientHeight || 150;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function renderDrive(drive, timings) {
  $("drive-empty").style.display = "none";
  $("drive-shape").textContent = drive.shape.join("×");
  $("drive-steps").textContent = drive.shape[0];
  $("drive-ms").textContent = `${fmtNumber(timings.processing_ms)} ms`;
  const { ctx, width, height } = prepareCanvas($("drive-canvas"));
  const pad = { left: 34, right: 12, top: 10, bottom: 18 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.font = `9px ${MONO}`;
  ctx.fillStyle = "rgba(138,164,150,.8)";
  for (let value = 0; value <= 1.001; value += .25) {
    const y = pad.top + (1 - value) * plotH;
    ctx.strokeStyle = "rgba(170,220,186,.10)";
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(value.toFixed(2), 5, y + 3);
  }
  for (let channel = 0; channel < 9; channel++) {
    ctx.beginPath();
    ctx.strokeStyle = DRIVE_COLORS[channel];
    ctx.lineWidth = channel < 3 ? 1.6 : 1.1;
    ctx.globalAlpha = channel < 3 ? .95 : .8;
    drive.frames.forEach((frame, step) => {
      const x = pad.left + step / 63 * plotW;
      const y = pad.top + (1 - frame[channel]) * plotH;
      if (step === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "rgba(138,164,150,.8)";
  [0, 16, 32, 48, 63].forEach((step) => ctx.fillText(String(step + 1), pad.left + step / 63 * plotW - 4, height - 4));
  const link = $("download-drive");
  if (link.href.startsWith("blob:")) URL.revokeObjectURL(link.href);
  link.href = URL.createObjectURL(driveCsv(drive));
}

function driveCsv(drive) {
  const lines = ["step," + DRIVE_CHANNELS.join(",")];
  drive.frames.forEach((frame, step) => lines.push(`${step + 1},${frame.map((v) => Number(v).toFixed(7)).join(",")}`));
  return new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
}

function renderResponse(response, timings) {
  $("response-empty").style.display = "none";
  $("response-backend").textContent = response.backend;
  $("response-mean").textContent = `${fmtNumber(response.mean_current_uA[state.channel], 4)} µA`;
  $("hardware-badge").textContent = response.measured ? "MEASURED" : "SIMULATED";
  const tabs = $("channel-tabs");
  tabs.innerHTML = RESPONSE_CHANNELS.map((name, index) =>
    `<button type="button" class="${index === state.channel ? "active" : ""}" data-channel="${index}">${name}</button>`
  ).join("");
  tabs.querySelectorAll("button").forEach((button) => {
    button.onclick = () => { state.channel = Number(button.dataset.channel); renderResponse(response, timings); };
  });
  const { ctx, width, height } = prepareCanvas($("response-canvas"));
  const pad = { left: 42, right: 10, top: 8, bottom: 20 };
  const matrix = response.currents_amperes[state.channel];
  const values = matrix.flat();
  let min = Math.min(...values), max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min < 1e-15) { min = 0; max = 1; }
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const cellW = plotW / matrix[0].length, cellH = plotH / matrix.length;
  for (let row = 0; row < matrix.length; row++) {
    for (let point = 0; point < matrix[row].length; point++) {
      const t = (matrix[row][point] - min) / (max - min);
      ctx.fillStyle = `rgb(${Math.round(20 + t * 235)},${Math.round(55 + t * 143)},${Math.round(85 + t * 25)})`;
      ctx.fillRect(pad.left + point * cellW, pad.top + row * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
  ctx.font = `8px ${MONO}`;
  ctx.fillStyle = "rgba(138,164,150,.9)";
  for (let row = 0; row < matrix.length; row++) ctx.fillText(`r${String(row + 1).padStart(2, "0")}`, 8, pad.top + (row + .7) * cellH);
  for (let point = 0; point < 65; point += 16) ctx.fillText(String(point + 1), pad.left + point * cellW - 3, height - 4);
}

function renderDecision(decision) {
  $("decision-badge").textContent = decision.model.toUpperCase();
  $("predicted-class").textContent = shortClass(decision.predicted_class);
  $("confidence").textContent = `${(decision.probability * 100).toFixed(1)}%`;
  $("probability-bars").innerHTML = (decision.top_k || []).slice(0, 7).map((row) => `
    <div class="bar-row" title="${row.class_name}">
      <span class="name">${shortClass(row.class_name)}</span>
      <span class="bar-track"><span class="bar-fill" data-width="${(row.probability * 100).toFixed(3)}"></span></span>
      <span class="bar-value">${(row.probability * 100).toFixed(1)}%</span>
    </div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar-fill").forEach((bar) => { bar.style.width = `${bar.dataset.width}%`; }));
}

function runRecorded() {
  if (state.mode === "recorded" && state.samples.length) runPipeline();
}

bindSourceControls();
boot().then(() => { if (state.samples.length) setTimeout(runRecorded, 350); });
window.addEventListener("resize", () => {
  if (state.drive && state.response) { renderDrive(state.drive, {}); renderResponse(state.response, {}); }
});
