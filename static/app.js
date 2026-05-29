/* ─── DOM refs ─── */
const form          = document.getElementById("upload-form");
const fileInput     = document.getElementById("xray-file");
const fileDropzone  = document.getElementById("file-dropzone");
const fileName      = document.getElementById("file-name");
const uploadState   = document.getElementById("upload-state");
const preview       = document.getElementById("preview");
const emptyPreview  = document.getElementById("empty-preview");
const heatmapEmpty  = document.getElementById("heatmap-empty");
const heatmapFigure = document.getElementById("heatmap-figure");
const heatmapImg    = document.getElementById("heatmap");
const heatmapLabel  = document.getElementById("heatmap-label");
const heatmapCaption= document.getElementById("heatmap-caption");
const statusChip    = document.getElementById("status");
const modelStatus   = document.getElementById("model-status");
const imageStatus   = document.getElementById("image-status");
const messageEl     = document.getElementById("message");
const reportEl      = document.getElementById("report");
const resultsEl     = document.getElementById("results");
const button        = document.getElementById("analyze-button");
const buttonLabel   = document.getElementById("button-label");

/* ─── Init ─── */
setInitialState();
refreshModelStatus();

/* ─── File picker ─── */
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) { setInitialState(); return; }

  fileName.textContent = file.name;
  fileDropzone.classList.add("has-file");
  uploadState.textContent = "Image selected — ready to analyze.";
  button.disabled = false;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";

  setChip(statusChip, "Ready", "");
  setChip(imageStatus, "Selected", "chip-ok");

  if (preview.src && preview.src.startsWith("blob:")) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyPreview.hidden = true;

  hideHeatmap();
  hideAlerts();
  renderEmpty("Upload an image to see findings", "Predictions will appear here after analysis.");
});

/* ─── Form submit ─── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) { setInitialState(); return; }

  const payload = new FormData();
  payload.append("file", file);
  setLoadingState();

  try {
    const res  = await fetch("/api/predict", { method: "POST", body: payload });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    renderResponse(data);
  } catch (err) {
    setChip(statusChip, "Error", "chip-err");
    uploadState.textContent = "Analysis failed. Please try again.";
    showAlert(messageEl, err.message || "Prediction failed.");
    renderEmpty("No results", "The analysis did not complete.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    buttonLabel.textContent = "Analyze";
  }
});

/* ─── States ─── */
function setInitialState() {
  fileName.textContent = "No file selected";
  fileDropzone.classList.remove("has-file");
  uploadState.textContent = "Choose a PNG, JPG, or JPEG image to begin.";
  button.disabled = true;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";
  setChip(statusChip,  "Waiting", "");
  setChip(imageStatus, "Waiting", "");

  if (preview.src && preview.src.startsWith("blob:")) URL.revokeObjectURL(preview.src);
  preview.removeAttribute("src");
  preview.hidden = true;
  emptyPreview.hidden = false;

  hideHeatmap();
  hideAlerts();
  renderEmpty("Upload an image to see findings", "Predictions will appear here after analysis.");
}

function setLoadingState() {
  button.disabled = true;
  button.classList.add("is-loading");
  buttonLabel.textContent = "Analyzing…";
  setChip(statusChip, "Analyzing", "chip-busy");
  uploadState.textContent = "Running model inference…";
  hideAlerts();
  hideHeatmap();
  renderEmpty("Running model", "Findings will appear here shortly.");
}

/* ─── Model status badge ─── */
async function refreshModelStatus() {
  try {
    const res  = await fetch("/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    const loaded = Boolean(data.model_loaded);
    modelStatus.textContent = loaded ? "✓ Model loaded" : "Waiting for checkpoint";
    modelStatus.className = "badge " + (loaded ? "badge-loaded" : "badge-waiting");
  } catch {
    modelStatus.textContent = "Model status unknown";
    modelStatus.className = "badge badge-error";
  }
}

/* ─── Render API response ─── */
function renderResponse(data) {
  const findings = Array.isArray(data.findings) ? data.findings : [];
  const ok = data.status === "ok";

  setChip(statusChip, ok ? "Complete" : "No model", ok ? "chip-ok" : "chip-err");
  uploadState.textContent = ok ? "Analysis complete." : "Model checkpoint not loaded.";

  if (data.message) showAlert(messageEl, data.message);
  if (data.report)  showAlert(reportEl,  data.report, "alert-info");

  if (data.heatmap?.image_data_url) {
    const prob = clamp(data.heatmap.probability);
    heatmapImg.src = data.heatmap.image_data_url;
    setChip(heatmapLabel, `${data.heatmap.label || "Top finding"} ${pct(prob)}`, "chip-ok");
    heatmapCaption.textContent = "Visualization is for research interpretation only.";
    heatmapEmpty.hidden  = true;
    heatmapFigure.hidden = false;
  } else {
    hideHeatmap();
  }

  if (!findings.length) {
    renderEmpty("No findings returned", "The API did not return prediction labels.");
    return;
  }

  const frag = document.createDocumentFragment();
  for (const item of findings) frag.append(buildFindingCard(item));
  resultsEl.replaceChildren(frag);
}

/* ─── Finding card ─── */
function buildFindingCard(item) {
  const prob     = clamp(item.probability);
  const level    = suspicionLevel(prob);
  const article  = el("article", `finding-card ${level.cls}`);

  const titleDiv = el("div", "finding-title");
  const label    = el("strong");
  label.textContent = item.label || "Unlabeled";

  const badge = el("span", "suspicion-badge");
  badge.textContent = level.label;

  const probDiv = el("div", "probability");
  probDiv.textContent = pct(prob);

  const track = el("div", "progress-track");
  const fill  = el("div", "progress-fill");
  fill.style.width = pct(prob);
  track.append(fill);

  titleDiv.append(label, badge);
  article.append(titleDiv, probDiv, track);
  return article;
}

function suspicionLevel(p) {
  if (p >= 0.65) return { cls: "high",     label: "High suspicion" };
  if (p >= 0.35) return { cls: "moderate", label: "Moderate suspicion" };
  return               { cls: "",          label: "Low suspicion" };
}

/* ─── Helpers ─── */
function el(tag, className = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function clamp(v)  { const n = Number(v); return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0; }
function pct(v)    { return `${Math.round(v * 100)}%`; }

function setChip(chipEl, text, cls) {
  chipEl.textContent = text;
  chipEl.className   = "chip" + (cls ? " " + cls : "");
}

function showAlert(node, text, cls = "alert-error") {
  node.textContent = text;
  node.className   = `alert ${cls}`;
  node.hidden      = false;
}

function hideAlerts() {
  messageEl.hidden = true; messageEl.textContent = "";
  reportEl.hidden  = true; reportEl.textContent  = "";
}

function hideHeatmap() {
  heatmapImg.removeAttribute("src");
  setChip(heatmapLabel, "Not generated", "");
  heatmapEmpty.hidden  = false;
  heatmapFigure.hidden = true;
}

function renderEmpty(title, detail) {
  const wrap   = el("div", "empty-state compact");
  const strong = el("strong");
  const span   = el("span");
  strong.textContent = title;
  span.textContent   = detail;
  wrap.append(strong, span);
  resultsEl.replaceChildren(wrap);
}
