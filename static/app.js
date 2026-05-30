const form = document.getElementById("upload-form");
const fileInput = document.getElementById("xray-file");
const fileDropzone = document.getElementById("file-dropzone");
const fileName = document.getElementById("file-name");
const uploadState = document.getElementById("upload-state");
const preview = document.getElementById("preview");
const emptyPreview = document.getElementById("empty-preview");
const heatmapEmpty = document.getElementById("heatmap-empty");
const heatmapFigure = document.getElementById("heatmap-figure");
const heatmapImg = document.getElementById("heatmap");
const heatmapLabel = document.getElementById("heatmap-label");
const heatmapCaption = document.getElementById("heatmap-caption");
const statusChip = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
const modelInfo = document.getElementById("model-info");
const imageStatus = document.getElementById("image-status");
const messageEl = document.getElementById("message");
const reportEl = document.getElementById("report");
const resultsEl = document.getElementById("results");
const button = document.getElementById("analyze-button");
const buttonLabel = document.getElementById("button-label");

let thresholdsLoaded = false;
let thresholds = {};

setInitialState();
refreshModelInfo();

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    setInitialState();
    return;
  }

  fileName.textContent = file.name;
  fileDropzone.classList.add("has-file");
  uploadState.textContent = "Image selected - ready to analyze.";
  button.disabled = false;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";

  setChip(statusChip, "Ready", "");
  setChip(imageStatus, "Selected", "chip-ok");

  if (preview.src && preview.src.startsWith("blob:")) {
    URL.revokeObjectURL(preview.src);
  }
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyPreview.hidden = true;

  hideHeatmap();
  hideAlerts();
  renderEmpty("Upload an image to see findings", "Predictions will appear here after analysis.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    setInitialState();
    return;
  }

  const payload = new FormData();
  payload.append("file", file);
  setLoadingState();

  try {
    const response = await fetch("/api/predict", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }
    renderResponse(data);
  } catch (error) {
    setChip(statusChip, "Error", "chip-err");
    uploadState.textContent = "Analysis failed. Please try again.";
    showAlert(messageEl, error.message || "Prediction failed.");
    renderEmpty("No results", "The analysis did not complete.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    buttonLabel.textContent = "Analyze";
  }
});

function setInitialState() {
  fileName.textContent = "No file selected";
  fileDropzone.classList.remove("has-file");
  uploadState.textContent = "PNG, JPG, JPEG supported";
  button.disabled = true;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";
  setChip(statusChip, "Waiting", "");
  setChip(imageStatus, "Waiting", "");

  if (preview.src && preview.src.startsWith("blob:")) {
    URL.revokeObjectURL(preview.src);
  }
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
  buttonLabel.textContent = "Analyzing...";
  setChip(statusChip, "Analyzing", "chip-busy");
  uploadState.textContent = "Running model inference...";
  hideAlerts();
  hideHeatmap();
  renderEmpty("Running model", "Findings will appear here shortly.");
}

async function refreshModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) {
      throw new Error("Model info unavailable.");
    }
    const data = await response.json();
    const loaded = Boolean(data.model_loaded);
    thresholdsLoaded = Boolean(data.thresholds_loaded);
    thresholds = data.thresholds || {};
    modelStatus.textContent = loaded ? "Model loaded" : "Waiting for checkpoint";
    modelStatus.className = `badge ${loaded ? "badge-loaded" : "badge-waiting"}`;

    const info = data.model_info || {};
    const architecture = info.architecture || "DenseNet121";
    const labelCount = info.label_count || (Array.isArray(data.labels) ? data.labels.length : 5);
    const auc = info.mean_auc_display || formatNumber(info.mean_auc, 4) || "0.8764";
    const validRows = info.valid_rows || 202;
    const thresholdText = thresholdsLoaded ? "thresholds loaded" : "probability only";
    modelInfo.textContent = `${architecture} / ${labelCount} labels / mean AUC ${auc} / valid rows ${validRows} / ${thresholdText}`;
  } catch {
    modelStatus.textContent = "Model status unknown";
    modelStatus.className = "badge badge-error";
    modelInfo.textContent = "DenseNet121 / 5 labels / mean AUC 0.8764 / valid rows 202";
  }
}

function renderResponse(data) {
  const findings = Array.isArray(data.findings) ? data.findings : [];
  const ok = data.status === "ok";

  setChip(statusChip, ok ? "Complete" : "No model", ok ? "chip-ok" : "chip-err");
  uploadState.textContent = ok ? "Analysis complete." : "Model checkpoint not loaded.";

  if (data.message) {
    showAlert(messageEl, data.message);
  }
  if (data.report) {
    showAlert(reportEl, data.report, "alert-info");
  }

  if (data.heatmap?.image_data_url) {
    const probability = clamp(data.heatmap.probability);
    heatmapImg.src = data.heatmap.image_data_url;
    setChip(heatmapLabel, `${data.heatmap.label || "Top finding"} ${pct(probability)}`, "chip-ok");
    heatmapCaption.textContent = "Visualization is for research interpretation only.";
    heatmapEmpty.hidden = true;
    heatmapFigure.hidden = false;
  } else {
    hideHeatmap();
  }

  if (!findings.length) {
    renderEmpty("No findings returned", "The API did not return prediction labels.");
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of findings) {
    fragment.append(buildFindingCard(item));
  }
  resultsEl.replaceChildren(fragment);
}

function buildFindingCard(item) {
  const probability = clamp(item.probability);
  const threshold = Number.isFinite(Number(item.threshold))
    ? Number(item.threshold)
    : Number.isFinite(Number(thresholds[item.label]))
      ? Number(thresholds[item.label])
      : null;
  const level = suspicionLevel(probability, threshold);
  const article = el("article", `finding-card ${level.cls}`);

  const titleDiv = el("div", "finding-title");
  const label = el("strong");
  label.textContent = item.label || "Unlabeled";

  const badge = el("span", "suspicion-badge");
  badge.textContent = level.label;

  const thresholdText = el("span", "threshold-note");
  thresholdText.textContent = threshold === null ? "No threshold loaded" : `Threshold ${pct(threshold)}`;

  const probabilityDiv = el("div", "probability");
  probabilityDiv.textContent = pct(probability);

  const track = el("div", "progress-track");
  const fill = el("div", "progress-fill");
  fill.style.width = pct(probability);
  track.append(fill);

  titleDiv.append(label, badge, thresholdText);
  article.append(titleDiv, probabilityDiv, track);
  return article;
}

function suspicionLevel(probability, threshold) {
  if (threshold === null) {
    return { cls: "", label: "Probability only" };
  }
  const highCutoff = Math.min(1, threshold + 0.15);
  if (probability >= highCutoff) {
    return { cls: "high", label: "High suspicion" };
  }
  if (probability >= threshold) {
    return { cls: "moderate", label: "Moderate suspicion" };
  }
  return { cls: "", label: "Low suspicion" };
}

function el(tag, className = "") {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  return node;
}

function clamp(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(1, Math.max(0, number)) : 0;
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "";
}

function setChip(chipEl, text, cls) {
  chipEl.textContent = text;
  chipEl.className = `chip${cls ? ` ${cls}` : ""}`;
}

function showAlert(node, text, cls = "alert-error") {
  node.textContent = text;
  node.className = `alert ${cls}`;
  node.hidden = false;
}

function hideAlerts() {
  messageEl.hidden = true;
  messageEl.textContent = "";
  reportEl.hidden = true;
  reportEl.textContent = "";
}

function hideHeatmap() {
  heatmapImg.removeAttribute("src");
  setChip(heatmapLabel, "Not generated", "");
  heatmapEmpty.hidden = false;
  heatmapFigure.hidden = true;
}

function renderEmpty(title, detail) {
  const wrap = el("div", "empty-state compact");
  const strong = el("strong");
  const span = el("span");
  strong.textContent = title;
  span.textContent = detail;
  wrap.append(strong, span);
  resultsEl.replaceChildren(wrap);
}
