// -----------------------------------------
// CheXpert Medical AI Workstation - Engine
// -----------------------------------------

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("xray-file");
const fileDropzone = document.getElementById("file-dropzone");
const fileName = document.getElementById("file-name");
const uploadState = document.getElementById("upload-state");
const preview = document.getElementById("preview");
const emptyPreview = document.getElementById("empty-preview");

// PACS Elements
const pacsViewport = document.getElementById("pacs-viewport");
const pacsCanvasWrapper = document.getElementById("pacs-canvas-wrapper");
const btnZoomIn = document.getElementById("btn-zoom-in");
const btnZoomOut = document.getElementById("btn-zoom-out");
const btnZoomFit = document.getElementById("btn-zoom-fit");
const btnInvert = document.getElementById("btn-invert");
const btnResetView = document.getElementById("btn-reset-view");
const sliderBrightness = document.getElementById("slider-brightness");
const sliderContrast = document.getElementById("slider-contrast");
const valBrightness = document.getElementById("val-brightness");
const valContrast = document.getElementById("val-contrast");

// Heatmap Elements
const heatmapEmpty = document.getElementById("heatmap-empty");
const heatmapFigure = document.getElementById("heatmap-figure");
const heatmapBase = document.getElementById("heatmap-base");
const overlayUnderlay = document.getElementById("overlay-underlay");
const heatmapPure = document.getElementById("heatmap-pure");
const heatmapFallback = document.getElementById("heatmap");
const heatmapLabel = document.getElementById("heatmap-label");
const heatmapControlsBar = document.getElementById("heatmap-controls-bar");
const findingTabs = document.getElementById("finding-tabs");
const boxLabelFinding = document.getElementById("box-label-finding");
const sliderOpacity = document.getElementById("slider-opacity");
const valOpacity = document.getElementById("val-opacity");

// Status, DICOM, Export
const statusChip = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
const modelInfo = document.getElementById("model-info");
const qualityAlert = document.getElementById("quality-alert");
const messageEl = document.getElementById("message");
const reportEl = document.getElementById("report");
const resultsEl = document.getElementById("results");
const button = document.getElementById("analyze-button");
const buttonLabel = document.getElementById("button-label");
const exportPdfBtn = document.getElementById("export-pdf-btn");

// DICOM banner elements
const dicomBanner = document.getElementById("dicom-banner");
const dicomPatient = document.getElementById("dicom-patient");
const dicomModality = document.getElementById("dicom-modality");
const dicomView = document.getElementById("dicom-view");
const dicomDate = document.getElementById("dicom-date");
const dicomMatrix = document.getElementById("dicom-matrix");

// State
let thresholdsLoaded = false;
let thresholds = {};
let selectedFile = null;
let currentPredictionData = null;
let currentActiveFinding = null;

// PACS State
let zoomLevel = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;
let isInverted = false;
let brightnessVal = 100;
let contrastVal = 100;

setInitialState();
refreshModelInfo();
initPacsControls();

// -----------------------------------------
// PACS Image Viewer Controls
// -----------------------------------------
function initPacsControls() {
  btnZoomIn.addEventListener("click", () => setZoom(zoomLevel + 0.25));
  btnZoomOut.addEventListener("click", () => setZoom(zoomLevel - 0.25));
  btnZoomFit.addEventListener("click", resetPanZoom);
  btnInvert.addEventListener("click", () => {
    isInverted = !isInverted;
    btnInvert.style.background = isInverted ? "var(--primary)" : "";
    btnInvert.style.color = isInverted ? "#042f2e" : "";
    applyPacsFilters();
  });
  btnResetView.addEventListener("click", () => {
    resetPanZoom();
    isInverted = false;
    btnInvert.style.background = "";
    btnInvert.style.color = "";
    sliderBrightness.value = 100;
    sliderContrast.value = 100;
    valBrightness.textContent = "100%";
    valContrast.textContent = "100%";
    brightnessVal = 100;
    contrastVal = 100;
    applyPacsFilters();
  });

  sliderBrightness.addEventListener("input", (e) => {
    brightnessVal = e.target.value;
    valBrightness.textContent = `${brightnessVal}%`;
    applyPacsFilters();
  });

  sliderContrast.addEventListener("input", (e) => {
    contrastVal = e.target.value;
    valContrast.textContent = `${contrastVal}%`;
    applyPacsFilters();
  });

  // Pan & Drag
  pacsViewport.addEventListener("mousedown", (e) => {
    if (!preview.src || preview.hidden) return;
    isDragging = true;
    pacsViewport.classList.add("is-dragging");
    startX = e.clientX - panX;
    startY = e.clientY - panY;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    updateTransform();
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
    pacsViewport.classList.remove("is-dragging");
  });

  // Mouse wheel zoom
  pacsViewport.addEventListener("wheel", (e) => {
    if (!preview.src || preview.hidden) return;
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom(zoomLevel + delta);
  }, { passive: false });
}

function setZoom(val) {
  zoomLevel = Math.min(4.0, Math.max(0.5, val));
  updateTransform();
}

function resetPanZoom() {
  zoomLevel = 1.0;
  panX = 0;
  panY = 0;
  updateTransform();
}

function updateTransform() {
  pacsCanvasWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
}

function applyPacsFilters() {
  const invertStr = isInverted ? "invert(100%) " : "";
  const filterStr = `${invertStr}brightness(${brightnessVal}%) contrast(${contrastVal}%)`;
  preview.style.filter = filterStr;
  heatmapBase.style.filter = filterStr;
  overlayUnderlay.style.filter = filterStr;
}

// -----------------------------------------
// Heatmap Opacity Control
// -----------------------------------------
sliderOpacity.addEventListener("input", (e) => {
  const val = e.target.value;
  valOpacity.textContent = `${val}%`;
  heatmapPure.style.opacity = (val / 100).toString();
});

// -----------------------------------------
// File Selection & Drag/Paste
// -----------------------------------------
function applySelectedFile(file) {
  if (!file) return;
  selectedFile = file;

  try {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  } catch {}

  fileName.textContent = file.name || "radiograph.dcm";
  fileDropzone.classList.add("has-file");
  uploadState.textContent = "Radiograph loaded & ready to analyze.";
  button.disabled = false;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";

  setChip(statusChip, "Ready", "");
  resetPanZoom();

  // Create preview
  const isDicom = file.name.toLowerCase().endsWith(".dcm");
  if (!isDicom) {
    if (preview.src && preview.src.startsWith("blob:")) {
      URL.revokeObjectURL(preview.src);
    }
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    emptyPreview.hidden = true;
  }

  hideHeatmap();
  hideAlerts();
  dicomBanner.hidden = true;
  renderEmpty("Click Analyze to start diagnostic inference", "Multi-label predictions & Grad-CAM will generate.");
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    setInitialState();
    return;
  }
  applySelectedFile(file);
});

// Comprehensive Clipboard Paste (Ctrl+V)
async function handlePasteEvent(event) {
  const clipboardData = event.clipboardData || window.clipboardData;
  if (!clipboardData) return;

  const files = clipboardData.files;
  if (files && files.length > 0) {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|dcm)$/i.test(file.name)) {
        event.preventDefault();
        event.stopPropagation();
        applySelectedFile(file);
        return;
      }
    }
  }

  const items = clipboardData.items;
  if (items && items.length > 0) {
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith("image/") || item.kind === "file") {
        const blob = item.getAsFile();
        if (blob) {
          event.preventDefault();
          event.stopPropagation();
          const ext = (blob.type && blob.type.split("/")[1]) || "png";
          const dateStr = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
          const file = new File([blob], `pasted_cxr_${dateStr}.${ext}`, { type: blob.type || "image/png" });
          applySelectedFile(file);
          return;
        }
      }
    }
  }
}

window.addEventListener("paste", handlePasteEvent, true);
document.addEventListener("paste", handlePasteEvent, true);
fileDropzone.addEventListener("paste", handlePasteEvent, true);

// Drag & Drop
["dragenter", "dragover"].forEach((eventName) => {
  window.addEventListener(eventName, (e) => e.preventDefault(), false);
  fileDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    fileDropzone.classList.add("has-file");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  window.addEventListener(eventName, (e) => e.preventDefault(), false);
  fileDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!selectedFile) {
      fileDropzone.classList.remove("has-file");
    }
  });
});

fileDropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    applySelectedFile(file);
  }
});

// -----------------------------------------
// Form Submission & Analysis
// -----------------------------------------
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = selectedFile || fileInput.files?.[0];
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
      throw new Error(data.detail || "Diagnostic inference failed.");
    }
    currentPredictionData = data;
    renderResponse(data);
  } catch (error) {
    setChip(statusChip, "Error", "chip-err");
    uploadState.textContent = "Analysis failed. Please try again.";
    showAlert(messageEl, error.message || "Diagnostic inference failed.");
    renderEmpty("Analysis Interrupted", "Could not complete model inference.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    buttonLabel.textContent = "Analyze";
  }
});

function setInitialState() {
  selectedFile = null;
  currentPredictionData = null;
  currentActiveFinding = null;
  fileName.textContent = "No file selected • Press Ctrl+V to paste";
  fileDropzone.classList.remove("has-file");
  uploadState.textContent = "DICOM (.dcm), PNG, JPG supported &bull; Press Ctrl+V";
  button.disabled = true;
  button.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze";
  setChip(statusChip, "Waiting", "");

  if (preview.src && preview.src.startsWith("blob:")) {
    URL.revokeObjectURL(preview.src);
  }
  preview.removeAttribute("src");
  preview.hidden = true;
  emptyPreview.hidden = false;
  dicomBanner.hidden = true;

  hideHeatmap();
  hideAlerts();
  renderEmpty("Upload an image to see findings", "Predictions with calibrated thresholds will appear here.");
}

function setLoadingState() {
  button.disabled = true;
  button.classList.add("is-loading");
  buttonLabel.textContent = "Analyzing...";
  setChip(statusChip, "Analyzing", "chip-busy");
  uploadState.textContent = "Running neural network inference...";
  hideAlerts();
  hideHeatmap();
  renderEmpty("Model Inference in Progress", "Computing multi-label probabilities & Grad-CAM...");
}

// -----------------------------------------
// Refresh Model Info
// -----------------------------------------
async function refreshModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) throw new Error("Model info unavailable.");
    const data = await response.json();
    const loaded = Boolean(data.model_loaded);
    thresholdsLoaded = Boolean(data.thresholds_loaded);
    thresholds = data.thresholds || {};
    
    const info = data.model_info || {};
    const architecture = info.architecture || "ConvNeXt-Small";
    const labelCount = info.label_count || (Array.isArray(data.labels) ? data.labels.length : 5);
    const auc = info.mean_auc_display || "0.8944";
    
    modelStatus.textContent = loaded ? `${architecture} Active` : "No Checkpoint";
    modelStatus.className = `badge ${loaded ? "badge-loaded" : "badge-waiting"}`;
    modelInfo.textContent = `${architecture} / ${labelCount} labels / mean AUC ${auc} / Stanford U-Ones Calibrated`;
  } catch {
    modelStatus.textContent = "Model status unknown";
    modelStatus.className = "badge badge-error";
  }
}

// -----------------------------------------
// Render Prediction Results
// -----------------------------------------
function renderResponse(data) {
  const findings = Array.isArray(data.findings) ? data.findings : [];
  const ok = data.status === "ok";

  setChip(statusChip, ok ? "Complete" : "No model", ok ? "chip-ok" : "chip-err");
  uploadState.textContent = ok ? "Analysis complete." : "Model checkpoint not loaded.";

  // Handle DICOM metadata banner
  if (data.dicom?.is_dicom) {
    dicomPatient.textContent = data.dicom.patient_id || "ANONYMIZED";
    dicomModality.textContent = data.dicom.modality || "CR/DX";
    dicomView.textContent = data.dicom.view_position || "PA";
    dicomDate.textContent = data.dicom.study_date || "N/A";
    dicomMatrix.textContent = `${data.width} x ${data.height}`;
    dicomBanner.hidden = false;
  } else {
    dicomBanner.hidden = true;
  }

  // Quality warnings
  if (data.quality?.warnings?.length > 0) {
    qualityAlert.textContent = `⚠️ Clinical Quality Advisory: ${data.quality.warnings.join(" | ")}`;
    qualityAlert.style.display = "block";
  } else {
    qualityAlert.textContent = "This model expects frontal chest radiographs (PA/AP view). Results are for clinical research and decision support only.";
  }

  if (data.message) showAlert(messageEl, data.message);
  if (data.report) showAlert(reportEl, data.report, "alert-info");

  // Render findings list
  if (!findings.length) {
    renderEmpty("No findings returned", "The model did not return prediction labels.");
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of findings) {
    fragment.append(buildFindingCard(item));
  }
  resultsEl.replaceChildren(fragment);

  // Render Heatmap
  if (data.heatmap?.image_data_url) {
    renderHeatmap(data.heatmap);
    renderFindingTabs(findings, data.heatmap.label);
  } else {
    hideHeatmap();
  }
}

// -----------------------------------------
// Build Interactive Finding Card
// -----------------------------------------
function buildFindingCard(item) {
  const probability = clamp(item.probability);
  const threshold = Number.isFinite(Number(item.threshold))
    ? Number(item.threshold)
    : Number.isFinite(Number(thresholds[item.label]))
      ? Number(thresholds[item.label])
      : null;
  
  const suspicion = item.suspicion_level || suspicionLevel(probability, threshold).label;
  const suspCls = suspicion.toLowerCase().includes("high") ? "high" : (suspicion.toLowerCase().includes("moderate") ? "moderate" : "low");

  const article = el("article", `finding-card ${suspCls}`);
  article.dataset.label = item.label;

  const headerDiv = el("div", "finding-header");

  const leftDiv = el("div", "finding-left");
  const label = el("strong", "finding-name");
  label.textContent = item.label || "Unlabeled";
  const badge = el("span", "suspicion-badge");
  badge.textContent = suspicion;
  leftDiv.append(label, badge);

  const rightDiv = el("div", "finding-right");
  if (threshold !== null) {
    const thresholdText = el("span", "threshold-note");
    thresholdText.textContent = `Threshold: ${pct(threshold)}`;
    rightDiv.append(thresholdText);
  }
  const probabilityDiv = el("span", "probability");
  probabilityDiv.textContent = pct(probability);
  rightDiv.append(probabilityDiv);

  headerDiv.append(leftDiv, rightDiv);

  const track = el("div", "progress-track");
  const fill = el("div", "progress-fill");
  fill.style.width = pct(probability);
  track.append(fill);

  article.append(headerDiv, track);

  // Click on finding card to view specific Grad-CAM!
  article.addEventListener("click", () => {
    selectFindingHeatmap(item.label);
  });

  return article;
}

// -----------------------------------------
// On-Demand Specific Finding Heatmap
// -----------------------------------------
async function selectFindingHeatmap(label) {
  if (!selectedFile && !fileInput.files?.[0]) return;
  currentActiveFinding = label;

  // Highlight active finding card
  document.querySelectorAll(".finding-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.label === label);
  });

  // Highlight active tab
  document.querySelectorAll(".tab-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.label === label);
  });

  setChip(heatmapLabel, `Generating ${label}...`, "chip-busy");

  const formData = new FormData();
  formData.append("file", selectedFile || fileInput.files[0]);
  formData.append("label", label);

  try {
    const res = await fetch("/api/explain", { method: "POST", body: formData });
    if (!res.ok) throw new Error("Could not compute Grad-CAM for this finding.");
    const explanation = await res.json();
    renderHeatmap(explanation);
  } catch (err) {
    setChip(heatmapLabel, `Error on ${label}`, "chip-err");
  }
}

function renderFindingTabs(findings, activeLabel) {
  findingTabs.replaceChildren();
  findings.forEach((f) => {
    const chip = el("button", `tab-chip ${f.label === activeLabel ? "active" : ""}`);
    chip.type = "button";
    chip.dataset.label = f.label;
    chip.textContent = `${f.label} (${pct(f.probability)})`;
    chip.addEventListener("click", () => selectFindingHeatmap(f.label));
    findingTabs.append(chip);
  });
  heatmapControlsBar.hidden = false;
}

function renderHeatmap(heatmapData) {
  const prob = clamp(heatmapData.probability);
  const label = heatmapData.label || "Top Finding";

  boxLabelFinding.textContent = `${label} (${pct(prob)})`;
  setChip(heatmapLabel, `${label} ${pct(prob)}`, "chip-ok");

  // Synchronize base radiograph preview
  if (preview.src) {
    heatmapBase.src = preview.src;
    overlayUnderlay.src = preview.src;
  }

  if (heatmapData.pure_heatmap_url) {
    heatmapPure.src = heatmapData.pure_heatmap_url;
    heatmapPure.hidden = false;
    heatmapPure.style.opacity = (sliderOpacity.value / 100).toString();
    heatmapFallback.hidden = true;
  } else if (heatmapData.image_data_url) {
    heatmapFallback.src = heatmapData.image_data_url;
    heatmapFallback.hidden = false;
    heatmapPure.hidden = true;
  }

  heatmapEmpty.hidden = true;
  heatmapFigure.hidden = false;
  applyPacsFilters();
}

function hideHeatmap() {
  heatmapEmpty.hidden = false;
  heatmapFigure.hidden = true;
  heatmapControlsBar.hidden = true;
  setChip(heatmapLabel, "Not generated", "");
}

// -----------------------------------------
// Export PDF Medical Report (A4 Format)
// -----------------------------------------
exportPdfBtn.addEventListener("click", () => {
  if (!currentPredictionData || !currentPredictionData.findings?.length) {
    alert("Please upload and analyze a chest X-ray before exporting a report.");
    return;
  }

  const data = currentPredictionData;
  const dateStr = new Date().toLocaleString();

  document.getElementById("print-date").textContent = dateStr;
  document.getElementById("print-patient").textContent = data.dicom?.patient_id || "ANONYMIZED_PATIENT";
  document.getElementById("print-modality").textContent = data.dicom?.modality || "CR/DX Radiography";
  document.getElementById("print-view").textContent = data.dicom?.view_position || data.quality?.suggested_view || "Frontal (PA/AP)";
  document.getElementById("print-filename").textContent = data.filename || "cxr_scan.png";

  document.getElementById("print-img-original").src = preview.src || "";
  document.getElementById("print-img-heatmap").src = heatmapPure.src || heatmapFallback.src || "";
  document.getElementById("print-active-finding").textContent = currentActiveFinding || data.heatmap?.label || "Top Finding";

  // Populate findings table
  const tbody = document.getElementById("print-table-body");
  tbody.replaceChildren();

  data.findings.forEach((item) => {
    const tr = document.createElement("tr");
    const threshold = item.threshold !== null ? pct(item.threshold) : "N/A";
    const statusText = item.positive ? "POSITIVE" : "NEGATIVE";
    const statusColor = item.positive ? "#dc2626" : "#16a34a";

    tr.innerHTML = `
      <td><strong>${item.label}</strong></td>
      <td>${pct(item.probability)}</td>
      <td>${threshold}</td>
      <td style="color: ${statusColor}; font-weight: bold;">${statusText}</td>
      <td>${item.suspicion_level || "Standard"}</td>
    `;
    tbody.appendChild(tr);
  });

  // Populate impression
  document.getElementById("print-impression").textContent = data.report || "No acute findings.";

  // Trigger print dialog
  window.print();
});

// -----------------------------------------
// Helpers
// -----------------------------------------
function suspicionLevel(probability, threshold) {
  if (threshold === null) return { cls: "", label: "Probability only" };
  const highCutoff = Math.min(1, threshold + 0.15);
  if (probability >= highCutoff) return { cls: "high", label: "High suspicion" };
  if (probability >= threshold) return { cls: "moderate", label: "Moderate suspicion" };
  return { cls: "", label: "Low suspicion" };
}

function el(tag, className = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function clamp(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(1, Math.max(0, number)) : 0;
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
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

function renderEmpty(title, detail) {
  const wrap = el("div", "empty-state compact");
  const strong = el("strong");
  const span = el("span");
  strong.textContent = title;
  span.textContent = detail;
  wrap.append(strong, span);
  resultsEl.replaceChildren(wrap);
}
