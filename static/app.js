// -----------------------------------------
// CheXpert Medical AI Workstation - Engine
// -----------------------------------------

// Mode switchers
const tabSingleMode = document.getElementById("tab-single-mode");
const tabBatchMode = document.getElementById("tab-batch-mode");
const singleModeContainer = document.getElementById("single-mode-container");
const batchModeContainer = document.getElementById("batch-mode-container");

// Single Mode Elements
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
const langToggleBtn = document.getElementById("lang-toggle-btn");
const langLabel = document.getElementById("lang-label");

// DICOM banner elements
const dicomBanner = document.getElementById("dicom-banner");
const dicomPatient = document.getElementById("dicom-patient");
const dicomModality = document.getElementById("dicom-modality");
const dicomView = document.getElementById("dicom-view");
const dicomDate = document.getElementById("dicom-date");
const dicomMatrix = document.getElementById("dicom-matrix");

// Batch Processing Elements
const batchForm = document.getElementById("batch-form");
const batchFilesInput = document.getElementById("batch-files");
const batchDropzone = document.getElementById("batch-dropzone");
const batchFileCount = document.getElementById("batch-file-count");
const batchAnalyzeBtn = document.getElementById("batch-analyze-btn");
const batchBtnLabel = document.getElementById("batch-btn-label");
const batchTableBody = document.getElementById("batch-table-body");
const batchStatsChip = document.getElementById("batch-stats-chip");
const btnExportCsv = document.getElementById("btn-export-csv");
const btnClearBatch = document.getElementById("btn-clear-batch");

// -----------------------------------------
// Universal On-The-Fly Translation Filter
// -----------------------------------------
const DICTIONARY_VI = {
  // Sentences & Clinical Reports (Backend generated)
  "No pathological findings exceeded calibrated thresholds. Frontal lung fields appear clear of acute radiopaque consolidation, pulmonary edema, or overt pleural effusion.":
    "Không có tổn thương bệnh lý nào vượt ngưỡng chẩn đoán. Phế trường hai bên sáng đều, không thấy hình ảnh đông đặc cấp tính, phù phổi hay tràn dịch màng phổi rõ rệt.",
  "Automated analysis indicates": "Kết quả phân tích tự động ghi nhận",
  "Clinical correlation and specialist radiologist overread are recommended.":
    "Khuyến nghị đối chiếu lâm sàng và có ý kiến hội chẩn của bác sĩ chuyên khoa chẩn đoán hình ảnh.",
  "This model expects frontal chest radiographs (PA/AP view). Results are for clinical research and decision support only.":
    "Mô hình chỉ áp dụng cho phim X-quang ngực thẳng (PA/AP). Kết quả phục vụ nghiên cứu và hỗ trợ quyết định lâm sàng.",
  "Research prototype only. Do not use these automated results for definitive medical diagnosis or treatment decisions without board-certified radiologist validation. Model calibrated on CheXpert frontal CXR.":
    "Phiên bản nghiên cứu thử nghiệm. Không dùng kết quả tự động này để đưa ra chẩn đoán hay quyết định điều trị y tế mà không có sự xác nhận của bác sĩ chẩn đoán hình ảnh. Mô hình được hiệu chuẩn trên bộ dữ liệu X-quang ngực thẳng CheXpert.",
  "Visual explanation is intended for research interpretation and quality assurance.":
    "Hình ảnh giải thích phục vụ mục đích nghiên cứu và kiểm định chất lượng chẩn đoán.",

  // Disease names
  "Atelectasis": "Atelectasis (Xẹp phổi)",
  "Cardiomegaly": "Cardiomegaly (Bóng tim to)",
  "Consolidation": "Consolidation (Đông đặc)",
  "Edema": "Edema (Phù phổi)",
  "Pleural Effusion": "Pleural Effusion (Tràn dịch)",

  // Suspicion levels
  "High suspicion": "Nghi ngờ cao",
  "Moderate suspicion": "Nghi ngờ vừa",
  "Low suspicion": "Nguy cơ thấp",
  "Probability only": "Chỉ số xác suất",

  // UI elements
  "Clinical Decision Support System": "Hệ thống hỗ trợ chẩn đoán hình ảnh",
  "Deep Learning Chest Radiograph Multi-Label Analysis": "Phân tích đa bệnh lý X-quang ngực bằng Deep Learning",
  "Export Report (PDF)": "Xuất báo cáo (PDF)",
  "Research Protocol": "Nghiên cứu lâm sàng",
  "Import Radiograph": "Tải lên phim X-quang",
  "DICOM (.dcm), PNG, JPG, WebP supported • Press Ctrl+V to paste": "Hỗ trợ DICOM (.dcm), PNG, JPG, WebP • Nhấn Ctrl+V để dán ảnh",
  "Choose image or Paste (Ctrl+V)": "Chọn ảnh hoặc dán (Ctrl+V)",
  "No file selected • Press Ctrl+V anywhere": "Chưa chọn file • Bấm Ctrl+V ở bất kỳ đâu",
  "Radiograph loaded & ready to analyze.": "Đã nạp phim • Sẵn sàng phân tích.",
  "Running neural network inference...": "Đang chạy mạng nơ-ron chẩn đoán...",
  "Analysis complete.": "Phân tích hoàn tất.",
  "Analyze": "Phân tích",
  "Analyzing...": "Đang phân tích...",
  "PACS Radiograph Viewer": "Trình đọc phim PACS",
  "Input Image": "Ảnh X-quang đầu vào",
  "Brightness": "Độ sáng",
  "Contrast": "Độ tương phản",
  "Fit": "Vừa khung",
  "Invert": "Đảo âm bản",
  "No radiograph loaded": "Chưa tải phim X-quang",
  "Select, drop, or press Ctrl+V to import an X-ray.": "Chọn, kéo thả hoặc nhấn Ctrl+V để nạp phim.",
  "Pathology Predictions": "Dự đoán bệnh lý",
  "Clinical Findings": "Kết quả bệnh lý",
  "💡 Click on any disease card below to view its specific Grad-CAM heatmap:": "💡 Nhấp vào thẻ bệnh lý bất kỳ để xem bản đồ nhiệt Grad-CAM tương ứng:",
  "Upload an image to see findings": "Tải ảnh lên để xem kết quả",
  "Predictions with calibrated thresholds will appear here.": "Dự đoán kèm ngưỡng tối ưu sẽ hiển thị tại đây.",
  "Explainable AI (XAI)": "Giải thích AI (XAI)",
  "Attention Heatmap (Grad-CAM)": "Bản đồ tập trung (Grad-CAM)",
  "Visual activation map highlighting anatomical regions driving model prediction": "Vùng kích hoạt làm nổi bật các tổn thương chi phối quyết định của AI",
  "Heatmap Blend": "Độ mờ nhiệt",
  "Heatmap will appear after analysis": "Bản đồ nhiệt sẽ xuất hiện sau khi phân tích",
  "Grad-CAM activation overlays are generated during inference.": "Vùng kích hoạt Grad-CAM được tạo tự động khi chạy suy luận.",
  "Original Radiograph": "Phim X-quang gốc",
  "Grad-CAM Overlay": "Bản đồ nhiệt Grad-CAM",
  "Waiting": "Chờ ảnh",
  "Ready": "Sẵn sàng",
  "Analyzing": "Đang xử lý",
  "Complete": "Hoàn tất",
  "Error": "Lỗi",
  "Not generated": "Chưa tạo",
  "No threshold": "Chưa có ngưỡng",
  "Threshold:": "Ngưỡng:",
  "Top Finding": "Tổn thương nổi bật",

  // Modes & Batch
  "Single Radiograph Diagnostics": "Chẩn đoán từng ca X-quang",
  "Batch Multi-File Queuing (CSV Export)": "Xử lý hàng loạt (Xuất file CSV)",
  "Batch Multi-File Analysis": "Xử lý & Chẩn đoán hàng loạt",
  "Select multiple DICOM / CXR files or entire study folder for high-speed parallel inference.":
    "Chọn nhiều file DICOM / X-quang hoặc cả thư mục ca bệnh để chạy suy luận song song tốc độ cao.",
  "Select multiple CXRs or Drop Folder": "Chọn nhiều file hoặc kéo thả thư mục",
  "Run Batch Analysis": "Bắt đầu quét hàng loạt",
  "High-Throughput Diagnostic Queue": "Hàng chờ chẩn đoán thông lượng cao",
  "Batch Analysis Overview": "Bảng tổng hợp chẩn đoán hàng loạt",
  "Export Summary (CSV)": "Xuất bảng tổng hợp (CSV)",
  "No batch studies loaded": "Chưa có danh sách ca bệnh",
  "Import multiple chest radiographs or DICOM files to begin parallel batch analysis.":
    "Tải lên nhiều ảnh X-quang hoặc file DICOM để bắt đầu quét song song.",
  "Preview": "Xem trước",
  "Patient ID / File": "Mã bệnh nhân / Tập tin",
  "Positive Pathologies": "Bệnh lý phát hiện",
  "Status": "Trạng thái",
  "Interactive PACS": "Mở xem PACS",
  "CLEAR": "BÌNH THƯỜNG",
  "POSITIVE": "DƯƠNG TÍNH",
  "Open": "Mở",

  // Print Layout
  "Chest Radiograph Automated Diagnostic & Pathology Report": "Phiếu Báo Cáo Kết Quả Xét Nghiệm & Chẩn Đoán X-quang Tự Động",
  "Original Frontal Radiograph": "Ảnh X-quang Ngực Thẳng Gốc",
  "1. Multi-Label Pathology Probabilities": "1. Xác suất Đa Bệnh Lý & Ngưỡng Chẩn Đoán",
  "2. Automated Radiological Impression": "2. Tóm Tắt Kết Luận Hình Ảnh Tự Động",
  "Finding / Pathology": "Tổn thương / Bệnh lý",
  "Probability": "Xác suất",
  "Calibrated Threshold": "Ngưỡng tối ưu",
  "Clinical Suspicion": "Mức độ nghi ngờ",
  "Disclaimer: This automated report is generated by a deep neural network research prototype. It is not an FDA-cleared diagnostic device. Overread by a licensed radiologist is mandatory.":
    "Miễn trừ trách nhiệm: Phiên bản nghiên cứu thử nghiệm bằng mạng nơ-ron học sâu. Không thay thế chẩn đoán xác định của bác sĩ chuyên khoa.",
  "Reviewing Radiologist Signature: ___________________________":
    "Chữ ký Bác sĩ Chẩn đoán hình ảnh: ___________________________"
};

// Current Language: Default is 'en'
let currentLang = localStorage.getItem("chex_lang") || "en";

// Sort keys by length descending so longer sentences are always matched and translated first!
const SORTED_KEYS_VI = Object.keys(DICTIONARY_VI).sort((a, b) => b.length - a.length);

// Universal translation filter function
function t(text) {
  if (!text || typeof text !== "string") return text;
  if (currentLang === "en") return text;
  const trimmed = text.trim();
  if (DICTIONARY_VI[trimmed]) return DICTIONARY_VI[trimmed];

  let translated = text;
  for (const enKey of SORTED_KEYS_VI) {
    if (translated.includes(enKey)) {
      translated = translated.replaceAll(enKey, DICTIONARY_VI[enKey]);
    }
  }
  return translated;
}

// State
let thresholdsLoaded = false;
let thresholds = {};
let selectedFile = null;
let currentPredictionData = null;
let currentActiveFinding = null;

// Batch State
let queuedBatchFiles = [];
let batchResultsData = [];

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
applyLanguage(currentLang);
refreshModelInfo();
initPacsControls();
initModeSwitchers();
initBatchControls();

// -----------------------------------------
// Mode Switchers
// -----------------------------------------
function initModeSwitchers() {
  tabSingleMode.addEventListener("click", () => switchWorkspaceMode("single"));
  tabBatchMode.addEventListener("click", () => switchWorkspaceMode("batch"));
}

function switchWorkspaceMode(mode) {
  if (mode === "single") {
    tabSingleMode.classList.add("active");
    tabBatchMode.classList.remove("active");
    singleModeContainer.hidden = false;
    batchModeContainer.hidden = true;
  } else {
    tabBatchMode.classList.add("active");
    tabSingleMode.classList.remove("active");
    singleModeContainer.hidden = true;
    batchModeContainer.hidden = false;
  }
}

// -----------------------------------------
// Language Switcher Logic
// -----------------------------------------
langToggleBtn.addEventListener("click", () => {
  currentLang = currentLang === "en" ? "vi" : "en";
  localStorage.setItem("chex_lang", currentLang);
  applyLanguage(currentLang);
});

function applyLanguage(lang) {
  langLabel.textContent = lang === "en" ? "VI" : "EN";

  // Mode Tabs
  setText("i18n-tab-single", t("Single Radiograph Diagnostics"));
  setText("i18n-tab-batch", t("Batch Multi-File Queuing (CSV Export)"));

  // Header & Upload
  setText("i18n-eyebrow", t("Clinical Decision Support System"));
  setText("i18n-subtitle", t("Deep Learning Chest Radiograph Multi-Label Analysis"));
  setText("i18n-export-report", t("Export Report (PDF)"));
  setText("i18n-research-protocol", t("Research Protocol"));
  setText("quality-alert", t("This model expects frontal chest radiographs (PA/AP view). Results are for clinical research and decision support only."));
  setText("upload-title", t("Import Radiograph"));
  setText("i18n-dropzone-title", t("Choose image or Paste (Ctrl+V)"));

  // PACS
  setText("i18n-pacs-label", t("PACS Radiograph Viewer"));
  setText("image-heading", t("Input Image"));
  setText("i18n-brightness-lbl", t("Brightness"));
  setText("i18n-contrast-lbl", t("Contrast"));
  setText("btn-zoom-fit", t("Fit"));
  setText("btn-invert", t("Invert"));
  setText("i18n-no-img-title", t("No radiograph loaded"));
  setText("i18n-no-img-sub", t("Select, drop, or press Ctrl+V to import an X-ray."));

  // Findings
  setText("i18n-pred-label", t("Pathology Predictions"));
  setText("findings-heading", t("Clinical Findings"));
  setText("i18n-findings-hint", t("💡 Click on any disease card below to view its specific Grad-CAM heatmap:"));
  setText("i18n-empty-find-title", t("Upload an image to see findings"));
  setText("i18n-empty-find-sub", t("Predictions with calibrated thresholds will appear here."));

  // XAI Heatmap
  setText("i18n-xai-label", t("Explainable AI (XAI)"));
  setText("heatmap-heading", t("Attention Heatmap (Grad-CAM)"));
  setText("i18n-xai-desc", t("Visual activation map highlighting anatomical regions driving model prediction"));
  setText("i18n-blend-lbl", t("Heatmap Blend"));
  setText("i18n-heat-empty-title", t("Heatmap will appear after analysis"));
  setText("i18n-heat-empty-sub", t("Grad-CAM activation overlays are generated during inference."));
  setText("i18n-orig-label", t("Original Radiograph"));
  setText("i18n-cam-label", t("Grad-CAM Overlay"));
  setText("heatmap-caption", t("Visual explanation is intended for research interpretation and quality assurance."));

  // Batch Mode Texts
  setText("batch-title", t("Batch Multi-File Analysis"));
  setText("i18n-batch-sub", t("Select multiple DICOM / CXR files or entire study folder for high-speed parallel inference."));
  setText("i18n-batch-drop-title", t("Select multiple CXRs or Drop Folder"));
  setText("i18n-batch-queue-label", t("High-Throughput Diagnostic Queue"));
  setText("i18n-batch-queue-title", t("Batch Analysis Overview"));
  setText("i18n-export-csv", t("Export Summary (CSV)"));
  setText("i18n-th-thumb", t("Preview"));
  setText("i18n-th-patient", t("Patient ID / File"));
  setText("i18n-th-top", t("Top Finding"));
  setText("i18n-th-pos", t("Positive Pathologies"));
  setText("i18n-th-status", t("Status"));
  setText("i18n-th-action", t("Interactive PACS"));

  // Printable Report elements
  setText("print-sub-header", t("Chest Radiograph Automated Diagnostic & Pathology Report"));
  setText("print-lbl-orig", t("Original Frontal Radiograph"));
  setText("print-lbl-cam", t("Grad-CAM Heatmap Overlay"));
  setText("print-sec1", t("1. Multi-Label Pathology Probabilities"));
  setText("print-sec2", t("2. Automated Radiological Impression"));
  setText("print-th-finding", t("Finding / Pathology"));
  setText("print-th-prob", t("Probability"));
  setText("print-th-thresh", t("Calibrated Threshold"));
  setText("print-th-status", t("Status"));
  setText("print-th-susp", t("Clinical Suspicion"));
  setText("print-footer-disc", t("Disclaimer: This automated report is generated by a deep neural network research prototype. It is not an FDA-cleared diagnostic device. Overread by a licensed radiologist is mandatory."));
  setText("print-sig-line", t("Reviewing Radiologist Signature: ___________________________"));

  // Disclaimer
  setText("i18n-disclaimer-text", t("Research prototype only. Do not use these automated results for definitive medical diagnosis or treatment decisions without board-certified radiologist validation. Model calibrated on CheXpert frontal CXR."));

  // Button state
  if (!button.disabled) {
    buttonLabel.textContent = t("Analyze");
  }

  // Re-filter findings and report box if data already present
  if (currentPredictionData && currentPredictionData.findings) {
    renderResponse(currentPredictionData);
  }

  // Re-render batch table if data present
  if (batchResultsData.length > 0) {
    renderBatchTable(batchResultsData);
  }
}

function setText(id, text) {
  const elNode = document.getElementById(id);
  if (elNode) elNode.textContent = text;
}

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
// Single File Selection & Drag/Paste
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
  uploadState.textContent = t("Radiograph loaded & ready to analyze.");
  button.disabled = false;
  button.classList.remove("is-loading");
  buttonLabel.textContent = t("Analyze");

  setChip(statusChip, t("Ready"), "");
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
  renderEmpty(t("Upload an image to see findings"), t("Predictions with calibrated thresholds will appear here."));
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    setInitialState();
    return;
  }
  applySelectedFile(file);
});

// Clipboard Paste (Ctrl+V)
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
// Single Form Submission & Analysis
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
    setChip(statusChip, t("Error"), "chip-err");
    uploadState.textContent = "Analysis failed. Please try again.";
    showAlert(messageEl, error.message || "Diagnostic inference failed.");
    renderEmpty(t("Error"), "Could not complete model inference.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    buttonLabel.textContent = t("Analyze");
  }
});

function setInitialState() {
  selectedFile = null;
  currentPredictionData = null;
  currentActiveFinding = null;
  fileName.textContent = t("No file selected • Press Ctrl+V anywhere");
  fileDropzone.classList.remove("has-file");
  uploadState.textContent = t("DICOM (.dcm), PNG, JPG, WebP supported • Press Ctrl+V to paste");
  button.disabled = true;
  button.classList.remove("is-loading");
  buttonLabel.textContent = t("Analyze");
  setChip(statusChip, t("Waiting"), "");

  if (preview.src && preview.src.startsWith("blob:")) {
    URL.revokeObjectURL(preview.src);
  }
  preview.removeAttribute("src");
  preview.hidden = true;
  emptyPreview.hidden = false;
  dicomBanner.hidden = true;

  hideHeatmap();
  hideAlerts();
  renderEmpty(t("Upload an image to see findings"), t("Predictions with calibrated thresholds will appear here."));
}

function setLoadingState() {
  button.disabled = true;
  button.classList.add("is-loading");
  buttonLabel.textContent = t("Analyzing...");
  setChip(statusChip, t("Analyzing"), "chip-busy");
  uploadState.textContent = t("Running neural network inference...");
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
    
    modelStatus.textContent = loaded ? `${architecture} ${t("Active")}` : t("No Checkpoint");
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

  setChip(statusChip, ok ? t("Complete") : t("No Checkpoint"), ok ? "chip-ok" : "chip-err");
  uploadState.textContent = ok ? t("Analysis complete.") : "Model checkpoint not loaded.";

  // DICOM metadata banner
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

  // Quality warnings filtered through t()
  if (data.quality?.warnings?.length > 0) {
    qualityAlert.textContent = `⚠️ Clinical Quality Advisory: ${data.quality.warnings.map(t).join(" | ")}`;
    qualityAlert.style.display = "block";
  } else {
    qualityAlert.textContent = t("This model expects frontal chest radiographs (PA/AP view). Results are for clinical research and decision support only.");
  }

  if (data.message) showAlert(messageEl, t(data.message));
  
  // Localized report filtered directly through t()
  if (data.report) {
    showAlert(reportEl, t(data.report), "alert-info");
  }

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
  
  const rawSuspicion = item.suspicion_level || suspicionLevel(probability, threshold).label;
  const translatedSuspicion = t(rawSuspicion);
  const suspCls = rawSuspicion.toLowerCase().includes("high") ? "high" : (rawSuspicion.toLowerCase().includes("moderate") ? "moderate" : "low");

  const displayName = t(item.label) || item.label || "Unlabeled";

  const article = el("article", `finding-card ${suspCls}`);
  article.dataset.label = item.label;

  const headerDiv = el("div", "finding-header");

  const leftDiv = el("div", "finding-left");
  const label = el("strong", "finding-name");
  label.textContent = displayName;
  const badge = el("span", "suspicion-badge");
  badge.textContent = translatedSuspicion;
  leftDiv.append(label, badge);

  const rightDiv = el("div", "finding-right");
  if (threshold !== null) {
    const thresholdText = el("span", "threshold-note");
    thresholdText.textContent = `${t("Threshold:")} ${pct(threshold)}`;
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

  setChip(heatmapLabel, `${t("Analyzing...")} ${t(label)}`, "chip-busy");

  const formData = new FormData();
  formData.append("file", selectedFile || fileInput.files[0]);
  formData.append("label", label);

  try {
    const res = await fetch("/api/explain", { method: "POST", body: formData });
    if (!res.ok) throw new Error("Could not compute Grad-CAM for this finding.");
    const explanation = await res.json();
    renderHeatmap(explanation);
  } catch (err) {
    setChip(heatmapLabel, `${t("Error")} ${t(label)}`, "chip-err");
  }
}

function renderFindingTabs(findings, activeLabel) {
  findingTabs.replaceChildren();
  findings.forEach((f) => {
    const chip = el("button", `tab-chip ${f.label === activeLabel ? "active" : ""}`);
    chip.type = "button";
    chip.dataset.label = f.label;
    chip.textContent = `${t(f.label)} (${pct(f.probability)})`;
    chip.addEventListener("click", () => selectFindingHeatmap(f.label));
    findingTabs.append(chip);
  });
  heatmapControlsBar.hidden = false;
}

function renderHeatmap(heatmapData) {
  const prob = clamp(heatmapData.probability);
  const rawLabel = heatmapData.label || "Top Finding";
  const displayName = t(rawLabel);

  boxLabelFinding.textContent = `${displayName} (${pct(prob)})`;
  setChip(heatmapLabel, `${displayName} ${pct(prob)}`, "chip-ok");

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
  setChip(heatmapLabel, t("Not generated"), "");
}

// -----------------------------------------
// Batch Processing Logic
// -----------------------------------------
function initBatchControls() {
  batchFilesInput.addEventListener("change", () => {
    const files = Array.from(batchFilesInput.files || []);
    if (!files.length) return;
    queueBatchFiles(files);
  });

  // Batch Drag & Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    batchDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      batchDropzone.classList.add("has-file");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    batchDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!queuedBatchFiles.length) batchDropzone.classList.remove("has-file");
    });
  });

  batchDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length) queueBatchFiles(files);
  });

  batchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!queuedBatchFiles.length) return;
    await runBatchInference();
  });

  btnExportCsv.addEventListener("click", exportBatchCsv);
  btnClearBatch.addEventListener("click", clearBatch);
}

function queueBatchFiles(files) {
  queuedBatchFiles = files;
  batchFileCount.textContent = `${files.length} files queued • Parallel vectorized GPU/CPU forward pass`;
  batchDropzone.classList.add("has-file");
  batchAnalyzeBtn.disabled = false;
  batchStatsChip.textContent = `${files.length} Studies Queued`;
}

async function runBatchInference() {
  if (!queuedBatchFiles.length) return;
  batchAnalyzeBtn.disabled = true;
  batchAnalyzeBtn.classList.add("is-loading");
  batchBtnLabel.textContent = t("Analyzing...");

  const formData = new FormData();
  for (const f of queuedBatchFiles) {
    formData.append("files", f);
  }

  try {
    const res = await fetch("/api/predict-batch", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Batch processing failed.");

    batchResultsData = data.results || [];
    renderBatchTable(batchResultsData);
    btnExportCsv.disabled = batchResultsData.length === 0;
    batchStatsChip.textContent = `${batchResultsData.length} Processed`;
  } catch (err) {
    alert(`Batch error: ${err.message}`);
  } finally {
    batchAnalyzeBtn.disabled = false;
    batchAnalyzeBtn.classList.remove("is-loading");
    batchBtnLabel.textContent = t("Run Batch Analysis");
  }
}

function renderBatchTable(results) {
  if (!results.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-table-row";
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "empty-table-cell";
    const strong = document.createElement("strong");
    strong.textContent = t("No batch studies loaded");
    const span = document.createElement("span");
    span.textContent = t("Import multiple chest radiographs or DICOM files to begin parallel batch analysis.");
    td.append(strong, span);
    tr.appendChild(td);
    batchTableBody.replaceChildren(tr);
    return;
  }

  const fragment = document.createDocumentFragment();
  results.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const hasPos = item.positive_count > 0;
    const statusText = hasPos ? t("POSITIVE") : t("CLEAR");
    const statusColor = hasPos ? "#ef4444" : "#10b981";

    // 1. Index
    const tdIdx = document.createElement("td");
    tdIdx.textContent = `${idx + 1}`;
    tr.appendChild(tdIdx);

    // 2. Thumbnail preview
    const tdThumb = document.createElement("td");
    const imgThumb = document.createElement("img");
    imgThumb.className = "batch-thumb-img";
    imgThumb.src = item.preview_url || "";
    imgThumb.alt = "Thumbnail";
    tdThumb.appendChild(imgThumb);
    tr.appendChild(tdThumb);

    // 3. Patient ID & filename
    const tdPatient = document.createElement("td");
    const strongPat = document.createElement("strong");
    strongPat.textContent = item.patient_id || "ANONYMIZED";
    const br = document.createElement("br");
    const smallFile = document.createElement("small");
    smallFile.style.color = "var(--text-muted)";
    smallFile.textContent = item.filename;
    tdPatient.append(strongPat, br, smallFile);
    tr.appendChild(tdPatient);

    // 4. Top Finding
    const tdTop = document.createElement("td");
    const strongTop = document.createElement("strong");
    strongTop.textContent = t(item.top_finding);
    const spanProb = document.createElement("span");
    spanProb.textContent = ` (${pct(item.top_probability)})`;
    tdTop.append(strongTop, spanProb);
    tr.appendChild(tdTop);

    // 5. Positive tags
    const tdPos = document.createElement("td");
    if (item.positive_labels && item.positive_labels.length > 0) {
      item.positive_labels.forEach((lbl) => {
        const tag = document.createElement("span");
        tag.className = "batch-tag-positive";
        tag.textContent = t(lbl);
        tdPos.appendChild(tag);
      });
    } else {
      const clearTag = document.createElement("span");
      clearTag.className = "batch-tag-clear";
      clearTag.textContent = t("CLEAR");
      tdPos.appendChild(clearTag);
    }
    tr.appendChild(tdPos);

    // 6. Status badge
    const tdStatus = document.createElement("td");
    tdStatus.style.color = statusColor;
    tdStatus.style.fontWeight = "bold";
    tdStatus.textContent = statusText;
    tr.appendChild(tdStatus);

    // 7. Interactive action button
    const tdAction = document.createElement("td");
    const btnOpen = document.createElement("button");
    btnOpen.className = "btn-open-pacs";
    btnOpen.type = "button";
    btnOpen.textContent = `${t("Open")} ↗`;
    btnOpen.addEventListener("click", (e) => {
      e.stopPropagation();
      openBatchItemInPacs(item.filename, idx);
    });
    tdAction.appendChild(btnOpen);
    tr.appendChild(tdAction);

    // Click anywhere on row to view in PACS
    tr.addEventListener("click", () => openBatchItemInPacs(item.filename, idx));
    fragment.appendChild(tr);
  });

  batchTableBody.replaceChildren(fragment);
}

function openBatchItemInPacs(filename, fallbackIdx) {
  // Find file by exact filename or fallback index for guaranteed integrity
  let file = queuedBatchFiles.find((f) => f.name === filename);
  if (!file && fallbackIdx !== undefined) {
    file = queuedBatchFiles[fallbackIdx];
  }
  if (!file) return;

  // Switch to single workspace mode
  switchWorkspaceMode("single");
  applySelectedFile(file);

  // Auto trigger analysis
  form.dispatchEvent(new Event("submit"));
}

function clearBatch() {
  queuedBatchFiles = [];
  batchResultsData = [];
  batchFilesInput.value = "";
  batchFileCount.textContent = "0 files queued • Parallel vectorized GPU/CPU forward pass";
  batchDropzone.classList.remove("has-file");
  batchAnalyzeBtn.disabled = true;
  btnExportCsv.disabled = true;
  batchStatsChip.textContent = "0 Patients Scanned";
  renderBatchTable([]);
}

function escapeCsvCell(val) {
  if (val === null || val === undefined) return '""';
  let s = String(val);
  // CSV Formula Injection defense: prepend single-quote if string starts with risky chars (=, +, -, @)
  if (/^[=+\-@\t\r]/.test(s)) {
    s = "'" + s;
  }
  return `"${s.replace(/"/g, '""')}"`;
}

function exportBatchCsv() {
  if (!batchResultsData.length) return;

  const headers = [
    "No",
    "Filename",
    "Patient ID",
    "Is DICOM",
    "Atelectasis (Prob)",
    "Cardiomegaly (Prob)",
    "Consolidation (Prob)",
    "Edema (Prob)",
    "Pleural Effusion (Prob)",
    "Positive Count",
    "Positive Pathologies",
    "Impression Report",
  ];

  const rows = batchResultsData.map((r, i) => {
    const probMap = {};
    (r.findings || []).forEach((f) => {
      probMap[f.label] = `${(f.probability * 100).toFixed(1)}%`;
    });

    return [
      escapeCsvCell(i + 1),
      escapeCsvCell(r.filename),
      escapeCsvCell(r.patient_id || 'ANONYMIZED'),
      escapeCsvCell(r.is_dicom ? "YES" : "NO"),
      escapeCsvCell(probMap["Atelectasis"] || "N/A"),
      escapeCsvCell(probMap["Cardiomegaly"] || "N/A"),
      escapeCsvCell(probMap["Consolidation"] || "N/A"),
      escapeCsvCell(probMap["Edema"] || "N/A"),
      escapeCsvCell(probMap["Pleural Effusion"] || "N/A"),
      escapeCsvCell(r.positive_count),
      escapeCsvCell((r.positive_labels || []).join('; ')),
      escapeCsvCell(r.report || ''),
    ].join(",");
  });

  const csvContent = "data:text/csv;charset=utf-8,\uFEFF" + [headers.map(escapeCsvCell).join(","), ...rows].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `chexpert_batch_summary_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
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
  
  const activeLabel = currentActiveFinding || data.heatmap?.label || "Top Finding";
  document.getElementById("print-active-finding").textContent = t(activeLabel);

  // Populate findings table using safe DOM construction
  const tbody = document.getElementById("print-table-body");
  tbody.replaceChildren();

  data.findings.forEach((item) => {
    const tr = document.createElement("tr");
    const threshold = item.threshold !== null ? pct(item.threshold) : "N/A";
    const statusText = item.positive ? "POSITIVE" : "NEGATIVE";
    const statusColor = item.positive ? "#dc2626" : "#16a34a";
    const name = t(item.label);
    const susp = t(item.suspicion_level || "Standard");

    const tdName = document.createElement("td");
    const strongName = document.createElement("strong");
    strongName.textContent = name;
    tdName.appendChild(strongName);

    const tdProb = document.createElement("td");
    tdProb.textContent = pct(item.probability);

    const tdThresh = document.createElement("td");
    tdThresh.textContent = threshold;

    const tdStatus = document.createElement("td");
    tdStatus.style.color = statusColor;
    tdStatus.style.fontWeight = "bold";
    tdStatus.textContent = statusText;

    const tdSusp = document.createElement("td");
    tdSusp.textContent = susp;

    tr.append(tdName, tdProb, tdThresh, tdStatus, tdSusp);
    tbody.appendChild(tr);
  });

  // Localized Impression for print filtered through t()
  document.getElementById("print-impression").textContent = t(data.report || "No acute findings.");

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
