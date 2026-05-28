const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#xray-file");
const fileName = document.querySelector("#file-name");
const preview = document.querySelector("#preview");
const emptyPreview = document.querySelector("#empty-preview");
const heatmapWrap = document.querySelector("#heatmap-wrap");
const heatmap = document.querySelector("#heatmap");
const heatmapLabel = document.querySelector("#heatmap-label");
const statusBadge = document.querySelector("#status");
const message = document.querySelector("#message");
const report = document.querySelector("#report");
const results = document.querySelector("#results");
const button = form.querySelector("button");

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    return;
  }

  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyPreview.hidden = true;
  heatmapWrap.hidden = true;
  report.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    return;
  }

  const payload = new FormData();
  payload.append("file", file);
  button.disabled = true;
  statusBadge.textContent = "Analyzing";
  message.hidden = true;
  report.hidden = true;
  heatmapWrap.hidden = true;
  results.replaceChildren();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: payload,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }
    renderResponse(data);
  } catch (error) {
    statusBadge.textContent = "Error";
    message.textContent = error.message;
    message.hidden = false;
  } finally {
    button.disabled = false;
  }
});

function renderResponse(data) {
  statusBadge.textContent = data.status === "ok" ? "Complete" : "Needs model";

  if (data.message) {
    message.textContent = data.message;
    message.hidden = false;
  }

  if (data.report) {
    report.textContent = data.report;
    report.hidden = false;
  }

  if (data.heatmap) {
    heatmap.src = data.heatmap.image_data_url;
    heatmapLabel.textContent = `${data.heatmap.label} ${Math.round(data.heatmap.probability * 100)}%`;
    heatmapWrap.hidden = false;
  }

  if (!data.findings.length) {
    results.innerHTML = "";
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of data.findings) {
    const card = document.createElement("article");
    card.className = "finding";

    const label = document.createElement("strong");
    label.textContent = item.label;

    const probability = document.createElement("div");
    probability.className = `probability${item.positive ? " positive" : ""}`;
    probability.textContent = `${Math.round(item.probability * 100)}%`;

    card.append(label, probability);
    fragment.append(card);
  }
  results.replaceChildren(fragment);
}
