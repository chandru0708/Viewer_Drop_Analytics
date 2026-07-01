let monthlyChart = null;
let segmentChart = null;

const genreFilter = document.getElementById("genreFilter");
const deviceFilter = document.getElementById("deviceFilter");
const planFilter = document.getElementById("planFilter");
const contentTypeFilter = document.getElementById("contentTypeFilter");
const regionFilter = document.getElementById("regionFilter");
const startDate = document.getElementById("startDate");
const endDate = document.getElementById("endDate");
const analysisBtn = document.getElementById("analysisBtn");

function getSelectedValues(selectEl) {
  return selectEl.value ? [selectEl.value] : [];
}

function populateSelect(selectEl, values, placeholder) {
  selectEl.innerHTML = "";
  const first = document.createElement("option");
  first.value = "";
  first.textContent = placeholder;
  selectEl.appendChild(first);

  values.forEach(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectEl.appendChild(option);
  });
}

function buildPayload() {
  return {
    genres: getSelectedValues(genreFilter),
    device_types: getSelectedValues(deviceFilter),
    subscription_plans: getSelectedValues(planFilter),
    content_types: getSelectedValues(contentTypeFilter),
    regions: getSelectedValues(regionFilter),
    start_date: startDate.value || null,
    end_date: endDate.value || null
  };
}

function renderKpis(kpis) {
  document.getElementById("totalSessions").textContent = kpis.total_sessions ?? "—";
  document.getElementById("totalViewers").textContent = kpis.total_viewers ?? "—";
  document.getElementById("avgCompletion").textContent = `${kpis.avg_completion_rate ?? "—"}%`;
  document.getElementById("dropoffRate").textContent = `${kpis.dropoff_rate ?? "—"}%`;
  document.getElementById("anomalyRate").textContent = `${kpis.anomaly_rate ?? "—"}%`;
  document.getElementById("avgEngagement").textContent = kpis.avg_engagement_score ?? "—";
}

function renderMonthlyChart(monthly) {
  const ctx = document.getElementById("monthlyChart").getContext("2d");
  const labels = monthly.map(item => item.session_month);
  const sessions = monthly.map(item => item.sessions);

  if (monthlyChart) monthlyChart.destroy();

  monthlyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Sessions",
        data: sessions,
        borderColor: "#4cc9f0",
        backgroundColor: "rgba(76, 201, 240, 0.12)",
        pointBackgroundColor: "#2dd4bf",
        pointRadius: 3,
        tension: 0.35,
        fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: "#96abc5" },
          grid: { color: "rgba(255,255,255,0.06)" }
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#96abc5" },
          grid: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });
}

function renderSegmentChips(segmentSummary) {
  const wrap = document.getElementById("segmentChips");
  wrap.innerHTML = "";

  segmentSummary.forEach(item => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.dataset.segment = item.segment;
    chip.textContent = `${item.segment} (${item.viewers})`;
    wrap.appendChild(chip);
  });
}

function renderSegmentationChart(segmentSummary) {
  const ctx = document.getElementById("segmentBarChart").getContext("2d");
  const labels = segmentSummary.map(item => item.segment);
  const counts = segmentSummary.map(item => item.viewers);

  if (segmentChart) segmentChart.destroy();

  segmentChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Viewers",
        data: counts,
        backgroundColor: [
          "#4cc9f0",
          "#2dd4bf",
          "#34d399",
          "#fbbf24",
          "#fb7185",
          "#a78bfa"
        ],
        borderRadius: 10
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: "#96abc5" },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#96abc5" },
          grid: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });
}

function renderSegmentationLegend(segmentSummary) {
  const legend = document.getElementById("segmentationLegend");
  if (!segmentSummary.length) {
    legend.textContent = "";
    return;
  }

  const top = segmentSummary[0];
  legend.textContent = `Largest segment: ${top.segment} with ${top.viewers} viewers.`;
}

function renderViewerSample(rows) {
  const tbody = document.querySelector("#segmentViewerTable tbody");
  tbody.innerHTML = "";

  rows.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.viewer_id ?? "—"}</td>
      <td>${row.segment ?? "—"}</td>
      <td>${row.F_score ?? "—"}</td>
      <td>${row.M_score ?? "—"}</td>
      <td>${row.R_score ?? "—"}</td>
      <td>${row.avg_completion_rate ?? "—"}</td>
      <td>${row.avg_engagement_score ?? "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderNlp(nlpResult, keywords) {
  document.getElementById("nlpSentiment").textContent = nlpResult.sentiment ?? "—";
  document.getElementById("nlpPositiveScore").textContent = nlpResult.positive_score ?? "—";
  document.getElementById("nlpNegativeScore").textContent = nlpResult.negative_score ?? "—";
  document.getElementById("nlpCleanText").textContent = nlpResult.clean_text ? `Clean text: ${nlpResult.clean_text}` : "";

  const wrap = document.getElementById("keywordChips");
  wrap.innerHTML = "";

  (keywords || []).forEach(item => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const word = item.keyword ?? item[0] ?? "";
    const count = item.count ?? item[1] ?? 0;
    chip.textContent = `${word} (${count})`;
    wrap.appendChild(chip);
  });
}

function renderDl(dlResult) {
  document.getElementById("dlTrend").textContent = dlResult.trend ?? "—";
  document.getElementById("dlConfidence").textContent = dlResult.confidence ?? "—";

  const forecast = Array.isArray(dlResult.forecast) ? dlResult.forecast : [];
  document.getElementById("dlForecastText").textContent = forecast.length
    ? `Next values: ${forecast.join(", ")}`
    : "";
}

async function loadFilterOptions() {
  const response = await fetch("/api/filter-options");
  const json = await response.json();

  if (json.status !== "success") return;

  const options = json.options || {};
  populateSelect(genreFilter, options.genres || [], "All Genres");
  populateSelect(deviceFilter, options.device_types || [], "All Devices");
  populateSelect(planFilter, options.subscription_plans || [], "All Plans");
  populateSelect(contentTypeFilter, options.content_types || [], "All Content Types");
  populateSelect(regionFilter, options.regions || [], "All Regions");

  if (options.min_date) startDate.value = options.min_date;
  if (options.max_date) endDate.value = options.max_date;
}

async function runAnalysis() {
  const response = await fetch("/api/run-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload())
  });

  const json = await response.json();
  if (json.status !== "success") return;

  renderKpis(json.kpis || {});
  renderMonthlyChart(json.monthly || []);
  renderNlp(json.nlp_result || {}, json.keywords || []);
  renderDl(json.dl_result || {});
}

async function loadSegmentation() {
  const response = await fetch("/api/segmentation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload())
  });

  const json = await response.json();
  if (json.status !== "success") return;

  const summary = json.segment_summary || [];
  renderSegmentChips(summary);
  renderSegmentationChart(summary);
  renderSegmentationLegend(summary);
  renderViewerSample(json.viewer_segments_sample || []);
}

document.addEventListener("click", e => {
  if (e.target.matches(".chip[data-segment]")) {
    const segment = e.target.dataset.segment;
    const legend = document.getElementById("segmentationLegend");
    if (legend) legend.textContent = `Selected segment: ${segment}`;
  }
});

analysisBtn.addEventListener("click", async () => {
  await runAnalysis();
  await loadSegmentation();
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadFilterOptions();
  await runAnalysis();
  await loadSegmentation();
});