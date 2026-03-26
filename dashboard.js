/**
 * dashboard.js  —  FMCG AI Frontend Logic  (v2 — Automated Forecast)
 * ====================================================================
 *
 * WHAT CHANGED IN v2
 * ──────────────────
 * runForecast() previously sent 7 fields:
 *   { product, month, price, promotion_flag, lag_1, lag_2, lag_3 }
 *
 * It now sends ONLY 2 fields:
 *   { product, month }
 *
 * The backend (forecast_model.py) derives everything else automatically.
 *
 * The API response now includes extra transparency fields:
 *   data_source, lag_1, lag_2, lag_3, price_used, promotion_flag
 * These are displayed in the "Auto-Generated Features" panel so the user
 * (and viva examiner) can see exactly what the model used.
 *
 * SECTION MAP
 * ───────────
 * Navigation      → nav-item click handlers
 * Init            → DOMContentLoaded (loads KPIs + overview charts)
 * runForecast()   → POST /api/forecast  ← v2 (product + month only)
 * runInventory()  → POST /api/inventory
 * loadSupply()    → GET  /api/supply-chain
 * loadAnomalies() → GET  /api/anomalies
 * loadHistory()   → GET  /api/history  (overview chart)
 */

"use strict";

// ── Chart.js global defaults (dark theme) ──────────────────────────────────
Chart.defaults.color        = "#6b6b85";
Chart.defaults.borderColor  = "#1a1a2e";
Chart.defaults.font.family  = "'DM Mono', monospace";

const ACCENT  = "#00d4aa";
const ACCENT2 = "#7c3aed";
const WARN    = "#f59e0b";
const DANGER  = "#ef4444";
const MUTED   = "#1a1a2e";

// Registry of active Chart.js instances — destroy before redraw to avoid leaks
const charts = {};
function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

// Month name lookup for display
const MONTH_NAMES = [
  "", "January","February","March","April","May","June",
  "July","August","September","October","November","December"
];


// ════════════════════════════════════════
// NAVIGATION
// ════════════════════════════════════════
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const sec = item.dataset.section;

    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));

    item.classList.add("active");
    document.getElementById("sec-" + sec)?.classList.add("active");

    const titles = {
      overview : "Overview",
      forecast : "AI Demand Forecast",
      inventory: "Inventory Optimisation",
      supply   : "Supply Chain Dispatch",
      anomaly  : "Risk & Anomaly Detection",
    };
    document.getElementById("pageTitle").textContent = titles[sec] || sec;
  });
});


// ════════════════════════════════════════
// INIT on page load
// ════════════════════════════════════════
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("lastUpdated").textContent =
    "Last updated: " + new Date().toLocaleTimeString();

  // Load model metrics into KPI cards
  fetch("/api/metrics", { credentials: "include" })
    .then(r => r.json())
    .then(d => {
      if (d.accuracy !== undefined) {
        document.querySelector("#kpiAccuracy .kpi-value").textContent  = d.accuracy + "%";
        document.querySelector("#kpiAccuracy .kpi-value").style.color  = ACCENT;
        document.querySelector("#kpiMAE  .kpi-value").textContent      = d.mae;
        document.querySelector("#kpiRMSE .kpi-value").textContent      = d.rmse;
      }
    })
    .catch(() => {});

  loadHistory();
  loadOverviewAnomalyChart();
});


// ════════════════════════════════════════
// AUTH
// ════════════════════════════════════════
async function logout() {
  await fetch("/api/logout", { method: "POST", credentials: "include" });
  window.location.href = "/";
}


// ════════════════════════════════════════
// LOADING STATE HELPERS
// ════════════════════════════════════════
function setLoading(btn, loading) {
  if (loading) {
    btn._orig    = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML =
      '<span style="display:inline-block;width:14px;height:14px;border:2px solid rgba(0,0,0,.3);' +
      'border-top-color:#071a14;border-radius:50%;animation:spin .6s linear infinite;' +
      'vertical-align:middle;margin-right:8px"></span>Processing…';
  } else {
    btn.disabled = false;
    btn.innerHTML = btn._orig;
  }
}


// ════════════════════════════════════════
// OVERVIEW — Actual vs Predicted Chart
// ════════════════════════════════════════
async function loadHistory() {
  const product = document.getElementById("productSelect")?.value || "Product_A";
  try {
    const res  = await fetch(`/api/history?product=${product}&n=16`, { credentials: "include" });
    const data = await res.json();
    if (data.error) return;

    destroyChart("forecast");
    const ctx = document.getElementById("forecastChart").getContext("2d");
    charts.forecast = new Chart(ctx, {
      type: "line",
      data: {
        labels  : data.labels,
        datasets: [
          {
            label          : "Actual Demand",
            data           : data.actual,
            borderColor    : ACCENT,
            backgroundColor: "rgba(0,212,170,.08)",
            tension        : 0.4,
            fill           : true,
            pointRadius    : 3,
          },
          {
            label          : "Predicted Demand",
            data           : data.predicted,
            borderColor    : ACCENT2,
            backgroundColor: "rgba(124,58,237,.08)",
            tension        : 0.4,
            borderDash     : [5, 3],
            fill           : true,
            pointRadius    : 3,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales : {
          x: { ticks: { maxTicksLimit: 8, maxRotation: 30 } },
          y: { beginAtZero: false },
        },
      },
    });
  } catch (e) { console.warn("History load failed:", e); }
}

async function loadOverviewAnomalyChart() {
  try {
    const res  = await fetch("/api/anomalies", { credentials: "include" });
    const data = await res.json();
    if (data.error) return;

    const riskEl = document.querySelector("#kpiRisk .kpi-value");
    if (riskEl) {
      riskEl.textContent = data.risk_score;
      riskEl.style.color = data.risk_level === "HIGH" ? DANGER : WARN;
    }

    const labels = data.points.map(p => p.date.slice(5, 10));
    const values = data.points.map(p => p.value);
    const colors = data.points.map(p => p.anomaly ? DANGER : ACCENT);

    destroyChart("anomalyOverview");
    const ctx = document.getElementById("anomalyChart").getContext("2d");
    charts.anomalyOverview = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: "Demand", data: values, backgroundColor: colors, borderRadius: 4 }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales : { y: { beginAtZero: false } },
      },
    });
  } catch (e) { console.warn(e); }
}


// ════════════════════════════════════════
// FORECAST  (v2 — product + month only)
// ════════════════════════════════════════
async function runForecast() {
  const btn = document.getElementById("fcBtn");
  setLoading(btn, true);

  /*
   * ╔════════════════════════════════════════════════════════════════╗
   * ║  v2 PAYLOAD — only 2 fields sent to the server                ║
   * ║  All features (lags, rolling means, price, promo) are AUTO    ║
   * ╚════════════════════════════════════════════════════════════════╝
   */
  const payload = {
    product : document.getElementById("fc_product").value,
    month   : parseInt(document.getElementById("fc_month").value, 10),
  };

  try {
    const res  = await fetch("/api/forecast", {
      method     : "POST",
      headers    : { "Content-Type": "application/json" },
      body       : JSON.stringify(payload),
      credentials: "include",
    });
    const data = await res.json();

    if (data.error) {
      alert("Forecast Error: " + data.error);
      return;
    }

    // ── Show results container ─────────────────────────────────────────────
    document.getElementById("fcResults").style.display = "block";

    // ── Main forecast number ───────────────────────────────────────────────
    document.getElementById("fcVal").textContent  = data.forecasted_demand.toFixed(1);
    document.getElementById("fcMeta").textContent =
      `${payload.product.replace("_", " ")}  ·  ${MONTH_NAMES[data.month]}  ·  Data: ${data.data_source}`;

    // ── KPI cards ──────────────────────────────────────────────────────────
    document.getElementById("fcConf").textContent  = data.confidence;
    document.getElementById("fcAcc").textContent   = data.model_accuracy;
    document.getElementById("fcMAE").textContent   = data.mae;
    document.getElementById("fcRMSE").textContent  = data.rmse;

    // ── Data source badge ──────────────────────────────────────────────────
    const srcBadge = document.getElementById("fcSourceBadge");
    srcBadge.textContent  = data.data_source === "csv" ? "📊 Real Data" : "🔬 Synthetic";
    srcBadge.className    = "af-source " +
      (data.data_source === "csv" ? "source-csv" : "source-synthetic");

    // ── Confidence gauge chart ─────────────────────────────────────────────
    destroyChart("fcGauge");
    const ctx = document.getElementById("fcGaugeChart").getContext("2d");
    charts.fcGauge = new Chart(ctx, {
      type: "bar",
      data: {
        labels  : ["Confidence", "Remaining"],
        datasets: [{
          data           : [data.confidence, 100 - data.confidence],
          backgroundColor: [ACCENT, MUTED],
          borderRadius   : 6,
        }],
      },
      options: {
        indexAxis : "y",
        responsive: true,
        plugins   : { legend: { display: false } },
        scales    : {
          x: { max: 100, ticks: { callback: v => v + "%" } },
          y: { ticks: { display: false } },
        },
      },
    });

    // ── Auto-features transparency panel ──────────────────────────────────
    // Shows exactly what the AI engine computed, useful for demo + viva
    _renderAutoFeatures(data);

  } catch (e) {
    alert("Network error. Is Flask running on port 5000?");
    console.error(e);
  } finally {
    setLoading(btn, false);
  }
}


/**
 * Renders the "Auto-Generated Features" panel.
 * Shows the 7 features the model derived automatically — for transparency.
 */
function _renderAutoFeatures(data) {
  const panel = document.getElementById("afPanel");
  const grid  = document.getElementById("afGrid");

  const items = [
    { label: "lag_1 (last week)",    value: data.lag_1 },
    { label: "lag_2 (2 weeks ago)",  value: data.lag_2 },
    { label: "lag_3 (3 weeks ago)",  value: data.lag_3 },
    { label: "rolling_mean_3",       value: ((data.lag_1 + data.lag_2 + data.lag_3) / 3).toFixed(2) },
    { label: "price used (₹)",       value: data.price_used },
    { label: "promotion_flag",       value: data.promotion_flag === 1 ? "Yes (1)" : "No (0)" },
    { label: "data_source",          value: data.data_source },
  ];

  grid.innerHTML = items.map(item => `
    <div class="af-item">
      <div class="af-label">${item.label}</div>
      <div class="af-value">${item.value}</div>
    </div>
  `).join("");

  panel.classList.add("show");
}


// ════════════════════════════════════════
// INVENTORY
// ════════════════════════════════════════
async function runInventory() {
  const btn = document.querySelector("#sec-inventory .btn-primary");
  setLoading(btn, true);

  const body = {
    product              : document.getElementById("inv_product").value,
    current_stock        : +document.getElementById("inv_stock").value,
    forecasted_demand    : +document.getElementById("inv_demand").value,
    lead_time_days       : +document.getElementById("inv_lead").value,
    ordering_cost        : +document.getElementById("inv_order_cost").value,
    holding_cost_per_unit: +document.getElementById("inv_hold").value,
  };

  try {
    const res = await fetch("/api/inventory", {
      method     : "POST",
      headers    : { "Content-Type": "application/json" },
      body       : JSON.stringify(body),
      credentials: "include",
    });
    const d = await res.json();
    if (d.error) { alert(d.error); return; }

    document.getElementById("invResults").style.display = "block";

    const alertEl      = document.getElementById("invAlert");
    alertEl.textContent = d.alert;
    alertEl.className   = "alert-banner alert-" + d.alert_level;

    document.getElementById("invStock").textContent = d.current_stock;
    document.getElementById("invROP").textContent   = d.reorder_point;
    document.getElementById("invEOQ").textContent   = d.reorder_quantity;
    document.getElementById("invSS").textContent    = d.safety_stock;
    document.getElementById("invDays").textContent  = d.days_of_stock;

    destroyChart("invChart");
    const ctx = document.getElementById("invChart").getContext("2d");
    charts.invChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels  : ["Current Stock", "Reorder Point", "Order Qty (EOQ)", "Safety Stock"],
        datasets: [{
          data           : [d.current_stock, d.reorder_point, d.reorder_quantity, d.safety_stock],
          backgroundColor: [d.should_reorder ? DANGER : ACCENT, WARN, ACCENT2, "#6b6b85"],
          borderRadius   : 6,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales : { y: { beginAtZero: true } },
      },
    });
  } catch (e) {
    alert("Backend error. Is Flask running?");
  } finally {
    setLoading(btn, false);
  }
}


// ════════════════════════════════════════
// SUPPLY CHAIN
// ════════════════════════════════════════
async function loadSupply() {
  const btn = document.querySelector("#sec-supply .btn-primary");
  setLoading(btn, true);

  try {
    const res = await fetch("/api/supply-chain", { credentials: "include" });
    const d   = await res.json();
    if (d.error) { alert(d.error); return; }

    document.getElementById("supplyResults").style.display = "block";
    document.getElementById("scWarehouse").textContent  = d.warehouse_stock;
    document.getElementById("scDispatched").textContent = d.total_dispatched;
    document.getElementById("scEfficiency").textContent = d.efficiency_pct + "%";
    document.getElementById("scFulfilled").textContent  = d.nodes_fulfilled + "/" + d.total_nodes;

    let html = `<table class="dispatch-table">
      <tr><th>Retailer</th><th>Location</th><th>Priority</th><th>Ideal</th><th>Dispatched</th><th>Days Cover</th><th>Status</th></tr>`;
    d.dispatch_plan.forEach(row => {
      html += `<tr>
        <td>${row.node}</td><td>${row.location}</td><td>${row.priority_score}</td>
        <td>${row.ideal_qty}</td><td>${row.dispatched_qty}</td>
        <td>${row.days_after}</td><td>${row.status}</td>
      </tr>`;
    });
    html += "</table>";
    document.getElementById("dispatchTable").innerHTML = html;

    destroyChart("supplyChart");
    const ctx = document.getElementById("supplyChart").getContext("2d");
    charts.supplyChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels  : d.dispatch_plan.map(r => r.location),
        datasets: [
          { label: "Ideal Qty",  data: d.dispatch_plan.map(r => r.ideal_qty),
            backgroundColor: "rgba(124,58,237,.4)", borderRadius: 4 },
          { label: "Dispatched", data: d.dispatch_plan.map(r => r.dispatched_qty),
            backgroundColor: ACCENT, borderRadius: 4 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales : { y: { beginAtZero: true } },
      },
    });
  } catch (e) {
    alert("Backend error.");
  } finally {
    setLoading(btn, false);
  }
}


// ════════════════════════════════════════
// ANOMALY DETECTION
// ════════════════════════════════════════
async function loadAnomalies() {
  const btn = document.querySelector("#sec-anomaly .btn-primary");
  setLoading(btn, true);

  try {
    const res = await fetch("/api/anomalies", { credentials: "include" });
    const d   = await res.json();
    if (d.error) { alert(d.error); return; }

    document.getElementById("anomalyResults").style.display = "block";
    document.getElementById("riskLevel").textContent = d.risk_level;
    document.getElementById("riskScore").textContent = d.risk_score;
    document.getElementById("riskCount").textContent = d.total_anomalies;
    document.getElementById("riskMean").textContent  = d.mean;

    const labels   = d.points.map(p => p.date.slice(5));
    const values   = d.points.map(p => p.value);
    const ptColors = d.points.map(p => p.anomaly ? DANGER : ACCENT);

    destroyChart("anomalyTimeline");
    const ctx = document.getElementById("anomalyTimeline").getContext("2d");
    charts.anomalyTimeline = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label              : "Demand",
            data               : values,
            borderColor        : ACCENT,
            backgroundColor    : "rgba(0,212,170,.06)",
            fill               : true,
            tension            : 0.35,
            pointBackgroundColor: ptColors,
            pointRadius        : d.points.map(p => p.anomaly ? 8 : 3),
            pointBorderColor   : ptColors,
          },
          {
            label      : "Mean",
            data       : Array(values.length).fill(d.mean),
            borderColor: WARN,
            borderDash : [4, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales : { y: { beginAtZero: false } },
      },
    });

    const alertsEl = document.getElementById("alertsList");
    if (!d.alerts || d.alerts.length === 0) {
      alertsEl.innerHTML = `<div class="alert-item" style="border-left:3px solid var(--accent)">
        ✅ No significant anomalies detected in the current period.
      </div>`;
    } else {
      alertsEl.innerHTML = d.alerts.map(a => `
        <div class="alert-item ${a.severity}">
          <div class="alert-date">${a.date}</div>
          <div class="alert-msg">⚠ ${a.message} — Value: ${a.value}</div>
          <div class="alert-badge badge-${a.severity}">${a.severity}</div>
        </div>
      `).join("");
    }
  } catch (e) {
    alert("Anomaly detection failed.");
  } finally {
    setLoading(btn, false);
  }
}