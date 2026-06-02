/**
 * Drift monitor dashboard — cadence cards (OK / ALERT / MISSED).
 */

const Shell = globalThis.MLBotConsole;

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const CADENCE_LABELS = {
  daily: "日更",
  weekly: "周更",
  monthly: "月更",
  quarterly: "季更",
  yearly: "年更",
};

function statusClass(st) {
  if (st === "ALERT") return "monitor-card-alert";
  if (st === "MISSED") return "monitor-card-missed";
  return "monitor-card-ok";
}

function fmtAge(hours) {
  if (hours == null || Number.isNaN(hours)) return "从未运行";
  if (hours < 48) return `${hours.toFixed(1)} 小时前`;
  return `${(hours / 24).toFixed(1)} 天前`;
}

function renderCards(cards) {
  const el = document.getElementById("cadenceCards");
  if (!cards.length) {
    el.innerHTML = '<p class="muted">无调度记录（远程需 enable systemd timer）</p>';
    return;
  }
  el.innerHTML = cards
    .map((c) => {
      const label = CADENCE_LABELS[c.cadence] || c.cadence;
      const st = c.display_status || "—";
      const wd = c.watchdog_any_alert ? "ALERT" : "OK";
      let dr = "—";
      if (c.drift_any_alert) dr = "ALERT";
      else if (c.drift_no_plateaus) dr = "未校准";
      else if (c.drift_any_alert === false) dr = "OK";
      const out = c.output_dir
        ? `<div class="monitor-card-meta"><code>${esc(c.output_dir)}</code></div>`
        : "";
      return `<article class="monitor-card ${statusClass(st)}">
        <div class="monitor-card-head">
          <span class="monitor-card-title">${esc(label)}</span>
          <span class="monitor-card-badge">${esc(st)}</span>
        </div>
        <div class="monitor-card-body">
          <div>最近运行：${esc(c.run_ts || "—")}（${esc(fmtAge(c.age_hours))}）</div>
          <div>watchdog：${esc(wd)} · drift：${esc(dr)}</div>
          <div class="muted">上限 ${esc(c.max_age_hours)}h 内有效</div>
          ${out}
        </div>
      </article>`;
    })
    .join("");
}

function renderBanner(summary, indexUpdated) {
  const el = document.getElementById("summaryBanner");
  const parts = [];
  if (summary.any_alert) parts.push('<strong class="pnl-neg">存在 ALERT</strong>');
  if (summary.any_missed) parts.push('<strong class="monitor-missed-text">存在缺勤</strong>');
  if (summary.any_uncalibrated) {
    parts.push('<strong class="monitor-uncal-text">plateau 未校准（需 Tier-0）</strong>');
  }
  if (!parts.length) parts.push('<strong class="pnl-pos">全部 cadence 正常</strong>');
  const ts = indexUpdated ? ` · 索引 ${esc(indexUpdated)}` : "";
  el.innerHTML = `${parts.join(" · ")}${ts}`;
  el.className = summary.any_alert || summary.any_missed || summary.any_uncalibrated
    ? "monitoring-banner monitor-banner-warn"
    : "monitoring-banner monitor-banner-ok";
}

function renderAlerts(strategyAlerts) {
  const tbody = document.getElementById("alertsBody");
  const rows = [];
  for (const [cadence, items] of Object.entries(strategyAlerts || {})) {
    for (const it of items || []) {
      rows.push({ cadence, source: it.source, strategy: it.strategy });
    }
  }
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">无策略级 ALERT</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => `<tr>
      <td>${esc(CADENCE_LABELS[r.cadence] || r.cadence)}</td>
      <td>${esc(r.source)}</td>
      <td><strong>${esc(r.strategy)}</strong></td>
    </tr>`
    )
    .join("");
}

async function refresh() {
  const status = document.getElementById("statusLine");
  status.textContent = "加载中…";
  try {
    const { data } = await Shell.api("/api/monitoring/dashboard");
    renderBanner(data.summary || {}, data.index_updated_at);
    renderCards(data.cards || []);
    renderAlerts(data.strategy_alerts || {});
    status.textContent = `已刷新 ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    status.textContent = String(e.message || e);
    document.getElementById("cadenceCards").innerHTML = "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  Shell.initAppNav("monitoring");
  Shell.initExtLinks?.();
  document.getElementById("refreshBtn")?.addEventListener("click", refresh);
  refresh();
});
