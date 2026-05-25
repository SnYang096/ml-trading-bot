/**
 * Regime ops — read-only regime.yaml + drift monitor snapshot.
 */

const Shell = globalThis.MLBotConsole;

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtLc(lc) {
  if (!lc || typeof lc !== "object") return "—";
  const notes = lc.notes ? String(lc.notes).slice(0, 80) : "";
  if (notes) return notes;
  const ts = lc.timestamp || lc.data_source || "";
  return ts || "—";
}

function fmtIsoShort(iso) {
  const s = String(iso || "").trim();
  if (!s) return "";
  const norm = s.includes("T") ? s.replace("T", " ") : s;
  return norm.length > 19 ? norm.slice(0, 19) : norm;
}

/** Drift check time (report) or config baseline (calibrate / eval writeback). */
function fmtDriftTime(row, meta) {
  const checked = row.drift_checked_at || meta?.drift_generated_at;
  if (checked) return { prefix: "检测", at: fmtIsoShort(checked) };
  const baseline = row.config_reference_at;
  if (baseline) return { prefix: "基准", at: fmtIsoShort(baseline) };
  return null;
}

function fmtDriftCell(row, meta) {
  const st = row.drift_status || "—";
  const detail = row.drift_detail || "";
  const cls =
    st === "漂移" || st === "告警"
      ? "pnl-neg"
      : st === "正常"
        ? "pnl-pos"
        : "";
  const title = detail ? ` title="${esc(detail)}"` : "";
  const time = fmtDriftTime(row, meta);
  const timeHtml = time
    ? `<br/><span class="account-sub">${esc(time.prefix)} ${esc(time.at)}</span>`
    : "";
  return `<span class="${cls}"${title}>${esc(st)}</span>${timeHtml}`;
}

function renderRows(rows, meta) {
  const tbody = document.getElementById("regimeBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">无数据</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const present = r.present ? "✓" : "—";
      const layer = r.account_layer_title || r.account_layer || "—";
      const src = r.regime_source || "—";
      return `<tr data-account-layer="${esc(r.account_layer || "")}">
        <td>${esc(layer)}</td>
        <td><strong>${esc(r.strategy)}</strong></td>
        <td class="muted regime-config-cell">${present} ${esc(src)}<br/><span class="account-sub">${esc(r.regime_path)}</span></td>
        <td>${r.n_rules ?? 0}</td>
        <td>${esc((r.allowed_sides || []).join(", "))}</td>
        <td>${esc(fmtLc(r.last_calibration))}</td>
        <td>${fmtDriftCell(r, meta)}</td>
      </tr>`;
    })
    .join("");
}

async function refresh() {
  document.getElementById("statusLine").textContent = "加载中…";
  const { data, meta } = await Shell.api("/api/trend/regime-ops");
  renderRows(data || [], meta || {});
  const driftHint = meta.drift_report_path
    ? ` · drift ${meta.drift_generated_at ? String(meta.drift_generated_at).slice(0, 19) : meta.drift_report_path}`
    : " · 无 drift 报告";
  document.getElementById("statusLine").textContent =
    `${meta.count ?? 0} strategies · ${meta.strategies_root || ""}${driftHint} · ${new Date().toLocaleTimeString()}`;
}

document.getElementById("refreshBtn").addEventListener("click", () => refresh().catch((e) => {
  document.getElementById("statusLine").textContent = String(e);
}));

(async () => {
  Shell.initAppNav("regime");
  if (Shell.loadExtLinks) await Shell.loadExtLinks();
  await refresh();
})();
