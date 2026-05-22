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
  const ts = lc.timestamp || lc.data_source || "";
  const notes = lc.notes ? String(lc.notes).slice(0, 80) : "";
  return [ts, notes].filter(Boolean).join(" · ") || "—";
}

function fmtDrift(d) {
  if (!d) return "—";
  const st = d.status || d.drift_status || d.overall || "";
  return esc(st || JSON.stringify(d).slice(0, 120));
}

function renderRows(rows) {
  const tbody = document.getElementById("regimeBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="muted">无数据</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const present = r.present ? "✓" : "缺失";
      return `<tr>
        <td><strong>${esc(r.strategy)}</strong></td>
        <td class="muted">${present} ${esc(r.regime_path)}</td>
        <td>${r.n_rules ?? 0}</td>
        <td>${esc((r.allowed_sides || []).join(", "))}</td>
        <td>${esc(fmtLc(r.last_calibration))}</td>
        <td>${fmtDrift(r.drift)}</td>
      </tr>`;
    })
    .join("");
}

async function refresh() {
  document.getElementById("statusLine").textContent = "加载中…";
  const { data, meta } = await Shell.api("/api/trend/regime-ops");
  renderRows(data || []);
  document.getElementById("statusLine").textContent =
    `${meta.count ?? 0} strategies · ${meta.strategies_root || ""} · ${new Date().toLocaleTimeString()}`;
}

document.getElementById("refreshBtn").addEventListener("click", () => refresh().catch((e) => {
  document.getElementById("statusLine").textContent = String(e);
}));

(async () => {
  Shell.initAppNav("regime");
  if (Shell.loadExtLinks) await Shell.loadExtLinks();
  await refresh();
})();
