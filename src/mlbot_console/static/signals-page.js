/**
 * Strategy signals overview — universe table (no sidebar).
 */

const Shell = globalThis.MLBotConsole;
const POLL_MS = 20000;

let pollTimer;

function setStatus(msg) {
  document.getElementById("statusLine").textContent = msg;
}

function fmtBarTime(meta) {
  if (!meta || !meta.timestamp) return "—";
  const s = String(meta.timestamp);
  return s.length >= 16 ? s.slice(0, 16).replace("T", " ") : s;
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderRows(rows) {
  const tbody = document.getElementById("signalsBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="muted">无数据</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const t = r.strategies?.trend || {};
      const s = r.strategies?.spot || {};
      const m = r.strategies?.multi_leg || {};
      const mapHref = r.map_href || `/trade-map?symbol=${r.symbol}`;
      return `<tr>
        <td><strong>${esc(r.symbol)}</strong></td>
        <td>${esc(fmtBarTime(r.latest_bar))}</td>
        <td>${r.bars_1min_rows ?? "—"}</td>
        <td><a href="${esc(mapHref)}">地图</a></td>
        <td title="${esc(t.last_summary)}">${esc(t.summary)}</td>
        <td class="muted">${esc(t.last_summary)}</td>
        <td title="${esc((s.blockers || []).join(", "))}">${esc(s.summary)}</td>
        <td title="${esc(m.last_summary)}">${esc(m.summary)}</td>
        <td class="muted">${esc(m.last_summary)}</td>
      </tr>`;
    })
    .join("");
}

function renderFunnelRows(rows) {
  const tbody = document.getElementById("funnelBody");
  if (!tbody) return;
  const flat = [];
  for (const snap of rows || []) {
    const bys = snap.by_strategy || {};
    for (const [strat, st] of Object.entries(bys)) {
      if (!st || typeof st !== "object") continue;
      flat.push({
        timestamp: snap.timestamp,
        symbol: snap.symbol || "",
        strategy: strat,
        regime_passed: st.regime_passed ?? 0,
        regime_denied: st.regime_denied ?? 0,
        prefilter_passed: st.prefilter_passed ?? 0,
        prefilter_denied: st.prefilter_denied ?? 0,
        direction: st.direction ?? 0,
        gate_passed: st.gate_passed ?? 0,
      });
    }
  }
  if (!flat.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="muted">无 funnel 数据（需实盘 stats_15min）</td></tr>';
    return;
  }
  tbody.innerHTML = flat
    .slice(0, 80)
    .map(
      (r) => `<tr>
        <td>${esc(String(r.timestamp || "").slice(0, 16))}</td>
        <td>${esc(r.symbol)}</td>
        <td>${esc(r.strategy)}</td>
        <td>${r.regime_passed}</td>
        <td>${r.regime_denied}</td>
        <td>${r.prefilter_passed}</td>
        <td>${r.prefilter_denied}</td>
        <td>${r.direction}</td>
        <td>${r.gate_passed}</td>
      </tr>`
    )
    .join("");
}

async function refreshFunnel() {
  const symSel = document.getElementById("funnelSymbolSelect");
  const sym = symSel?.value || "";
  const q = new URLSearchParams({ limit: "48" });
  if (sym) q.set("symbol", sym);
  try {
    const { data } = await Shell.api(`/api/trend/funnel?${q}`);
    renderFunnelRows(data || []);
    const symbols = new Set((data || []).map((r) => r.symbol).filter(Boolean));
    if (symSel && symSel.options.length <= 1) {
      for (const s of [...symbols].sort()) {
        const o = document.createElement("option");
        o.value = s;
        o.textContent = s;
        symSel.appendChild(o);
      }
    }
  } catch (_) {
    renderFunnelRows([]);
  }
}

let funnelLoaded = false;

async function refreshSignals() {
  const timeframe = document.getElementById("timeframeSelect").value;
  const lookback = document.getElementById("lookbackSelect").value;
  setStatus("加载中…");
  const q = new URLSearchParams({ timeframe, lookback_days: lookback });
  const { data, meta } = await Shell.api(`/api/trade-map/signals?${q}`);
  renderRows(data || []);
  
  const funnelPanel = document.querySelector(".funnel-panel");
  if (funnelPanel && funnelPanel.open) {
    await refreshFunnel();
  }
  
  setStatus(
    `${meta.count ?? (data || []).length} symbols · ${timeframe} · ${lookback}d · ${new Date().toLocaleTimeString()}`
  );
}

function bindControls() {
  const rerun = () => refreshSignals().catch((e) => setStatus(String(e)));
  ["timeframeSelect", "lookbackSelect"].forEach((id) =>
    document.getElementById(id).addEventListener("change", rerun)
  );
  document.getElementById("refreshBtn").addEventListener("click", rerun);
  const funnelSym = document.getElementById("funnelSymbolSelect");
  if (funnelSym) funnelSym.addEventListener("change", () => refreshFunnel().catch(() => {}));
  
  const funnelPanel = document.querySelector(".funnel-panel");
  if (funnelPanel) {
    funnelPanel.addEventListener("toggle", () => {
      if (funnelPanel.open && !funnelLoaded) {
        refreshFunnel().then(() => {
          funnelLoaded = true;
        }).catch(() => {});
      }
    });
  }
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => refreshSignals().catch(() => {}), POLL_MS);
}

(async () => {
  try {
    Shell.initAppNav("signals");
    bindControls();
    await Shell.loadExtLinks();
    await refreshSignals();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
