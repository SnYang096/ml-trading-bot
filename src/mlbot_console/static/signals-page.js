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

async function refreshSignals() {
  const timeframe = document.getElementById("timeframeSelect").value;
  const lookback = document.getElementById("lookbackSelect").value;
  setStatus("加载中…");
  const q = new URLSearchParams({ timeframe, lookback_days: lookback });
  const { data, meta } = await Shell.api(`/api/trade-map/signals?${q}`);
  renderRows(data || []);
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
