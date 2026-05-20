/**
 * Orders page — full-table view with scope filters (separate from Trade Map).
 */

const Core = globalThis.MLBotTradeMapCore;
const Shell = globalThis.MLBotConsole;
const POLL_MS = 15000;

let pollTimer;

function layersState() {
  return {
    trend: document.getElementById("layerTrend").checked,
    spot: document.getElementById("layerSpot").checked,
    multiLeg: document.getElementById("layerMultiLeg").checked,
    pending: false,
  };
}

function scopesParam() {
  return Core.scopesFromLayers(layersState());
}

function setStatus(msg) {
  document.getElementById("statusLine").textContent = msg;
}

function persistScopes() {
  Shell.setScopesState(layersState());
}

function applyScopesFromStorage() {
  const saved = Shell.getScopesDefault();
  if (!saved) return;
  if (saved.trend != null) document.getElementById("layerTrend").checked = !!saved.trend;
  if (saved.spot != null) document.getElementById("layerSpot").checked = !!saved.spot;
  if (saved.multiLeg != null) document.getElementById("layerMultiLeg").checked = !!saved.multiLeg;
}

async function showOrderDetail(row, markerId) {
  const body = document.getElementById("orderDetailBody");
  const linkEl = document.getElementById("orderMapLink");
  linkEl.innerHTML = "";
  body.textContent = JSON.stringify(row, null, 2);
  if (markerId) {
    const symbol = document.getElementById("symbolSelect").value;
    const href = `/trade-map?symbol=${encodeURIComponent(symbol)}&marker_id=${encodeURIComponent(markerId)}`;
    linkEl.innerHTML = `<a href="${href}">在交易地图中查看标记</a>`;
    try {
      const { data } = await Shell.api(
        `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(markerId)}`
      );
      body.textContent = JSON.stringify({ order: row, db: data }, null, 2);
    } catch (e) {
      body.textContent += `\n\n(marker-detail: ${e})`;
    }
  }
}

async function refreshOrders() {
  const symbol = document.getElementById("symbolSelect").value;
  Shell.setSymbol(symbol);
  persistScopes();
  const scopes = scopesParam();
  const status = document.getElementById("statusFilter").value;
  const tbody = document.getElementById("ordersBody");
  const countEl = document.getElementById("ordersCount");
  setStatus("加载中…");
  const q = new URLSearchParams({
    symbol,
    scopes,
    limit: "500",
  });
  if (status) q.set("status", status);
  try {
    const { data, meta } = await Shell.api(`/api/orders/list?${q}`);
    const rows = data || [];
    countEl.textContent = `(${meta.count ?? rows.length})`;
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">无订单</td></tr>';
      setStatus(`${symbol} · 0 条 · ${new Date().toLocaleTimeString()}`);
      return;
    }
    tbody.innerHTML = rows
      .map((r, i) => {
        const mid = r.marker_id || "";
        return `<tr data-idx="${i}" data-marker-id="${mid}">
          <td>${r.scope}</td>
          <td>${Shell.formatOrderTime(r.time)}</td>
          <td>${r.side || ""}</td>
          <td>${r.status || ""}</td>
          <td>${r.filled_quantity ?? r.quantity ?? ""}</td>
          <td>${r.average_price ?? r.price ?? ""}</td>
          <td class="id-cell" title="${r.order_id || ""}">${r.order_id || ""}</td>
        </tr>`;
      })
      .join("");
    tbody.querySelectorAll("tr[data-idx]").forEach((tr) => {
      tr.addEventListener("click", () => {
        tbody.querySelectorAll("tr").forEach((x) => x.classList.remove("selected"));
        tr.classList.add("selected");
        const idx = Number(tr.getAttribute("data-idx"));
        const row = rows[idx];
        const mid = tr.getAttribute("data-marker-id");
        showOrderDetail(row, mid || null);
      });
    });
    setStatus(
      `${symbol} · ${rows.length} 条 · ${scopes} · ${new Date().toLocaleTimeString()}`
    );
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="muted">${e}</td></tr>`;
    countEl.textContent = "";
    setStatus(String(e));
  }
}

function bindControls() {
  const rerun = () => refreshOrders().catch((e) => setStatus(String(e)));
  [
    "symbolSelect",
    "statusFilter",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
  ].forEach((id) => document.getElementById(id).addEventListener("change", rerun));
  document.getElementById("refreshBtn").addEventListener("click", rerun);
  Shell.bindSymbolPersist("symbolSelect");
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => refreshOrders().catch(() => {}), POLL_MS);
}

(async () => {
  try {
    Shell.initAppNav("orders");
    applyScopesFromStorage();
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect");
    await refreshOrders();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
