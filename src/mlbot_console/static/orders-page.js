/**
 * Orders page — full-table view with scope filters (separate from Trade Map).
 */

const Core = globalThis.MLBotTradeMapCore;
const Shell = globalThis.MLBotConsole;
const POLL_MS = 15000;

let pollTimer;
let lastOrderRows = [];

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

function symbolFilterValue() {
  return document.getElementById("symbolSelect").value;
}

function showSymbolColumn() {
  return Shell.isAllSymbols(symbolFilterValue());
}

function updateOrdersTableHead() {
  const showSym = showSymbolColumn();
  const symTh = document.getElementById("ordersThSymbol");
  if (symTh) symTh.classList.toggle("hidden", !showSym);
}

async function showOrderDetail(row, markerId) {
  const body = document.getElementById("orderDetailBody");
  const linkEl = document.getElementById("orderMapLink");
  linkEl.innerHTML = "";
  body.className = "order-detail-body";
  body.innerHTML = Shell.renderOrderDetailHtml(row, null);
  if (markerId) {
    const sym = row.symbol || symbolFilterValue();
    const mapSym = Shell.isAllSymbols(sym) ? row.symbol : sym;
    const href = `/trade-map?symbol=${encodeURIComponent(mapSym || "ETHUSDT")}&marker_id=${encodeURIComponent(markerId)}`;
    linkEl.innerHTML = `<a href="${href}">在交易地图中查看标记</a>`;
    try {
      const { data } = await Shell.api(
        `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(markerId)}`
      );
      body.innerHTML = Shell.renderOrderDetailHtml(row, data);
    } catch (e) {
      body.innerHTML =
        Shell.renderOrderDetailHtml(row, null) +
        `<p class="order-detail-error muted">marker-detail: ${Shell.escHtml(String(e))}</p>`;
    }
  }
}

function bindOrdersTable(rows) {
  const tbody = document.getElementById("ordersBody");
  const colspan = Shell.ordersTableColspan(showSymbolColumn());
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
  if (!tbody.querySelector("tr[data-idx]")) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">无订单</td></tr>`;
  }
}

async function refreshOrders() {
  const symbol = symbolFilterValue();
  if (!Shell.isAllSymbols(symbol)) Shell.setSymbol(symbol);
  persistScopes();
  const scopes = scopesParam();
  const status = document.getElementById("statusFilter").value;
  const tbody = document.getElementById("ordersBody");
  const countEl = document.getElementById("ordersCount");
  const colspan = Shell.ordersTableColspan(showSymbolColumn());
  updateOrdersTableHead();
  setStatus("加载中…");
  const q = new URLSearchParams({
    symbol,
    scopes,
    limit: "500",
  });
  if (status) q.set("status", status);
  const exclude = [];
  if (document.getElementById("hideExpired")?.checked) exclude.push("expired");
  if (document.getElementById("hideCanceled")?.checked) exclude.push("canceled");
  if (exclude.length) q.set("exclude_status", exclude.join(","));
  try {
    const { data, meta } = await Shell.api(`/api/orders/list?${q}`);
    const rows = data || [];
    lastOrderRows = rows;
    countEl.textContent = `(${meta.count ?? rows.length})`;
    const symLabel = Shell.isAllSymbols(symbol) ? "全部" : symbol;
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">无订单</td></tr>`;
      document.getElementById("orderDetailBody").innerHTML =
        '<p class="muted">选择一行查看详情</p>';
      setStatus(`${symLabel} · 0 条 · ${new Date().toLocaleTimeString()}`);
      return;
    }
    tbody.innerHTML = Shell.buildOrdersTableRows(rows, {
      showSymbol: showSymbolColumn(),
      escHtml: Shell.escHtml,
    });
    bindOrdersTable(rows);
    setStatus(`${symLabel} · ${rows.length} 条 · ${scopes} · ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">${Shell.escHtml(String(e))}</td></tr>`;
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
    "hideExpired",
    "hideCanceled",
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
    await Shell.loadSymbols("symbolSelect", null, { includeAll: true });
    await refreshOrders();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
