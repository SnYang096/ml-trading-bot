/**
 * Orders page — full-table view with scope filters (separate from Trade Map).
 */

const Core = globalThis.MLBotTradeMapCore;
const Shell = globalThis.MLBotConsole;
const POLL_MS = 15000;

let pollTimer;
let lastOrderRows = [];
let lastRowsSignature = "";
let selectedRowIdx = -1;

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

function rowsSignature(rows) {
  return (rows || [])
    .map((r) =>
      [
        r.order_id,
        r.status,
        r.time,
        r.filled_quantity,
        r.take_profit_price,
        r.stop_loss_price,
        r.pnl_usdt,
        r.realized_pnl,
        r.unrealized_pnl,
      ].join("|")
    )
    .join("\n");
}

function fmtSummaryPnl(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}`;
}

async function refreshSummaryStrip(symbol) {
  const el = document.getElementById("ordersSummaryStrip");
  if (!el) return;
  try {
    const q = new URLSearchParams({ symbol, lookback_days: "0", scopes: scopesParam() });
    const { data } = await Shell.api(`/api/account/summary?${q}`);
    const t = data?.totals || {};
    const recent = data?.recent_realized || {};
    const symLabel = Shell.isAllSymbols(symbol) ? "全部" : symbol;
    const scopeNote = scopesParam().replace(/,/g, " · ");
    el.innerHTML =
      `${symLabel}汇总（${scopeNote}）· 已实现 <strong class="${t.realized_pnl > 0 ? "pnl-pos" : t.realized_pnl < 0 ? "pnl-neg" : ""}">${fmtSummaryPnl(t.realized_pnl)}</strong> USDT` +
      ` · 浮盈 <strong>${fmtSummaryPnl(t.unrealized_pnl)}</strong> USDT` +
      ` · 已平仓 ${t.closed_trades ?? 0} 笔` +
      (recent.last_day
        ? ` · 最近日 ${recent.last_day} <strong class="${recent.last_day_pnl > 0 ? "pnl-pos" : recent.last_day_pnl < 0 ? "pnl-neg" : ""}">${fmtSummaryPnl(recent.last_day_pnl)}</strong>`
        : "") +
      ` · <a href="/account?symbol=${encodeURIComponent(symbol)}">账户总览</a>`;
  } catch {
    el.textContent = "";
  }
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
      selectedRowIdx = idx;
      const row = rows[idx];
      const mid = tr.getAttribute("data-marker-id");
      showOrderDetail(row, mid || null);
    });
  });
  if (!tbody.querySelector("tr[data-idx]")) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">无订单</td></tr>`;
  }
}

async function refreshOrders(opts = {}) {
  const silent = !!opts.silent;
  const symbol = symbolFilterValue();
  if (!Shell.isAllSymbols(symbol)) Shell.setSymbol(symbol);
  persistScopes();
  const scopes = scopesParam();
  const status = document.getElementById("statusFilter").value;
  const tbody = document.getElementById("ordersBody");
  const countEl = document.getElementById("ordersCount");
  const colspan = Shell.ordersTableColspan(showSymbolColumn());
  updateOrdersTableHead();
  const q = new URLSearchParams({
    symbol,
    scopes,
    limit: "500",
  });
  if (status) q.set("status", status);
  const exclude = Shell.ordersExcludeStatusParamFromFilter(Shell.ordersFilterFromControls());
  if (exclude) q.set("exclude_status", exclude);
  try {
    const { data, meta } = await Shell.api(`/api/orders/list?${q}`);
    const rows = data || [];
    const sig = rowsSignature(rows);
    if (silent && sig === lastRowsSignature && rows.length) {
      countEl.textContent = `(${meta.count ?? rows.length})`;
      return;
    }
    lastOrderRows = rows;
    lastRowsSignature = sig;
    countEl.textContent = `(${meta.count ?? rows.length})`;
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">无订单</td></tr>`;
      document.getElementById("orderDetailBody").innerHTML =
        '<p class="muted">选择一行查看详情</p>';
      return;
    }
    tbody.innerHTML = Shell.buildOrdersTableRows(rows, {
      showSymbol: showSymbolColumn(),
      escHtml: Shell.escHtml,
    });
    Shell.bindOrdersTableResize(document.getElementById("ordersTable"));
    bindOrdersTable(rows);
    if (
      selectedRowIdx >= 0 &&
      selectedRowIdx < rows.length &&
      tbody.querySelector(`tr[data-idx="${selectedRowIdx}"]`)
    ) {
      const tr = tbody.querySelector(`tr[data-idx="${selectedRowIdx}"]`);
      tr.classList.add("selected");
      const mid = tr.getAttribute("data-marker-id");
      showOrderDetail(rows[selectedRowIdx], mid || null);
    }
    refreshSummaryStrip(symbol).catch(() => {});
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted">${Shell.escHtml(String(e))}</td></tr>`;
    countEl.textContent = "";
    console.error(e);
  }
}

function bindControls() {
  const rerun = () => refreshOrders({ silent: false }).catch((e) => console.error(e));
  [
    "symbolSelect",
    "statusFilter",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "hideExpired",
    "hideCanceled",
    "hideRejected",
    "hidePending",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      if (["hideExpired", "hideCanceled", "hideRejected", "hidePending"].includes(id)) {
        Shell.saveOrdersFilter(Shell.ordersFilterFromControls());
      }
      rerun();
    });
  });
  document.getElementById("refreshBtn").addEventListener("click", rerun);
  Shell.bindSymbolPersist("symbolSelect");
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(
    () => refreshOrders({ silent: true }).catch((e) => console.error(e)),
    POLL_MS
  );
}

(async () => {
  try {
    Shell.initAppNav("orders");
    Shell.bindOrdersTableResize(document.getElementById("ordersTable"));
    Shell.applyOrdersFilterToControls(Shell.loadOrdersFilter());
    Shell.bindOrdersFilterSync(() => refreshOrders().catch(() => {}));
    applyScopesFromStorage();
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect", null, { includeAll: true });
    await refreshOrders();
    startPoll();
  } catch (e) {
    console.error("orders page init failed:", e);
  }
})();
