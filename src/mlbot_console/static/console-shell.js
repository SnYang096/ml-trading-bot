/**
 * Shared shell: app nav, API helper, symbol/scopes persistence across pages.
 */
(function (root) {
  const SYMBOL_KEY = "mlbot_console_symbol";
  const SCOPES_KEY = "mlbot_console_scopes";
  const ORDERS_FILTER_KEY = "mlbot_orders_filter";
  const TRADE_MAP_LAYOUT_KEY = "mlbot_trade_map_layout_v2";
  const SYMBOL_ALL = "*";

  const SCOPE_LABELS = {
    trend: "B·Trend",
    spot: "A·Spot",
    multi_leg: "C·Multi-leg",
  };

  const PAGES = [
    { id: "signals", href: "/signals", label: "策略信号" },
    { id: "trade-map", href: "/trade-map", label: "交易地图" },
    { id: "orders", href: "/orders", label: "订单" },
    { id: "regime", href: "/regime", label: "Regime" },
    { id: "account", href: "/account", label: "账户总览" },
  ];

  async function api(path) {
    const r = await fetch(path);
    const text = await r.text();
    let j;
    try {
      j = JSON.parse(text);
    } catch (_) {
      const snippet = String(text || r.statusText || "").slice(0, 120);
      throw new Error(
        r.ok ? `Invalid JSON from ${path}` : `${r.status} ${path}: ${snippet}`
      );
    }
    if (!j.ok) throw new Error(j.error?.message || j.detail || r.statusText || "API error");
    return j;
  }

  function getSymbol() {
    return localStorage.getItem(SYMBOL_KEY) || "";
  }

  function isAllSymbols(sym) {
    const s = String(sym || "").trim();
    return !s || s === SYMBOL_ALL || s.toUpperCase() === "ALL";
  }

  function setSymbol(sym) {
    if (sym && !isAllSymbols(sym)) localStorage.setItem(SYMBOL_KEY, sym);
  }

  function escHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getScopesDefault() {
    try {
      return JSON.parse(localStorage.getItem(SCOPES_KEY) || "null");
    } catch (_) {
      return null;
    }
  }

  function setScopesState(state) {
    localStorage.setItem(SCOPES_KEY, JSON.stringify(state));
  }

  function initAppNav(activePage) {
    const nav = document.getElementById("appNav");
    if (!nav) return;
    nav.innerHTML = PAGES.map((p) => {
      const cls = p.id === activePage ? "app-nav-link active" : "app-nav-link";
      return `<a class="${cls}" href="${p.href}">${p.label}</a>`;
    }).join("");
  }

  function browserLocalUrl(port, path) {
    const host =
      (typeof globalThis !== "undefined" &&
        globalThis.location &&
        globalThis.location.hostname) ||
      "127.0.0.1";
    return `http://${host}:${port}${path || ""}`;
  }

  function resolveLinkUrl(link) {
    if (link && link.id === "grafana") return browserLocalUrl(3000);
    const raw = (link && link.url) || "";
    if (raw.includes("host.docker.internal")) {
      try {
        const u = new URL(raw);
        return browserLocalUrl(u.port || "3000", u.pathname);
      } catch (_) {
        return browserLocalUrl(3000);
      }
    }
    return raw;
  }

  async function loadExtLinks() {
    const nav = document.getElementById("extLinks");
    if (!nav) return;
    try {
      const { data } = await api("/api/links");
      for (const link of data.links || []) {
        const a = document.createElement("a");
        a.href = resolveLinkUrl(link);
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = link.label;
        nav.appendChild(a);
      }
    } catch (_) {
      /* optional */
    }
  }

  async function loadSymbols(selectId, preferred, options) {
    const opts = options || {};
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const { data } = await api("/api/trade-map/symbols");
    sel.innerHTML = "";
    if (opts.includeAll) {
      const allOpt = document.createElement("option");
      allOpt.value = SYMBOL_ALL;
      allOpt.textContent = "全部";
      sel.appendChild(allOpt);
    }
    const list = data.length ? data : [{ symbol: "ETHUSDT" }];
    for (const row of list) {
      const sym = row.symbol || row;
      const opt = document.createElement("option");
      opt.value = sym;
      opt.textContent = sym;
      sel.appendChild(opt);
    }
    const saved = preferred || getSymbol();
    if (saved && [...sel.options].some((o) => o.value === saved)) {
      sel.value = saved;
    } else if (opts.includeAll) {
      sel.value = SYMBOL_ALL;
    } else if (list[0]) {
      sel.value = list[0].symbol || "ETHUSDT";
    }
    if (!isAllSymbols(sel.value)) setSymbol(sel.value);
  }

  function bindSymbolPersist(selectId) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    sel.addEventListener("change", () => {
      if (!isAllSymbols(sel.value)) setSymbol(sel.value);
    });
  }

  function scopeBadge(scope) {
    const id = String(scope || "");
    const label = SCOPE_LABELS[id] || id;
    return `<span class="scope-badge scope-${escHtml(id)}">${escHtml(label)}</span>`;
  }

  function strategyBadge(strategy, scope) {
    const sid = String(strategy || "")
      .trim()
      .toLowerCase();
    if (!sid || sid === "unknown" || sid === "shared") {
      return scopeBadge(scope);
    }
    const label = sid.replace(/_/g, " ");
    return `<span class="strategy-badge strategy-${escHtml(sid)}" title="${escHtml(sid)}">${escHtml(label)}</span>`;
  }

  function statusBadge(status) {
    const st = String(status || "").toLowerCase();
    return `<span class="status-badge status-${escHtml(st)}">${escHtml(status || "—")}</span>`;
  }

  function sideClass(side) {
    const s = String(side || "").toLowerCase();
    if (s === "buy" || s === "long") return "side-long";
    if (s === "sell" || s === "short") return "side-short";
    return "";
  }

  function tryParseJson(value) {
    if (value == null || value === "") return null;
    if (typeof value === "object") return value;
    if (typeof value !== "string") return value;
    try {
      return JSON.parse(value);
    } catch (_) {
      return value;
    }
  }

  function prettyJson(value) {
    const v = tryParseJson(value);
    if (v == null) return "";
    if (typeof v === "string") return v;
    return JSON.stringify(v, null, 2);
  }

  function formatDetailValue(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—";
    if (typeof value === "object") return prettyJson(value);
    const s = String(value);
    if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
      const parsed = tryParseJson(s);
      if (typeof parsed === "object") return prettyJson(parsed);
    }
    return s;
  }

  function renderOrderDetailHtml(order, markerDetail) {
    const o = order || {};
    const rows = [
      ["账户层", SCOPE_LABELS[o.scope] || o.scope],
      ["交易对", o.symbol],
      ["方向", o.side],
      ["状态", o.status],
      ["类型", o.order_type],
      ["数量", o.filled_quantity ?? o.quantity],
      ["价格", o.average_price ?? o.price],
      ["止盈价", formatTpPrice(o)],
      ["止损价", formatSlPrice(o)],
      ["盈亏", formatPnl(o)],
      ["用途", o.purpose ?? o.order_type],
      ["策略", o.strategy],
      ["网格批次", o.grid_batch],
      ["档位", o.leg_label],
      ["成交时间", formatOrderTime(o.time)],
      ["创建", o.created_at],
      ["成交", o.filled_at],
      ["订单号", o.order_id],
      ["标记 ID", o.marker_id],
    ];
    let html = '<div class="order-detail-card">';
    html += '<dl class="order-detail-dl">';
    for (const [label, val] of rows) {
      if (val == null || val === "") continue;
      html += `<dt>${escHtml(label)}</dt><dd class="${sideClass(o.side)}">${escHtml(formatDetailValue(val))}</dd>`;
    }
    html += "</dl>";
    if (markerDetail) {
      html += '<details class="order-detail-block"><summary>标记 / 数据库</summary>';
      html += `<pre class="order-json-pre">${escHtml(prettyJson(markerDetail))}</pre>`;
      html += "</details>";
    }
    html += '<details class="order-detail-block"><summary>原始 JSON</summary>';
    html += `<pre class="order-json-pre">${escHtml(prettyJson(o))}</pre>`;
    html += "</details></div>";
    return html;
  }

  function shortGridBatchLabel(batch) {
    const s = String(batch || "");
    if (!s) return "";
    const m = s.match(/^([A-Z0-9]+)_(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})/);
    if (m) return `${m[1]} · ${m[2].replace("T", " ")}`;
    if (s.length > 42) return `${s.slice(0, 40)}…`;
    return s;
  }

  function legBadge(row) {
    const parts = [];
    const purpose = String(row?.purpose || "").toLowerCase();
    if (purpose === "inventory" || String(row?.order_type || "") === "inventory_leg") {
      parts.push('<span class="leg-badge leg-inv" title="引擎库存腿">库存</span>');
    }
    if (row?.is_repair_tp) {
      parts.push('<span class="leg-badge leg-repair-tp" title="手动补挂止盈">补挂</span>');
    }
    const label = String(row?.leg_label || "").trim();
    if (label) {
      const cls = label.endsWith("_tp") ? "leg-badge leg-tp" : "leg-badge";
      parts.push(`<span class="${cls}" title="网格档位">${escHtml(label)}</span>`);
    }
    return parts.join(" ");
  }

  function isTpLegRow(row) {
    return String(row?.leg_label || "").trim().endsWith("_tp");
  }

  function ordersLegCell(row, esc) {
    const leg = legBadge(row);
    const meta = `${scopeBadge(row.scope)} ${strategyBadge(row.strategy, row.scope)}`;
    return (
      `<td class="orders-leg-cell">` +
      `<div class="orders-leg-cell-inner">` +
      `<span class="orders-leg-meta">${meta}</span>` +
      `<span class="orders-leg-slot">${leg}</span>` +
      `</div></td>`
    );
  }

  function buildOrdersTableRows(rows, options) {
    const opts = options || {};
    const showSymbol = opts.showSymbol !== false;
    const esc = opts.escHtml || escHtml;
    const colspan = ordersTableColspan(showSymbol);
    const parts = [];
    let lastBatch = null;
    (rows || []).forEach((r, i) => {
      const batch = r.scope === "multi_leg" && r.grid_batch ? String(r.grid_batch) : "";
      if (batch && batch !== lastBatch) {
        parts.push(
          `<tr class="grid-batch-header"><td colspan="${colspan}">` +
            `<span class="grid-batch-tag">网格批次</span> ` +
            `${esc(shortGridBatchLabel(batch))}` +
            (showSymbol && r.symbol ? ` · ${esc(r.symbol)}` : "") +
            `</td></tr>`
        );
        lastBatch = batch;
      } else if (!batch) {
        lastBatch = null;
      }
      const mid = r.marker_id || "";
      const symCell = showSymbol ? `<td>${esc(r.symbol || "")}</td>` : "";
      const tpRow = isTpLegRow(r);
      parts.push(
        `<tr data-idx="${i}" data-marker-id="${esc(mid)}" data-symbol="${esc(r.symbol || "")}"` +
          (batch ? ` data-grid-batch="${esc(batch)}"` : "") +
          (tpRow ? ` class="orders-leg-tp-row"` : "") +
          `>
          ${ordersLegCell(r, esc)}
          ${symCell}
          <td>${esc(formatOrderTime(r.time))}</td>
          <td class="${sideClass(r.side)}">${esc(r.side || "")}</td>
          <td>${statusBadge(r.status)}</td>
          <td>${esc(String(r.filled_quantity ?? r.quantity ?? ""))}</td>
          <td>${esc(String(r.average_price ?? r.price ?? ""))}</td>
          <td>${esc(formatTpPrice(r))}</td>
          <td>${esc(formatSlPrice(r))}</td>
          <td class="${pnlClass(r)}">${esc(formatPnl(r))}</td>
          <td class="id-cell" title="${esc(r.order_id || "")}">${esc(r.order_id || "")}</td>
        </tr>`
      );
    });
    return parts.join("");
  }

  function formatTpPrice(row) {
    const v = row?.take_profit_price;
    if (v == null || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    const hint = row?.take_profit_hint ? ` (${row.take_profit_hint})` : "";
    return `${n}${hint}`;
  }

  function formatSlPrice(row) {
    const v = row?.stop_loss_price;
    if (v == null || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    const hint = row?.stop_loss_hint ? ` (${row.stop_loss_hint})` : "";
    return `${n}${hint}`;
  }

  function formatPnl(row) {
    const v = row?.pnl_usdt ?? row?.realized_pnl ?? row?.unrealized_pnl;
    if (v == null || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    const hint = row?.pnl_hint ? ` (${row.pnl_hint})` : "";
    const sign = n > 0 ? "+" : "";
    return `${sign}${n.toFixed(2)}${hint}`;
  }

  function pnlClass(row) {
    const v = row?.pnl_usdt ?? row?.realized_pnl ?? row?.unrealized_pnl;
    const n = Number(v);
    if (!Number.isFinite(n)) return "pnl-cell";
    if (n > 0) return "pnl-cell pnl-pos";
    if (n < 0) return "pnl-cell pnl-neg";
    return "pnl-cell";
  }

  function ordersTableColspan(showSymbol) {
    return showSymbol ? 11 : 10;
  }

  function formatOrderTime(ts) {
    if (!ts) return "—";
    const d = new Date(Number(ts) * 1000);
    return d.toISOString().slice(0, 19).replace("T", " ");
  }

  function defaultOrdersFilter() {
    return {
      hideExpired: true,
      hideCanceled: true,
      hideRejected: false,
      hidePending: false,
    };
  }

  function migrateOrdersFilterFromTradeMapLayout() {
    try {
      const raw = localStorage.getItem(TRADE_MAP_LAYOUT_KEY);
      if (!raw) return null;
      const layout = JSON.parse(raw);
      if (layout == null || typeof layout !== "object") return null;
      return {
        hideExpired: layout.hideExpired !== false,
        hideCanceled: layout.hideCanceled !== false,
        hideRejected: false,
        hidePending: false,
      };
    } catch (_) {
      return null;
    }
  }

  function loadOrdersFilter() {
    try {
      const raw = localStorage.getItem(ORDERS_FILTER_KEY);
      if (raw) {
        const stored = JSON.parse(raw);
        if (stored && typeof stored === "object") {
          return { ...defaultOrdersFilter(), ...stored };
        }
      }
    } catch (_) {
      /* fall through */
    }
    const migrated = migrateOrdersFilterFromTradeMapLayout();
    if (migrated) {
      saveOrdersFilter(migrated);
      return migrated;
    }
    return defaultOrdersFilter();
  }

  function saveOrdersFilter(filter) {
    const merged = { ...defaultOrdersFilter(), ...(filter || {}) };
    localStorage.setItem(ORDERS_FILTER_KEY, JSON.stringify(merged));
    return merged;
  }

  function applyOrdersFilterToControls(filter, ids) {
    const f = { ...defaultOrdersFilter(), ...(filter || {}) };
    const map = ids || {
      hideExpired: "hideExpired",
      hideCanceled: "hideCanceled",
      hideRejected: "hideRejected",
      hidePending: "hidePending",
    };
    for (const [key, elId] of Object.entries(map)) {
      const el = document.getElementById(elId);
      if (el) el.checked = !!f[key];
    }
    return f;
  }

  function ordersFilterFromControls(ids) {
    const map = ids || {
      hideExpired: "hideExpired",
      hideCanceled: "hideCanceled",
      hideRejected: "hideRejected",
      hidePending: "hidePending",
    };
    const out = { ...defaultOrdersFilter() };
    for (const [key, elId] of Object.entries(map)) {
      const el = document.getElementById(elId);
      if (el) out[key] = !!el.checked;
    }
    return out;
  }

  function ordersExcludeStatusParamFromFilter(filter) {
    const f = { ...defaultOrdersFilter(), ...(filter || {}) };
    const parts = [];
    if (f.hideExpired) parts.push("expired");
    if (f.hideCanceled) parts.push("canceled");
    if (f.hideRejected) parts.push("rejected");
    if (f.hidePending) parts.push("pending");
    return parts.join(",");
  }

  function bindOrdersFilterSync(onChange) {
    window.addEventListener("storage", (ev) => {
      if (ev.key === ORDERS_FILTER_KEY && ev.newValue) {
        try {
          const filter = { ...defaultOrdersFilter(), ...JSON.parse(ev.newValue) };
          applyOrdersFilterToControls(filter);
          if (typeof onChange === "function") onChange(filter);
        } catch (_) {}
      }
    });
  }

  const ORDERS_COL_WIDTH_KEY = "mlbot.console.ordersColWidths";
  const DEFAULT_ORDERS_COL_WIDTHS = {
    ordersTable: {
      account: 128,
      symbol: 76,
      time: 112,
      side: 52,
      status: 68,
      qty: 56,
      price: 76,
      tp: 100,
      sl: 76,
      pnl: 72,
      order_id: 340,
    },
    ordersDockTable: {
      account: 96,
      symbol: 72,
      time: 100,
      side: 48,
      status: 64,
      qty: 52,
      price: 68,
      tp: 84,
      order_id: 240,
    },
  };

  function loadOrdersColWidths(tableId) {
    try {
      const all = JSON.parse(localStorage.getItem(ORDERS_COL_WIDTH_KEY) || "{}");
      return { ...(DEFAULT_ORDERS_COL_WIDTHS[tableId] || {}), ...(all[tableId] || {}) };
    } catch (_) {
      return { ...(DEFAULT_ORDERS_COL_WIDTHS[tableId] || {}) };
    }
  }

  function saveOrdersColWidth(tableId, colKey, widthPx) {
    try {
      const all = JSON.parse(localStorage.getItem(ORDERS_COL_WIDTH_KEY) || "{}");
      const per = { ...(all[tableId] || {}), [colKey]: Math.round(widthPx) };
      all[tableId] = per;
      localStorage.setItem(ORDERS_COL_WIDTH_KEY, JSON.stringify(all));
    } catch (_) {}
  }

  function applyOrdersColWidths(tableEl) {
    if (!tableEl) return;
    const tableId = tableEl.id || "ordersTable";
    const widths = loadOrdersColWidths(tableId);
    tableEl.querySelectorAll("thead th[data-col]").forEach((th) => {
      const key = th.getAttribute("data-col");
      const w = widths[key];
      if (w && w > 24) {
        th.style.width = `${w}px`;
        th.style.minWidth = `${w}px`;
      }
    });
  }

  function bindOrdersTableResize(tableEl) {
    if (!tableEl || tableEl.dataset.resizeBound === "1") return;
    tableEl.dataset.resizeBound = "1";
    const tableId = tableEl.id || "ordersTable";
    applyOrdersColWidths(tableEl);

    tableEl.querySelectorAll("thead th[data-col]").forEach((th) => {
      if (th.querySelector(".col-resize-grip")) return;
      const grip = document.createElement("span");
      grip.className = "col-resize-grip";
      grip.setAttribute("aria-hidden", "true");
      th.appendChild(grip);

      grip.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const colKey = th.getAttribute("data-col") || "";
        const startX = ev.clientX;
        const startW = th.offsetWidth;
        tableEl.classList.add("resizing");

        const onMove = (moveEv) => {
          const w = Math.max(36, startW + moveEv.clientX - startX);
          th.style.width = `${w}px`;
          th.style.minWidth = `${w}px`;
        };
        const onUp = () => {
          tableEl.classList.remove("resizing");
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          if (colKey) saveOrdersColWidth(tableId, colKey, th.offsetWidth);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
    });
  }

  root.MLBotConsole = {
    api,
    SYMBOL_ALL,
    isAllSymbols,
    getSymbol,
    setSymbol,
    getScopesDefault,
    setScopesState,
    initAppNav,
    loadExtLinks,
    loadSymbols,
    bindSymbolPersist,
    formatOrderTime,
    escHtml,
    scopeBadge,
    strategyBadge,
    statusBadge,
    renderOrderDetailHtml,
    buildOrdersTableRows,
    ordersTableColspan,
    formatTpPrice,
    formatSlPrice,
    formatPnl,
    pnlClass,
    defaultOrdersFilter,
    loadOrdersFilter,
    saveOrdersFilter,
    applyOrdersFilterToControls,
    ordersFilterFromControls,
    ordersExcludeStatusParamFromFilter,
    bindOrdersFilterSync,
    bindOrdersTableResize,
    applyOrdersColWidths,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
