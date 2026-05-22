/**
 * Shared shell: app nav, API helper, symbol/scopes persistence across pages.
 */
(function (root) {
  const SYMBOL_KEY = "mlbot_console_symbol";
  const SCOPES_KEY = "mlbot_console_scopes";
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

  function buildOrdersTableRows(rows, options) {
    const opts = options || {};
    const showSymbol = opts.showSymbol !== false;
    const esc = opts.escHtml || escHtml;
    return (rows || [])
      .map((r, i) => {
        const mid = r.marker_id || "";
        const symCell = showSymbol
          ? `<td>${esc(r.symbol || "")}</td>`
          : "";
        return `<tr data-idx="${i}" data-marker-id="${esc(mid)}" data-symbol="${esc(r.symbol || "")}">
          <td>${scopeBadge(r.scope)} ${strategyBadge(r.strategy, r.scope)}</td>
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
        </tr>`;
      })
      .join("");
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
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
