/**
 * Shared shell: app nav, API helper, symbol/scopes persistence across pages.
 */
(function (root) {
  const SYMBOL_KEY = "mlbot_console_symbol";
  const SCOPES_KEY = "mlbot_console_scopes";

  const PAGES = [
    { id: "trade-map", href: "/trade-map", label: "交易地图" },
    { id: "orders", href: "/orders", label: "订单" },
  ];

  async function api(path) {
    const r = await fetch(path);
    const j = await r.json();
    if (!j.ok) throw new Error(j.error?.message || r.statusText || "API error");
    return j;
  }

  function getSymbol() {
    return localStorage.getItem(SYMBOL_KEY) || "";
  }

  function setSymbol(sym) {
    if (sym) localStorage.setItem(SYMBOL_KEY, sym);
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

  async function loadSymbols(selectId, preferred) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const { data } = await api("/api/trade-map/symbols");
    sel.innerHTML = "";
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
    } else if (list[0]) {
      sel.value = list[0].symbol || "ETHUSDT";
    }
    setSymbol(sel.value);
  }

  function bindSymbolPersist(selectId) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    sel.addEventListener("change", () => setSymbol(sel.value));
  }

  function formatOrderTime(ts) {
    if (!ts) return "—";
    const d = new Date(Number(ts) * 1000);
    return d.toISOString().slice(0, 19).replace("T", " ");
  }

  root.MLBotConsole = {
    api,
    getSymbol,
    setSymbol,
    getScopesDefault,
    setScopesState,
    initAppNav,
    loadExtLinks,
    loadSymbols,
    bindSymbolPersist,
    formatOrderTime,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
