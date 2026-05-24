/**
 * Strategy signals overview — universe table (no sidebar).
 */

const Shell = globalThis.MLBotConsole;
const POLL_MS = 20000;

const LAYER_LABELS = {
  trend: "B·Trend",
  spot: "A·Spot",
  multi_leg: "C·Multi-leg",
};

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

function strategyLayerLabel(layer) {
  return LAYER_LABELS[layer] || layer || "—";
}

function renderStrategyLines(scopeBlock) {
  const by = scopeBlock?.by_strategy || {};
  const keys = Object.keys(by).sort();
  if (!keys.length) {
    return `<span class="muted">${esc(scopeBlock?.summary || "—")}</span>`;
  }
  return keys
    .map((sid) => {
      const row = by[sid] || {};
      const funnel = row.funnel_summary ? ` <span class="muted">· ${esc(row.funnel_summary)}</span>` : "";
      const title = esc(row.last_summary || row.summary || "");
      return `<div class="strategy-line"><strong>${esc(sid)}</strong> ${esc(row.summary || "—")}${funnel}</div>`;
    })
    .join("");
}

function renderLastStrategyLines(scopeBlock) {
  const by = scopeBlock?.by_strategy || {};
  const keys = Object.keys(by).sort();
  if (!keys.length) return esc(scopeBlock?.last_summary || "—");
  return keys
    .map((sid) => {
      const row = by[sid] || {};
      const last = row.last_summary || "—";
      if (last === "—") return "";
      return `<div><strong>${esc(sid)}</strong> ${esc(last)}</div>`;
    })
    .filter(Boolean)
    .join("") || esc(scopeBlock?.last_summary || "—");
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
      const spotTitle = Object.values(s.by_strategy || {})
        .map((x) => (x.blockers || []).join(", "))
        .filter(Boolean)
        .join("; ");
      return `<tr>
        <td><strong>${esc(r.symbol)}</strong></td>
        <td>${esc(fmtBarTime(r.latest_bar))}</td>
        <td>${r.bars_1min_rows ?? "—"}</td>
        <td><a href="${esc(mapHref)}">地图</a></td>
        <td class="strategy-cell">${renderStrategyLines(t)}</td>
        <td class="muted strategy-cell">${renderLastStrategyLines(t)}</td>
        <td class="strategy-cell" title="${esc(spotTitle)}">${renderStrategyLines(s)}</td>
        <td class="strategy-cell">${renderStrategyLines(m)}</td>
        <td class="muted strategy-cell">${renderLastStrategyLines(m)}</td>
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
        account_layer: inferStrategyLayer(strat),
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
    tbody.innerHTML =
      '<tr><td colspan="10" class="muted">无 funnel 数据（B 层需 quant-trend 写 stats_15min；A/C 见上表各策略成交行）</td></tr>';
    return;
  }
  tbody.innerHTML = flat
    .slice(0, 120)
    .map(
      (r) => `<tr>
        <td>${esc(String(r.timestamp || "").slice(0, 16))}</td>
        <td>${esc(r.symbol)}</td>
        <td>${esc(strategyLayerLabel(r.account_layer))}</td>
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

function inferStrategyLayer(strategyId) {
  const sid = String(strategyId || "").toLowerCase();
  if (sid.includes("spot")) return "spot";
  if (sid === "chop_grid" || sid === "trend_scalp") return "multi_leg";
  return "trend";
}

function fillSelectOptions(sel, values, keepFirst = true) {
  if (!sel) return;
  const current = sel.value;
  const existing = new Set(
    [...sel.options].slice(keepFirst ? 1 : 0).map((o) => o.value)
  );
  for (const v of values) {
    if (existing.has(v)) continue;
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    sel.appendChild(o);
    existing.add(v);
  }
  if ([...sel.options].some((o) => o.value === current)) {
    sel.value = current;
  }
}

async function refreshFunnel() {
  const symSel = document.getElementById("funnelSymbolSelect");
  const layerSel = document.getElementById("funnelLayerSelect");
  const stratSel = document.getElementById("funnelStrategySelect");
  const sym = symSel?.value || "";
  const layer = layerSel?.value || "";
  const strat = stratSel?.value || "";
  const q = new URLSearchParams({ limit: "48" });
  if (sym) q.set("symbol", sym);
  if (layer) q.set("account_layer", layer);
  if (strat) q.set("strategy", strat);
  try {
    const { data } = await Shell.api(`/api/trend/funnel?${q}`);
    renderFunnelRows(data || []);
    const symbols = new Set((data || []).map((r) => r.symbol).filter(Boolean));
    const strategies = new Set();
    for (const snap of data || []) {
      for (const sid of Object.keys(snap.by_strategy || {})) strategies.add(sid);
    }
    fillSelectOptions(symSel, [...symbols].sort());
    fillSelectOptions(stratSel, [...strategies].sort());
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
  const funnelLayer = document.getElementById("funnelLayerSelect");
  const funnelStrat = document.getElementById("funnelStrategySelect");
  if (funnelSym) funnelSym.addEventListener("change", () => refreshFunnel().catch(() => {}));
  if (funnelLayer) funnelLayer.addEventListener("change", () => refreshFunnel().catch(() => {}));
  if (funnelStrat) funnelStrat.addEventListener("change", () => refreshFunnel().catch(() => {}));

  const funnelPanel = document.querySelector(".funnel-panel");
  if (funnelPanel) {
    funnelPanel.addEventListener("toggle", () => {
      if (funnelPanel.open && !funnelLoaded) {
        refreshFunnel()
          .then(() => {
            funnelLoaded = true;
          })
          .catch(() => {});
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
