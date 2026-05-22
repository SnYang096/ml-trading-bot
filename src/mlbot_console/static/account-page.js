/**
 * Account overview — global exchange ledger vs symbol-scoped strategy PnL.
 */

const Shell = globalThis.MLBotConsole;

function fmtPnlNum(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}`;
}

function pnlClassNum(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "";
  if (v > 0) return "pnl-pos";
  if (v < 0) return "pnl-neg";
  return "";
}

function setStatus(msg) {
  const el = document.getElementById("statusLine");
  if (el) el.textContent = msg;
}

function fmtUsdt(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function renderGlobalKpis(totals, ledger) {
  const t = totals || {};
  const lt = ledger?.totals || {};
  const cards = [
    {
      label: "总权益（总账）",
      value: lt.equity_usdt ?? t.equity_usdt,
      hint: "USDT · 币安各账户之和",
      fmt: fmtUsdt,
    },
    {
      label: "总钱包余额",
      value: lt.wallet_balance_usdt ?? t.wallet_balance_usdt,
      hint: "USDT",
      fmt: fmtUsdt,
    },
    {
      label: "总可用",
      value: lt.available_usdt ?? t.available_usdt,
      hint: "USDT",
      fmt: fmtUsdt,
    },
    {
      label: "合约未实现",
      value: lt.exchange_unrealized_pnl_usdt ?? t.exchange_unrealized_pnl_usdt,
      hint: "USDT · 交易所",
      fmt: fmtUsdt,
    },
  ];
  return cards
    .map((c) => {
      const raw = c.fmt ? c.fmt(c.value) : fmtPnlNum(c.value);
      return `<div class="account-kpi-card account-kpi-global">
        <div class="account-kpi-label">${Shell.escHtml(c.label)}</div>
        <div class="account-kpi-value">${Shell.escHtml(raw)}</div>
        <div class="account-kpi-hint muted">${Shell.escHtml(c.hint)}</div>
      </div>`;
    })
    .join("");
}

function renderScopedKpis(totals) {
  const t = totals || {};
  const cards = [
    { label: "已实现盈亏", value: t.realized_pnl, hint: "USDT · 本地 DB" },
    { label: "持仓浮盈", value: t.unrealized_pnl, hint: "USDT · 本地估算" },
    { label: "已平仓笔数", value: t.closed_trades, hint: "笔", fmt: (v) => String(v ?? 0) },
    { label: "未平仓位/批次", value: t.open_positions, hint: "个", fmt: (v) => String(v ?? 0) },
  ];
  return cards
    .map((c) => {
      const raw = c.fmt ? c.fmt(c.value) : fmtPnlNum(c.value);
      const cls = c.fmt ? "" : pnlClassNum(c.value);
      return `<div class="account-kpi-card">
        <div class="account-kpi-label">${Shell.escHtml(c.label)}</div>
        <div class="account-kpi-value ${cls}">${Shell.escHtml(raw)}</div>
        <div class="account-kpi-hint muted">${Shell.escHtml(c.hint)}</div>
      </div>`;
    })
    .join("");
}

function exCell(ex, field) {
  if (!ex?.ok) {
    const err = ex?.error || (ex?.configured ? "拉取失败" : "未配置密钥");
    return `<span class="muted" title="${Shell.escHtml(err)}">—</span>`;
  }
  return Shell.escHtml(fmtUsdt(ex[field]));
}

function renderScopesTable(scopes) {
  if (!scopes?.length) return '<p class="muted">无数据</p>';
  const rows = scopes
    .map((s) => {
      const label = s.label || s.scope || "—";
      const ex = s.exchange || {};
      const binance = ex.binance_label
        ? `<div class="muted account-sub">${Shell.escHtml(ex.binance_label)}</div>`
        : "";
      return `<tr>
        <td>${Shell.escHtml(label)}${binance}</td>
        <td>${exCell(ex, "wallet_balance_usdt")}</td>
        <td>${exCell(ex, "equity_usdt")}</td>
        <td>${exCell(ex, "available_usdt")}</td>
        <td class="${pnlClassNum(ex.unrealized_pnl_usdt)}">${exCell(ex, "unrealized_pnl_usdt")}</td>
        <td class="${pnlClassNum(s.realized_pnl)}">${Shell.escHtml(fmtPnlNum(s.realized_pnl))}</td>
        <td class="${pnlClassNum(s.unrealized_pnl)}">${Shell.escHtml(fmtPnlNum(s.unrealized_pnl))}</td>
        <td>${Shell.escHtml(String(s.closed_trades ?? 0))}</td>
        <td>${Shell.escHtml(String(s.open_positions ?? 0))}</td>
      </tr>`;
    })
    .join("");
  return `<table class="account-table">
    <thead><tr>
      <th>账户层</th><th>钱包余额</th><th>权益</th><th>可用</th><th>交易所浮盈</th>
      <th>已实现</th><th>本地浮盈</th><th>已平仓</th><th>未平</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderLedgerPanel(ledger) {
  const lt = ledger?.totals || {};
  const ok = lt.accounts_ok ?? 0;
  const total = lt.accounts_total ?? 0;
  
  let hint = "";
  if (ok < total) {
    const errs = [];
    (ledger?.accounts || []).forEach(a => {
      if (!a.ok) {
        errs.push(`${a.label}: ${a.error || "未配置"}`);
      }
    });
    hint = `<p class="muted" style="color: #f85149">已连接 ${ok}/${total} 个币安账户。失败: ${errs.join(" | ")}</p>`;
  } else if (total > 0) {
    hint = `<p class="muted" style="color: #3fb950">已连接 ${ok}/${total} 个币安账户</p>`;
  }
  
  return `<div class="account-ledger-strip">
    <span>总账权益 <strong>${Shell.escHtml(fmtUsdt(lt.equity_usdt))}</strong> USDT</span>
    <span>总钱包 <strong>${Shell.escHtml(fmtUsdt(lt.wallet_balance_usdt))}</strong></span>
    <span>总可用 <strong>${Shell.escHtml(fmtUsdt(lt.available_usdt))}</strong></span>
    <span>合约未实现 <strong>${Shell.escHtml(fmtUsdt(lt.exchange_unrealized_pnl_usdt))}</strong></span>
  </div>${hint}`;
}

function renderStrategiesTable(strategies) {
  if (!strategies?.length) return '<p class="muted">无数据</p>';
  const rows = strategies
    .map((s) => {
      const name = `${s.scope || ""} · ${s.strategy || ""}`;
      return `<tr>
        <td>${Shell.escHtml(name)}</td>
        <td class="${pnlClassNum(s.realized_pnl)}">${Shell.escHtml(fmtPnlNum(s.realized_pnl))}</td>
        <td class="${pnlClassNum(s.unrealized_pnl)}">${Shell.escHtml(fmtPnlNum(s.unrealized_pnl))}</td>
        <td>${Shell.escHtml(String(s.closed_trades ?? 0))}</td>
        <td>${Shell.escHtml(String(s.open_positions ?? 0))}</td>
      </tr>`;
    })
    .join("");
  return `<table class="account-table">
    <thead><tr>
      <th>策略</th><th>已实现</th><th>浮盈</th><th>已平仓</th><th>未平</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderDailyChart(daily) {
  const pts = daily || [];
  if (!pts.length) return '<p class="muted">回看期内无已实现盈亏记录</p>';
  const values = pts.map((p) => Math.abs(Number(p.pnl) || 0));
  const maxAbs = Math.max(...values, 1e-6);
  const bars = pts
    .map((p) => {
      const v = Number(p.pnl) || 0;
      const h = Math.max(4, (Math.abs(v) / maxAbs) * 72);
      const cls = v >= 0 ? "bar-pos" : "bar-neg";
      const title = `${p.date}: ${fmtPnlNum(v)}`;
      return `<div class="daily-bar ${cls}" style="height:${h}px" title="${Shell.escHtml(title)}"></div>`;
    })
    .join("");
  const labels = pts
    .filter((_, i) => i === 0 || i === pts.length - 1 || i % Math.ceil(pts.length / 6) === 0)
    .map((p) => `<span>${Shell.escHtml(p.date)}</span>`)
    .join("");
  return `<div class="daily-bars">${bars}</div><div class="daily-labels">${labels}</div>`;
}

async function refreshGlobalAccount() {
  const globalQ = new URLSearchParams({ symbol: Shell.SYMBOL_ALL, lookback_days: "0" });
  try {
    const { data: globalData } = await Shell.api(`/api/account/summary?${globalQ}`);
    document.getElementById("kpiRow").innerHTML = renderGlobalKpis(
      globalData.totals,
      globalData.exchange_ledger
    );
    const ledgerEl = document.getElementById("ledgerPanel");
    if (ledgerEl) {
      ledgerEl.innerHTML = globalData.exchange_ledger
        ? renderLedgerPanel(globalData.exchange_ledger)
        : '<p class="muted">—</p>';
    }
  } catch (e) {
    document.getElementById("kpiRow").innerHTML = `<span class="muted">${Shell.escHtml(String(e))}</span>`;
    setStatus(String(e));
  }
}

function renderSpotHoldings(scopes) {
  const spot = (scopes || []).find(s => s.scope === "spot");
  if (!spot || !spot.exchange || !spot.exchange.ok) return "";
  
  const ex = spot.exchange;
  const holdings = ex.holdings || [];
  if (!holdings.length) return "";
  
  const totalValue = ex.holdings_value_usdt || 0;
  
  const rows = holdings.map(h => {
    const pct = totalValue > 0 ? (h.value_usdt / totalValue) * 100 : 0;
    const src = h.price_source === "ticker" ? " (API)" : (h.price_source === "stablecoin" ? "" : " (Parquet)");
    return `<tr>
      <td>${Shell.escHtml(h.asset)}</td>
      <td>${Shell.escHtml(String(h.qty))}</td>
      <td>${Shell.escHtml(fmtUsdt(h.price_usdt))}<span class="muted" style="font-size: 0.85em">${src}</span></td>
      <td>${Shell.escHtml(fmtUsdt(h.value_usdt))}</td>
      <td>${pct.toFixed(1)}%</td>
    </tr>`;
  }).join("");
  
  // Create a stacked bar chart for holdings
  const colors = ["#58a6ff", "#3fb950", "#f0883e", "#d2a8ff", "#f85149", "#e3b341"];
  let chartHtml = "";
  if (totalValue > 0) {
    const segments = holdings.map((h, i) => {
      const pct = (h.value_usdt / totalValue) * 100;
      const color = colors[i % colors.length];
      return `<div style="width: ${pct}%; background-color: ${color}; height: 100%; float: left;" title="${h.asset}: ${pct.toFixed(1)}%"></div>`;
    }).join("");
    chartHtml = `<div style="height: 12px; width: 100%; background: #21262d; border-radius: 6px; overflow: hidden; margin-top: 10px; margin-bottom: 10px;">${segments}</div>`;
  }
  
  let ledgerDiff = "";
  if (ex.ledger_holdings) {
    const diff = ex.holdings_value_usdt - (ex.ledger_holdings_value_usdt || 0);
    ledgerDiff = `<p class="muted" style="margin-top: 5px; font-size: 0.9em;">与本地母仓市值差额: ${fmtPnlNum(diff)} USDT</p>`;
  }
  
  return `<section class="panel account-panel" style="grid-column: 1 / -1;">
    <h3>现货持仓明细 (Spot)</h3>
    ${chartHtml}
    <div class="account-table-wrap">
      <table class="account-table">
        <thead><tr><th>资产</th><th>数量</th><th>现价 (USDT)</th><th>市值 (USDT)</th><th>占比</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${ledgerDiff}
  </section>`;
}

async function refreshReconciliation() {
  const container = document.getElementById("reconciliationContainer");
  if (!container) return;
  
  container.innerHTML = '<p class="muted">正在对账...</p>';
  
  try {
    const scopes = ["trend", "multi_leg", "spot"];
    const promises = scopes.map(s => Shell.api(`/api/account/reconciliation?scope=${s}`).catch(e => ({ error: e })));
    const results = await Promise.all(promises);
    
    let html = '<div class="account-grid">';
    
    results.forEach((res, i) => {
      const scope = scopes[i];
      const label = scope === "trend" ? "B·Trend" : (scope === "multi_leg" ? "C·Multi-leg" : "A·Spot");
      
      if (res.error || !res.data) {
        html += `<section class="panel account-panel">
          <h3>${label} 对账</h3>
          <p class="muted" style="color: #f85149">失败: ${Shell.escHtml(String(res.error || "未知错误"))}</p>
        </section>`;
        return;
      }
      
      const data = res.data;
      if (!data.ok && data.error) {
        html += `<section class="panel account-panel">
          <h3>${label} 对账</h3>
          <p class="muted" style="color: #f85149">交易所拉取失败: ${Shell.escHtml(data.error)}</p>
        </section>`;
        return;
      }
      
      const issues = data.issues || [];
      if (issues.length === 0) {
        html += `<section class="panel account-panel">
          <h3>${label} 对账</h3>
          <p style="color: #3fb950; font-weight: bold;">✓ 交易所与本地数据一致</p>
        </section>`;
      } else {
        const issueRows = issues.map(iss => {
          if (iss.kind === "qty_mismatch") {
            return `<tr><td>数量不符</td><td>${iss.asset}</td><td>${iss.exchange}</td><td>${iss.local}</td><td>${fmtPnlNum(iss.delta)}</td></tr>`;
          } else if (iss.kind === "missing_exchange_order") {
            return `<tr><td>交易所缺单</td><td>${iss.symbol}</td><td>—</td><td>${iss.order_id}</td><td>—</td></tr>`;
          } else if (iss.kind === "orphan_exchange_order") {
            return `<tr><td>孤儿单</td><td>${iss.symbol}</td><td>${iss.order_id}</td><td>—</td><td>—</td></tr>`;
          } else if (iss.kind === "position_mismatch") {
            return `<tr><td>仓位不符</td><td>${iss.symbol}</td><td>${iss.exchange}</td><td>${iss.local}</td><td>${fmtPnlNum(iss.delta)}</td></tr>`;
          } else if (iss.kind === "wallet_extra") {
            const note = iss.note ? ` <span class="muted">${Shell.escHtml(iss.note)}</span>` : "";
            return `<tr><td>钱包未入账</td><td>${iss.asset}${note}</td><td>${iss.exchange}</td><td>—</td><td>—</td></tr>`;
          }
          return `<tr><td colspan="5">${JSON.stringify(iss)}</td></tr>`;
        }).join("");
        
        html += `<section class="panel account-panel">
          <h3>${label} 对账 <span style="color: #f85149; font-size: 0.8em; font-weight: normal;">(⚠ ${issues.length} 项差异)</span></h3>
          <div class="account-table-wrap">
            <table class="account-table" style="font-size: 0.9em">
              <thead><tr><th>类型</th><th>资产/标的</th><th>交易所</th><th>本地</th><th>差额</th></tr></thead>
              <tbody>${issueRows}</tbody>
            </table>
          </div>
        </section>`;
      }
    });
    
    html += '</div>';
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = `<p class="muted" style="color: #f85149">对账请求失败: ${Shell.escHtml(String(e))}</p>`;
  }
}

async function refreshScopedAccount() {
  const symbol = document.getElementById("symbolSelect").value;
  const lookback = document.getElementById("lookbackSelect").value;
  if (!Shell.isAllSymbols(symbol)) Shell.setSymbol(symbol);
  setStatus("加载中…");
  const scopedQ = new URLSearchParams({ symbol, lookback_days: lookback });
  try {
    const { data: scopedData, meta } = await Shell.api(`/api/account/summary?${scopedQ}`);
    
    const scopedKpi = document.getElementById("scopedKpiRow");
    if (scopedKpi) {
      scopedKpi.innerHTML = renderScopedKpis(scopedData.totals);
    }
    document.getElementById("scopesTable").innerHTML = renderScopesTable(scopedData.scopes);
    document.getElementById("strategiesTable").innerHTML = renderStrategiesTable(
      scopedData.strategies
    );
    
    const spotHoldingsContainer = document.getElementById("spotHoldingsContainer");
    if (spotHoldingsContainer) {
      spotHoldingsContainer.innerHTML = renderSpotHoldings(scopedData.scopes);
    }
    
    document.getElementById("dailyChart").innerHTML = renderDailyChart(scopedData.daily_realized);
    const notes = (scopedData.notes || []).map((n) => `· ${n}`).join("\n");
    document.getElementById("accountNotes").textContent = notes;
    const symLabel = meta?.symbol || scopedData.symbol || symbol;
    const lb = lookback === "0" ? "全部历史" : `${lookback} 天`;
    setStatus(`${symLabel} · ${lb} · ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    setStatus(String(e));
  }
}

async function refreshAccount() {
  setStatus("加载中…");
  await Promise.all([
    refreshGlobalAccount(),
    refreshScopedAccount(),
    refreshReconciliation(),
  ]);
}

function bindControls() {
  const rerunAll = () => refreshAccount().catch((e) => setStatus(String(e)));
  const rerunScoped = () => refreshScopedAccount().catch((e) => setStatus(String(e)));
  document.getElementById("refreshBtn").addEventListener("click", rerunAll);
  document.getElementById("symbolSelect").addEventListener("change", rerunScoped);
  document.getElementById("lookbackSelect").addEventListener("change", rerunScoped);
  Shell.bindSymbolPersist("symbolSelect");
}

(async () => {
  try {
    Shell.initAppNav("account");
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect", null, { includeAll: true });
    await refreshAccount();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();
