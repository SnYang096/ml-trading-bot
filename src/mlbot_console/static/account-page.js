/**
 * Account overview — aggregate PnL by scope/strategy and daily realized chart.
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

function renderKpis(totals) {
  const t = totals || {};
  const cards = [
    { label: "总权益（总账）", value: t.equity_usdt, hint: "USDT · 币安各账户之和", fmt: fmtUsdt },
    { label: "总钱包余额", value: t.wallet_balance_usdt, hint: "USDT", fmt: fmtUsdt },
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
      const binance = ex.binance_label ? `<div class="muted account-sub">${Shell.escHtml(ex.binance_label)}</div>` : "";
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
  const hint =
    ok < total
      ? `<p class="muted">已连接 ${ok}/${total} 个币安账户；未配置的账户请在容器环境变量中设置对应 API Key。</p>`
      : "";
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

async function refreshAccount() {
  const symbol = document.getElementById("symbolSelect").value;
  const lookback = document.getElementById("lookbackSelect").value;
  if (!Shell.isAllSymbols(symbol)) Shell.setSymbol(symbol);
  setStatus("加载中…");
  const q = new URLSearchParams({ symbol, lookback_days: lookback });
  try {
    const { data, meta } = await Shell.api(`/api/account/summary?${q}`);
    document.getElementById("kpiRow").innerHTML = renderKpis(data.totals);
    const ledgerEl = document.getElementById("ledgerPanel");
    if (ledgerEl) {
      ledgerEl.innerHTML = data.exchange_ledger
        ? renderLedgerPanel(data.exchange_ledger)
        : '<p class="muted">—</p>';
    }
    document.getElementById("scopesTable").innerHTML = renderScopesTable(data.scopes);
    document.getElementById("strategiesTable").innerHTML = renderStrategiesTable(data.strategies);
    document.getElementById("dailyChart").innerHTML = renderDailyChart(data.daily_realized);
    const notes = (data.notes || []).map((n) => `· ${n}`).join("\n");
    document.getElementById("accountNotes").textContent = notes;
    const symLabel = meta?.symbol || data.symbol || symbol;
    setStatus(`${symLabel} · ${lookback} 天 · ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    document.getElementById("kpiRow").innerHTML = `<span class="muted">${Shell.escHtml(String(e))}</span>`;
    setStatus(String(e));
  }
}

function bindControls() {
  const rerun = () => refreshAccount().catch((e) => setStatus(String(e)));
  document.getElementById("refreshBtn").addEventListener("click", rerun);
  document.getElementById("symbolSelect").addEventListener("change", rerun);
  document.getElementById("lookbackSelect").addEventListener("change", rerun);
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
