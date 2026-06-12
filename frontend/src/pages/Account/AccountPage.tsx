import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type { AccountReconciliationAll, AccountSummary, SymbolRow } from '@/api/types.ts';
import { fmtPnl, getSymbol, isAllSymbols, pnlClass, setSymbol, SYMBOL_ALL } from '@/lib/shell.ts';
import {
  DailyPnlChart,
  EquityCurveChart,
  AccountEquityChart,
  fmtUsdt,
  KpiCard,
  ReconciliationPanels,
  ScopesTable,
  SpotHoldingsPanel,
  StrategiesTable,
  WeeklyPnlChart,
  WeeklyPnlTable,
} from './accountViews.tsx';
import styles from './AccountPage.module.css';

export function AccountPage() {
  const [searchParams] = useSearchParams();
  const [symbol, setSym] = useState(searchParams.get('symbol') || getSymbol() || 'ETHUSDT');
  const [lookback, setLookback] = useState('0');

  const symbolsQuery = useQuery({
    queryKey: ['symbols'],
    queryFn: () => apiGet<SymbolRow[]>('/api/trade-map/symbols'),
  });

  const globalQuery = useQuery({
    queryKey: ['account-summary-global'],
    queryFn: () =>
      apiGet<AccountSummary>('/api/account/summary', {
        symbol: SYMBOL_ALL,
        lookback_days: '0',
        scopes: 'trend,spot,multi_leg',
      }),
  });

  const summaryQuery = useQuery({
    queryKey: ['account-summary', symbol, lookback],
    queryFn: () =>
      apiGet<AccountSummary>('/api/account/summary', {
        symbol,
        lookback_days: lookback,
        scopes: 'trend,spot,multi_leg',
      }),
  });

  const reconQuery = useQuery({
    queryKey: ['account-recon', symbol, lookback],
    queryFn: () =>
      apiGet<AccountReconciliationAll>('/api/account/reconciliation/all', {
        symbol,
        lookback_days: lookback,
      }),
  });

  const global = globalQuery.data?.data;
  const scoped = summaryQuery.data?.data;
  const recon = reconQuery.data?.data;
  const gTotals = global?.totals || {};
  const gLedger = global?.exchange_ledger?.totals || global?.ledger?.totals || {};
  const sTotals = scoped?.totals || {};
  const recent = scoped?.recent_realized || {};
  const reconIssues = recon?.issues?.length ?? 0;

  const refreshAll = () => {
    globalQuery.refetch();
    summaryQuery.refetch();
    reconQuery.refetch();
  };

  return (
    <div className={styles.page}>
      <div className="toolbar-row">
        <h2>账户总览</h2>
        <button type="button" onClick={refreshAll}>
          刷新
        </button>
        <a href="/orders">订单列表</a>
      </div>

      <section className={styles.section}>
        <header>
          <h2>全局资产</h2>
          <p className={`muted ${styles.sectionNote}`}>
            总账权益与钱包余额为币安全账户汇总，不受下方 Symbol 筛选影响。
          </p>
        </header>
        <div className={styles.kpiRow}>
          <KpiCard
            label="总权益（总账）"
            value={`${fmtUsdt(gLedger.equity_usdt ?? gTotals.equity_usdt)} USDT`}
            hint="币安各账户之和"
            valueClass={styles.kpiGlobal}
          />
          <KpiCard
            label="总钱包余额"
            value={`${fmtUsdt(gLedger.wallet_balance_usdt ?? gTotals.wallet_balance_usdt)} USDT`}
            hint="USDT"
          />
          <KpiCard
            label="总可用"
            value={`${fmtUsdt(gLedger.available_usdt ?? gTotals.available_usdt)} USDT`}
            hint="USDT"
          />
          <KpiCard
            label="合约未实现"
            value={`${fmtUsdt(gLedger.exchange_unrealized_pnl_usdt ?? gTotals.exchange_unrealized_pnl_usdt)} USDT`}
            hint="交易所"
          />
        </div>
        {global?.exchange_ledger ? (
          <div className={styles.ledgerStrip}>
            <span>
              总账权益 <strong>{fmtUsdt(gLedger.equity_usdt)}</strong> USDT
            </span>
            <span>
              总钱包 <strong>{fmtUsdt(gLedger.wallet_balance_usdt)}</strong>
            </span>
            <span>
              总可用 <strong>{fmtUsdt(gLedger.available_usdt)}</strong>
            </span>
            <span>
              合约未实现 <strong>{fmtUsdt(gLedger.exchange_unrealized_pnl_usdt)}</strong>
            </span>
          </div>
        ) : null}
      </section>

      <section className={styles.section}>
        <header className={styles.sectionHead}>
          <div>
            <h2>策略与账户层盈亏</h2>
            <p className={`muted ${styles.sectionNote}`}>
              已实现盈亏、持仓与按日统计受 Symbol 与回看期筛选。交易所浮盈列为全账户；选单品种时另显示该品种浮盈。
            </p>
          </div>
          <div className="toolbar-row">
            <label>
              Symbol
              <select
                value={symbol}
                onChange={(e) => {
                  setSym(e.target.value);
                  if (!isAllSymbols(e.target.value)) setSymbol(e.target.value);
                }}
              >
                <option value="*">全部</option>
                {(symbolsQuery.data?.data || [{ symbol: 'ETHUSDT' }]).map((r) => (
                  <option key={r.symbol} value={r.symbol}>
                    {r.symbol}
                  </option>
                ))}
              </select>
            </label>
            <label>
              回看
              <select value={lookback} onChange={(e) => setLookback(e.target.value)}>
                <option value="0">全部</option>
                <option value="7">7 天</option>
                <option value="14">14 天</option>
                <option value="30">30 天</option>
                <option value="90">90 天</option>
                <option value="365">365 天</option>
              </select>
            </label>
          </div>
        </header>

        <div className={styles.kpiRow}>
          <KpiCard label="已实现盈亏" value={`${fmtPnl(sTotals.realized_pnl)} USDT`} hint="本地 DB" valueClass={pnlClass(sTotals.realized_pnl)} />
          <KpiCard label="持仓浮盈" value={`${fmtPnl(sTotals.unrealized_pnl)} USDT`} hint="本地估算" valueClass={pnlClass(sTotals.unrealized_pnl)} />
          <KpiCard label="已平仓笔数" value={String(sTotals.closed_trades ?? 0)} hint="笔" />
          <KpiCard label="未平仓位/批次" value={String(sTotals.open_positions ?? 0)} hint="个" />
        </div>

        {recent.last_day != null || recent.this_week_pnl != null ? (
          <section className="panel">
            <h3>已实现盈亏速览</h3>
            <p className="muted">自然周按 UTC 周一重计。</p>
            <div className={styles.kpiRow}>
              <KpiCard
                label="最近交易日"
                value={`${String(recent.last_day || '—')} · ${fmtPnl(recent.last_day_pnl)} USDT`}
                valueClass={pnlClass(recent.last_day_pnl)}
              />
              <KpiCard
                label={`本周已实现 (${String(recent.this_week_start || '—')} 起)`}
                value={`${fmtPnl(recent.this_week_pnl)} USDT`}
                valueClass={pnlClass(recent.this_week_pnl)}
              />
              <KpiCard
                label={`上周已实现 (${String(recent.last_week_start || '—')} 起)`}
                value={`${fmtPnl(recent.last_week_pnl)} USDT`}
                valueClass={pnlClass(recent.last_week_pnl)}
              />
            </div>
          </section>
        ) : null}

        <div className={styles.grid2}>
          <section className="panel">
            <h3>账户层汇总</h3>
            <p className="muted" style={{ margin: '0 0 8px', fontSize: '0.85rem' }}>
              钱包/权益/交易所浮盈 = 币安 API；已实现/本地浮盈/本地未平 = SQLite 账本（可能滞后）。
            </p>
            <ScopesTable scopes={scoped?.scopes || []} symbolFilter={symbol} />
          </section>
          <section className="panel">
            <h3>策略汇总 (本地 DB)</h3>
            <p className="muted" style={{ margin: '0 0 8px', fontSize: '0.85rem' }}>
              仅 constitution 启用的 live 策略；无成交时显示 0（灰色行）。
            </p>
            <StrategiesTable strategies={scoped?.strategies || []} />
          </section>
        </div>

        <div className={styles.grid2}>
          <SpotHoldingsPanel scopes={scoped?.scopes || []} />
        </div>

        <section className="panel" style={{ marginTop: 16 }}>
          <h3>已实现盈亏</h3>
          <p className="muted" style={{ margin: '0 0 12px' }}>
            按 UTC 自然周汇总。累计曲线为交易所钱包/权益（USDT）；历史权益≈钱包，最新点为币安实时值。
          </p>
          <h4 className={styles.pnlSubhead}>按周统计</h4>
          <WeeklyPnlChart weekly={scoped?.weekly_realized || []} />
          <WeeklyPnlTable weekly={scoped?.weekly_realized || []} />
          <h4 className={styles.pnlSubhead}>钱包 / 权益曲线</h4>
          <AccountEquityChart curves={scoped?.account_curves} />
          <h4 className={styles.pnlSubhead}>已实现盈亏（本地 DB · 累计）</h4>
          <EquityCurveChart curve={scoped?.cumulative_realized || []} />
          <h4 className={styles.pnlSubhead}>按日明细</h4>
          <DailyPnlChart daily={scoped?.daily_realized || []} />
          {scoped?.notes?.length ? (
            <p className={styles.notes}>{scoped.notes.map((n) => `· ${n}`).join('\n')}</p>
          ) : null}
        </section>

        <header style={{ marginTop: 32 }}>
          <h2>交易所对账</h2>
          <p className={`muted ${styles.sectionNote}`}>比对币安 API 实际资产与本地数据库记录的差异。</p>
        </header>
        {reconQuery.isFetching && !recon ? (
          <p className="muted">加载对账…</p>
        ) : reconQuery.isError ? (
          <p className="pnl-neg">对账加载失败：{String(reconQuery.error)}</p>
        ) : (
          <>
            <p>
              {recon?.ok ? (
                <span className="pnl-pos">✓ 交易所与本地数据一致</span>
              ) : (
                <span className="pnl-neg">{reconIssues} 项差异需复核</span>
              )}
              {recon?.pnl?.fetched_at ? (
                <span className="muted"> · 交易所快照 {String(recon.pnl.fetched_at)}</span>
              ) : null}
            </p>
            <ReconciliationPanels recon={recon} />
          </>
        )}
      </section>

      <p className="status-line">
        {summaryQuery.isFetching ? '加载中…' : `${symbol} · ${lookback === '0' ? '全部历史' : `${lookback}d`}`}
      </p>
    </div>
  );
}
