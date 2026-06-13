import type {
  AccountCurves,
  AccountReconciliationAll,
  AccountReconIssue,
  AccountScopeBlock,
  AccountStrategyRow,
  DailyPnlPoint,
} from '@/api/types.ts';
import { fmtPnl, pnlClass, SCOPE_LABELS } from '@/lib/shell.ts';
import { Fragment, useMemo, useState, type ReactNode } from 'react';
import styles from './AccountPage.module.css';

export function fmtUsdt(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return '—';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function exCell(ex: Record<string, unknown> | undefined, field: string): string {
  if (!ex?.ok) {
    return '—';
  }
  return fmtUsdt(ex[field]);
}

export function KpiCard({
  label,
  value,
  hint,
  valueClass,
}: {
  label: string;
  value: string;
  hint?: string;
  valueClass?: string;
}) {
  return (
    <div className={styles.kpiCard}>
      <div className={styles.kpiLabel}>{label}</div>
      <div className={`${styles.kpiValue} ${valueClass || ''}`}>{value}</div>
      {hint ? <div className={`${styles.kpiHint} muted`}>{hint}</div> : null}
    </div>
  );
}

function scopeRowCells(s: AccountScopeBlock, symbolScoped: boolean) {
  const ex = (s.exchange || {}) as Record<string, unknown>;
  const label = s.label || SCOPE_LABELS[String(s.scope || '')] || s.scope || '—';
  const accountUpnl = ex.account_unrealized_pnl_usdt ?? ex.unrealized_pnl_usdt;
  const symbolUpnl = ex.symbol_unrealized_pnl_usdt;
  const displayUpnl = symbolScoped ? (symbolUpnl ?? accountUpnl) : accountUpnl;
  const exOpenCount = ex.exchange_open_position_count;
  const localOpen = Number(s.open_positions ?? 0);
  const compareUpnlNum = Number(symbolScoped ? (symbolUpnl ?? 0) : accountUpnl);
  const localUpnlNum = Number(s.unrealized_pnl ?? 0);
  const pnlMismatch =
    Number.isFinite(compareUpnlNum) &&
    Math.abs(compareUpnlNum) > 0.5 &&
    localOpen === 0 &&
    Math.abs(localUpnlNum) < 0.5;
  return {
    label,
    sub: ex.binance_label ? String(ex.binance_label) : undefined,
    wallet: exCell(ex, 'wallet_balance_usdt'),
    equity: exCell(ex, 'equity_usdt'),
    available: exCell(ex, 'available_usdt'),
    displayUpnl,
    accountUpnl,
    exOpenCount,
    localOpen,
    pnlMismatch,
    realized: s.realized_pnl,
    unrealized: s.unrealized_pnl,
    closed: s.closed_trades,
  };
}

function isSymbolScopedFilter(symbolFilter: string): boolean {
  const sym = symbolFilter.trim();
  return sym !== '' && sym !== '*' && sym.toUpperCase() !== 'ALL';
}

/** Account scope rows with expandable per-strategy breakdown (local DB). */
export function AccountHierarchyTable({
  scopes,
  strategies,
  symbolFilter = '*',
}: {
  scopes: AccountScopeBlock[];
  strategies: AccountStrategyRow[];
  symbolFilter?: string;
}) {
  const strategiesByScope = useMemo(() => {
    const map = new Map<string, AccountStrategyRow[]>();
    for (const row of strategies || []) {
      const key = String(row.scope || '');
      const list = map.get(key) || [];
      list.push(row);
      map.set(key, list);
    }
    return map;
  }, [strategies]);

  const defaultExpanded = useMemo(
    () =>
      new Set(
        (scopes || [])
          .map((s) => String(s.scope || ''))
          .filter((scope) => (strategiesByScope.get(scope)?.length ?? 0) > 0),
      ),
    [scopes, strategiesByScope],
  );

  const [expanded, setExpanded] = useState<Set<string> | null>(null);
  const effectiveExpanded = expanded ?? defaultExpanded;

  if (!scopes?.length) return <p className="muted">无数据</p>;

  const symbolScoped = isSymbolScopedFilter(symbolFilter);
  const dash = '—';

  const toggleScope = (scope: string) => {
    setExpanded((prev) => {
      const next = new Set(prev ?? defaultExpanded);
      if (next.has(scope)) next.delete(scope);
      else next.add(scope);
      return next;
    });
  };

  return (
    <div className={styles.tableWrap}>
      <table className={`${styles.table} ${styles.hierarchyTable}`}>
        <thead>
          <tr>
            <th className={styles.hierarchyNameCol}>账户 / 策略</th>
            <th>钱包余额</th>
            <th>权益</th>
            <th>可用</th>
            <th>{symbolScoped ? '品种浮盈' : '交易所浮盈'}</th>
            {symbolScoped ? <th>全账户浮盈</th> : null}
            <th>交易所未平</th>
            <th>已实现</th>
            <th>本地浮盈</th>
            <th>已平仓</th>
            <th>本地未平</th>
          </tr>
        </thead>
        <tbody>
          {scopes.map((scopeBlock) => {
            const scopeKey = String(scopeBlock.scope || '');
            const cells = scopeRowCells(scopeBlock, symbolScoped);
            const childRows = strategiesByScope.get(scopeKey) || [];
            const canExpand = childRows.length > 0;
            const isOpen = canExpand && effectiveExpanded.has(scopeKey);
            return (
              <Fragment key={scopeKey}>
                <tr
                  className={`${styles.scopeRow} ${cells.pnlMismatch ? styles.rowWarn : ''}`}
                >
                  <td>
                    <div className={styles.hierarchyNameCell}>
                      {canExpand ? (
                        <button
                          type="button"
                          className={`${styles.expandBtn} ${isOpen ? styles.expandBtnOpen : ''}`}
                          aria-expanded={isOpen}
                          aria-label={isOpen ? '收起策略' : '展开策略'}
                          onClick={() => toggleScope(scopeKey)}
                        >
                          ▶
                        </button>
                      ) : (
                        <span className={styles.expandSpacer} aria-hidden="true" />
                      )}
                      <div>
                        <div className={styles.scopeTitle}>{cells.label}</div>
                        {cells.sub ? (
                          <div className={`muted ${styles.sub}`}>{cells.sub}</div>
                        ) : null}
                        {canExpand ? (
                          <div className={`muted ${styles.sub}`}>{childRows.length} 个策略</div>
                        ) : null}
                      </div>
                    </div>
                  </td>
                  <td>{cells.wallet}</td>
                  <td>{cells.equity}</td>
                  <td>{cells.available}</td>
                  <td className={pnlClass(cells.displayUpnl)}>{fmtUsdt(cells.displayUpnl)}</td>
                  {symbolScoped ? (
                    <td className={pnlClass(cells.accountUpnl)}>
                      {fmtUsdt(cells.accountUpnl)}
                    </td>
                  ) : null}
                  <td>{cells.exOpenCount != null ? String(cells.exOpenCount) : dash}</td>
                  <td className={pnlClass(cells.realized)}>{fmtPnl(cells.realized)}</td>
                  <td className={pnlClass(cells.unrealized)}>{fmtPnl(cells.unrealized)}</td>
                  <td>{String(cells.closed ?? 0)}</td>
                  <td>
                    {String(cells.localOpen)}
                    {cells.pnlMismatch ? (
                      <div className={`muted ${styles.sub} pnl-neg`}>与交易所不同步</div>
                    ) : null}
                  </td>
                </tr>
                {isOpen
                  ? childRows.map((st) => {
                      const inactive =
                        (st.realized_pnl ?? 0) === 0 &&
                        (st.unrealized_pnl ?? 0) === 0 &&
                        (st.closed_trades ?? 0) === 0 &&
                        (st.open_positions ?? 0) === 0;
                      const title = st.strategy_title || st.strategy || dash;
                      return (
                        <tr
                          key={`${scopeKey}-${st.strategy}`}
                          className={`${styles.strategyRow} ${inactive ? 'muted' : ''}`}
                        >
                          <td>
                            <div className={styles.strategyNameCell}>{title}</div>
                          </td>
                          <td className={styles.inheritedDash}>{dash}</td>
                          <td className={styles.inheritedDash}>{dash}</td>
                          <td className={styles.inheritedDash}>{dash}</td>
                          <td className={styles.inheritedDash}>{dash}</td>
                          {symbolScoped ? <td className={styles.inheritedDash}>{dash}</td> : null}
                          <td className={styles.inheritedDash}>{dash}</td>
                          <td className={pnlClass(st.realized_pnl)}>{fmtPnl(st.realized_pnl)}</td>
                          <td className={pnlClass(st.unrealized_pnl)}>
                            {fmtPnl(st.unrealized_pnl)}
                          </td>
                          <td>{String(st.closed_trades ?? 0)}</td>
                          <td>{String(st.open_positions ?? 0)}</td>
                        </tr>
                      );
                    })
                  : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** @deprecated use AccountHierarchyTable — flat scope-only table */
export function ScopesTable({
  scopes,
  symbolFilter = '*',
}: {
  scopes: AccountScopeBlock[];
  symbolFilter?: string;
}) {
  if (!scopes?.length) return <p className="muted">无数据</p>;
  const symbolScoped = isSymbolScopedFilter(symbolFilter);
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>账户层</th>
            <th>钱包余额</th>
            <th>权益</th>
            <th>可用</th>
            <th>{symbolScoped ? '品种浮盈' : '交易所浮盈'}</th>
            {symbolScoped ? <th>全账户浮盈</th> : null}
            <th>交易所未平</th>
            <th>已实现</th>
            <th>本地浮盈</th>
            <th>已平仓</th>
            <th>本地未平</th>
          </tr>
        </thead>
        <tbody>
          {scopes.map((s) => {
            const cells = scopeRowCells(s, symbolScoped);
            return (
              <tr
                key={String(s.scope)}
                className={cells.pnlMismatch ? styles.rowWarn : undefined}
              >
                <td>
                  {cells.label}
                  {cells.sub ? <div className={`muted ${styles.sub}`}>{cells.sub}</div> : null}
                </td>
                <td>{cells.wallet}</td>
                <td>{cells.equity}</td>
                <td>{cells.available}</td>
                <td className={pnlClass(cells.displayUpnl)}>{fmtUsdt(cells.displayUpnl)}</td>
                {symbolScoped ? (
                  <td className={pnlClass(cells.accountUpnl)}>{fmtUsdt(cells.accountUpnl)}</td>
                ) : null}
                <td>{cells.exOpenCount != null ? String(cells.exOpenCount) : '—'}</td>
                <td className={pnlClass(cells.realized)}>{fmtPnl(cells.realized)}</td>
                <td className={pnlClass(cells.unrealized)}>{fmtPnl(cells.unrealized)}</td>
                <td>{String(cells.closed ?? 0)}</td>
                <td>
                  {String(cells.localOpen)}
                  {cells.pnlMismatch ? (
                    <div className={`muted ${styles.sub} pnl-neg`}>与交易所不同步</div>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function StrategiesTable({ strategies }: { strategies: AccountStrategyRow[] }) {
  if (!strategies?.length) return <p className="muted">无数据</p>;
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>策略</th>
            <th>已实现</th>
            <th>浮盈</th>
            <th>已平仓</th>
            <th>未平</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((s) => {
            const scopeLabel =
              s.scope_label || SCOPE_LABELS[String(s.scope || '')] || s.scope || '—';
            const title = s.strategy_title || s.strategy || '—';
            const inactive =
              (s.realized_pnl ?? 0) === 0 &&
              (s.unrealized_pnl ?? 0) === 0 &&
              (s.closed_trades ?? 0) === 0 &&
              (s.open_positions ?? 0) === 0;
            return (
              <tr key={`${s.scope}-${s.strategy}`} className={inactive ? 'muted' : undefined}>
                <td>
                  {scopeLabel} · {title}
                </td>
                <td className={pnlClass(s.realized_pnl)}>{fmtPnl(s.realized_pnl)}</td>
                <td className={pnlClass(s.unrealized_pnl)}>{fmtPnl(s.unrealized_pnl)}</td>
                <td>{String(s.closed_trades ?? 0)}</td>
                <td>{String(s.open_positions ?? 0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function SpotHoldingsPanel({ scopes }: { scopes: AccountScopeBlock[] }) {
  const spot = scopes?.find((s) => s.scope === 'spot');
  const ex = spot?.exchange as Record<string, unknown> | undefined;
  if (!ex?.ok) return null;
  const holdings = (ex.holdings || []) as Array<Record<string, unknown>>;
  if (!holdings.length) return null;
  const totalValue = Number(ex.holdings_value_usdt) || 0;
  const colors = ['#00ff41', '#5cffff', '#ffcc44', '#d966ff', '#ff3366', '#ffff66'];

  return (
    <section className={`panel ${styles.spotPanel}`}>
      <h3>现货持仓明细 (Spot)</h3>
      {totalValue > 0 ? (
        <div className={styles.holdingsBar}>
          {holdings.map((h, i) => {
            const pct = ((Number(h.value_usdt) || 0) / totalValue) * 100;
            return (
              <div
                key={String(h.asset)}
                className={styles.holdingsSeg}
                style={{ width: `${pct}%`, backgroundColor: colors[i % colors.length] }}
                title={`${h.asset}: ${pct.toFixed(1)}%`}
              />
            );
          })}
        </div>
      ) : null}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>资产</th>
              <th>数量</th>
              <th>买入均价</th>
              <th>现价</th>
              <th>市值</th>
              <th>浮盈</th>
              <th>占比</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h) => {
              const pct = totalValue > 0 ? ((Number(h.value_usdt) || 0) / totalValue) * 100 : 0;
              const avg = Number(h.avg_entry_usdt);
              const upnl = Number(h.unrealized_pnl_usdt);
              return (
                <tr key={String(h.asset)}>
                  <td>{String(h.asset)}</td>
                  <td>{String(h.qty)}</td>
                  <td>{Number.isFinite(avg) && avg > 0 ? fmtUsdt(avg) : '—'}</td>
                  <td>{fmtUsdt(h.price_usdt)}</td>
                  <td>{fmtUsdt(h.value_usdt)}</td>
                  <td className={pnlClass(upnl)}>{Number.isFinite(upnl) ? fmtPnl(upnl) : '—'}</td>
                  <td>{pct.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function pnlBars(pts: DailyPnlPoint[], valueKey: 'pnl' | 'cumulative' = 'pnl') {
  if (!pts.length) return <p className="muted">回看期内无已实现盈亏记录</p>;
  const values = pts.map((p) => Math.abs(Number(p[valueKey]) || 0));
  const maxAbs = Math.max(...values, 1e-6);
  const step = Math.max(1, Math.ceil(pts.length / 6));
  return (
    <>
      <div className={styles.pnlBars}>
        {pts.map((p, i) => {
          const v = Number(p[valueKey]) || 0;
          const h = Math.max(4, (Math.abs(v) / maxAbs) * 72);
          const label = p.label || p.date || p.week_start || '';
          const cls = v >= 0 ? styles.barPos : styles.barNeg;
          return (
            <div
              key={`${label}-${i}`}
              className={`${styles.pnlBar} ${cls}`}
              style={{ height: `${h}px` }}
              title={`${label}: ${fmtPnl(v)}`}
            />
          );
        })}
      </div>
      <div className={styles.pnlBarLabels}>
        {pts
          .filter((_, i) => i === 0 || i === pts.length - 1 || i % step === 0)
          .map((p) => {
            const t = p.label || p.week_start || p.date || '';
            return (
              <span key={t} title={t}>
                {String(t).slice(5)}
              </span>
            );
          })}
      </div>
    </>
  );
}

export function WeeklyPnlChart({ weekly }: { weekly: DailyPnlPoint[] }) {
  return <div className={styles.pnlChart}>{pnlBars(weekly || [], 'pnl')}</div>;
}

export function DailyPnlChart({ daily }: { daily: DailyPnlPoint[] }) {
  return <div className={styles.pnlChart}>{pnlBars(daily || [], 'pnl')}</div>;
}

export function WeeklyPnlTable({ weekly }: { weekly: DailyPnlPoint[] }) {
  const pts = weekly || [];
  if (!pts.length) return null;
  return (
    <div className={`${styles.tableWrap} ${styles.weeklyTable}`}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>自然周</th>
            <th>已实现 (USDT)</th>
          </tr>
        </thead>
        <tbody>
          {[...pts].reverse().map((w) => (
            <tr key={w.week_start || w.label || w.date}>
              <td>{w.label || w.week_start || '—'}</td>
              <td className={pnlClass(w.pnl)}>{fmtPnl(w.pnl)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AccountEquityChart({ curves }: { curves: AccountCurves | undefined }) {
  const balance = curves?.balance || [];
  const equity = curves?.equity || [];
  if (!balance.length && !equity.length) {
    return <p className="muted">{curves?.note || '无钱包/权益曲线数据'}</p>;
  }
  const dates = balance.length >= equity.length ? balance : equity;
  const balByDate = new Map(balance.map((p) => [String(p.date || ''), Number(p.value_usdt) || 0]));
  const eqByDate = new Map(equity.map((p) => [String(p.date || ''), Number(p.value_usdt) || 0]));
  const vals: number[] = [];
  for (const pt of dates) {
    const d = String(pt.date || '');
    vals.push(balByDate.get(d) ?? 0, eqByDate.get(d) ?? 0);
  }
  let minV = Math.min(...vals);
  let maxV = Math.max(...vals);
  if (minV === maxV) {
    minV -= 1;
    maxV += 1;
  }
  const span = maxV - minV;
  const w = 900;
  const h = 180;
  const padX = 12;
  const padY = 18;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;
  const yFor = (v: number) => padY + innerH - ((v - minV) / span) * innerH;
  const xFor = (i: number) => padX + (i / Math.max(1, dates.length - 1)) * innerW;
  const lineCoords = (series: Map<string, number>) =>
    dates
      .map((pt, i) => {
        const v = series.get(String(pt.date || '')) ?? 0;
        return `${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`;
      })
      .join(' ');
  const lastBal = balance.length ? Number(balance[balance.length - 1].value_usdt) : 0;
  const lastEq = equity.length ? Number(equity[equity.length - 1].value_usdt) : 0;
  return (
    <div className={styles.equityWrap}>
      <div className={styles.curveLegend}>
        <span className={styles.legendBalance}>钱包 balance</span>
        <span className={styles.legendEquity}>权益 equity</span>
      </div>
      <svg className={styles.equitySvg} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img">
        <polyline className={`${styles.equityLine} ${styles.balanceLine}`} points={lineCoords(balByDate)} />
        <polyline className={`${styles.equityLine} ${styles.equityEquityLine}`} points={lineCoords(eqByDate)} />
      </svg>
      <div className={styles.equityMeta}>
        <span>{dates[0]?.date || ''}</span>
        <span>
          钱包 {fmtUsdt(lastBal)} · 权益 {fmtUsdt(lastEq)}
        </span>
        <span>{dates[dates.length - 1]?.date || ''}</span>
      </div>
      {curves?.note ? (
        <p className="muted" style={{ margin: '6px 0 0', fontSize: '0.8rem' }}>
          {curves.note}
        </p>
      ) : null}
    </div>
  );
}

/** @deprecated use AccountEquityChart — kept for tests */
export function EquityCurveChart({ curve }: { curve: DailyPnlPoint[] }) {
  const pts = curve || [];
  if (!pts.length) return <p className="muted">回看期内无已实现盈亏记录</p>;
  const vals = pts.map((p) => Number(p.cumulative) || 0);
  let minV = Math.min(...vals, 0);
  let maxV = Math.max(...vals, 0);
  if (minV === maxV) {
    minV -= 1;
    maxV += 1;
  }
  const span = maxV - minV;
  const w = 900;
  const h = 160;
  const padX = 12;
  const padY = 14;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;
  const zeroY = padY + innerH - ((0 - minV) / span) * innerH;
  const coords = pts.map((_, i) => {
    const x = padX + (i / Math.max(1, pts.length - 1)) * innerW;
    const y = padY + innerH - ((vals[i] - minV) / span) * innerH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = vals[vals.length - 1];
  const lineCls = last >= 0 ? styles.equityPos : styles.equityNeg;
  return (
    <div className={styles.equityWrap}>
      <svg className={styles.equitySvg} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img">
        <line className={styles.equityZero} x1={padX} y1={zeroY} x2={w - padX} y2={zeroY} />
        <polyline className={`${styles.equityLine} ${lineCls}`} points={coords.join(' ')} />
      </svg>
      <div className={styles.equityMeta}>
        <span>{pts[0].date || ''}</span>
        <span className={pnlClass(last)}>累计 {fmtPnl(last)} USDT</span>
        <span>{pts[pts.length - 1].date || ''}</span>
      </div>
    </div>
  );
}

function issueRow(issue: AccountReconIssue, idx: number): ReactNode {
  const kind = String(issue.kind || 'unknown');
  if (kind === 'qty_mismatch') {
    return (
      <tr key={idx}>
        <td>数量不符</td>
        <td>{String(issue.asset || '—')}</td>
        <td>{String(issue.exchange ?? '—')}</td>
        <td>{String(issue.local ?? '—')}</td>
        <td>{fmtPnl(issue.delta)}</td>
      </tr>
    );
  }
  if (kind === 'missing_exchange_order') {
    return (
      <tr key={idx}>
        <td>交易所缺单</td>
        <td>{String(issue.symbol || '—')}</td>
        <td>—</td>
        <td>{String(issue.order_id || '—')}</td>
        <td>—</td>
      </tr>
    );
  }
  if (kind === 'orphan_exchange_order') {
    return (
      <tr key={idx}>
        <td>孤儿单</td>
        <td>{String(issue.symbol || '—')}</td>
        <td>{String(issue.order_id || '—')}</td>
        <td>—</td>
        <td>—</td>
      </tr>
    );
  }
  if (kind === 'position_mismatch') {
    return (
      <tr key={idx}>
        <td>仓位不符</td>
        <td>{String(issue.symbol || '—')}</td>
        <td>{String(issue.exchange ?? '—')}</td>
        <td>{String(issue.local ?? '—')}</td>
        <td>{fmtPnl(issue.delta)}</td>
      </tr>
    );
  }
  if (kind === 'exchange_position_not_in_local_db') {
    return (
      <tr key={idx}>
        <td>交易所有仓·本地无</td>
        <td colSpan={4}>{issue.message || JSON.stringify(issue)}</td>
      </tr>
    );
  }
  if (kind === 'local_position_not_on_exchange') {
    return (
      <tr key={idx}>
        <td>本地有仓·交易所无</td>
        <td>{String(issue.symbol || '—')}</td>
        <td>{String(issue.exchange ?? 0)}</td>
        <td>{String(issue.local ?? '—')}</td>
        <td>{fmtPnl(issue.delta)}</td>
      </tr>
    );
  }
  if (kind === 'unrealized_pnl_mismatch') {
    return (
      <tr key={idx}>
        <td>浮盈不符</td>
        <td colSpan={4}>{issue.message || JSON.stringify(issue)}</td>
      </tr>
    );
  }
  if (kind === 'wallet_extra') {
    return (
      <tr key={idx}>
        <td>钱包未入账</td>
        <td>
          {String(issue.asset || '—')}
          {issue.note ? <span className="muted"> {String(issue.note)}</span> : null}
        </td>
        <td>{String(issue.exchange ?? '—')}</td>
        <td>—</td>
        <td>—</td>
      </tr>
    );
  }
  return (
    <tr key={idx}>
      <td>
        <span className="muted">{issue.layer || 'pnl'}</span> · {kind}
      </td>
      <td colSpan={4}>{issue.message || JSON.stringify(issue)}</td>
    </tr>
  );
}

export function ReconciliationPanels({ recon }: { recon: AccountReconciliationAll | undefined }) {
  const scopes = ['spot', 'trend', 'multi_leg'] as const;
  return (
    <div className={styles.reconGrid}>
      {scopes.map((scope) => {
        const label = SCOPE_LABELS[scope] || scope;
        const engine = recon?.engine?.[scope];
        const pnl = recon?.pnl?.scopes?.[scope];
        const issues: AccountReconIssue[] = [
          ...(engine?.issues || []).map((i) => ({ ...i, layer: 'engine' })),
          ...(pnl?.issues || []).map((i) => ({ ...i, layer: 'pnl' })),
        ];
        const ok = (engine?.ok ?? true) && (pnl?.ok ?? true) && issues.length === 0;
        const exErr =
          engine?.exchange_snapshot && !(engine.exchange_snapshot as { ok?: boolean }).ok
            ? String((engine.exchange_snapshot as { error?: string }).error || '交易所不可用')
            : null;
        return (
          <section key={scope} className={`panel ${styles.reconPanel}`}>
            <h3>
              {label} 对账{' '}
              {!ok ? <span className={`${styles.reconWarn} pnl-neg`}>(⚠ {issues.length} 项差异)</span> : null}
            </h3>
            {exErr ? <p className="muted pnl-neg">{exErr}</p> : null}
            {pnl?.local ? (
              <p className="muted" style={{ fontSize: '0.85rem' }}>
                本地 已实现 {fmtPnl(pnl.local.realized_pnl)} · 浮盈 {fmtPnl(pnl.local.unrealized_pnl)} · 未平{' '}
                {pnl.local.open_positions ?? 0}
              </p>
            ) : null}
            {engine?.local_snapshot?.note ? (
              <p className="muted" style={{ fontSize: '0.85rem' }}>
                {String(engine.local_snapshot.note)}
              </p>
            ) : null}
            {issues.length === 0 ? (
              <p className="pnl-pos">✓ 交易所与本地数据一致</p>
            ) : (
              <div className={styles.tableWrap}>
                <table className={`${styles.table} ${styles.reconTable}`}>
                  <thead>
                    <tr>
                      <th>类型</th>
                      <th>资产/标的</th>
                      <th>交易所</th>
                      <th>本地</th>
                      <th>差额</th>
                    </tr>
                  </thead>
                  <tbody>{issues.map((iss, idx) => issueRow(iss, idx))}</tbody>
                </table>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
