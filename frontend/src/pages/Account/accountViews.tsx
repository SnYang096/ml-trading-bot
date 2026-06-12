import type { ReactNode } from 'react';
import type {
  AccountReconIssue,
  AccountReconciliationAll,
  AccountScopeBlock,
  AccountStrategyRow,
  DailyPnlPoint,
} from '@/api/types.ts';
import { fmtPnl, pnlClass, SCOPE_LABELS } from '@/lib/shell.ts';
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

export function ScopesTable({
  scopes,
  symbolFilter = '*',
}: {
  scopes: AccountScopeBlock[];
  symbolFilter?: string;
}) {
  if (!scopes?.length) return <p className="muted">无数据</p>;
  const symbolScoped =
    symbolFilter.trim() !== '' &&
    symbolFilter.trim() !== '*' &&
    symbolFilter.toUpperCase() !== 'ALL';
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>账户层</th>
            <th>钱包余额</th>
            <th>权益</th>
            <th>可用</th>
            <th>交易所浮盈</th>
            {symbolScoped ? <th>品种浮盈</th> : null}
            <th>已实现</th>
            <th>本地浮盈</th>
            <th>已平仓</th>
            <th>未平</th>
          </tr>
        </thead>
        <tbody>
          {scopes.map((s) => {
            const ex = (s.exchange || {}) as Record<string, unknown>;
            const label = s.label || SCOPE_LABELS[String(s.scope || '')] || s.scope || '—';
            const accountUpnl =
              ex.account_unrealized_pnl_usdt ?? ex.unrealized_pnl_usdt;
            const symbolUpnl = ex.symbol_unrealized_pnl_usdt;
            return (
              <tr key={String(s.scope)}>
                <td>
                  {label}
                  {ex.binance_label ? (
                    <div className={`muted ${styles.sub}`}>{String(ex.binance_label)}</div>
                  ) : null}
                </td>
                <td>{exCell(ex, 'wallet_balance_usdt')}</td>
                <td>{exCell(ex, 'equity_usdt')}</td>
                <td>{exCell(ex, 'available_usdt')}</td>
                <td className={pnlClass(accountUpnl)}>{fmtUsdt(accountUpnl)}</td>
                {symbolScoped ? (
                  <td className={pnlClass(symbolUpnl)}>{fmtUsdt(symbolUpnl)}</td>
                ) : null}
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
