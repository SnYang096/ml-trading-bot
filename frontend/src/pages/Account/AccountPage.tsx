import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type {
  AccountReconIssue,
  AccountReconScopeBlock,
  AccountReconciliationAll,
  AccountSummary,
  SymbolRow,
} from '@/api/types.ts';
import { fmtPnl, getSymbol, isAllSymbols, pnlClass, setSymbol, SCOPE_LABELS } from '@/lib/shell.ts';

function fmtUsdt(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return '—';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

const RECON_SCOPES = ['spot', 'trend', 'multi_leg'] as const;

function scopeIssues(
  recon: AccountReconciliationAll | undefined,
  scope: (typeof RECON_SCOPES)[number],
): {
  engine?: AccountReconScopeBlock;
  pnl?: AccountReconScopeBlock;
  issues: AccountReconIssue[];
} {
  const engine = recon?.engine?.[scope];
  const pnl = recon?.pnl?.scopes?.[scope];
  const issues: AccountReconIssue[] = [
    ...(engine?.issues || []).map((i) => ({ ...i, layer: i.layer || 'engine' })),
    ...(pnl?.issues || []).map((i) => ({ ...i, layer: i.layer || 'pnl' })),
  ];
  return { engine, pnl, issues };
}

export function AccountPage() {
  const [searchParams] = useSearchParams();
  const [symbol, setSym] = useState(searchParams.get('symbol') || getSymbol() || 'ETHUSDT');
  const [lookback, setLookback] = useState('0');
  const [reconOpen, setReconOpen] = useState(false);

  const symbolsQuery = useQuery({
    queryKey: ['symbols'],
    queryFn: () => apiGet<SymbolRow[]>('/api/trade-map/symbols'),
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
    enabled: reconOpen,
  });

  const totals = summaryQuery.data?.data?.totals || {};
  const ledger = summaryQuery.data?.data?.ledger?.totals || {};
  const recent = summaryQuery.data?.data?.recent_realized || {};
  const recon = reconQuery.data?.data;
  const reconIssues = recon?.issues?.length ?? 0;

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>账户总览</h2>
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
          Lookback
          <select value={lookback} onChange={(e) => setLookback(e.target.value)}>
            <option value="0">0d</option>
            <option value="7">7d</option>
            <option value="30">30d</option>
          </select>
        </label>
        <button type="button" onClick={() => summaryQuery.refetch()}>
          刷新
        </button>
      </div>
      <div className="panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
        <div>
          <div className="muted">总权益（总账）</div>
          <div>{fmtUsdt(ledger.equity_usdt ?? totals.equity_usdt)} USDT</div>
        </div>
        <div>
          <div className="muted">已实现盈亏</div>
          <div className={pnlClass(totals.realized_pnl)}>{fmtPnl(totals.realized_pnl)} USDT</div>
        </div>
        <div>
          <div className="muted">浮盈</div>
          <div className={pnlClass(totals.unrealized_pnl)}>{fmtPnl(totals.unrealized_pnl)} USDT</div>
        </div>
        <div>
          <div className="muted">已平仓</div>
          <div>{totals.closed_trades ?? 0} 笔</div>
        </div>
      </div>
      {recent.last_day ? (
        <section className="panel">
          <h3>已实现盈亏速览</h3>
          <p>
            最近日 {String(recent.last_day)} ·{' '}
            <span className={pnlClass(recent.last_day_pnl)}>{fmtPnl(recent.last_day_pnl)} USDT</span>
          </p>
          <p>
            本周 <span className={pnlClass(recent.this_week_pnl)}>{fmtPnl(recent.this_week_pnl)} USDT</span>
          </p>
        </section>
      ) : null}
      <section className="panel">
        <div className="toolbar-row" style={{ marginBottom: 8 }}>
          <h3 style={{ margin: 0 }}>对账</h3>
          {!reconOpen ? (
            <button type="button" onClick={() => setReconOpen(true)}>
              展开对账
            </button>
          ) : (
            <button type="button" onClick={() => reconQuery.refetch()}>
              刷新对账
            </button>
          )}
        </div>
        {!reconOpen ? (
          <p className="muted">点击「展开对账」加载 A·Spot / B·Trend / C·Multi-leg 与交易所比对。</p>
        ) : reconQuery.isFetching && !recon ? (
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
            <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
              {RECON_SCOPES.map((scope) => {
                const { engine, pnl, issues } = scopeIssues(recon, scope);
                const ok = (engine?.ok ?? true) && (pnl?.ok ?? true) && issues.length === 0;
                const exErr = engine?.exchange_snapshot && !(engine.exchange_snapshot as { ok?: boolean }).ok
                  ? String((engine.exchange_snapshot as { error?: string }).error || '交易所不可用')
                  : null;
                return (
                  <div key={scope} className="panel" style={{ margin: 0, padding: 12 }}>
                    <div className="toolbar-row" style={{ marginBottom: 6 }}>
                      <strong>{SCOPE_LABELS[scope] || scope}</strong>
                      <span className={ok ? 'pnl-pos' : 'pnl-neg'}>{ok ? '一致' : `${issues.length} 项`}</span>
                    </div>
                    {exErr ? <p className="muted">{exErr}</p> : null}
                    {pnl?.local ? (
                      <p className="muted" style={{ fontSize: '0.85rem', margin: '4px 0' }}>
                        本地 已实现 {fmtPnl(pnl.local.realized_pnl)} · 浮盈 {fmtPnl(pnl.local.unrealized_pnl)} ·
                        未平 {pnl.local.open_positions ?? 0}
                      </p>
                    ) : null}
                    {engine?.local_snapshot?.note ? (
                      <p className="muted" style={{ fontSize: '0.85rem' }}>
                        {String(engine.local_snapshot.note)}
                      </p>
                    ) : null}
                    {issues.length ? (
                      <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: '0.85rem' }}>
                        {issues.map((issue, idx) => (
                          <li key={`${scope}-${issue.kind}-${idx}`}>
                            <span className="muted">{issue.layer || 'issue'}</span> · {issue.kind || 'unknown'} —{' '}
                            {issue.message || JSON.stringify(issue)}
                          </li>
                        ))}
                      </ul>
                    ) : ok ? (
                      <p className="pnl-pos" style={{ fontSize: '0.85rem', margin: '8px 0 0' }}>
                        订单/持仓与交易所一致
                      </p>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </section>
      <p className="status-line">
        {summaryQuery.isFetching ? '加载中…' : `${symbol} · ${lookback}d lookback`}
      </p>
    </div>
  );
}
