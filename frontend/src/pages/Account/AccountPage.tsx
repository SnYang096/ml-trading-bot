import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type { AccountSummary, SymbolRow } from '@/api/types.ts';
import { fmtPnl, getSymbol, isAllSymbols, pnlClass, setSymbol } from '@/lib/shell.ts';

function fmtUsdt(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return '—';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
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
    queryKey: ['account-recon', symbol],
    queryFn: () => apiGet<unknown>('/api/account/reconciliation', { symbol }),
    enabled: reconOpen,
  });

  const totals = summaryQuery.data?.data?.totals || {};
  const ledger = summaryQuery.data?.data?.ledger?.totals || {};
  const recent = summaryQuery.data?.data?.recent_realized || {};

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
        <h3>对账</h3>
        {!reconOpen ? (
          <button type="button" onClick={() => setReconOpen(true)}>
            展开对账
          </button>
        ) : (
          <>
            {reconQuery.isFetching ? <p className="muted">加载对账…</p> : null}
            <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
              {JSON.stringify(reconQuery.data?.data ?? {}, null, 2)}
            </pre>
          </>
        )}
      </section>
      <p className="status-line">
        {summaryQuery.isFetching ? '加载中…' : `${symbol} · ${lookback}d lookback`}
      </p>
    </div>
  );
}
