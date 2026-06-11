import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import type { OrderRow, SymbolRow } from '@/api/types.ts';
import {
  displayOrderQty,
  getScopesDefault,
  getSymbol,
  isAllSymbols,
  setScopesState,
  setSymbol,
  SCOPE_LABELS,
  SYMBOL_ALL,
} from '@/lib/shell.ts';
import { listStrategiesForLayers, scopesFromLayers as scopesFromLayersLib } from '@/lib/tradeMap';

interface LayerState {
  trend: boolean;
  spot: boolean;
  multiLeg: boolean;
}

function scopesFromLayers(layers: LayerState): string {
  return scopesFromLayersLib({ trend: layers.trend, spot: layers.spot, multiLeg: layers.multiLeg });
}

const PAGE_SIZE = 50;
const DEFAULT_EXCLUDE_STATUS = 'expired,canceled,rejected';

export function OrdersPage() {
  const pageVisible = usePageVisible();
  const [searchParams] = useSearchParams();
  const [symbol, setSym] = useState(searchParams.get('symbol') || getSymbol() || 'ETHUSDT');
  const [layers, setLayers] = useState<LayerState>(() => {
    const saved = getScopesDefault();
    return {
      trend: saved?.trend ?? true,
      spot: saved?.spot ?? true,
      multiLeg: saved?.multiLeg ?? true,
    };
  });
  const [statusFilter, setStatusFilter] = useState('all');
  const [strategyFilter, setStrategyFilter] = useState('');
  const [page, setPage] = useState(0);
  const [selectedIdx, setSelectedIdx] = useState(-1);

  const strategies = useMemo(
    () =>
      listStrategiesForLayers({
        trend: layers.trend,
        spot: layers.spot,
        multiLeg: layers.multiLeg,
      }),
    [layers],
  );

  const symbolsQuery = useQuery({
    queryKey: ['symbols'],
    queryFn: () => apiGet<SymbolRow[]>('/api/trade-map/symbols'),
  });

  const ordersQuery = useQuery({
    queryKey: ['orders', symbol, layers, statusFilter, strategyFilter],
    queryFn: () =>
      apiGet<OrderRow[]>('/api/orders/list', {
        symbol,
        scopes: scopesFromLayers(layers),
        status: statusFilter === 'all' ? undefined : statusFilter,
        strategy: strategyFilter || undefined,
        exclude_status: DEFAULT_EXCLUDE_STATUS,
        limit: isAllSymbols(symbol) ? 100 : 200,
      }),
    refetchInterval: visibleRefetchInterval(pageVisible, 15_000),
  });

  useEffect(() => {
    if (!strategyFilter) return;
    if (!strategies.some((s) => s.id === strategyFilter)) {
      setStrategyFilter('');
    }
  }, [strategies, strategyFilter]);

  useEffect(() => {
    if (!isAllSymbols(symbol)) setSymbol(symbol);
    setScopesState({ ...layers, pending: false });
  }, [symbol, layers]);

  useEffect(() => {
    setPage(0);
    setSelectedIdx(-1);
  }, [symbol, layers, statusFilter, strategyFilter]);

  const rows = ordersQuery.data?.data || [];
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = useMemo(
    () => rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE),
    [rows, safePage],
  );
  const showSymbol = isAllSymbols(symbol);
  const selected = selectedIdx >= 0 ? rows[selectedIdx] : null;

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>订单</h2>
        <label>
          Symbol
          <select value={symbol} onChange={(e) => setSym(e.target.value)}>
            <option value={SYMBOL_ALL}>全部</option>
            {(symbolsQuery.data?.data || [{ symbol: 'ETHUSDT' }]).map((r) => (
              <option key={r.symbol} value={r.symbol}>
                {r.symbol}
              </option>
            ))}
          </select>
        </label>
        <label className="chk-pill">
          <input
            type="checkbox"
            checked={layers.trend}
            onChange={(e) => setLayers((l) => ({ ...l, trend: e.target.checked }))}
          />
          {SCOPE_LABELS.trend}
        </label>
        <label className="chk-pill">
          <input
            type="checkbox"
            checked={layers.spot}
            onChange={(e) => setLayers((l) => ({ ...l, spot: e.target.checked }))}
          />
          {SCOPE_LABELS.spot}
        </label>
        <label className="chk-pill">
          <input
            type="checkbox"
            checked={layers.multiLeg}
            onChange={(e) => setLayers((l) => ({ ...l, multiLeg: e.target.checked }))}
          />
          {SCOPE_LABELS.multi_leg}
        </label>
        <label>
          策略
          <select value={strategyFilter} onChange={(e) => setStrategyFilter(e.target.value)}>
            <option value="">全部</option>
            {strategies.map((s) => (
              <option key={s.id} value={s.id}>
                {s.account_layer_title ? `${s.account_layer_title} · ` : ''}
                {s.title || s.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Status
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">全部</option>
            <option value="filled">filled</option>
            <option value="open">open</option>
            <option value="pending">pending</option>
          </select>
        </label>
        <button type="button" onClick={() => ordersQuery.refetch()}>
          刷新
        </button>
        <span className="muted">
          第 {safePage + 1}/{pageCount} 页（共 {rows.length} 条）
        </span>
        <button type="button" disabled={safePage <= 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>
          上一页
        </button>
        <button
          type="button"
          disabled={safePage >= pageCount - 1}
          onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
        >
          下一页
        </button>
        <Link to={`/account?symbol=${encodeURIComponent(symbol)}`}>账户总览</Link>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            {showSymbol ? <th>Symbol</th> : null}
            <th>Scope</th>
            <th>Strategy</th>
            <th>Side</th>
            <th>Status</th>
            <th>Qty</th>
            <th>Price</th>
            <th>PNL</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {pageRows.length ? (
            pageRows.map((r, i) => {
              const globalIdx = safePage * PAGE_SIZE + i;
              return (
                <tr
                  key={`${r.order_id}-${globalIdx}`}
                  className={globalIdx === selectedIdx ? 'selected' : undefined}
                  onClick={() => setSelectedIdx(globalIdx)}
                  style={{ cursor: 'pointer' }}
                >
                  {showSymbol ? <td>{r.symbol}</td> : null}
                  <td>{SCOPE_LABELS[r.scope] || r.scope}</td>
                  <td>{r.strategy}</td>
                  <td>{r.side}</td>
                  <td>{r.status}</td>
                  <td>{displayOrderQty(r)}</td>
                  <td>{r.average_price ?? r.price ?? '—'}</td>
                  <td>{r.pnl_usdt != null ? String(r.pnl_usdt) : r.realized_pnl != null ? String(r.realized_pnl) : '—'}</td>
                  <td>{String(r.filled_at || r.created_at || r.time || '—').slice(0, 19)}</td>
                </tr>
              );
            })
          ) : (
            <tr>
              <td colSpan={showSymbol ? 9 : 8} className="muted">
                无订单
              </td>
            </tr>
          )}
        </tbody>
      </table>
      {selected ? (
        <section className="panel">
          <h3>订单详情</h3>
          <dl>
            <dt>订单号</dt>
            <dd>{selected.order_id}</dd>
            <dt>标记</dt>
            <dd>
              {selected.marker_id ? (
                <Link
                  to={`/trade-map?symbol=${encodeURIComponent(selected.symbol || symbol)}&marker_id=${encodeURIComponent(String(selected.marker_id))}`}
                >
                  在地图查看
                </Link>
              ) : (
                '—'
              )}
            </dd>
          </dl>
          <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
            {JSON.stringify(selected, null, 2)}
          </pre>
        </section>
      ) : null}
      <p className="status-line">
        {ordersQuery.isFetching ? '加载中…' : `${rows.length} orders · ${symbol}${strategyFilter ? ` · ${strategyFilter}` : ''}`}
      </p>
    </div>
  );
}
