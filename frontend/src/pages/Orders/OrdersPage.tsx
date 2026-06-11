import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import type { OrderRow, SymbolRow, TradeLink } from '@/api/types.ts';
import {
  displayExitKind,
  displayOrderAction,
  displayOrderPrice,
  displayOrderQty,
  displayPositionSideLabel,
  fmtPnl,
  getScopesDefault,
  getSymbol,
  isAllSymbols,
  pnlClass,
  setScopesState,
  setSymbol,
  SCOPE_LABELS,
  SYMBOL_ALL,
  formatUnixTs,
} from '@/lib/shell.ts';
import { listStrategiesForLayers, scopesFromLayers as scopesFromLayersLib } from '@/lib/tradeMap';

interface LayerState {
  trend: boolean;
  spot: boolean;
  multiLeg: boolean;
}

type ViewMode = 'legs' | 'orders';

function scopesFromLayers(layers: LayerState): string {
  return scopesFromLayersLib({ trend: layers.trend, spot: layers.spot, multiLeg: layers.multiLeg });
}

const PAGE_SIZE = 50;
const DEFAULT_EXCLUDE_STATUS = 'expired,canceled,rejected';

export function OrdersPage() {
  const pageVisible = usePageVisible();
  const [searchParams] = useSearchParams();
  const [symbol, setSym] = useState(searchParams.get('symbol') || getSymbol() || 'ETHUSDT');
  const [viewMode, setViewMode] = useState<ViewMode>('legs');
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

  const legsEnabled = viewMode === 'legs' && !isAllSymbols(symbol);

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
    enabled: viewMode === 'orders',
    refetchInterval: visibleRefetchInterval(pageVisible, 15_000),
  });

  const linksQuery = useQuery({
    queryKey: ['orders-trade-links', symbol, layers, strategyFilter],
    queryFn: () =>
      apiGet<TradeLink[]>('/api/orders/trade-links', {
        symbol,
        scopes: scopesFromLayers(layers),
        strategy: strategyFilter || undefined,
        limit: 200,
      }),
    enabled: legsEnabled,
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
  }, [symbol, layers, statusFilter, strategyFilter, viewMode]);

  useEffect(() => {
    if (viewMode === 'legs' && isAllSymbols(symbol)) {
      setViewMode('orders');
    }
  }, [symbol, viewMode]);

  const orderRows = ordersQuery.data?.data || [];
  const linkRows = linksQuery.data?.data || [];
  const rows = viewMode === 'legs' ? linkRows : orderRows;
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageOrderRows = useMemo(
    () => orderRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE),
    [orderRows, safePage],
  );
  const pageLinkRows = useMemo(
    () => linkRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE),
    [linkRows, safePage],
  );
  const showSymbol = isAllSymbols(symbol);
  const selectedOrder = viewMode === 'orders' && selectedIdx >= 0 ? orderRows[selectedIdx] : null;
  const selectedLink = viewMode === 'legs' && selectedIdx >= 0 ? linkRows[selectedIdx] : null;
  const isFetching = viewMode === 'legs' ? linksQuery.isFetching : ordersQuery.isFetching;

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>订单</h2>
        <label>
          视图
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value as ViewMode)}
          >
            <option value="legs">回合（开平一行）</option>
            <option value="orders">原始订单</option>
          </select>
        </label>
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
        {viewMode === 'orders' ? (
          <label>
            Status
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="all">全部</option>
              <option value="filled">filled</option>
              <option value="open">open</option>
              <option value="pending">pending</option>
            </select>
          </label>
        ) : null}
        <button
          type="button"
          onClick={() => (viewMode === 'legs' ? linksQuery : ordersQuery).refetch()}
        >
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

      {viewMode === 'legs' ? (
        <p className="muted" style={{ margin: '0 0 10px', fontSize: '0.85rem' }}>
          回合视图：同一腿的<strong>开→平</strong>合并为一行（与交易地图连线一致）。挂单、未平仓请切「原始订单」。
        </p>
      ) : (
        <p className="muted" style={{ margin: '0 0 10px', fontSize: '0.85rem' }}>
          原始订单：交易所逐笔记录。「动作」列用开多/平多/开空/平空代替 BUY/LONG 混排。
        </p>
      )}

      {viewMode === 'legs' ? (
        <table className="data-table">
          <thead>
            <tr>
              <th>Scope</th>
              <th>Strategy</th>
              <th>Leg</th>
              <th>方向</th>
              <th>开仓价</th>
              <th>平仓价</th>
              <th>PNL</th>
              <th>平仓方式</th>
              <th>开仓时间</th>
              <th>平仓时间</th>
            </tr>
          </thead>
          <tbody>
            {pageLinkRows.length ? (
              pageLinkRows.map((r, i) => {
                const globalIdx = safePage * PAGE_SIZE + i;
                return (
                  <tr
                    key={`${r.entry_marker_id}-${r.exit_marker_id}-${globalIdx}`}
                    className={globalIdx === selectedIdx ? 'selected' : undefined}
                    onClick={() => setSelectedIdx(globalIdx)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>{SCOPE_LABELS[r.scope || ''] || r.scope || '—'}</td>
                    <td>{r.strategy}</td>
                    <td>{r.leg || '—'}</td>
                    <td>{displayPositionSideLabel(r.side)}</td>
                    <td>{Number.isFinite(Number(r.entry_price)) ? String(r.entry_price) : '—'}</td>
                    <td>{Number.isFinite(Number(r.exit_price)) ? String(r.exit_price) : '—'}</td>
                    <td className={pnlClass(r.pnl_usdt)}>
                      {r.pnl_usdt != null ? fmtPnl(r.pnl_usdt) : '—'}
                    </td>
                    <td>{displayExitKind(r.exit_kind)}</td>
                    <td>{formatUnixTs(r.entry_time)}</td>
                    <td>{formatUnixTs(r.exit_time)}</td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={10} className="muted">
                  {legsEnabled ? '无已平仓回合' : '回合视图需选择单个 Symbol'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              {showSymbol ? <th>Symbol</th> : null}
              <th>Scope</th>
              <th>Strategy</th>
              <th>动作</th>
              <th>Status</th>
              <th>Qty</th>
              <th>Price</th>
              <th>PNL</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {pageOrderRows.length ? (
              pageOrderRows.map((r, i) => {
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
                    <td>{displayOrderAction(r)}</td>
                    <td>{r.status}</td>
                    <td>{displayOrderQty(r)}</td>
                    <td>{displayOrderPrice(r)}</td>
                    <td className={pnlClass(r.pnl_usdt ?? r.realized_pnl)}>
                      {r.pnl_usdt != null ? fmtPnl(r.pnl_usdt) : r.realized_pnl != null ? fmtPnl(r.realized_pnl) : '—'}
                    </td>
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
      )}

      {selectedOrder ? (
        <section className="panel">
          <h3>订单详情</h3>
          <dl>
            <dt>订单号</dt>
            <dd>{selectedOrder.order_id}</dd>
            <dt>动作</dt>
            <dd>{displayOrderAction(selectedOrder)}</dd>
            <dt>标记</dt>
            <dd>
              {selectedOrder.marker_id ? (
                <Link
                  to={`/trade-map?symbol=${encodeURIComponent(selectedOrder.symbol || symbol)}&marker_id=${encodeURIComponent(String(selectedOrder.marker_id))}`}
                >
                  在地图查看
                </Link>
              ) : (
                '—'
              )}
            </dd>
          </dl>
          <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
            {JSON.stringify(selectedOrder, null, 2)}
          </pre>
        </section>
      ) : null}

      {selectedLink ? (
        <section className="panel">
          <h3>回合详情</h3>
          <dl>
            <dt>方向</dt>
            <dd>{displayPositionSideLabel(selectedLink.side)}</dd>
            <dt>开仓</dt>
            <dd>
              {selectedLink.entry_price} @ {formatUnixTs(selectedLink.entry_time)}
            </dd>
            <dt>平仓</dt>
            <dd>
              {selectedLink.exit_price} @ {formatUnixTs(selectedLink.exit_time)}
            </dd>
            <dt>地图</dt>
            <dd>
              {selectedLink.entry_marker_id ? (
                <Link
                  to={`/trade-map?symbol=${encodeURIComponent(symbol)}&marker_id=${encodeURIComponent(String(selectedLink.entry_marker_id))}`}
                >
                  查看开平标记
                </Link>
              ) : (
                '—'
              )}
            </dd>
          </dl>
          <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
            {JSON.stringify(selectedLink, null, 2)}
          </pre>
        </section>
      ) : null}

      <p className="status-line">
        {isFetching
          ? '加载中…'
          : viewMode === 'legs'
            ? `${linkRows.length} 回合 · ${symbol}${strategyFilter ? ` · ${strategyFilter}` : ''}`
            : `${orderRows.length} orders · ${symbol}${strategyFilter ? ` · ${strategyFilter}` : ''}`}
      </p>
    </div>
  );
}
