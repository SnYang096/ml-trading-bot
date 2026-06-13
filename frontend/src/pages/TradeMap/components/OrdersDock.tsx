import { apiGet } from '@/api/client.ts';
import type { OrderRow, TradeLink } from '@/api/types.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import {
  SCOPE_LABELS,
  displayExitKind,
  displayLinkQty,
  displayPositionSideLabel,
  fmtPnl,
  formatUnixTs,
  pnlClass,
} from '@/lib/shell.ts';
import { barSecForTimeframe, orderOnBar } from '@/lib/tradeMap/orderTime.ts';
import { scopesFromLayers, type LayerState } from '@/stores/tradeMapStore.ts';
import { useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './OrdersDock.module.css';

interface Props {
  symbol: string;
  layers: LayerState;
  strategyFocus?: string;
  timeframe: string;
  layout?: 'bottom' | 'side';
  selectedOrderId: string | null;
  selectedMarkerId?: string | null;
  highlightBarTime: number | null;
  onSelectOrder: (order: OrderRow) => void;
}

type ViewMode = 'orders' | 'legs';

export function OrdersDock({
  symbol,
  layers,
  strategyFocus = '',
  timeframe,
  layout = 'bottom',
  selectedOrderId,
  selectedMarkerId,
  highlightBarTime,
  onSelectOrder,
}: Props) {
  const pageVisible = usePageVisible();
  const strat = String(strategyFocus || '').trim();
  const [viewMode, setViewMode] = useState<ViewMode>('orders');

  const ordersQuery = useQuery({
    queryKey: ['trade-map-orders-dock', symbol, layers, strat],
    queryFn: () =>
      apiGet<OrderRow[]>('/api/orders/list', {
        symbol,
        scopes: scopesFromLayers(layers),
        strategy: strat || undefined,
        limit: 80,
      }),
    enabled: viewMode === 'orders',
    refetchInterval: visibleRefetchInterval(pageVisible, 15_000),
  });

  const linksQuery = useQuery({
    queryKey: ['trade-map-orders-links', symbol, layers, strat],
    queryFn: () =>
      apiGet<TradeLink[]>('/api/orders/trade-links', {
        symbol,
        scopes: scopesFromLayers(layers),
        strategy: strat || undefined,
        limit: 200,
      }),
    enabled: viewMode === 'legs',
    refetchInterval: visibleRefetchInterval(pageVisible, 15_000),
  });

  const orderRows = ordersQuery.data?.data || [];
  const linkRows = linksQuery.data?.data || [];
  const barSec = barSecForTimeframe(timeframe);
  const wrapClass = layout === 'side' ? styles.dockSide : styles.dockBottom;

  const selectedLegId = useMemo(() => {
    if (selectedMarkerId == null) return null;
    const match = linkRows.find(
      (r) =>
        String(r.entry_marker_id) === String(selectedMarkerId) ||
        String(r.exit_marker_id) === String(selectedMarkerId),
    );
    return match ? `${match.entry_marker_id}-${match.exit_marker_id}` : null;
  }, [linkRows, selectedMarkerId]);

  const highlightLegId = useMemo(() => {
    if (highlightBarTime == null || !barSec) return null;
    const t = highlightBarTime;
    const match = linkRows.find(
      (r) =>
        (r.entry_time <= t && t <= r.exit_time) ||
        (r.entry_time >= t && t >= r.exit_time),
    );
    return match ? `${match.entry_marker_id}-${match.exit_marker_id}` : null;
  }, [linkRows, highlightBarTime, barSec]);

  return (
    <section className={wrapClass} aria-label="订单表">
      <div className={styles.head}>
        <h3>订单表</h3>
        <span className={styles.hint}>
          点击行定位主图 · 十字线悬停高亮
          {strat ? ` · 策略=${strat}` : ''}
        </span>
        <select
          value={viewMode}
          onChange={(e) => setViewMode(e.target.value as ViewMode)}
          className={styles.viewModeSelect}
        >
          <option value="orders">原始订单</option>
          <option value="legs">回合（开平一行）</option>
        </select>
        <Link to={`/orders?symbol=${encodeURIComponent(symbol)}&view=positions`}>完整订单页</Link>
      </div>
      <div className={styles.scroll}>
        {viewMode === 'orders' ? (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Scope</th>
                <th>Strategy</th>
                <th>Side</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {orderRows.length ? (
                orderRows.map((r) => {
                  const selected =
                    (selectedOrderId != null && r.order_id === selectedOrderId) ||
                    (selectedMarkerId != null &&
                      r.marker_id != null &&
                      String(r.marker_id) === String(selectedMarkerId));
                  const hovered =
                    highlightBarTime != null && orderOnBar(r, highlightBarTime, barSec);
                  const rowClass = selected
                    ? styles.rowSelected
                    : hovered
                      ? styles.rowHovered
                      : undefined;
                  return (
                    <tr
                      key={r.order_id}
                      className={rowClass}
                      onClick={() => onSelectOrder(r)}
                    >
                      <td>{SCOPE_LABELS[r.scope] || r.scope}</td>
                      <td>{r.strategy}</td>
                      <td>{r.side}</td>
                      <td>{r.status}</td>
                      <td>{String(r.filled_at || r.created_at || r.time || '—').slice(0, 16)}</td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={5} className="muted">
                    {ordersQuery.isFetching ? '加载中…' : '无订单'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Leg</th>
                <th>方向</th>
                <th>Qty</th>
                <th>开仓价</th>
                <th>平仓价</th>
                <th>PNL</th>
                <th>平仓</th>
                <th>开仓时间</th>
                <th>平仓时间</th>
              </tr>
            </thead>
            <tbody>
              {linkRows.length ? (
                linkRows.map((r) => {
                  const key = `${r.entry_marker_id}-${r.exit_marker_id}`;
                  const selected =
                    selectedLegId === key ||
                    (selectedMarkerId != null &&
                      (String(r.entry_marker_id) === String(selectedMarkerId) ||
                        String(r.exit_marker_id) === String(selectedMarkerId)));
                  const hovered = highlightLegId === key;
                  const rowClass = selected
                    ? styles.rowSelected
                    : hovered
                      ? styles.rowHovered
                      : undefined;
                  const onRowClick = () => {
                    if (r.entry_marker_id) {
                      onSelectOrder({ order_id: '', marker_id: r.entry_marker_id } as OrderRow);
                    }
                  };
                  return (
                    <tr
                      key={key}
                      className={rowClass}
                      onClick={onRowClick}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>{r.strategy || '—'}</td>
                      <td>{r.leg || '—'}</td>
                      <td>{displayPositionSideLabel(r.side)}</td>
                      <td>{displayLinkQty(r)}</td>
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
                    {linksQuery.isFetching ? '加载中…' : '无已平仓回合'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
