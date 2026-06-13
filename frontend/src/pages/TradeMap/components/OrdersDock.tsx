import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { apiGet } from '@/api/client.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import type { OrderRow } from '@/api/types.ts';
import { barSecForTimeframe, orderOnBar, orderRowUnixSec } from '@/lib/tradeMap/orderTime.ts';
import { SCOPE_LABELS } from '@/lib/shell.ts';
import { scopesFromLayers, type LayerState } from '@/stores/tradeMapStore.ts';
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
  const ordersQuery = useQuery({
    queryKey: ['trade-map-orders-dock', symbol, layers, strat],
    queryFn: () =>
      apiGet<OrderRow[]>('/api/orders/list', {
        symbol,
        scopes: scopesFromLayers(layers),
        strategy: strat || undefined,
        limit: 80,
      }),
    refetchInterval: visibleRefetchInterval(pageVisible, 15_000),
  });

  const rows = ordersQuery.data?.data || [];
  const barSec = barSecForTimeframe(timeframe);
  const wrapClass = layout === 'side' ? styles.dockSide : styles.dockBottom;

  return (
    <section className={wrapClass} aria-label="订单表">
      <div className={styles.head}>
        <h3>订单表</h3>
        <span className={styles.hint}>
          点击行定位主图 · 十字线悬停高亮
          {strat ? ` · 策略=${strat}` : ''}
        </span>
        <Link to={`/orders?symbol=${encodeURIComponent(symbol)}`}>完整订单页</Link>
      </div>
      <div className={styles.scroll}>
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
            {rows.length ? (
              rows.map((r) => {
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
      </div>
    </section>
  );
}

export { orderRowUnixSec };
