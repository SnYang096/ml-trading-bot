import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { apiGet } from '@/api/client.ts';
import type { OrderRow } from '@/api/types.ts';
import { SCOPE_LABELS } from '@/lib/shell.ts';
import { scopesFromLayers, type LayerState } from '@/stores/tradeMapStore.ts';
import styles from './OrdersDock.module.css';

interface Props {
  symbol: string;
  layers: LayerState;
}

export function OrdersDock({ symbol, layers }: Props) {
  const ordersQuery = useQuery({
    queryKey: ['trade-map-orders-dock', symbol, layers],
    queryFn: () =>
      apiGet<OrderRow[]>('/api/orders/list', {
        symbol,
        scopes: scopesFromLayers(layers),
        limit: 80,
      }),
    refetchInterval: 15_000,
  });

  const rows = ordersQuery.data?.data || [];

  return (
    <section className={styles.dock} aria-label="订单表">
      <div className={styles.head}>
        <h3>订单表</h3>
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
              rows.map((r, i) => (
                <tr key={`${r.order_id}-${i}`}>
                  <td>{SCOPE_LABELS[r.scope] || r.scope}</td>
                  <td>{r.strategy}</td>
                  <td>{r.side}</td>
                  <td>{r.status}</td>
                  <td>{String(r.filled_at || r.created_at || r.time || '—').slice(0, 16)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} className="muted">
                  无订单
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
