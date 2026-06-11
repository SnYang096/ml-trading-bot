import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/api/client.ts';
import type { TradeMarker } from '@/api/types.ts';
import styles from './MarkerDetailDrawer.module.css';

interface Props {
  markerId: string | null;
  marker: TradeMarker | null | undefined;
  onClose: () => void;
}

export function MarkerDetailDrawer({ markerId, marker, onClose }: Props) {
  const detailQuery = useQuery({
    queryKey: ['marker-detail', markerId],
    queryFn: () =>
      apiGet<Record<string, unknown>>(
        `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(String(markerId))}`,
      ),
    enabled: !!markerId,
  });

  if (!markerId) return null;

  return (
    <section className={styles.drawer} aria-label="标记详情">
      <div className={styles.head}>
        <h3>标记详情</h3>
        <button type="button" onClick={onClose} aria-label="关闭">
          ×
        </button>
      </div>
      <pre className={styles.body}>
        {JSON.stringify(
          { marker: marker || { id: markerId }, db: detailQuery.data?.data },
          null,
          2,
        )}
      </pre>
      {detailQuery.isFetching ? <p className={styles.status}>加载 DB 详情…</p> : null}
      {detailQuery.error ? (
        <p className={styles.status}>DB lookup failed: {String(detailQuery.error)}</p>
      ) : null}
    </section>
  );
}
