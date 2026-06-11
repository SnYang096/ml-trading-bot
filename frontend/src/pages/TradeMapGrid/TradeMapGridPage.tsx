import { useMemo, useState } from 'react';
import { SCOPE_LABELS } from '@/lib/shell.ts';
import { type LayerState } from '@/stores/tradeMapStore.ts';
import { useStaggeredGridQueries } from '@/hooks/useStaggeredGridQueries.ts';
import { GRID_SYMBOLS } from '@/lib/tradeMap/grid.ts';
import { MiniTradeMapChart } from './MiniTradeMapChart.tsx';
import styles from './TradeMapGridPage.module.css';

const defaultLayers: LayerState = {
  trend: true,
  spot: true,
  multiLeg: true,
  pending: false,
  chopGrid: false,
  prefilter: false,
  gate: false,
};

export function TradeMapGridPage() {
  const [timeframe, setTimeframe] = useState('2h');
  const [layers, setLayers] = useState<LayerState>(defaultLayers);

  const queries = useStaggeredGridQueries(timeframe, layers);

  const statusText = useMemo(() => {
    const ok = queries.filter((q) => q.isSuccess).length;
    const pending = queries.filter((q) => q.isFetching).length;
    return `${ok}/${GRID_SYMBOLS.length} loaded${pending ? ` · ${pending} fetching` : ''}`;
  }, [queries]);

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <div className={styles.module}>
          <span className={styles.moduleLabel}>多品种地图</span>
          <span className={styles.hint}>21d · markers + links · 点击跳转单品种</span>
        </div>
        <label>
          周期
          <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            <option value="15min">15min</option>
            <option value="2h">2h</option>
            <option value="1d">1d</option>
          </select>
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.trend}
            onChange={(e) => setLayers((l) => ({ ...l, trend: e.target.checked }))}
          />
          {SCOPE_LABELS.trend}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.spot}
            onChange={(e) => setLayers((l) => ({ ...l, spot: e.target.checked }))}
          />
          {SCOPE_LABELS.spot}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.multiLeg}
            onChange={(e) => setLayers((l) => ({ ...l, multiLeg: e.target.checked }))}
          />
          {SCOPE_LABELS.multi_leg}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.pending}
            onChange={(e) => setLayers((l) => ({ ...l, pending: e.target.checked }))}
          />
          含挂单
        </label>
        <span className={styles.status}>{statusText}</span>
      </div>

      <div className={styles.grid}>
        {GRID_SYMBOLS.map((symbol, idx) => {
          const q = queries[idx];
          const data = q.data;
          return (
            <MiniTradeMapChart
              key={symbol}
              symbol={symbol}
              candles={data?.ohlcv?.candles || []}
              markers={data?.markers || []}
              tradeLinks={data?.trade_links || []}
              layers={layers}
              timeframe={timeframe}
              loading={q.isLoading}
              error={q.error ? String(q.error) : null}
            />
          );
        })}
      </div>
    </div>
  );
}
