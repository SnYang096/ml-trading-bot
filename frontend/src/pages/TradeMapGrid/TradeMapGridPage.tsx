import { useQueries } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type { BundleData } from '@/api/types.ts';
import { buildMiniGridQuery } from '@/lib/tradeMap/bundleQuery.ts';
import { GRID_SYMBOLS, gridOhlcvQueryRange } from '@/lib/tradeMap/grid.ts';
import { SCOPE_LABELS } from '@/lib/shell.ts';
import { scopesFromLayers, type LayerState } from '@/stores/tradeMapStore.ts';
import { MiniTradeMapChart } from './MiniTradeMapChart.tsx';
import styles from './TradeMapGridPage.module.css';

const GRID_POLL_MS = 30_000;

const defaultLayers: LayerState = {
  trend: true,
  spot: true,
  multiLeg: true,
  pending: false,
  chopGrid: false,
  prefilter: false,
  gate: false,
};

async function fetchGridCell(
  symbol: string,
  timeframe: string,
  layers: LayerState,
): Promise<BundleData> {
  const range = gridOhlcvQueryRange(timeframe);
  const q = buildMiniGridQuery(symbol, timeframe, layers, range);
  const { data } = await apiGet<BundleData>(`/api/trade-map/bundle?${q}`);
  return data;
}

export function TradeMapGridPage() {
  const [timeframe, setTimeframe] = useState('2h');
  const [layers, setLayers] = useState<LayerState>(defaultLayers);
  const scopeKey = scopesFromLayers(layers);

  const queries = useQueries({
    queries: GRID_SYMBOLS.map((symbol) => ({
      queryKey: ['trade-map-grid', symbol, timeframe, scopeKey, layers.pending],
      queryFn: () => fetchGridCell(symbol, timeframe, layers),
      refetchInterval: GRID_POLL_MS,
      staleTime: GRID_POLL_MS / 2,
    })),
  });

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
