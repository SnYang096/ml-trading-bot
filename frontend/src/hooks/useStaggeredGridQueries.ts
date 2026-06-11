import { useQueries } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type { BundleData, Candle } from '@/api/types.ts';
import { visibleRefetchInterval, usePageVisible } from '@/hooks/usePageVisible.ts';
import { buildGridPollQuery, buildMiniGridQuery } from '@/lib/tradeMap/bundleQuery.ts';
import { GRID_SYMBOLS, gridOhlcvQueryRange } from '@/lib/tradeMap/grid.ts';
import {
  mergeCandlesByTime,
  mergeMarkersById,
  mergeTradeLinks,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap';

const GRID_POLL_MS = 30_000;
const STAGGER_MS = 300;

export interface GridCellData {
  ohlcv: BundleData['ohlcv'];
  markers: BundleData['markers'];
  trade_links: BundleData['trade_links'];
}

function mergeGridBundle(prev: GridCellData | undefined, incoming: BundleData): GridCellData {
  const prevCandles = prev?.ohlcv?.candles || [];
  const tail = sanitizeCandlesForLwc(incoming.ohlcv?.candles || []);
  const mergedCandles = tail.length
    ? (mergeCandlesByTime(tail, prevCandles) as Candle[])
    : prevCandles;
  return {
    ohlcv: {
      ...incoming.ohlcv,
      candles: mergedCandles.length ? mergedCandles : incoming.ohlcv?.candles || [],
    },
    markers: mergeMarkersById(prev?.markers, incoming.markers || []),
    trade_links: mergeTradeLinks(prev?.trade_links, incoming.trade_links || []),
  };
}

async function fetchGridFull(symbol: string, timeframe: string): Promise<GridCellData> {
  const range = gridOhlcvQueryRange(timeframe);
  const q = buildMiniGridQuery(symbol, timeframe, range);
  const { data } = await apiGet<BundleData>(`/api/trade-map/bundle?${q}`);
  return {
    ohlcv: data.ohlcv,
    markers: data.markers || [],
    trade_links: data.trade_links || [],
  };
}

async function fetchGridPoll(
  symbol: string,
  timeframe: string,
  prev: GridCellData | undefined,
  lastPollSince: string | null,
): Promise<{ data: GridCellData; pollSince: string }> {
  const range = gridOhlcvQueryRange(timeframe);
  const candles = prev?.ohlcv?.candles || [];
  const q = buildGridPollQuery(symbol, timeframe, range, candles, lastPollSince);
  const { data } = await apiGet<BundleData>(`/api/trade-map/bundle?${q}`);
  return {
    data: mergeGridBundle(prev, data),
    pollSince: new Date().toISOString(),
  };
}

/** OHLCV/markers fetched once per symbol+TF; B/A/C/pending toggles are client-side in MiniTradeMapChart. */
export function useStaggeredGridQueries(timeframe: string) {
  const pageVisible = usePageVisible();
  const [enabledCount, setEnabledCount] = useState(1);
  const pollSinceRef = useRef<Record<string, string>>({});
  const dataRef = useRef<Record<string, GridCellData>>({});

  useEffect(() => {
    setEnabledCount(1);
    pollSinceRef.current = {};
    dataRef.current = {};
    const timers = GRID_SYMBOLS.slice(1).map((_, i) =>
      window.setTimeout(() => setEnabledCount((c) => Math.max(c, i + 2)), STAGGER_MS * (i + 1)),
    );
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [timeframe]);

  const pollInterval = visibleRefetchInterval(pageVisible, GRID_POLL_MS);

  return useQueries({
    queries: GRID_SYMBOLS.map((symbol, idx) => ({
      queryKey: ['trade-map-grid', symbol, timeframe],
      queryFn: async (): Promise<GridCellData> => {
        const prev = dataRef.current[symbol];
        const since = pollSinceRef.current[symbol] || null;
        if (prev && since) {
          const { data, pollSince } = await fetchGridPoll(symbol, timeframe, prev, since);
          dataRef.current[symbol] = data;
          pollSinceRef.current[symbol] = pollSince;
          return data;
        }
        const full = await fetchGridFull(symbol, timeframe);
        dataRef.current[symbol] = full;
        pollSinceRef.current[symbol] = new Date().toISOString();
        return full;
      },
      enabled: idx < enabledCount,
      refetchInterval: idx < enabledCount ? pollInterval : false,
      staleTime: GRID_POLL_MS / 2,
    })),
  });
}
