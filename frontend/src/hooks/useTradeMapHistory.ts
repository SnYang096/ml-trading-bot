import { useCallback, useEffect, useRef } from 'react';
import type { IChartApi } from 'lightweight-charts';
import { apiGet, apiQuery } from '@/api/client.ts';
import type { BundleData, Candle } from '@/api/types.ts';
import {
  featureColumnsParam,
  isoFromUnixSec,
  mainOverlaysQueryParam,
  mergeCandlesByTime,
  mergeFeatureOverlays,
  mergeMarkersById,
  mergeTradeLinks,
  ohlcvInitialQueryRange,
  sanitizeCandlesForLwc,
  stageRegionsQueryParam,
  tradeMapHistoryChunkDays,
} from '@/lib/tradeMap';
import {
  resetHistoryState,
  scopesFromLayers,
  useTradeMapStore,
} from '@/stores/tradeMapStore.ts';

const PREFETCH_THRESHOLD = 25;
const PREFETCH_DELAY_MS = 350;

async function fetchBundle(query: string) {
  return apiGet<BundleData>(`/api/trade-map/bundle?${query}`);
}

export function useTradeMapHistory(mainChart: IChartApi | null) {
  const inFlightRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const refreshMarkersOnly = useCallback(async () => {
    const state = useTradeMapStore.getState();
    const init = ohlcvInitialQueryRange(state.timeframe);
    const from = state.markerQueryFromIso || state.ohlcvLoadedFrom || init.from;
    const q = apiQuery({
      symbol: state.symbol,
      timeframe: state.timeframe,
      scopes: scopesFromLayers(state.layers),
      include_pending: String(state.layers.pending),
      include_ohlcv: 'none',
      include_features: 'false',
      include_markers: 'true',
      include_trade_links: 'true',
      include_chop: 'false',
      from,
      to: new Date().toISOString(),
      full_range: 'false',
    });
    const { data } = await fetchBundle(q);
    const mergedMarkers = mergeMarkersById(state.markers, data.markers || []);
    const mergedLinks = mergeTradeLinks(state.lastTradeLinks, data.trade_links || []);
    useTradeMapStore.setState({
      markers: mergedMarkers,
      lastTradeLinks: mergedLinks,
    });
  }, []);

  const loadMoreHistory = useCallback(async () => {
    const state = useTradeMapStore.getState();
    if (inFlightRef.current || state.historyExhausted || !state.lastCandles.length) return;

    const oldest = state.lastCandles[0].time;
    const chunkDays = tradeMapHistoryChunkDays(state.timeframe);
    const newFromMs = Number(oldest) * 1000 - chunkDays * 86400000;
    const newFromIso = new Date(newFromMs).toISOString();
    const featParam = featureColumnsParam(state.selectedFeatureColumns);
    const mainOl = mainOverlaysQueryParam(state.mainEma1200, state.mainWeeklyEma200);
    const stageRg = stageRegionsQueryParam(state.layers.prefilter, state.layers.gate);

    inFlightRef.current = true;
    useTradeMapStore.setState({ statusText: '加载更早历史…' });

    try {
      const q = apiQuery({
        symbol: state.symbol,
        timeframe: state.timeframe,
        scopes: scopesFromLayers(state.layers),
        include_pending: String(state.layers.pending),
        from: newFromIso,
        to: isoFromUnixSec(Number(oldest)),
        include_ohlcv: 'full',
        include_features: state.selectedFeatureColumns.length > 0 ? 'true' : 'false',
        include_markers: 'false',
        include_trade_links: 'false',
        include_chop: state.layers.chopGrid ? 'true' : 'false',
        full_range: 'false',
        feature_columns: featParam || undefined,
        main_overlays: mainOl || undefined,
        stage_regions: stageRg || undefined,
        strategy: state.featureStrategyFocus.trim() || undefined,
      });

      const { data, meta } = await fetchBundle(q);
      const more = sanitizeCandlesForLwc(data.ohlcv?.candles || []);
      if (!more.length) {
        useTradeMapStore.setState({ historyExhausted: true });
        return;
      }

      const chart = mainChart;
      const snap = chart?.timeScale().getVisibleLogicalRange() || null;
      const prevLen = state.lastCandles.length;
      const merged = mergeCandlesByTime(more, state.lastCandles) as Candle[];

      if (merged.length === prevLen) {
        useTradeMapStore.setState({ historyExhausted: true });
        return;
      }

      const added = merged.length - prevLen;
      let nextMainOverlays = state.lastMainOverlays;
      if (data.main_overlays && Object.keys(data.main_overlays).length) {
        nextMainOverlays = { ...state.lastMainOverlays, ...data.main_overlays };
      }

      let nextOverlays = state.lastOverlays;
      if (data.overlays && Object.keys(data.overlays).length) {
        nextOverlays = mergeFeatureOverlays(state.lastOverlays, data.overlays, merged);
      }

      const nextChop = data.chop_grid_overlay || state.lastChopMapData;
      const nextRegime = data.chop_regime_regions?.length
        ? data.chop_regime_regions
        : state.chopRegimeRegions;
      const nextStages = data.strategy_stage_regions || state.strategyStageRegions;

      let markerFrom = state.markerQueryFromIso;
      if (
        markerFrom == null ||
        new Date(newFromIso).getTime() < new Date(markerFrom).getTime()
      ) {
        markerFrom = newFromIso;
      }

      useTradeMapStore.setState({
        lastCandles: merged,
        lastMainOverlays: nextMainOverlays,
        lastOverlays: nextOverlays,
        lastChopMapData: nextChop,
        chopRegimeRegions: nextRegime,
        strategyStageRegions: nextStages,
        ohlcvLoadedFrom: isoFromUnixSec(merged[0].time),
        markerQueryFromIso: markerFrom,
        chartFitPending: false,
        statusText: `${merged.length} bars (+${added} history)`,
      });

      if (chart && snap && added > 0) {
        const from = Number(snap.from) + added;
        const to = Number(snap.to) + added;
        if (Number.isFinite(from) && Number.isFinite(to)) {
          chart.timeScale().setVisibleLogicalRange({ from, to });
        }
      }

      if (meta?.range_start) {
        useTradeMapStore.setState({ ohlcvLoadedFrom: String(meta.range_start) });
      }

      await refreshMarkersOnly();
    } finally {
      inFlightRef.current = false;
    }
  }, [mainChart, refreshMarkersOnly]);

  const schedulePrefetch = useCallback(
    (range: { from: number; to: number } | null) => {
      if (!range || inFlightRef.current) return;
      const state = useTradeMapStore.getState();
      if (state.historyExhausted || !state.lastCandles.length) return;
      if (range.from > PREFETCH_THRESHOLD) return;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null;
        loadMoreHistory().catch((e) => {
          useTradeMapStore.setState({ statusText: String(e) });
        });
      }, PREFETCH_DELAY_MS);
    },
    [loadMoreHistory],
  );

  useEffect(() => {
    const chart = mainChart;
    if (!chart) return;
    const handler = (range: { from: number; to: number } | null) => {
      schedulePrefetch(range);
    };
    chart.timeScale().subscribeVisibleLogicalRangeChange(handler);
    return () => {
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
    };
  }, [mainChart, schedulePrefetch]);

  return { loadMoreHistory, resetHistory: resetHistoryState, refreshMarkersOnly };
}
