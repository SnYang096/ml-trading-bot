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
  sanitizeCandlesForLwc,
  stageRegionsQueryParam,
  tradeMapHistoryChunkDays,
} from '@/lib/tradeMap';
import { logicalRangeAfterHistoryPrepend } from '@/lib/tradeMap/candles.ts';
import { buildMarkersOnlyQuery, bundleFeatureColumns } from '@/lib/tradeMap/bundleQuery.ts';
import {
  resetHistoryState,
  scopesFromLayers,
  useTradeMapStore,
} from '@/stores/tradeMapStore.ts';

const PREFETCH_THRESHOLD = 25;
const PREFETCH_DELAY_MS = 600;

async function fetchBundle(query: string) {
  return apiGet<BundleData>(`/api/trade-map/bundle?${query}`);
}

export function useTradeMapHistory(mainChart: IChartApi | null) {
  const inFlightRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const refreshMarkersOnly = useCallback(async () => {
    const state = useTradeMapStore.getState();
    const { data } = await fetchBundle(buildMarkersOnlyQuery(state));
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
    const featParam = featureColumnsParam(bundleFeatureColumns(state));
    const mainOl = mainOverlaysQueryParam(state.mainEma1200, state.mainWeeklyEma200);
    const stageRg = stageRegionsQueryParam(state.layers.prefilter, state.layers.gate);

    inFlightRef.current = true;
    useTradeMapStore.setState({ statusText: '加载更早历史…' });

    const cols = bundleFeatureColumns(state);
    const needFeatures = cols.length > 0;
    const needChop = state.layers.chopGrid;
    const needStages = Boolean(stageRg);

    try {
      const ohlcvQ = apiQuery({
        symbol: state.symbol,
        timeframe: state.timeframe,
        scopes: scopesFromLayers(state.layers),
        include_pending: String(state.layers.pending),
        from: newFromIso,
        to: isoFromUnixSec(Number(oldest)),
        include_ohlcv: 'full',
        include_features: 'false',
        include_markers: 'false',
        include_trade_links: 'false',
        include_chop: 'false',
        full_range: 'false',
        main_overlays: mainOl || undefined,
      });

      const { data, meta } = await fetchBundle(ohlcvQ);
      const more = sanitizeCandlesForLwc(data.ohlcv?.candles || []);
      if (!more.length) {
        useTradeMapStore.setState({ historyExhausted: true });
        return;
      }

      const snap = mainChart?.timeScale().getVisibleLogicalRange() || null;
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

      let markerFrom = state.markerQueryFromIso;
      if (
        markerFrom == null ||
        new Date(newFromIso).getTime() < new Date(markerFrom).getTime()
      ) {
        markerFrom = newFromIso;
      }

      const scrollAdjust = logicalRangeAfterHistoryPrepend(snap, added, merged.length);

      useTradeMapStore.setState({
        lastCandles: merged,
        lastMainOverlays: nextMainOverlays,
        ohlcvLoadedFrom: isoFromUnixSec(merged[0].time),
        markerQueryFromIso: markerFrom,
        chartFitPending: false,
        historyScrollAdjust: scrollAdjust,
        statusText: `${merged.length} bars (+${added} history)`,
      });

      if (meta?.range_start) {
        useTradeMapStore.setState({ ohlcvLoadedFrom: String(meta.range_start) });
      }

      if (needFeatures || needChop || needStages) {
        const featQ = apiQuery({
          symbol: state.symbol,
          timeframe: state.timeframe,
          scopes: scopesFromLayers(state.layers),
          include_pending: String(state.layers.pending),
          from: newFromIso,
          to: isoFromUnixSec(Number(oldest)),
          include_ohlcv: 'none',
          include_features: needFeatures ? 'true' : 'false',
          include_markers: 'false',
          include_trade_links: 'false',
          include_chop: needChop ? 'true' : 'false',
          full_range: 'false',
          feature_columns: featParam || undefined,
          stage_regions: stageRg || undefined,
          strategy: state.featureStrategyFocus.trim() || undefined,
        });
        const { data: featData } = await fetchBundle(featQ);
        const overlayPatch: {
          lastOverlays?: typeof state.lastOverlays;
          lastChopMapData?: typeof state.lastChopMapData;
          chopRegimeRegions?: typeof state.chopRegimeRegions;
          strategyStageRegions?: typeof state.strategyStageRegions;
        } = {};
        if (featData.overlays && Object.keys(featData.overlays).length) {
          overlayPatch.lastOverlays = mergeFeatureOverlays(
            useTradeMapStore.getState().lastOverlays,
            featData.overlays,
            merged,
          );
        }
        if (featData.chop_grid_overlay) overlayPatch.lastChopMapData = featData.chop_grid_overlay;
        if (featData.chop_regime_regions?.length) {
          overlayPatch.chopRegimeRegions = featData.chop_regime_regions;
        }
        if (featData.strategy_stage_regions) {
          overlayPatch.strategyStageRegions = featData.strategy_stage_regions;
        }
        if (Object.keys(overlayPatch).length) {
          useTradeMapStore.setState(overlayPatch);
        }
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
