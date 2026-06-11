import { useCallback, useRef } from 'react';
import { apiGet } from '@/api/client.ts';
import type { BundleData, Candle, TradeLink } from '@/api/types.ts';
import { ohlcvInitialQueryRange } from '@/lib/tradeMap';
import {
  buildFullFeaturesQuery,
  buildFullMarkersQuery,
  buildFullShellQuery,
  buildMarkersOnlyQuery,
  buildPollQuery,
} from '@/lib/tradeMap/bundleQuery.ts';
import {
  mergeCandlesByTime,
  mergeMarkersById,
  mergeTradeLinks,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap';
import { setSymbol } from '@/lib/shell.ts';
import {
  loadLayout,
  resetHistoryState,
  saveLayout,
  useTradeMapStore,
} from '@/stores/tradeMapStore.ts';

async function fetchBundle(query: string): Promise<{ data: BundleData; meta?: Record<string, unknown> }> {
  return apiGet<BundleData>(`/api/trade-map/bundle?${query}`);
}

export function useTradeMapBundle() {
  const store = useTradeMapStore();
  const pollInFlightRef = useRef(false);

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

  const refreshPoll = useCallback(async () => {
    const state = useTradeMapStore.getState();
    if (state.loading || pollInFlightRef.current) return;
    if (!state.lastCandles.length) {
      return;
    }
    pollInFlightRef.current = true;
    try {
      const { data } = await fetchBundle(buildPollQuery(state));
      const tail = sanitizeCandlesForLwc(data.ohlcv?.candles || []);
      const mergedCandles = tail.length
        ? (mergeCandlesByTime(tail, state.lastCandles) as Candle[])
        : state.lastCandles;
      const mergedMarkers = mergeMarkersById(state.markers, data.markers || []);
      const mergedLinks = mergeTradeLinks(state.lastTradeLinks, data.trade_links || []);
      const markerCount = mergedMarkers.length;
      const linkCount = mergedLinks.length;
      useTradeMapStore.setState({
        lastCandles: mergedCandles,
        markers: mergedMarkers,
        lastTradeLinks: mergedLinks,
        lastMarkerPollSince: new Date().toISOString(),
        statusText: `${mergedCandles.length} bars · ${markerCount} markers · ${linkCount} links · ${state.selectedFeatureColumns.length} features`,
      });
    } finally {
      pollInFlightRef.current = false;
    }
  }, []);

  const refreshFull = useCallback(async () => {
    const state = useTradeMapStore.getState();
    state.setLoading(true);
    setSymbol(state.symbol);
    try {
      const init = ohlcvInitialQueryRange(state.timeframe);
      const markerFrom = state.markerQueryFromIso || init.from;

      const { data: shellData, meta: shellMeta } = await fetchBundle(buildFullShellQuery(state));
      const candles = shellData.ohlcv?.candles || [];
      store.setLastCandles(candles);
      if (shellMeta?.range_start && !state.ohlcvLoadedFrom) {
        store.setBundlePhase({
          ohlcvLoadedFrom: String(shellMeta.range_start),
          ohlcvLoadedTo: String(shellMeta.range_end || new Date().toISOString()),
        });
      }

      const [markersResp, featuresResp] = await Promise.all([
        fetchBundle(buildFullMarkersQuery(useTradeMapStore.getState(), markerFrom)),
        fetchBundle(buildFullFeaturesQuery(useTradeMapStore.getState(), markerFrom)),
      ]);

      store.setBundlePhase({
        markers: markersResp.data.markers || [],
        lastTradeLinks: (markersResp.data.trade_links || []) as TradeLink[],
        lastOverlays: (featuresResp.data.overlays || {}) as import('@/lib/tradeMap/types.ts').FeatureOverlays,
        lastMainOverlays: shellData.main_overlays || {},
        lastChopMapData: featuresResp.data.chop_grid_overlay,
        chopRegimeRegions: featuresResp.data.chop_regime_regions || [],
        strategyStageRegions: featuresResp.data.strategy_stage_regions || {},
        markerQueryFromIso: markerFrom || null,
        lastMarkerPollSince: new Date().toISOString(),
        historyExhausted: false,
        loading: false,
        chartFitPending: false,
        statusText: `${candles.length} bars · ${(markersResp.data.markers || []).length} markers · ${(markersResp.data.trade_links || []).length} links · ${state.selectedFeatureColumns.length} features`,
      });
      saveLayout({
        layers: state.layers,
        selectedFeatureColumns: state.selectedFeatureColumns,
        featureStrategyFocus: state.featureStrategyFocus,
        mainEma1200: state.mainEma1200,
        mainWeeklyEma200: state.mainWeeklyEma200,
        paneVolume: state.paneVolume,
        ordersDockOpen: state.ordersDockOpen,
      });
      return { shellData, shellMeta, markersResp, featuresResp, candles };
    } catch (e) {
      store.setLoading(false);
      store.setStatusText(String(e));
      throw e;
    }
  }, [store]);

  const refreshMainOverlays = useCallback(async () => {
    const state = useTradeMapStore.getState();
    const { data: shellData } = await fetchBundle(buildFullShellQuery(state));
    useTradeMapStore.setState({
      lastMainOverlays: shellData.main_overlays || {},
    });
  }, []);

  const initFromLayout = useCallback(() => {
    const layout = loadLayout();
    if (!layout) return;
    if (layout.layers && typeof layout.layers === 'object') {
      store.setLayers(layout.layers as Partial<typeof store.layers>);
    }
    if (Array.isArray(layout.selectedFeatureColumns)) {
      store.setSelectedFeatureColumns(layout.selectedFeatureColumns as string[]);
    }
    if (typeof layout.featureStrategyFocus === 'string') {
      store.setFeatureStrategyFocus(layout.featureStrategyFocus);
    }
    if (typeof layout.mainEma1200 === 'boolean') store.setBundlePhase({ mainEma1200: layout.mainEma1200 });
    if (typeof layout.mainWeeklyEma200 === 'boolean') {
      store.setBundlePhase({ mainWeeklyEma200: layout.mainWeeklyEma200 });
    }
    if (typeof layout.paneVolume === 'boolean') store.setPaneVolume(layout.paneVolume);
    if (typeof layout.ordersDockOpen === 'boolean') store.setOrdersDockOpen(layout.ordersDockOpen);
  }, [store]);

  return {
    refreshFull,
    refreshPoll,
    refreshMarkersOnly,
    refreshMainOverlays,
    initFromLayout,
    resetHistory: resetHistoryState,
  };
}
