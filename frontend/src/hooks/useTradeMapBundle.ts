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
  bundleFeatureColumns,
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
  const pollInFlightRef = useRef(false);
  const fullInFlightRef = useRef(false);

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
        ...(data.chop_grid_overlay ? { lastChopMapData: data.chop_grid_overlay } : {}),
        ...(data.chop_regime_regions ? { chopRegimeRegions: data.chop_regime_regions } : {}),
        ...(data.strategy_stage_regions ? { strategyStageRegions: data.strategy_stage_regions } : {}),
        statusText: `${mergedCandles.length} bars · ${markerCount} markers · ${linkCount} links · ${state.selectedFeatureColumns.length} features`,
      });
    } finally {
      pollInFlightRef.current = false;
    }
  }, []);

  const refreshFull = useCallback(async () => {
    if (fullInFlightRef.current) return;
    fullInFlightRef.current = true;
    const state = useTradeMapStore.getState();
    state.setLoading(true);
    setSymbol(state.symbol);
    try {
      const init = ohlcvInitialQueryRange(state.timeframe);
      const markerFrom = state.markerQueryFromIso || init.from;

      const { data: shellData, meta: shellMeta } = await fetchBundle(buildFullShellQuery(state));
      const candles = shellData.ohlcv?.candles || [];
      state.setLastCandles(candles);
      const shellPhase: {
        loading: boolean;
        lastMainOverlays: typeof shellData.main_overlays;
        statusText: string;
        ohlcvLoadedFrom?: string;
        ohlcvLoadedTo?: string;
      } = {
        loading: false,
        lastMainOverlays: shellData.main_overlays || {},
        statusText: candles.length
          ? `${candles.length} bars · loading markers…`
          : 'loading markers…',
      };
      if (shellMeta?.range_start && !state.ohlcvLoadedFrom) {
        shellPhase.ohlcvLoadedFrom = String(shellMeta.range_start);
        shellPhase.ohlcvLoadedTo = String(shellMeta.range_end || new Date().toISOString());
      }
      useTradeMapStore.setState(shellPhase);

      const [markersResp, featuresResp] = await Promise.all([
        fetchBundle(buildFullMarkersQuery(useTradeMapStore.getState(), markerFrom)),
        fetchBundle(buildFullFeaturesQuery(useTradeMapStore.getState(), markerFrom)),
      ]);

      const latest = useTradeMapStore.getState();
      latest.setBundlePhase({
        markers: markersResp.data.markers || [],
        lastTradeLinks: (markersResp.data.trade_links || []) as TradeLink[],
        lastOverlays: (featuresResp.data.overlays || {}) as import('@/lib/tradeMap/types.ts').FeatureOverlays,
        lastChopMapData: featuresResp.data.chop_grid_overlay,
        chopRegimeRegions: featuresResp.data.chop_regime_regions || [],
        strategyStageRegions: featuresResp.data.strategy_stage_regions || {},
        markerQueryFromIso: markerFrom || null,
        lastMarkerPollSince: new Date().toISOString(),
        historyExhausted: false,
        loading: false,
        featuresLoading: false,
        statusText: `${candles.length} bars · ${(markersResp.data.markers || []).length} markers · ${(markersResp.data.trade_links || []).length} links · ${bundleFeatureColumns(latest).length} features`,
      });
      saveLayout({
        layers: latest.layers,
        selectedFeatureColumns: latest.selectedFeatureColumns,
        featureStrategyFocus: latest.featureStrategyFocus,
        mainEma1200: latest.mainEma1200,
        mainWeeklyEma200: latest.mainWeeklyEma200,
        paneVolume: latest.paneVolume,
        ordersDockOpen: latest.ordersDockOpen,
      });
      return { shellData, shellMeta, markersResp, featuresResp, candles };
    } catch (e) {
      const errState = useTradeMapStore.getState();
      errState.setLoading(false);
      errState.setStatusText(String(e));
      throw e;
    } finally {
      fullInFlightRef.current = false;
    }
  }, []);

  const refreshMainOverlays = useCallback(async () => {
    const state = useTradeMapStore.getState();
    const { data: shellData } = await fetchBundle(buildFullShellQuery(state));
    useTradeMapStore.setState({
      lastMainOverlays: shellData.main_overlays || {},
    });
  }, []);

  /** Feature column / strategy focus changes — do not block main OHLCV chart. */
  const refreshFeaturesOnly = useCallback(async () => {
    const state = useTradeMapStore.getState();
    if (!state.lastCandles.length) {
      await refreshFull();
      return;
    }
    const cols = bundleFeatureColumns(state);
    if (!cols.length) {
      useTradeMapStore.setState({
        lastOverlays: {},
        lastChopMapData: null,
        chopRegimeRegions: [],
        strategyStageRegions: {},
        featuresLoading: false,
        statusText: `${state.lastCandles.length} bars · 0 features`,
      });
      return;
    }
    useTradeMapStore.setState({ featuresLoading: true, statusText: '加载特征附图…' });
    try {
      const init = ohlcvInitialQueryRange(state.timeframe);
      const markerFrom = state.markerQueryFromIso || init.from;
      const { data } = await fetchBundle(buildFullFeaturesQuery(state, markerFrom));
      const latest = useTradeMapStore.getState();
      latest.setBundlePhase({
        lastOverlays: (data.overlays || {}) as import('@/lib/tradeMap/types.ts').FeatureOverlays,
        lastChopMapData: data.chop_grid_overlay,
        chopRegimeRegions: data.chop_regime_regions || [],
        strategyStageRegions: data.strategy_stage_regions || {},
        featuresLoading: false,
        statusText: `${latest.lastCandles.length} bars · ${latest.markers.length} markers · ${cols.length} features`,
      });
    } catch (e) {
      useTradeMapStore.setState({ featuresLoading: false, statusText: String(e) });
      throw e;
    }
  }, [refreshFull]);

  const initFromLayout = useCallback(() => {
    const layout = loadLayout();
    if (!layout) return;
    const state = useTradeMapStore.getState();
    if (layout.layers && typeof layout.layers === 'object') {
      state.setLayers(layout.layers as Partial<typeof state.layers>);
    }
    if (Array.isArray(layout.selectedFeatureColumns)) {
      state.setSelectedFeatureColumns(layout.selectedFeatureColumns as string[]);
    }
    if (typeof layout.featureStrategyFocus === 'string') {
      state.setFeatureStrategyFocus(layout.featureStrategyFocus);
    }
    if (typeof layout.mainEma1200 === 'boolean') state.setBundlePhase({ mainEma1200: layout.mainEma1200 });
    if (typeof layout.mainWeeklyEma200 === 'boolean') {
      state.setBundlePhase({ mainWeeklyEma200: layout.mainWeeklyEma200 });
    }
    if (typeof layout.paneVolume === 'boolean') state.setPaneVolume(layout.paneVolume);
    if (typeof layout.ordersDockOpen === 'boolean') state.setOrdersDockOpen(layout.ordersDockOpen);
  }, []);

  return {
    refreshFull,
    refreshFeaturesOnly,
    refreshPoll,
    refreshMarkersOnly,
    refreshMainOverlays,
    initFromLayout,
    resetHistory: resetHistoryState,
  };
}
