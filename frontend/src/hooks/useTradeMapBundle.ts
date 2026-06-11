import { useCallback } from 'react';
import { apiGet, apiQuery } from '@/api/client.ts';
import type { BundleData, TradeLink } from '@/api/types.ts';
import {
  featureColumnsParam,
  mainOverlaysQueryParam,
  ohlcvInitialQueryRange,
  stageRegionsQueryParam,
} from '@/lib/tradeMap';
import { setSymbol } from '@/lib/shell.ts';
import {
  loadLayout,
  resetHistoryState,
  saveLayout,
  scopesFromLayers,
  useTradeMapStore,
} from '@/stores/tradeMapStore.ts';

function markerRangeParams(state: ReturnType<typeof useTradeMapStore.getState>) {
  const out: Record<string, string> = {};
  if (state.ohlcvLoadedFrom) out.from = state.ohlcvLoadedFrom;
  if (state.ohlcvLoadedTo) out.to = state.ohlcvLoadedTo;
  return out;
}

function buildBaseParams(state: ReturnType<typeof useTradeMapStore.getState>) {
  const featParam = featureColumnsParam(state.selectedFeatureColumns);
  const mainOl = mainOverlaysQueryParam(state.mainEma1200, state.mainWeeklyEma200);
  const stageRg = stageRegionsQueryParam(state.layers.prefilter, state.layers.gate);
  const stratFocus = state.featureStrategyFocus.trim();
  return {
    symbol: state.symbol,
    timeframe: state.timeframe,
    scopes: scopesFromLayers(state.layers),
    include_pending: String(state.layers.pending),
    ...markerRangeParams(state),
    featParam,
    mainOl,
    stageRg,
    stratFocus,
  };
}

async function fetchBundle(query: string): Promise<{ data: BundleData; meta?: Record<string, unknown> }> {
  return apiGet<BundleData>(`/api/trade-map/bundle?${query}`);
}

export function useTradeMapBundle() {
  const store = useTradeMapStore();

  const refreshFull = useCallback(async () => {
    const state = useTradeMapStore.getState();
    state.setLoading(true);
    setSymbol(state.symbol);
    try {
      const base = buildBaseParams(state);
      const init = ohlcvInitialQueryRange(state.timeframe);
      const markerFrom = state.markerQueryFromIso || init.from;
      const shellQ = apiQuery({
        symbol: base.symbol,
        timeframe: base.timeframe,
        scopes: base.scopes,
        include_pending: base.include_pending,
        include_ohlcv: 'full',
        include_features: 'false',
        include_markers: 'false',
        include_trade_links: 'false',
        include_chop: 'false',
        from: state.ohlcvLoadedFrom || init.from,
        to: state.ohlcvLoadedTo || init.to,
        full_range: state.ohlcvLoadedFrom ? 'false' : init.full_range || 'false',
        main_overlays: base.mainOl || undefined,
      });

      const { data: shellData, meta: shellMeta } = await fetchBundle(shellQ);
      const candles = shellData.ohlcv?.candles || [];
      store.setLastCandles(candles);
      if (shellMeta?.range_start && !state.ohlcvLoadedFrom) {
        store.setBundlePhase({
          ohlcvLoadedFrom: String(shellMeta.range_start),
          ohlcvLoadedTo: String(shellMeta.range_end || new Date().toISOString()),
        });
      }

      const range = markerRangeParams(useTradeMapStore.getState());
      const shared = {
        symbol: base.symbol,
        timeframe: base.timeframe,
        scopes: base.scopes,
        include_pending: base.include_pending,
        from: range.from || markerFrom || state.ohlcvLoadedFrom || undefined,
        to: range.to || state.ohlcvLoadedTo || new Date().toISOString(),
      };

      const markersQ = apiQuery({
        ...shared,
        include_ohlcv: 'none',
        include_features: 'false',
        include_markers: 'true',
        include_trade_links: 'true',
        include_chop: 'false',
      });
      const featuresQ = apiQuery({
        ...shared,
        include_ohlcv: 'none',
        include_features: 'true',
        include_markers: 'false',
        include_trade_links: 'false',
        include_chop: 'true',
        feature_columns: base.featParam || undefined,
        stage_regions: base.stageRg || undefined,
        strategy: base.stratFocus || undefined,
      });

      const [markersResp, featuresResp] = await Promise.all([
        fetchBundle(markersQ),
        fetchBundle(featuresQ),
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

  return { refreshFull, initFromLayout, resetHistory: resetHistoryState };
}
