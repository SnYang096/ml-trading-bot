import { apiGet } from '@/api/client.ts';
import type { OrderRow, SymbolRow } from '@/api/types.ts';
import { usePageVisible } from '@/hooks/usePageVisible.ts';
import { useTradeMapBundle } from '@/hooks/useTradeMapBundle.ts';
import { useTradeMapFeatureCatalog } from '@/hooks/useTradeMapFeatureCatalog.ts';
import { useTradeMapHistory } from '@/hooks/useTradeMapHistory.ts';
import { useTradeMapMainChart } from '@/hooks/useTradeMapMainChart.ts';
import { getSymbol, SCOPE_LABELS, setSymbol } from '@/lib/shell.ts';
import {
  barSecForTimeframe,
  chopGridOverlayEnabled,
  findMarkerOnBar,
  listStrategiesForLayers,
  orderRowUnixSec,
  scrollIndexForTime,
} from '@/lib/tradeMap';
import { POLL_MS, scopesFromLayers, useTradeMapStore } from '@/stores/tradeMapStore.ts';
import { useQuery } from '@tanstack/react-query';
import type { IChartApi } from 'lightweight-charts';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useShallow } from 'zustand/react/shallow';
import { ChopGridLabelLayer } from './components/ChopGridLabelLayer.tsx';
import { FeatureDrawer } from './components/FeatureDrawer.tsx';
import { MarkerDetailDrawer } from './components/MarkerDetailDrawer.tsx';
import { OrdersDock } from './components/OrdersDock.tsx';
import { SubchartStack } from './components/SubchartStack.tsx';
import { TradeMapBusyHud, TradeMapBusyStatus } from './components/TradeMapBusyHud.tsx';
import styles from './TradeMapPage.module.css';

export function TradeMapPage() {
  const [searchParams] = useSearchParams();
  const [mainChart, setMainChart] = useState<IChartApi | null>(null);
  const [selectedOrderId, setSelectedOrderId] = useState<string | null>(null);
  const pageVisible = usePageVisible();

  const symbol = useTradeMapStore((s) => s.symbol);
  const timeframe = useTradeMapStore((s) => s.timeframe);
  const layers = useTradeMapStore((s) => s.layers);
  const markers = useTradeMapStore((s) => s.markers);
  const lastCandles = useTradeMapStore((s) => s.lastCandles);
  const lastOverlays = useTradeMapStore((s) => s.lastOverlays);
  const lastMainOverlays = useTradeMapStore((s) => s.lastMainOverlays);
  const lastChopMapData = useTradeMapStore((s) => s.lastChopMapData);
  const lastTradeLinks = useTradeMapStore((s) => s.lastTradeLinks);
  const chopRegimeRegions = useTradeMapStore((s) => s.chopRegimeRegions);
  const strategyStageRegions = useTradeMapStore((s) => s.strategyStageRegions);
  const selectedMarkerId = useTradeMapStore((s) => s.selectedMarkerId);
  const featureStrategyFocus = useTradeMapStore((s) => s.featureStrategyFocus);
  const selectedFeatureColumns = useTradeMapStore((s) => s.selectedFeatureColumns);
  const statusText = useTradeMapStore((s) => s.statusText);
  const loading = useTradeMapStore((s) => s.loading);
  const historyLoading = useTradeMapStore((s) => s.historyLoading);
  const featuresLoading = useTradeMapStore((s) => s.featuresLoading);
  const chartBusy = loading || historyLoading;
  const busyMode = loading ? 'full' : 'history';
  const mainEma1200 = useTradeMapStore((s) => s.mainEma1200);
  const mainWeeklyEma200 = useTradeMapStore((s) => s.mainWeeklyEma200);
  const featureDrawerOpen = useTradeMapStore((s) => s.featureDrawerOpen);
  const paneVolume = useTradeMapStore((s) => s.paneVolume);
  const ordersDockOpen = useTradeMapStore((s) => s.ordersDockOpen);
  const highlightBarTime = useTradeMapStore((s) => s.highlightBarTime);
  const chartFitPending = useTradeMapStore((s) => s.chartFitPending);
  const hasCandles = useTradeMapStore((s) => s.lastCandles.length > 0);

  const {
    setSymbol: setStoreSymbol,
    setTimeframe,
    setLayers,
    setSelectedMarkerId,
    setBundlePhase,
    setHighlightBarTime,
    setFeatureDrawerOpen,
    setPaneVolume,
    setOrdersDockOpen,
  } = useTradeMapStore(
    useShallow((s) => ({
      setSymbol: s.setSymbol,
      setTimeframe: s.setTimeframe,
      setLayers: s.setLayers,
      setSelectedMarkerId: s.setSelectedMarkerId,
      setBundlePhase: s.setBundlePhase,
      setHighlightBarTime: s.setHighlightBarTime,
      setFeatureDrawerOpen: s.setFeatureDrawerOpen,
      setPaneVolume: s.setPaneVolume,
      setOrdersDockOpen: s.setOrdersDockOpen,
    })),
  );

  const {
    refreshFull,
    refreshFeaturesOnly,
    refreshPoll,
    refreshMarkersOnly,
    refreshMainOverlays,
    initFromLayout,
    resetHistory,
  } = useTradeMapBundle();
  const featureFetchKeyRef = useRef<string | null>(null);
  const { applyStrategyFocus, applyLayerDefaults } = useTradeMapFeatureCatalog({
    catalogEnabled: true,
  });
  useTradeMapHistory(mainChart);

  const onChartClick = useCallback(
    (barTime: number) => {
      setHighlightBarTime(barTime);
      const tol = barSecForTimeframe(timeframe);
      const m = findMarkerOnBar(markers, barTime, tol);
      if (m?.id) {
        setSelectedMarkerId(m.id);
        setSelectedOrderId(null);
        return;
      }
      setSelectedMarkerId(null);
      setSelectedOrderId(null);
    },
    [markers, timeframe, setHighlightBarTime, setSelectedMarkerId],
  );

  const chopMapData = useMemo(
    () =>
      ({
        chop_grid_overlay: (lastChopMapData || undefined) as import('@/lib/tradeMap/chartOverlay.ts').ChopMapPayload['chop_grid_overlay'],
        chop_regime_regions: chopRegimeRegions as import('@/lib/tradeMap/chartOverlay.ts').TimeSpan[],
        strategy_stage_regions: strategyStageRegions as import('@/lib/tradeMap/chartOverlay.ts').ChopMapPayload['strategy_stage_regions'],
      }) satisfies import('@/lib/tradeMap/chartOverlay.ts').ChopMapPayload,
    [lastChopMapData, chopRegimeRegions, strategyStageRegions],
  );

  const { containerRef, chartRef, candleSeries, labelSpecs } = useTradeMapMainChart({
    candles: lastCandles,
    markers,
    tradeLinks: lastTradeLinks,
    overlays: lastOverlays,
    mainOverlays: lastMainOverlays,
    chopMapData,
    layers,
    strategyFocus: featureStrategyFocus,
    timeframe,
    selectedMarkerId,
    chartFitPending,
    onHighlightBarTime: setHighlightBarTime,
    onChartClick,
    onChartReady: setMainChart,
  });

  const symbolsQuery = useQuery({
    queryKey: ['symbols'],
    queryFn: () => apiGet<SymbolRow[]>('/api/trade-map/symbols'),
  });

  const strategies = useMemo(() => listStrategiesForLayers(layers), [layers]);
  const chopLabelsEnabled = chopGridOverlayEnabled(layers, featureStrategyFocus);

  useEffect(() => {
    const focus = featureStrategyFocus.trim();
    if (!focus) return;
    if (strategies.some((s) => s.id === focus)) return;
    applyStrategyFocus('');
  }, [strategies, featureStrategyFocus, applyStrategyFocus]);
  const selectedMarker = useMemo(
    () => markers.find((m) => m.id === selectedMarkerId) || null,
    [markers, selectedMarkerId],
  );

  useEffect(() => {
    initFromLayout();
    const symParam = searchParams.get('symbol');
    const saved = getSymbol();
    if (symParam) setStoreSymbol(symParam);
    else if (saved) setStoreSymbol(saved);
    const markerId = searchParams.get('marker_id');
    if (markerId) setSelectedMarkerId(markerId);
  }, [initFromLayout, searchParams, setStoreSymbol, setSelectedMarkerId]);

  useEffect(() => {
    featureFetchKeyRef.current = null;
    refreshFull().catch(() => { });
  }, [refreshFull, symbol, timeframe]);

  const featureFetchKey = useMemo(
    () => JSON.stringify({ cols: selectedFeatureColumns, focus: featureStrategyFocus }),
    [selectedFeatureColumns, featureStrategyFocus],
  );

  useEffect(() => {
    if (!hasCandles) return;
    if (featureFetchKeyRef.current === null) {
      featureFetchKeyRef.current = featureFetchKey;
      return;
    }
    if (featureFetchKeyRef.current === featureFetchKey) return;
    featureFetchKeyRef.current = featureFetchKey;
    refreshFeaturesOnly().catch(() => { });
  }, [featureFetchKey, hasCandles, refreshFeaturesOnly]);


  useEffect(() => {
    if (!useTradeMapStore.getState().lastCandles.length) return;
    refreshMarkersOnly().catch(() => { });
  }, [refreshMarkersOnly, layers.trend, layers.spot, layers.multiLeg, layers.pending]);

  useEffect(() => {
    if (!pageVisible || !hasCandles) return;
    const t = window.setInterval(() => refreshPoll().catch(() => { }), POLL_MS);
    return () => window.clearInterval(t);
  }, [refreshPoll, pageVisible, hasCandles]);

  const scrollChartToBarTime = useCallback(
    (barTime: number) => {
      const chart = chartRef.current;
      if (!chart || !lastCandles.length) return;
      const idx = scrollIndexForTime(lastCandles, barTime);
      if (idx < 0) return;
      const pad = 15;
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, idx - pad),
        to: Math.min(lastCandles.length - 1, idx + pad),
      });
      setHighlightBarTime(barTime);
    },
    [chartRef, lastCandles, setHighlightBarTime],
  );

  const onOrderSelect = useCallback(
    (order: OrderRow) => {
      setSelectedOrderId(order.order_id);
      const t = orderRowUnixSec(order);
      if (t != null) scrollChartToBarTime(t);
      if (order.marker_id) setSelectedMarkerId(String(order.marker_id));
    },
    [scrollChartToBarTime, setSelectedMarkerId],
  );

  const symbolOptions = symbolsQuery.data?.data?.length
    ? symbolsQuery.data.data
    : [{ symbol: 'ETHUSDT' }];

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <div className={styles.module}>
          <span className={styles.moduleLabel}>数据</span>
          <label>
            Symbol
            <select
              value={symbol}
              onChange={(e) => {
                resetHistory();
                setStoreSymbol(e.target.value);
                setSymbol(e.target.value);
              }}
            >
              {symbolOptions.map((row) => (
                <option key={row.symbol} value={row.symbol}>
                  {row.symbol}
                </option>
              ))}
            </select>
          </label>
          <label>
            周期
            <select
              value={timeframe}
              onChange={(e) => {
                resetHistory();
                setTimeframe(e.target.value);
              }}
            >
              <option value="15min">15min</option>
              <option value="2h">2h</option>
              <option value="1d">1d</option>
              <option value="1w">1w</option>
            </select>
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={mainEma1200}
              onChange={(e) => {
                setBundlePhase({ mainEma1200: e.target.checked });
                refreshMainOverlays().catch(() => { });
              }}
            />
            EMA1200
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={mainWeeklyEma200}
              onChange={(e) => {
                setBundlePhase({ mainWeeklyEma200: e.target.checked });
                refreshMainOverlays().catch(() => { });
              }}
            />
            W-EMA200
          </label>
        </div>

        <div className={styles.module}>
          <span className={styles.moduleLabel}>账户层</span>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.trend}
              onChange={(e) => {
                setLayers({ trend: e.target.checked });
                applyLayerDefaults();
              }}
            />
            {SCOPE_LABELS.trend}
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.spot}
              onChange={(e) => {
                setLayers({ spot: e.target.checked });
                applyLayerDefaults();
              }}
            />
            {SCOPE_LABELS.spot}
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.multiLeg}
              onChange={(e) => {
                setLayers({ multiLeg: e.target.checked });
                applyLayerDefaults();
              }}
            />
            {SCOPE_LABELS.multi_leg}
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.pending}
              onChange={(e) => setLayers({ pending: e.target.checked })}
            />
            含挂单
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.chopGrid}
              onChange={(e) => setLayers({ chopGrid: e.target.checked })}
            />
            网格线
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.prefilter}
              onChange={(e) => setLayers({ prefilter: e.target.checked })}
            />
            Prefilter区
          </label>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={layers.gate}
              onChange={(e) => setLayers({ gate: e.target.checked })}
            />
            Gate区
          </label>
        </div>

        <div className={styles.module}>
          <span className={styles.moduleLabel}>策略</span>
          <div className={styles.chips}>
            <button
              type="button"
              className={!featureStrategyFocus ? styles.chipActive : styles.chip}
              onClick={() => applyStrategyFocus('')}
            >
              全部
            </button>
            {strategies.map((s) => (
              <button
                key={s.id}
                type="button"
                className={featureStrategyFocus === s.id ? styles.chipActive : styles.chip}
                onClick={() => applyStrategyFocus(s.id)}
              >
                {s.title || s.id}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.module}>
          <span className={styles.moduleLabel}>附图</span>
          <label className={styles.chk}>
            <input
              type="checkbox"
              checked={paneVolume}
              onChange={(e) => setPaneVolume(e.target.checked)}
            />
            成交量
          </label>
          <button
            type="button"
            className={styles.featureBtn}
            onClick={() => setFeatureDrawerOpen(true)}
          >
            特征 <span className={styles.badge}>{selectedFeatureColumns.length}</span>
          </button>
        </div>

        <button
          type="button"
          className={ordersDockOpen ? `${styles.dockBtn} ${styles.dockBtnActive}` : styles.dockBtn}
          onClick={() => setOrdersDockOpen(!ordersDockOpen)}
        >
          {ordersDockOpen ? '隐藏订单表' : '订单表'}
        </button>
        <button type="button" onClick={() => refreshFull().catch(() => { })}>
          刷新
        </button>
        <span className={styles.status}>
          {chartBusy ? (
            <TradeMapBusyStatus mode={busyMode} />
          ) : featuresLoading ? (
            <span className={styles.statusBusyFeatures}>加载特征附图…</span>
          ) : (
            statusText || scopesFromLayers(layers)
          )}
        </span>
      </div>

      <div className={styles.mainRow}>
        <div className={styles.chartColumn}>
          <div className={styles.chartStack}>
            <div
              className={
                chartBusy ? `${styles.chartPane} ${styles.chartPaneBusy}` : styles.chartPane
              }
            >
              <div ref={containerRef} className={styles.chart} />
              {chartBusy ? <TradeMapBusyHud mode={busyMode} /> : null}
              <ChopGridLabelLayer
                chart={mainChart}
                candleSeries={candleSeries}
                candles={lastCandles}
                specs={labelSpecs}
                enabled={chopLabelsEnabled}
              />
            </div>
            {hasCandles ? (
              <SubchartStack
                mainChart={mainChart}
                candles={lastCandles}
                overlays={lastOverlays}
                onBarClick={scrollChartToBarTime}
              />
            ) : null}
          </div>
          {ordersDockOpen ? (
            <OrdersDock
              symbol={symbol}
              layers={layers}
              strategyFocus={featureStrategyFocus}
              timeframe={timeframe}
              layout="bottom"
              selectedOrderId={selectedOrderId}
              selectedMarkerId={selectedMarkerId}
              highlightBarTime={highlightBarTime}
              onSelectOrder={onOrderSelect}
            />
          ) : null}
        </div>
        {selectedMarkerId ? (
          <MarkerDetailDrawer
            markerId={selectedMarkerId}
            marker={selectedMarker}
            onClose={() => setSelectedMarkerId(null)}
          />
        ) : null}
      </div>

      {featureDrawerOpen ? <FeatureDrawer onClose={() => setFeatureDrawerOpen(false)} /> : null}
    </div>
  );
}
