import { useCallback, useEffect, useRef, useState } from 'react';
import {
  CandlestickSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type CandlestickData,
  type LineData,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import type { Candle, MainOverlaySpec, TradeLink, TradeMarker } from '@/api/types.ts';
import {
  bandHighlightSeriesOptions,
  buildChopGridLineSpecs,
  chopGridOverlayEnabled,
  chopMapDataForStrategyFocus,
  chopOverlaySpans,
  flattenStageRegions,
  mainChartOverlaySeriesOptions,
  spanHighlightCandles,
  type ChopGridLabelSpec,
  type ChopMapPayload,
  type TimeSpan,
} from '@/lib/tradeMap/chartOverlay.ts';
import {
  CHART_THEME,
  CHOP_REGIME_FILL,
  PREFILTER_STAGE_FILL,
  GATE_STAGE_FILL,
} from '@/lib/tradeMap/constants.ts';
import {
  buildTradeLinkLines,
  expandPriceRangeForOverlays,
  markersForChartDisplay,
  markersToLwc,
  prepareChartMarkers,
  priceRangeForChartAutoscale,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap';
import { isValidLogicalRange, visibleLogicalRange } from '@/lib/tradeMap/candles.ts';
import type { FeatureOverlays } from '@/lib/tradeMap/types.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';
import { useTradeMapStore } from '@/stores/tradeMapStore.ts';

const CHART_OPTS = {
  layout: {
    background: { color: CHART_THEME.bg },
    textColor: CHART_THEME.text,
    attributionLogo: false,
  },
  grid: { vertLines: { color: CHART_THEME.grid }, horzLines: { color: CHART_THEME.grid } },
  timeScale: {
    timeVisible: true,
    secondsVisible: false,
    barSpacing: 3,
    minBarSpacing: 0.5,
    rightOffset: 8,
  },
  rightPriceScale: {
    borderColor: CHART_THEME.border,
    scaleMargins: { top: 0.08, bottom: 0.12 },
    minimumWidth: 72,
  },
  handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
  handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
};

export interface MainChartParams {
  candles: Candle[];
  markers: TradeMarker[];
  tradeLinks: TradeLink[];
  overlays: FeatureOverlays;
  mainOverlays: Record<string, MainOverlaySpec>;
  chopMapData: ChopMapPayload | null;
  layers: LayerState;
  strategyFocus: string;
  timeframe: string;
  selectedMarkerId: string | null;
  chartFitPending: boolean;
  onHighlightBarTime: (t: number | null) => void;
  onChartClick?: (barTimeSec: number) => void;
  onChartReady?: (chart: IChartApi) => void;
}

export function useTradeMapMainChart(params: MainChartParams) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const overlaySeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const chopBandRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const prefilterBandRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const gateBandRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const mainOverlaySeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
  const tradeLinkSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const [labelSpecs, setLabelSpecs] = useState<ChopGridLabelSpec[]>([]);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const clearOverlaySeries = useCallback((chart: IChartApi) => {
    for (const s of overlaySeriesRef.current) {
      try {
        chart.removeSeries(s);
      } catch {
        /* */
      }
    }
    overlaySeriesRef.current = [];
    for (const s of [chopBandRef, prefilterBandRef, gateBandRef]) {
      if (s.current) {
        try {
          chart.removeSeries(s.current);
        } catch {
          /* */
        }
        s.current = null;
      }
    }
    for (const s of mainOverlaySeriesRef.current.values()) {
      try {
        chart.removeSeries(s);
      } catch {
        /* */
      }
    }
    mainOverlaySeriesRef.current.clear();
  }, []);

  const refreshPriceAutoscale = useCallback(() => {
    const chart = chartRef.current;
    const candles = paramsRef.current.candles;
    if (!chart || !candles.length) return;
    const logical = chart.timeScale().getVisibleLogicalRange();
    let pr = priceRangeForChartAutoscale(candles, logical);
    if (pr && paramsRef.current.mainOverlays) {
      const overlayPts = new Map<string, Array<{ time: number; value: number }>>();
      for (const [k, spec] of Object.entries(paramsRef.current.mainOverlays)) {
        if (spec?.points?.length) overlayPts.set(k, spec.points);
      }
      if (overlayPts.size) {
        pr = expandPriceRangeForOverlays(pr, candles, logical, overlayPts) || pr;
      }
    }
    if (!pr) return;
    const ps = chart.priceScale('right');
    ps.applyOptions({ autoScale: false });
    if (typeof ps.setVisibleRange === 'function') {
      ps.setVisibleRange({ from: pr.minValue, to: pr.maxValue });
    } else {
      ps.applyOptions({ autoScale: true });
    }
  }, []);

  const applyChopLayers = useCallback(
    (chart: IChartApi, candleSeries: ISeriesApi<'Candlestick'>) => {
      const p = paramsRef.current;
      const candles = p.candles;
      clearOverlaySeries(chart);
      const labels: ChopGridLabelSpec[] = [];
      if (!candles.length) {
        setLabelSpecs([]);
        return;
      }

      const chopEnabled = chopGridOverlayEnabled(p.layers, p.strategyFocus);
      const payload = chopMapDataForStrategyFocus(p.chopMapData, p.strategyFocus, chopEnabled);
      const stageBy = payload.strategy_stage_regions as
        | Record<string, Record<string, TimeSpan[]>>
        | undefined;
      const prefilterSpans = flattenStageRegions(stageBy, 'prefilter');
      const gateSpans = flattenStageRegions(stageBy, 'gate');

      const ensureBand = (
        ref: React.MutableRefObject<ISeriesApi<'Candlestick'> | null>,
        rgba: string,
      ) => {
        if (!ref.current) {
          ref.current = chart.addSeries(CandlestickSeries, bandHighlightSeriesOptions(rgba));
        }
        return ref.current;
      };

      if (p.layers.prefilter && prefilterSpans.length) {
        ensureBand(prefilterBandRef, PREFILTER_STAGE_FILL).setData(
          spanHighlightCandles(candles, prefilterSpans) as CandlestickData<Time>[],
        );
      }
      if (p.layers.gate && gateSpans.length) {
        ensureBand(gateBandRef, GATE_STAGE_FILL).setData(
          spanHighlightCandles(candles, gateSpans) as CandlestickData<Time>[],
        );
      }

      if (chopEnabled) {
        const gridSpans = chopOverlaySpans(prefilterSpans, payload);
        if (gridSpans.length) {
          if (!chopBandRef.current) {
            chopBandRef.current = chart.addSeries(
              CandlestickSeries,
              bandHighlightSeriesOptions(CHOP_REGIME_FILL),
            );
          }
          chopBandRef.current.setData(
            spanHighlightCandles(candles, gridSpans) as CandlestickData<Time>[],
          );
        }
        const lineSpecs = buildChopGridLineSpecs(
          payload.chop_grid_overlay || {},
          candles,
          gridSpans.length ? gridSpans : null,
        );
        for (const spec of lineSpecs) {
          const line = chart.addSeries(LineSeries, {
            ...mainChartOverlaySeriesOptions({
              color: spec.color,
              lineWidth: spec.lineWidth ?? 1,
              lineStyle: spec.lineStyle ?? 2,
            }),
          });
          line.setData(
            spec.points.filter((pt) => Number.isFinite(pt.value)) as LineData<Time>[],
          );
          overlaySeriesRef.current.push(line);
          if (spec.label) labels.push(spec.label);
        }
      }
      setLabelSpecs(labels);
      void candleSeries;
      refreshPriceAutoscale();
    },
    [clearOverlaySeries, refreshPriceAutoscale],
  );

  const applyTradeLinks = useCallback((chart: IChartApi) => {
    for (const s of tradeLinkSeriesRef.current) {
      try {
        chart.removeSeries(s);
      } catch {
        /* */
      }
    }
    tradeLinkSeriesRef.current = [];
    const p = paramsRef.current;
    const prepared = prepareChartMarkers(
      p.markers,
      p.candles,
      p.overlays,
      p.layers,
      p.strategyFocus,
    );
    const lines = buildTradeLinkLines(
      p.tradeLinks,
      p.candles,
      p.layers,
      p.strategyFocus,
      p.timeframe,
      prepared,
    );
    for (const line of lines) {
      const series = chart.addSeries(
        LineSeries,
        mainChartOverlaySeriesOptions({
          color: line.color,
          lineWidth: 1.5,
          lineStyle: 0,
          lastValueVisible: false,
        }),
      );
      series.setData(line.points as LineData<Time>[]);
      tradeLinkSeriesRef.current.push(series);
    }
  }, []);

  const applyMainOverlays = useCallback((chart: IChartApi) => {
    const specs = paramsRef.current.mainOverlays || {};
    for (const [key, spec] of Object.entries(specs)) {
      if (!spec?.available || !spec.points?.length) continue;
      let series = mainOverlaySeriesRef.current.get(key);
      if (!series) {
        series = chart.addSeries(
          LineSeries,
          mainChartOverlaySeriesOptions({
            color: key.includes('weekly') ? CHART_THEME.emaSecondary : CHART_THEME.emaPrimary,
          }),
        );
        mainOverlaySeriesRef.current.set(key, series);
      }
      series.setData(spec.points as LineData<Time>[]);
    }
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, CHART_OPTS);
    const series = chart.addSeries(CandlestickSeries, {
      upColor: CHART_THEME.candleUp,
      downColor: CHART_THEME.candleDown,
      borderVisible: false,
      wickUpColor: CHART_THEME.candleUp,
      wickDownColor: CHART_THEME.candleDown,
    });
    const markerPlugin = createSeriesMarkers(series);
    chartRef.current = chart;
    seriesRef.current = series;
    markersRef.current = markerPlugin;
    paramsRef.current.onChartReady?.(chart);

    chart.subscribeCrosshairMove((param) => {
      if (!param.time) {
        paramsRef.current.onHighlightBarTime(null);
        return;
      }
      paramsRef.current.onHighlightBarTime(Number(param.time));
    });

    chart.subscribeClick((param) => {
      if (!param.time) return;
      paramsRef.current.onChartClick?.(Number(param.time));
    });

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
        refreshPriceAutoscale();
      }
    });
    ro.observe(containerRef.current);

    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      refreshPriceAutoscale();
    });

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
      overlaySeriesRef.current = [];
      chopBandRef.current = null;
      prefilterBandRef.current = null;
      gateBandRef.current = null;
      tradeLinkSeriesRef.current = [];
      mainOverlaySeriesRef.current.clear();
    };
  }, [refreshPriceAutoscale]);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    const p = paramsRef.current;
    const clean = sanitizeCandlesForLwc(p.candles) as CandlestickData<Time>[];
    if (!clean.length) return;
    series.setData(clean);

    const scrollAdjust = useTradeMapStore.getState().historyScrollAdjust;
    if (scrollAdjust) {
      if (isValidLogicalRange(scrollAdjust, clean.length)) {
        chart.timeScale().setVisibleLogicalRange(scrollAdjust);
      }
      useTradeMapStore.setState({ historyScrollAdjust: null });
    } else if (p.chartFitPending) {
      const lr = visibleLogicalRange(clean.length);
      if (lr) {
        chart.timeScale().setVisibleLogicalRange(lr);
      }
      useTradeMapStore.getState().setBundlePhase({ chartFitPending: false });
    }
    applyMainOverlays(chart);
    applyChopLayers(chart, series);
    applyTradeLinks(chart);
    refreshPriceAutoscale();
  }, [
    params.candles,
    params.mainOverlays,
    params.chopMapData,
    params.layers,
    params.strategyFocus,
    params.tradeLinks,
    params.timeframe,
    params.chartFitPending,
    applyChopLayers,
    applyMainOverlays,
    applyTradeLinks,
    refreshPriceAutoscale,
  ]);

  useEffect(() => {
    const plugin = markersRef.current;
    if (!plugin) return;
    const p = paramsRef.current;
    const prepared = prepareChartMarkers(
      p.markers,
      p.candles,
      p.overlays,
      p.layers,
      p.strategyFocus,
    );
    const display = markersForChartDisplay(prepared, p.strategyFocus, p.selectedMarkerId);
    plugin.setMarkers(markersToLwc(display, p.selectedMarkerId) as SeriesMarker<Time>[]);
  }, [
    params.markers,
    params.candles,
    params.overlays,
    params.layers,
    params.strategyFocus,
    params.selectedMarkerId,
  ]);

  return { containerRef, chartRef, candleSeriesRef: seriesRef, labelSpecs, refreshPriceAutoscale };
}
