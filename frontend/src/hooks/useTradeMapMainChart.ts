import type { Candle, MainOverlaySpec, TradeLink, TradeMarker } from '@/api/types.ts';
import { useTheme } from '@/context/ThemeContext.tsx';
import {
  buildTradeLinkLines,
  markersForChartDisplay,
  markersToLwc,
  prepareChartMarkers,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap';
import {
  isValidLogicalRange,
  visibleLogicalRange,
  type LogicalRange,
} from '@/lib/tradeMap/candles.ts';
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
import { candleSeriesOptions, chartLayoutOptions } from '@/lib/tradeMap/chartTheme.ts';
import {
  CHART_THEME,
  CHOP_REGIME_FILL,
  GATE_STAGE_FILL,
  PREFILTER_STAGE_FILL,
} from '@/lib/tradeMap/constants.ts';
import type { FeatureOverlays } from '@/lib/tradeMap/types.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';
import { useTradeMapStore } from '@/stores/tradeMapStore.ts';
import {
  CandlestickSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type LineData,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import { useCallback, useEffect, useRef, useState } from 'react';

function labelSpecsEqual(a: ChopGridLabelSpec[], b: ChopGridLabelSpec[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (
      x.price !== y.price ||
      x.text !== y.text ||
      x.side !== y.side ||
      x.kind !== y.kind ||
      x.color !== y.color
    ) {
      return false;
    }
    const xs = x.spans;
    const ys = y.spans;
    if (xs === ys) continue;
    if (!xs || !ys || xs.length !== ys.length) return false;
    for (let j = 0; j < xs.length; j++) {
      if (xs[j].start !== ys[j].start || xs[j].end !== ys[j].end) return false;
    }
  }
  return true;
}

const CHART_OPTS_BASE = {
  timeScale: {
    timeVisible: true,
    secondsVisible: false,
    barSpacing: 3,
    minBarSpacing: 0.5,
    rightOffset: 8,
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
  const { chartTheme } = useTheme();
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
  const [candleSeries, setCandleSeries] = useState<ISeriesApi<'Candlestick'> | null>(null);
  const [chartReadyTick, setChartReadyTick] = useState(0);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const applyChartViewport = useCallback(
    (
      chart: IChartApi,
      barCount: number,
      fitPending: boolean,
      rangeBeforeSetData?: LogicalRange | null,
    ) => {
      const scrollAdjust = useTradeMapStore.getState().historyScrollAdjust;
      if (scrollAdjust && isValidLogicalRange(scrollAdjust, barCount)) {
        chart.timeScale().setVisibleLogicalRange(scrollAdjust);
        useTradeMapStore.setState({ historyScrollAdjust: null });
        return;
      }
      if (scrollAdjust) {
        useTradeMapStore.setState({ historyScrollAdjust: null });
      }

      /* ── Initial load / symbol switch → snap to tail ── */
      if (fitPending) {
        const lr = visibleLogicalRange(barCount);
        if (lr && isValidLogicalRange(lr, barCount)) {
          chart.timeScale().setVisibleLogicalRange(lr);
          useTradeMapStore.getState().setBundlePhase({ chartFitPending: false });
        }
        return;
      }

      /*
       * ── Poll / overlay refresh → preserve user's scroll position ──
       *
       * With rightOffset > 0 the timeScale's visible range extends beyond
       * barCount (e.g. rightOffset=8 → to ≈ barCount+7), so the old
       * isValidLogicalRange(to < barCount) check always failed and forced
       * a snap-to-right on every poll tick.
       *
       * Now we simply trust the captured range and restore it.
       * rangeBeforeSetData is only set on history prepend (handled above
       * via scrollAdjust), so for normal poll we skip the restore too —
       * lightweight-charts preserves the visible logical range across
       * series.setData() when data is only appended.
       */
    },
    [],
  );

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
    // NOTE: mainOverlaySeriesRef (EMA1200, W-EMA200) is NOT cleared here.
    // It is managed independently by applyMainOverlays to avoid being
    // destroyed every time chop layers refresh.
  }, []);

  const refreshPriceAutoscale = useCallback(() => {
    const chart = chartRef.current;
    if (!chart || !paramsRef.current.candles.length) return;
    chart.priceScale('right').applyOptions({ autoScale: true });
  }, []);

  const applyChopLayers = useCallback(
    (chart: IChartApi, candleSeries: ISeriesApi<'Candlestick'>) => {
      const p = paramsRef.current;
      const candles = p.candles;
      clearOverlaySeries(chart);
      const labels: ChopGridLabelSpec[] = [];
      if (!candles.length) {
        setLabelSpecs((prev) => (prev.length === 0 ? prev : []));
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
          const lineSpecs = buildChopGridLineSpecs(
            payload.chop_grid_overlay || {},
            candles,
            gridSpans,
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
      }
      setLabelSpecs((prev) => (labelSpecsEqual(prev, labels) ? prev : labels));
      void candleSeries;
    },
    [clearOverlaySeries],
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
    const lines = buildTradeLinkLines(
      p.tradeLinks,
      p.candles,
      p.layers,
      p.strategyFocus,
      p.timeframe,
      p.markers,
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
    const chart = createChart(containerRef.current, {
      ...CHART_OPTS_BASE,
      ...chartLayoutOptions(chartTheme),
    });
    const series = chart.addSeries(CandlestickSeries, candleSeriesOptions(chartTheme));
    const markerPlugin = createSeriesMarkers(series);
    chartRef.current = chart;
    seriesRef.current = series;
    setCandleSeries(series);
    markersRef.current = markerPlugin;
    paramsRef.current.onChartReady?.(chart);
    setChartReadyTick((n) => n + 1);
    // 暴露给 E2E 测试
    if (typeof window !== 'undefined') (window as any).__lwcChart = chart;

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

    return () => {
      ro.disconnect();
      chart.remove();
      if (typeof window !== 'undefined' && (window as any).__lwcChart === chart) {
        delete (window as any).__lwcChart;
      }
      chartRef.current = null;
      seriesRef.current = null;
      setCandleSeries(null);
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
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart) return;
    chart.applyOptions(chartLayoutOptions(chartTheme));
    if (series) series.applyOptions(candleSeriesOptions(chartTheme));
  }, [chartTheme]);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    const p = paramsRef.current;
    const clean = sanitizeCandlesForLwc(p.candles) as CandlestickData<Time>[];
    if (!clean.length) return;
    const rangeBeforeSetData = chart.timeScale().getVisibleLogicalRange();
    series.setData(clean);
    applyChartViewport(chart, clean.length, p.chartFitPending, rangeBeforeSetData);
    applyMainOverlays(chart);
    applyChopLayers(chart, series);
    applyTradeLinks(chart);
  }, [
    params.candles,
    params.mainOverlays,
    params.chopMapData,
    params.layers,
    params.strategyFocus,
    params.tradeLinks,
    params.timeframe,
    params.chartFitPending,
    chartReadyTick,
    applyChartViewport,
    applyChopLayers,
    applyMainOverlays,
    applyTradeLinks,
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

  return { containerRef, chartRef, candleSeries, labelSpecs, refreshPriceAutoscale };
}
