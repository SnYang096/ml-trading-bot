import { useEffect, useRef } from 'react';
import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type CandlestickData,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import { markersForChartDisplay, markersToLwc, sanitizeCandlesForLwc } from '@/lib/tradeMap';
import type { Candle, TradeMarker } from '@/api/types.ts';

const CHART_OPTS = {
  layout: {
    background: { color: '#0f1419' },
    textColor: '#8b949e',
    attributionLogo: false,
  },
  grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
  timeScale: {
    timeVisible: true,
    secondsVisible: false,
    barSpacing: 3,
    minBarSpacing: 0.5,
    rightOffset: 8,
  },
  rightPriceScale: {
    borderColor: '#30363d',
    scaleMargins: { top: 0.08, bottom: 0.12 },
    minimumWidth: 72,
  },
  handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
  handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
};

export function useLightweightChart(
  candles: Candle[],
  markers: TradeMarker[],
  selectedMarkerId: string | null,
  featureStrategyFocus: string,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, CHART_OPTS);
    const series = chart.addSeries(CandlestickSeries, {});
    const markerPlugin = createSeriesMarkers(series);
    chartRef.current = chart;
    seriesRef.current = series;
    markersRef.current = markerPlugin;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    const clean = sanitizeCandlesForLwc(candles) as CandlestickData<Time>[];
    series.setData(clean);
    if (clean.length && chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [candles]);

  useEffect(() => {
    const plugin = markersRef.current;
    if (!plugin) return;
    const display = markersForChartDisplay(markers, featureStrategyFocus, selectedMarkerId);
    plugin.setMarkers(markersToLwc(display, selectedMarkerId) as SeriesMarker<Time>[]);
  }, [markers, selectedMarkerId, featureStrategyFocus]);

  return containerRef;
}
