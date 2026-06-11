import { memo, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
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
import type { Candle, TradeLink, TradeMarker } from '@/api/types.ts';
import {
  buildTradeLinkLines,
  mainChartOverlaySeriesOptions,
  markersForChartDisplay,
  markersToLwc,
  prepareChartMarkers,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap';
import { CHART_THEME } from '@/lib/tradeMap/constants.ts';
import type { LayerState } from '@/stores/tradeMapStore.ts';
import styles from './MiniTradeMapChart.module.css';

const MINI_CHART_OPTS = {
  layout: {
    background: { color: CHART_THEME.bg },
    textColor: CHART_THEME.text,
    attributionLogo: false,
  },
  grid: { vertLines: { visible: false }, horzLines: { color: CHART_THEME.grid } },
  timeScale: {
    timeVisible: false,
    secondsVisible: false,
    barSpacing: 2,
    minBarSpacing: 0.5,
    rightOffset: 4,
  },
  rightPriceScale: {
    borderVisible: false,
    scaleMargins: { top: 0.06, bottom: 0.06 },
  },
  handleScroll: false,
  handleScale: false,
};

interface Props {
  symbol: string;
  candles: Candle[];
  markers: TradeMarker[];
  tradeLinks: TradeLink[];
  layers: LayerState;
  timeframe: string;
  loading?: boolean;
  error?: string | null;
}

export const MiniTradeMapChart = memo(function MiniTradeMapChart({
  symbol,
  candles,
  markers,
  tradeLinks,
  layers,
  timeframe,
  loading,
  error,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const linkSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const fittedRef = useRef(false);

  useEffect(() => {
    fittedRef.current = false;
    const el = hostRef.current;
    if (!el) return;

    const chart = createChart(el, {
      ...MINI_CHART_OPTS,
      width: el.clientWidth,
      height: el.clientHeight,
    });
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

    const ro = new ResizeObserver(() => {
      if (hostRef.current) {
        chart.applyOptions({
          width: hostRef.current.clientWidth,
          height: hostRef.current.clientHeight,
        });
      }
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
      linkSeriesRef.current = [];
    };
  }, [symbol]);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    const clean = sanitizeCandlesForLwc(candles) as CandlestickData<Time>[];
    series.setData(clean);
    if (clean.length && !fittedRef.current) {
      chart.timeScale().fitContent();
      fittedRef.current = true;
    }
  }, [candles]);

  useEffect(() => {
    const plugin = markersRef.current;
    if (!plugin) return;
    const scoped = prepareChartMarkers(markers, candles, null, layers, '');
    const display = markersForChartDisplay(scoped, '', null);
    plugin.setMarkers(
      markersToLwc(display, null, { showText: 'compact' }) as SeriesMarker<Time>[],
    );
  }, [markers, candles, layers.trend, layers.spot, layers.multiLeg, layers.pending]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    for (const s of linkSeriesRef.current) {
      try {
        chart.removeSeries(s);
      } catch {
        /* */
      }
    }
    linkSeriesRef.current = [];
    const prepared = prepareChartMarkers(markers, candles, null, layers, '');
    const lines = buildTradeLinkLines(
      tradeLinks,
      candles,
      layers,
      '',
      timeframe,
      prepared,
    );
    for (const line of lines) {
      const ls = chart.addSeries(
        LineSeries,
        mainChartOverlaySeriesOptions({
          color: line.color,
          lineWidth: 1,
          lineStyle: 0,
          lastValueVisible: false,
          priceLineVisible: false,
        }),
      );
      ls.setData(line.points as LineData<Time>[]);
      linkSeriesRef.current.push(ls);
    }
  }, [tradeLinks, candles, layers, timeframe]);

  const recentCount = markers.filter((m) => {
    const last = candles[candles.length - 1]?.time;
    if (!last) return false;
    return Number(m.time) >= Number(last) - 86400 * 7;
  }).length;

  return (
    <article className={styles.cell}>
      <header className={styles.head}>
        <Link to={`/trade-map?symbol=${encodeURIComponent(symbol)}`} className={styles.symbolLink}>
          {symbol.replace('USDT', '')}
        </Link>
        <span className={styles.meta}>
          {loading ? '…' : error ? 'err' : `${candles.length} bars · ${markers.length} mk · ${recentCount} recent`}
        </span>
      </header>
      <div ref={hostRef} className={styles.chart} />
      {error ? <div className={styles.error}>{error}</div> : null}
    </article>
  );
});
