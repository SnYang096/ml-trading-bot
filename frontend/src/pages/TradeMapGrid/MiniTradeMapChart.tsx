import { memo, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  CandlestickSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
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

  useEffect(() => {
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

    const ro = new ResizeObserver(() => {
      if (hostRef.current) {
        chart.applyOptions({
          width: hostRef.current.clientWidth,
          height: hostRef.current.clientHeight,
        });
      }
    });
    ro.observe(el);

    const clean = sanitizeCandlesForLwc(candles) as CandlestickData<Time>[];
    series.setData(clean);
    if (clean.length) chart.timeScale().fitContent();

    const display = markersForChartDisplay(markers, '', null);
    markerPlugin.setMarkers(markersToLwc(display, null) as SeriesMarker<Time>[]);

    const lines = buildTradeLinkLines(tradeLinks, candles, layers, '', timeframe);
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
    }

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [symbol, candles, markers, tradeLinks, layers, timeframe]);

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
