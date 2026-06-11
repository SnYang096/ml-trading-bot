import type { Candle } from '@/api/types.ts';
import {
  clipOverlayPointsToCandles,
  forwardFillOverlayToCandles,
  subchartColor,
} from '@/lib/tradeMap';
import { mainChartOverlaySeriesOptions, subchartBaseOptions } from '@/lib/tradeMap/chartOverlay.ts';
import { syncSubchartToMain } from '@/lib/tradeMap/chartSync.ts';
import { CHART_THEME } from '@/lib/tradeMap/constants.ts';
import type { FeatureOverlay } from '@/lib/tradeMap/types.ts';
import {
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts';
import { useEffect, useRef } from 'react';
import styles from './SubchartStack.module.css';

function featureSeriesData(
  overlay: FeatureOverlay,
  candles: Candle[],
): { time: Time; value: number }[] {
  const pts = clipOverlayPointsToCandles(overlay.points || [], candles);
  return forwardFillOverlayToCandles(pts, candles)
    .filter((p) => p.value != null && Number.isFinite(Number(p.value)))
    .map((p) => ({ time: p.time as Time, value: Number(p.value) }));
}

interface VolumePaneProps {
  candles: Candle[];
  mainChart: IChartApi | null;
}

export function VolumePane({ candles, mainChart }: VolumePaneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const fittedRef = useRef(false);

  useEffect(() => {
    if (!hostRef.current) return;
    fittedRef.current = false;
    const chart = createChart(hostRef.current, {
      ...subchartBaseOptions,
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
    });
    const series = chart.addSeries(HistogramSeries, { color: CHART_THEME.volume });
    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      if (hostRef.current) {
        chart.applyOptions({
          width: hostRef.current.clientWidth,
          height: hostRef.current.clientHeight,
        });
      }
    });
    ro.observe(hostRef.current);
    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      fittedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart || !candles.length) return;
    const data = candles
      .filter((x) => x.volume != null && Number.isFinite(Number(x.volume)))
      .map((x) => ({ time: x.time as Time, value: Number(x.volume), color: CHART_THEME.volume }));
    series.setData(data);
    if (mainChart && !fittedRef.current) {
      syncSubchartToMain(mainChart, chart, candles.length);
      fittedRef.current = true;
    }
  }, [candles, mainChart]);

  useEffect(() => {
    const main = mainChart;
    const sub = chartRef.current;
    if (!main || !sub || !candles.length) return;
    const sync = () => syncSubchartToMain(main, sub, candles.length);
    const raf = requestAnimationFrame(sync);
    main.timeScale().subscribeVisibleLogicalRangeChange(sync);
    return () => {
      cancelAnimationFrame(raf);
      main.timeScale().unsubscribeVisibleLogicalRangeChange(sync);
    };
  }, [mainChart, candles.length]);

  return (
    <div className={styles.pane}>
      <span className={styles.label}>成交量</span>
      <div ref={hostRef} className={styles.inner} />
    </div>
  );
}

interface FeaturePaneProps {
  column: string;
  overlay: FeatureOverlay;
  candles: Candle[];
  colorIndex: number;
  mainChart: IChartApi | null;
}

export function FeaturePane({ column, overlay, candles, colorIndex, mainChart }: FeaturePaneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const fittedRef = useRef(false);
  const seriesData = featureSeriesData(overlay, candles);
  const hasData = seriesData.length > 0;

  useEffect(() => {
    if (!hostRef.current) return;
    fittedRef.current = false;
    const chart = createChart(hostRef.current, {
      ...subchartBaseOptions,
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
    });
    const color = subchartColor(colorIndex);
    const series = chart.addSeries(
      LineSeries,
      mainChartOverlaySeriesOptions({ color, lineWidth: 2 }),
    );
    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      if (hostRef.current) {
        chart.applyOptions({
          width: hostRef.current.clientWidth,
          height: hostRef.current.clientHeight,
        });
      }
    });
    ro.observe(hostRef.current);
    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      fittedRef.current = false;
    };
  }, [column, colorIndex]);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    series.setData(hasData ? seriesData : []);
    if (hasData && mainChart && !fittedRef.current) {
      syncSubchartToMain(mainChart, chart, candles.length);
      fittedRef.current = true;
    }
  }, [seriesData, hasData, candles.length, mainChart]);

  useEffect(() => {
    const main = mainChart;
    const sub = chartRef.current;
    if (!main || !sub || !candles.length) return;
    const sync = () => syncSubchartToMain(main, sub, candles.length);
    const raf = requestAnimationFrame(sync);
    main.timeScale().subscribeVisibleLogicalRangeChange(sync);
    return () => {
      cancelAnimationFrame(raf);
      main.timeScale().unsubscribeVisibleLogicalRangeChange(sync);
    };
  }, [mainChart, candles.length, column, colorIndex]);

  return (
    <div className={styles.pane}>
      <span className={styles.label}>{column}</span>
      <div ref={hostRef} className={styles.inner} />
      {!hasData ? <div className={styles.emptyOverlay}>特征数据对齐中…</div> : null}
    </div>
  );
}
