import { useEffect, useRef } from 'react';
import {
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type Time,
} from 'lightweight-charts';
import type { Candle } from '@/api/types.ts';
import { CHART_THEME } from '@/lib/tradeMap/constants.ts';
import { alignSeriesToCandleTimes, clipOverlayPointsToCandles, subchartColor } from '@/lib/tradeMap';
import { mainChartOverlaySeriesOptions, subchartBaseOptions } from '@/lib/tradeMap/chartOverlay.ts';
import type { FeatureOverlay } from '@/lib/tradeMap/types.ts';
import styles from './SubchartStack.module.css';

interface VolumePaneProps {
  candles: Candle[];
  mainChart: IChartApi | null;
}

export function VolumePane({ candles, mainChart }: VolumePaneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!hostRef.current) return;
    const chart = createChart(hostRef.current, {
      ...subchartBaseOptions,
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
    });
    const series = chart.addSeries(HistogramSeries, { color: CHART_THEME.volume });
    chartRef.current = chart;
    const data = candles
      .filter((x) => x.volume != null && Number.isFinite(Number(x.volume)))
      .map((x) => ({ time: x.time as Time, value: Number(x.volume), color: CHART_THEME.volume }));
    series.setData(data);

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
    };
  }, [candles]);

  useEffect(() => {
    const main = mainChart;
    const sub = chartRef.current;
    if (!main || !sub) return;
    const sync = () => {
      const range = main.timeScale().getVisibleRange?.();
      if (range?.from != null && range?.to != null) {
        try {
          sub.timeScale().setVisibleRange(range);
        } catch {
          /* */
        }
      }
    };
    sync();
    main.timeScale().subscribeVisibleLogicalRangeChange(sync);
    return () => main.timeScale().unsubscribeVisibleLogicalRangeChange(sync);
  }, [mainChart]);

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

  useEffect(() => {
    if (!hostRef.current) return;
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
    const pts = clipOverlayPointsToCandles(overlay.points || [], candles);
    series.setData(alignSeriesToCandleTimes(pts, candles) as { time: Time; value: number }[]);

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
    };
  }, [column, overlay, candles, colorIndex]);

  useEffect(() => {
    const main = mainChart;
    const sub = chartRef.current;
    if (!main || !sub) return;
    const sync = () => {
      const range = main.timeScale().getVisibleRange?.();
      if (range?.from != null && range?.to != null) {
        try {
          sub.timeScale().setVisibleRange(range);
        } catch {
          /* */
        }
      }
    };
    sync();
    main.timeScale().subscribeVisibleLogicalRangeChange(sync);
    return () => main.timeScale().unsubscribeVisibleLogicalRangeChange(sync);
  }, [mainChart]);

  return (
    <div className={styles.pane}>
      <span className={styles.label}>{column}</span>
      <div ref={hostRef} className={styles.inner} />
    </div>
  );
}
