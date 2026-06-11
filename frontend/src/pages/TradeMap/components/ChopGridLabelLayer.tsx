import { useEffect, useState } from 'react';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';
import type { Candle } from '@/api/types.ts';
import { chopGridLabelAnchor } from '@/lib/tradeMap/markers.ts';
import { labelTimeForSpans } from '@/lib/tradeMap/chartOverlay.ts';
import type { ChopGridLabelSpec } from '@/lib/tradeMap/chartOverlay.ts';
import styles from './ChopGridLabelLayer.module.css';

interface Props {
  chart: IChartApi | null;
  candleSeries: ISeriesApi<'Candlestick'> | null;
  candles: Candle[];
  specs: ChopGridLabelSpec[];
  enabled: boolean;
}

interface LabelPos {
  key: string;
  left: number;
  top: number;
  text: string;
  anchor: 'above' | 'below';
  color?: string;
}

export function ChopGridLabelLayer({ chart, candleSeries, candles, specs, enabled }: Props) {
  const [labels, setLabels] = useState<LabelPos[]>([]);

  useEffect(() => {
    if (!chart || !candleSeries || !enabled || !specs.length || !candles.length) {
      setLabels([]);
      return;
    }

    const layout = () => {
      const ts = chart.timeScale();
      const next: LabelPos[] = [];
      for (const spec of specs) {
        const anchor = chopGridLabelAnchor(String(spec.side || ''), spec.kind);
        const labelTime = labelTimeForSpans(spec.spans, candles);
        if (labelTime == null || !Number.isFinite(labelTime)) continue;
        const x = ts.timeToCoordinate(labelTime as never);
        const y = candleSeries.priceToCoordinate(Number(spec.price));
        if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
        next.push({
          key: `${spec.text}-${spec.price}`,
          left: Math.round(x),
          top: Math.round(y),
          text: spec.text,
          anchor,
          color: spec.color,
        });
      }
      setLabels(next);
    };

    layout();
    const handler = () => layout();
    chart.timeScale().subscribeVisibleLogicalRangeChange(handler);
    return () => chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
  }, [chart, candleSeries, candles, specs, enabled]);

  if (!enabled || !labels.length) return null;

  return (
    <div className={styles.layer} aria-hidden="true">
      {labels.map((l) => (
        <span
          key={l.key}
          className={`${styles.label} ${l.anchor === 'above' ? styles.above : styles.below}`}
          style={{ left: l.left, top: l.top, borderColor: l.color }}
        >
          {l.text}
        </span>
      ))}
    </div>
  );
}
