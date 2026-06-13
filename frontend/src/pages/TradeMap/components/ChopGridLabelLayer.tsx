import { useEffect, useRef, useState } from 'react';
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

function labelsEqual(a: LabelPos[], b: LabelPos[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (
      x.key !== y.key ||
      x.left !== y.left ||
      x.top !== y.top ||
      x.text !== y.text ||
      x.anchor !== y.anchor ||
      x.color !== y.color
    ) {
      return false;
    }
  }
  return true;
}

function computeLabelPositions(
  chart: IChartApi,
  candleSeries: ISeriesApi<'Candlestick'>,
  candles: Candle[],
  specs: ChopGridLabelSpec[],
): LabelPos[] {
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
  return next;
}

export function ChopGridLabelLayer({ chart, candleSeries, candles, specs, enabled }: Props) {
  const [labels, setLabels] = useState<LabelPos[]>([]);
  const layoutRafRef = useRef<number | null>(null);

  useEffect(() => {
    const clearLabels = () => {
      setLabels((prev) => (prev.length === 0 ? prev : []));
    };

    if (!chart || !candleSeries || !enabled || !specs.length || !candles.length) {
      clearLabels();
      return;
    }

    const scheduleLayout = () => {
      if (layoutRafRef.current != null) return;
      layoutRafRef.current = window.requestAnimationFrame(() => {
        layoutRafRef.current = null;
        const next = computeLabelPositions(chart, candleSeries, candles, specs);
        setLabels((prev) => (labelsEqual(prev, next) ? prev : next));
      });
    };

    scheduleLayout();
    const handler = () => scheduleLayout();
    chart.timeScale().subscribeVisibleLogicalRangeChange(handler);
    return () => {
      if (layoutRafRef.current != null) {
        window.cancelAnimationFrame(layoutRafRef.current);
        layoutRafRef.current = null;
      }
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
    };
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
