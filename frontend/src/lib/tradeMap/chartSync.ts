import type { IChartApi } from 'lightweight-charts';
import { isValidLogicalRange } from './candles.ts';

/** Keep subchart time axis aligned with main chart (same bar indices). */
export function syncSubchartToMain(
  main: IChartApi,
  sub: IChartApi,
  barCount: number,
): void {
  if (barCount <= 0) return;
  const lr = main.timeScale().getVisibleLogicalRange();
  if (!isValidLogicalRange(lr, barCount)) return;
  try {
    sub.timeScale().setVisibleLogicalRange(lr);
  } catch {
    /* chart may be mid-resize */
  }
}
