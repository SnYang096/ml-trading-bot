import { useEffect, useMemo, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import {
  chopRegimeExitBarTimes,
  chopRegimeHysteresisOnBarTimes,
  strategyBarGateEvaluator,
  strategyFocusLabel,
  strategyMetricsRowCell,
  strategyMetricsRowSpecs,
} from '@/lib/tradeMap';
import { formatMetricsBarHeader, visibleCandleIndexRange } from '@/lib/tradeMap/chartOverlay.ts';
import type { Candle } from '@/api/types.ts';
import type { FeatureOverlays } from '@/lib/tradeMap/types.ts';
import type { IChartApi } from 'lightweight-charts';
import styles from './FeatureMetricsTable.module.css';

interface Props {
  strategyId: string;
  columns: string[];
  candles: Candle[];
  overlays: FeatureOverlays;
  highlightTime: number | null;
  mainChart: IChartApi | null;
  onBarClick: (timeSec: number) => void;
}

function logicalRangeEqual(
  a: { from: number; to: number } | null,
  b: { from: number; to: number } | null,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.from === b.from && a.to === b.to;
}

function syncVisibleLogicalRange(
  mainChart: IChartApi,
  pendingRef: MutableRefObject<number | null>,
  setLogicalRange: Dispatch<SetStateAction<{ from: number; to: number } | null>>,
) {
  if (pendingRef.current != null) return;
  pendingRef.current = window.requestAnimationFrame(() => {
    pendingRef.current = null;
    const lr = mainChart.timeScale().getVisibleLogicalRange();
    setLogicalRange((prev) => (logicalRangeEqual(prev, lr) ? prev : lr));
  });
}

export function FeatureMetricsTable({
  strategyId,
  columns,
  candles,
  overlays,
  highlightTime,
  mainChart,
  onBarClick,
}: Props) {
  const sid = String(strategyId || 'chop_grid').toLowerCase();
  const rowSpecs = useMemo(
    () => strategyMetricsRowSpecs(sid, columns, overlays),
    [sid, columns, overlays],
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const rangeSyncRef = useRef<number | null>(null);

  const [logicalRange, setLogicalRange] = useState<{ from: number; to: number } | null>(null);

  useEffect(() => {
    if (!mainChart) return;
    const update = () => syncVisibleLogicalRange(mainChart, rangeSyncRef, setLogicalRange);
    update();
    mainChart.timeScale().subscribeVisibleLogicalRangeChange(update);
    return () => {
      if (rangeSyncRef.current != null) {
        cancelAnimationFrame(rangeSyncRef.current);
        rangeSyncRef.current = null;
      }
      mainChart.timeScale().unsubscribeVisibleLogicalRangeChange(update);
    };
  }, [mainChart, candles.length]);

  const indexRange = useMemo(
    () => visibleCandleIndexRange(candles, logicalRange),
    [candles, logicalRange],
  );

  const regimeExitTimes = useMemo(
    () => (sid === 'chop_grid' ? chopRegimeExitBarTimes(candles, overlays) : new Set<number>()),
    [candles, overlays, sid],
  );
  const regimeOnTimes = useMemo(
    () =>
      sid === 'chop_grid' ? chopRegimeHysteresisOnBarTimes(candles, overlays) : new Set<number>(),
    [candles, overlays, sid],
  );

  const bars = useMemo(() => {
    const { from, to } = indexRange;
    const barList: Array<{ time: number; label: string }> = [];
    for (let i = from; i <= to; i++) {
      const t = candles[i]?.time;
      if (t == null) continue;
      barList.push({ time: t, label: formatMetricsBarHeader(t) });
    }
    return barList;
  }, [candles, indexRange]);

  const atTail =
    candles.length > 0 && indexRange.to >= Math.max(0, candles.length - 2);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !bars.length) return;
    if (atTail) {
      el.scrollLeft = el.scrollWidth - el.clientWidth;
    }
  }, [bars, atTail, indexRange.from, indexRange.to]);

  if (!rowSpecs.length) {
    return (
      <p className="muted">
        无指标列（请选择「{strategyFocusLabel(sid) || sid}」预设或勾选该策略特征列）
      </p>
    );
  }
  if (!bars.length) {
    return <p className="muted">主图可见区间无 K 线</p>;
  }

  const activeTime = highlightTime;

  return (
    <div className={styles.wrap}>
      <div className={styles.caption}>{strategyFocusLabel(sid) || sid} · 指标矩阵</div>
      <div ref={scrollRef} className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.rowLabel}>可入场</th>
              {bars.map((b) => {
                const canEnter = strategyBarGateEvaluator(sid, columns, overlays, b.time, candles);
                const gate = canEnter === true ? '✓' : canEnter === false ? '×' : '·';
                const active = activeTime != null && Number(activeTime) === Number(b.time);
                return (
                  <th
                    key={`gate-${b.time}`}
                    className={`${styles.barCol} ${canEnter === true ? styles.enterOk : canEnter === false ? styles.enterFail : ''} ${active ? styles.active : ''}`}
                    onClick={() => onBarClick(b.time)}
                  >
                    {gate}
                  </th>
                );
              })}
            </tr>
            {sid === 'chop_grid' ? (
              <>
                <tr>
                  <th className={styles.rowLabel}>regime滞回</th>
                  {bars.map((b) => {
                    const on = regimeOnTimes.has(Number(b.time));
                    const active = activeTime != null && Number(activeTime) === Number(b.time);
                    return (
                      <th
                        key={`on-${b.time}`}
                        className={`${styles.barCol} ${on ? styles.regimeOn : ''} ${active ? styles.active : ''}`}
                        onClick={() => onBarClick(b.time)}
                      >
                        {on ? 'ON' : 'OFF'}
                      </th>
                    );
                  })}
                </tr>
                <tr>
                  <th className={styles.rowLabel}>regime退出</th>
                  {bars.map((b) => {
                    const isExit = regimeExitTimes.has(Number(b.time));
                    const active = activeTime != null && Number(activeTime) === Number(b.time);
                    return (
                      <th
                        key={`exit-${b.time}`}
                        className={`${styles.barCol} ${isExit ? styles.regimeExit : ''} ${active ? styles.active : ''}`}
                        onClick={() => onBarClick(b.time)}
                      >
                        {isExit ? '退出' : '·'}
                      </th>
                    );
                  })}
                </tr>
              </>
            ) : null}
            <tr>
              <th className={styles.rowLabel}>时间</th>
              {bars.map((b) => {
                const active = activeTime != null && Number(activeTime) === Number(b.time);
                return (
                  <th
                    key={`time-${b.time}`}
                    className={`${styles.barCol} ${active ? styles.active : ''}`}
                    onClick={() => onBarClick(b.time)}
                  >
                    {b.label}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rowSpecs.map((row) => (
              <tr key={row.column || row.label}>
                <th className={styles.rowLabel}>
                  <div>{row.label}</div>
                  <div className={styles.thresh}>{row.threshold || ''}</div>
                </th>
                {bars.map((b) => {
                  const cell = strategyMetricsRowCell(sid, row, overlays, b.time, candles);
                  const active = activeTime != null && Number(activeTime) === Number(b.time);
                  const cls =
                    cell.pass === true
                      ? styles.passOk
                      : cell.pass === false
                        ? styles.passFail
                        : '';
                  return (
                    <td
                      key={`${row.column}-${b.time}`}
                      className={`${styles.barCol} ${cls} ${active ? styles.active : ''}`}
                      onClick={() => onBarClick(b.time)}
                    >
                      {cell.value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
