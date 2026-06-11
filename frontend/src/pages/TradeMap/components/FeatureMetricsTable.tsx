import { useEffect, useMemo, useState } from 'react';
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
  const rowSpecs = strategyMetricsRowSpecs(sid, columns, overlays);

  const [logicalRange, setLogicalRange] = useState<{ from: number; to: number } | null>(null);

  useEffect(() => {
    if (!mainChart) return;
    const update = () => setLogicalRange(mainChart.timeScale().getVisibleLogicalRange());
    update();
    mainChart.timeScale().subscribeVisibleLogicalRangeChange(update);
    return () => mainChart.timeScale().unsubscribeVisibleLogicalRangeChange(update);
  }, [mainChart]);

  const { bars, regimeExitTimes, regimeOnTimes } = useMemo(() => {
    const { from, to } = visibleCandleIndexRange(candles, logicalRange);
    const barList: Array<{ time: number; label: string }> = [];
    for (let i = from; i <= to; i++) {
      const t = candles[i]?.time;
      if (t == null) continue;
      barList.push({ time: t, label: formatMetricsBarHeader(t) });
    }
    return {
      bars: barList,
      regimeExitTimes:
        sid === 'chop_grid' ? chopRegimeExitBarTimes(candles, overlays) : new Set<number>(),
      regimeOnTimes:
        sid === 'chop_grid' ? chopRegimeHysteresisOnBarTimes(candles, overlays) : new Set<number>(),
    };
  }, [candles, overlays, sid, logicalRange]);

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
      <div className={styles.scroll}>
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
