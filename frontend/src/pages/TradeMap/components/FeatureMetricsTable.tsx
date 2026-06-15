import { useMemo } from 'react';
import {
  chopRegimeExitBarTimes,
  chopRegimeHysteresisOnBarTimes,
  strategyBarGateEvaluator,
  strategyFocusLabel,
  strategyMetricsRowCell,
  strategyMetricsRowSpecs,
} from '@/lib/tradeMap';
import { formatMetricsBarHeader } from '@/lib/tradeMap/chartOverlay.ts';
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
  mainChart: _mainChart,
  onBarClick: _onBarClick,
}: Props) {
  const sid = String(strategyId || 'chop_grid').toLowerCase();
  const rowSpecs = useMemo(
    () => strategyMetricsRowSpecs(sid, columns, overlays),
    [sid, columns, overlays],
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

  const activeTime = highlightTime;

  // gate for the highlighted bar
  const gateEval = useMemo(() => {
    if (activeTime == null) return null;
    return strategyBarGateEvaluator(sid, columns, overlays, activeTime, candles);
  }, [sid, columns, overlays, activeTime, candles]);

  if (!rowSpecs.length) {
    return (
      <p className="muted">
        无指标列（请选择「{strategyFocusLabel(sid) || sid}」预设或勾选该策略特征列）
      </p>
    );
  }

  if (!candles.length) {
    return <p className="muted">无 K 线数据</p>;
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.caption}>
        {strategyFocusLabel(sid) || sid} · 指标矩阵
        {activeTime != null ? (
          <span className={styles.barTime}>@{formatMetricsBarHeader(activeTime)}</span>
        ) : null}
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.rowLabel}>可入场</th>
              <td className={styles.valCol}>
                {activeTime == null ? (
                  <span className={styles.mutedHint}>悬停图表查看</span>
                ) : gateEval === true ? (
                  <span className={styles.enterOk}>✓</span>
                ) : gateEval === false ? (
                  <span className={styles.enterFail}>×</span>
                ) : (
                  '·'
                )}
              </td>
            </tr>
            {sid === 'chop_grid' ? (
              <>
                <tr>
                  <th className={styles.rowLabel}>regime滞回</th>
                  <td className={styles.valCol}>
                    {activeTime != null && regimeOnTimes.has(Number(activeTime)) ? (
                      <span className={styles.regimeOn}>ON</span>
                    ) : activeTime != null ? (
                      'OFF'
                    ) : (
                      <span className={styles.mutedHint}>—</span>
                    )}
                  </td>
                </tr>
                <tr>
                  <th className={styles.rowLabel}>regime退出</th>
                  <td className={styles.valCol}>
                    {activeTime != null && regimeExitTimes.has(Number(activeTime)) ? (
                      <span className={styles.regimeExit}>退出</span>
                    ) : activeTime != null ? (
                      '·'
                    ) : (
                      <span className={styles.mutedHint}>—</span>
                    )}
                  </td>
                </tr>
              </>
            ) : null}
          </thead>
          <tbody>
            {rowSpecs.map((row) => {
              const cell =
                activeTime != null
                  ? strategyMetricsRowCell(sid, row, overlays, activeTime, candles)
                  : null;
              const cls =
                cell?.pass === true
                  ? styles.passOk
                  : cell?.pass === false
                    ? styles.passFail
                    : '';
              return (
                <tr key={row.column || row.label}>
                  <th className={styles.rowLabel}>
                    <div>{row.label}</div>
                    {row.threshold ? (
                      <div className={styles.thresh}>{row.threshold}</div>
                    ) : null}
                  </th>
                  <td className={`${styles.valCol} ${cls}`}>
                    {cell ? cell.value : <span className={styles.mutedHint}>—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
