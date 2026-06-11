import { useMemo } from 'react';
import type { IChartApi } from 'lightweight-charts';
import type { Candle } from '@/api/types.ts';
import {
  orderFeaturePaneItems,
  resolveSubchartColumns,
  strategyMetricsTableActive,
} from '@/lib/tradeMap';
import { MAX_FEATURE_SUBCHARTS } from '@/lib/tradeMap/constants.ts';
import type { FeatureOverlays } from '@/lib/tradeMap/types.ts';
import { useTradeMapStore } from '@/stores/tradeMapStore.ts';
import { FeatureMetricsTable } from './FeatureMetricsTable.tsx';
import { FeaturePane, VolumePane } from './FeatureSubchart.tsx';
import styles from './SubchartStack.module.css';

interface Props {
  mainChart: IChartApi | null;
  candles: Candle[];
  overlays: FeatureOverlays;
  onBarClick: (timeSec: number) => void;
}

export function SubchartStack({ mainChart, candles, overlays, onBarClick }: Props) {
  const layers = useTradeMapStore((s) => s.layers);
  const selected = useTradeMapStore((s) => s.selectedFeatureColumns);
  const available = useTradeMapStore((s) => s.availableFeatureColumns);
  const focus = useTradeMapStore((s) => s.featureStrategyFocus);
  const paneVolume = useTradeMapStore((s) => s.paneVolume);
  const highlightBarTime = useTradeMapStore((s) => s.highlightBarTime);

  const colsForPanes = resolveSubchartColumns(
    selected,
    available,
    layers,
    focus,
    MAX_FEATURE_SUBCHARTS,
  );
  const tableFirst = strategyMetricsTableActive(focus, colsForPanes);
  const panePlan = orderFeaturePaneItems(colsForPanes, layers, focus);

  const metricsItem = useMemo(
    () => panePlan.find((item) => item.type === 'metrics_table'),
    [panePlan],
  );

  if (!paneVolume && !colsForPanes.length) return null;

  return (
    <div className={styles.stack}>
      {paneVolume ? <VolumePane candles={candles} mainChart={mainChart} /> : null}
      {tableFirst && metricsItem ? (
        <FeatureMetricsTable
          strategyId={metricsItem.strategy || focus || 'chop_grid'}
          columns={metricsItem.columns || colsForPanes}
          candles={candles}
          overlays={overlays}
          highlightTime={highlightBarTime}
          mainChart={mainChart}
          onBarClick={onBarClick}
        />
      ) : (
        panePlan.map((item, idx) => {
          if (item.type === 'header') {
            return (
              <div key={`${item.title}-${item.strategy}`} className={styles.header}>
                {item.title}
              </div>
            );
          }
          if (item.type !== 'feature' || !item.column) return null;
          const overlay = overlays[item.column] || {
            available: false,
            points: [],
            reference_lines: [],
          };
          return (
            <FeaturePane
              key={item.column}
              column={item.column}
              overlay={overlay}
              candles={candles}
              colorIndex={idx}
              mainChart={mainChart}
            />
          );
        })
      )}
    </div>
  );
}
