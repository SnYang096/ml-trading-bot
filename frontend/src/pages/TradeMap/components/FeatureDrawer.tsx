import { useMemo } from 'react';
import {
  filterColumnsForFeaturePicker,
  filterFeatureColumns,
  groupFeatureColumnsByStrategy,
  listStrategiesForLayers,
  lookupFeatureMeta,
  presetColumnsForStrategy,
} from '@/lib/tradeMap';
import { MAX_FEATURE_SUBCHARTS } from '@/lib/tradeMap/constants.ts';
import { useTradeMapStore } from '@/stores/tradeMapStore.ts';
import styles from './FeatureDrawer.module.css';

const PRESETS = [
  { id: 'spot', label: 'Spot' },
  { id: 'tpc', label: 'TPC' },
  { id: 'bpc', label: 'BPC' },
  { id: 'chop_grid', label: 'Chop' },
  { id: 'trend_scalp', label: 'Scalp' },
] as const;

interface Props {
  onClose: () => void;
}

export function FeatureDrawer({ onClose }: Props) {
  const layers = useTradeMapStore((s) => s.layers);
  const available = useTradeMapStore((s) => s.availableFeatureColumns);
  const selected = useTradeMapStore((s) => s.selectedFeatureColumns);
  const focus = useTradeMapStore((s) => s.featureStrategyFocus);
  const search = useTradeMapStore((s) => s.featureSearchQuery);
  const setSearch = useTradeMapStore((s) => s.setFeatureSearchQuery);
  const setSelected = useTradeMapStore((s) => s.setSelectedFeatureColumns);
  const setFocus = useTradeMapStore((s) => s.setFeatureStrategyFocus);

  const strategies = listStrategiesForLayers(layers);
  const pool = filterColumnsForFeaturePicker(available, layers, focus);
  const filtered = filterFeatureColumns(pool, search);
  const groups = useMemo(
    () => groupFeatureColumnsByStrategy(filtered, layers),
    [filtered, layers],
  );

  const toggleColumn = (col: string) => {
    const has = selected.includes(col);
    if (has) setSelected(selected.filter((c) => c !== col));
    else if (selected.length < MAX_FEATURE_SUBCHARTS) setSelected([...selected, col]);
  };

  const applyPreset = (strategyId: string) => {
    const picks = presetColumnsForStrategy(strategyId, available, MAX_FEATURE_SUBCHARTS);
    setSelected(picks);
    setFocus(strategyId);
  };

  return (
    <>
      <div className={styles.backdrop} onClick={onClose} aria-hidden="true" />
      <aside className={styles.drawer} role="dialog" aria-label="特征列选择">
        <div className={styles.head}>
          <h3>特征附图</h3>
          <button type="button" className={styles.close} onClick={onClose} aria-label="关闭">
            ×
          </button>
        </div>
        <div className={styles.controls}>
          <input
            type="search"
            className={styles.search}
            placeholder="搜索特征列…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <label>
            策略筛选
            <select value={focus} onChange={(e) => setFocus(e.target.value)}>
              <option value="">全部</option>
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title || s.id}
                </option>
              ))}
            </select>
          </label>
          <div className={styles.presets}>
            {PRESETS.map((p) => (
              <button key={p.id} type="button" onClick={() => applyPreset(p.id)}>
                {p.label}
              </button>
            ))}
            <button type="button" onClick={() => setSelected([])}>
              清空
            </button>
          </div>
        </div>
        <div className={styles.chips}>
          {selected.map((col) => (
            <button key={col} type="button" className={styles.chip} onClick={() => toggleColumn(col)}>
              {col} ×
            </button>
          ))}
          {!selected.length ? <span className="muted">未选特征列</span> : null}
        </div>
        <div className={styles.list}>
          {groups.map(([group, cols]) => (
            <section key={group}>
              <h4>{group}</h4>
              {cols.map((col: string) => {
                const meta = lookupFeatureMeta(col);
                const on = selected.includes(col);
                return (
                  <label key={col} className={`${styles.row} ${on ? styles.rowOn : ''}`}>
                    <input type="checkbox" checked={on} onChange={() => toggleColumn(col)} />
                    <span>{col}</span>
                    {meta.stage_title ? (
                      <span className={styles.hint}>{meta.stage_title}</span>
                    ) : null}
                  </label>
                );
              })}
            </section>
          ))}
          {!groups.length ? <p className="muted">无匹配列</p> : null}
        </div>
      </aside>
    </>
  );
}
