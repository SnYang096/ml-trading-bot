import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { apiGet } from '@/api/client.ts';
import {
  filterSelectedFeaturesByLayers,
  inferStrategyFocusFromLayers,
  presetColumnsForStrategy,
  setFeatureTaxonomy,
} from '@/lib/tradeMap';
import { MAX_FEATURE_SUBCHARTS } from '@/lib/tradeMap/constants.ts';
import { useTradeMapStore } from '@/stores/tradeMapStore.ts';

interface FeatureColumnsResponse {
  columns?: string[];
  defaults?: string[];
  taxonomy?: unknown;
}

export function useTradeMapFeatureCatalog(opts?: { catalogEnabled?: boolean }) {
  const symbol = useTradeMapStore((s) => s.symbol);
  const timeframe = useTradeMapStore((s) => s.timeframe);
  const layers = useTradeMapStore((s) => s.layers);
  const selected = useTradeMapStore((s) => s.selectedFeatureColumns);
  const featureDrawerOpen = useTradeMapStore((s) => s.featureDrawerOpen);
  const setAvailable = useTradeMapStore((s) => s.setAvailableFeatureColumns);
  const setSelected = useTradeMapStore((s) => s.setSelectedFeatureColumns);
  const setFocus = useTradeMapStore((s) => s.setFeatureStrategyFocus);

  const catalogEnabled =
    opts?.catalogEnabled ?? (featureDrawerOpen || selected.length > 0);

  const query = useQuery({
    queryKey: ['feature-columns', symbol, timeframe],
    queryFn: () =>
      apiGet<FeatureColumnsResponse>(
        `/api/bus/features/columns?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`,
      ),
    enabled: catalogEnabled,
  });

  useEffect(() => {
    const data = query.data?.data;
    if (!data) return;
    const cols = data.columns || [];
    setAvailable(cols);
    if (data.taxonomy) setFeatureTaxonomy(data.taxonomy as never);

    const focus = useTradeMapStore.getState().featureStrategyFocus?.trim() || '';
    if (focus) {
      const picks = presetColumnsForStrategy(focus, cols, MAX_FEATURE_SUBCHARTS);
      if (picks.length) {
        const cur = useTradeMapStore.getState().selectedFeatureColumns;
        if (
          picks.length !== cur.length ||
          picks.some((c, i) => c !== cur[i])
        ) {
          setSelected(picks);
        }
        return;
      }
    }

    let next = selected.filter((c) => cols.includes(c));
    if (!next.length && data.defaults?.length) next = [...data.defaults];
    if (!next.length && cols.length) next = [cols[0]];
    if (next.length !== selected.length || next.some((c, i) => c !== selected[i])) {
      setSelected(next);
    }
  }, [query.data, selected, setAvailable, setSelected]);

  const applyStrategyFocus = (strategyId: string | null) => {
    const sid = strategyId?.trim() || '';
    setFocus(sid);
    if (sid) {
      const avail = useTradeMapStore.getState().availableFeatureColumns;
      const picks = presetColumnsForStrategy(sid, avail, MAX_FEATURE_SUBCHARTS);
      if (picks.length) setSelected(picks);
    }
  };

  const applyLayerDefaults = () => {
    const avail = useTradeMapStore.getState().availableFeatureColumns;
    const filtered = filterSelectedFeaturesByLayers(selected, layers);
    if (filtered.length !== selected.length) setSelected(filtered);
    const focus = useTradeMapStore.getState().featureStrategyFocus;
    if (!focus) {
      const inferred = inferStrategyFocusFromLayers(layers);
      if (inferred) applyStrategyFocus(inferred);
    }
    if (!useTradeMapStore.getState().selectedFeatureColumns.length && avail.length) {
      setSelected([avail[0]]);
    }
  };

  return { query, applyStrategyFocus, applyLayerDefaults };
}
