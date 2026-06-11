import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { useEffect, useMemo } from 'react';
import { apiGet } from '@/api/client.ts';
import type { SymbolRow } from '@/api/types.ts';
import { useLightweightChart } from '@/hooks/useLightweightChart.ts';
import { useTradeMapBundle } from '@/hooks/useTradeMapBundle.ts';
import { getSymbol, setSymbol, SCOPE_LABELS } from '@/lib/shell.ts';
import { scopesFromLayers, useTradeMapStore } from '@/stores/tradeMapStore.ts';
import styles from './TradeMapPage.module.css';

export function TradeMapPage() {
  const [searchParams] = useSearchParams();
  const {
    symbol,
    timeframe,
    layers,
    markers,
    lastCandles,
    selectedMarkerId,
    featureStrategyFocus,
    statusText,
    loading,
    mainEma1200,
    mainWeeklyEma200,
    setSymbol: setStoreSymbol,
    setTimeframe,
    setLayers,
    setSelectedMarkerId,
    setBundlePhase,
  } = useTradeMapStore();

  const { refreshFull, initFromLayout } = useTradeMapBundle();
  const chartRef = useLightweightChart(
    lastCandles,
    markers,
    selectedMarkerId,
    featureStrategyFocus,
  );

  const symbolsQuery = useQuery({
    queryKey: ['symbols'],
    queryFn: () => apiGet<SymbolRow[]>('/api/trade-map/symbols'),
  });

  useEffect(() => {
    initFromLayout();
    const symParam = searchParams.get('symbol');
    const saved = getSymbol();
    if (symParam) setStoreSymbol(symParam);
    else if (saved) setStoreSymbol(saved);
    const markerId = searchParams.get('marker_id');
    if (markerId) setSelectedMarkerId(markerId);
  }, [initFromLayout, searchParams, setStoreSymbol, setSelectedMarkerId]);

  useEffect(() => {
    refreshFull().catch(() => {});
    const t = window.setInterval(() => refreshFull().catch(() => {}), 10_000);
    return () => window.clearInterval(t);
  }, [refreshFull, symbol, timeframe, layers, mainEma1200, mainWeeklyEma200]);

  const symbolOptions = symbolsQuery.data?.data?.length
    ? symbolsQuery.data.data
    : [{ symbol: 'ETHUSDT' }];

  const scopeLabel = useMemo(() => scopesFromLayers(layers), [layers]);

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <label>
          Symbol
          <select
            value={symbol}
            onChange={(e) => {
              setStoreSymbol(e.target.value);
              setSymbol(e.target.value);
              setBundlePhase({ chartFitPending: true, ohlcvLoadedFrom: null });
            }}
          >
            {symbolOptions.map((row) => (
              <option key={row.symbol} value={row.symbol}>
                {row.symbol}
              </option>
            ))}
          </select>
        </label>
        <label>
          TF
          <select
            value={timeframe}
            onChange={(e) => {
              setTimeframe(e.target.value);
              setBundlePhase({ chartFitPending: true, ohlcvLoadedFrom: null });
            }}
          >
            <option value="15min">15min</option>
            <option value="2h">2h</option>
            <option value="1d">1d</option>
            <option value="1w">1w</option>
          </select>
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.trend}
            onChange={(e) => setLayers({ trend: e.target.checked })}
          />
          {SCOPE_LABELS.trend}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.spot}
            onChange={(e) => setLayers({ spot: e.target.checked })}
          />
          {SCOPE_LABELS.spot}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.multiLeg}
            onChange={(e) => setLayers({ multiLeg: e.target.checked })}
          />
          {SCOPE_LABELS.multi_leg}
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={layers.pending}
            onChange={(e) => setLayers({ pending: e.target.checked })}
          />
          Pending
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={mainEma1200}
            onChange={(e) => setBundlePhase({ mainEma1200: e.target.checked })}
          />
          EMA1200
        </label>
        <label className={styles.chk}>
          <input
            type="checkbox"
            checked={mainWeeklyEma200}
            onChange={(e) => setBundlePhase({ mainWeeklyEma200: e.target.checked })}
          />
          W-EMA200
        </label>
        <button type="button" onClick={() => refreshFull().catch(() => {})}>
          刷新
        </button>
        <Link to={`/orders?symbol=${encodeURIComponent(symbol)}`}>订单</Link>
        <span className={styles.status}>{loading ? '加载中…' : statusText || scopeLabel}</span>
      </div>
      <div ref={chartRef} className={styles.chart} />
      <div className={styles.markerList}>
        {markers.slice(0, 40).map((m) => (
          <button
            key={m.id}
            type="button"
            className={
              m.id === selectedMarkerId ? `${styles.markerBtn} ${styles.markerSelected}` : styles.markerBtn
            }
            onClick={() => setSelectedMarkerId(m.id)}
          >
            {m.strategy}:{m.event} @{m.time}
          </button>
        ))}
      </div>
    </div>
  );
}
