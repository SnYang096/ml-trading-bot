import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { useMemo, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import type { SignalRow } from '@/api/types.ts';

function fmtBarTime(meta: { timestamp?: string } | undefined): string {
  if (!meta?.timestamp) return '—';
  const s = String(meta.timestamp);
  return s.length >= 16 ? s.slice(0, 16).replace('T', ' ') : s;
}

type StrategyBlock = NonNullable<SignalRow['strategies']>[string];

function StrategyLines({ block }: { block?: StrategyBlock }) {
  const by = block?.by_strategy || {};
  const keys = Object.keys(by).sort();
  if (!keys.length) {
    return <span className="muted">{block?.summary || '—'}</span>;
  }
  return (
    <>
      {keys.map((sid) => {
        const row = by[sid] || {};
        return (
          <div key={sid} className="strategy-line">
            <strong>{sid}</strong> {row.summary || '—'}
            {row.funnel_summary ? (
              <span className="muted"> · {row.funnel_summary}</span>
            ) : null}
          </div>
        );
      })}
    </>
  );
}

function LastStrategyLines({ block }: { block?: StrategyBlock }) {
  const by = block?.by_strategy || {};
  const keys = Object.keys(by).sort();
  if (!keys.length) return <>{block?.last_summary || '—'}</>;
  const nodes = keys
    .map((sid) => {
      const last = by[sid]?.last_summary;
      if (!last || last === '—') return null;
      return (
        <div key={sid}>
          <strong>{sid}</strong> {last}
        </div>
      );
    })
    .filter(Boolean);
  return nodes.length ? <>{nodes}</> : <>{block?.last_summary || '—'}</>;
}

export function SignalsPage() {
  const pageVisible = usePageVisible();
  const [timeframe, setTimeframe] = useState('2h');
  const [lookback, setLookback] = useState('7');

  const { data, isFetching, error, refetch, dataUpdatedAt } = useQuery({
    queryKey: ['signals', timeframe, lookback],
    queryFn: () =>
      apiGet<SignalRow[]>('/api/trade-map/signals', {
        timeframe,
        lookback_days: lookback,
      }),
    refetchInterval: visibleRefetchInterval(pageVisible, 20_000),
  });

  const rows = data?.data || [];
  const meta = data?.meta || {};

  const status = useMemo(() => {
    if (error) return String(error);
    if (isFetching && !rows.length) return '加载中…';
    return `${meta.count ?? rows.length} symbols · ${timeframe} · ${lookback}d · ${new Date(dataUpdatedAt || Date.now()).toLocaleTimeString()}`;
  }, [error, isFetching, rows.length, meta.count, timeframe, lookback, dataUpdatedAt]);

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>策略信号</h2>
        <label>
          TF
          <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            <option value="15min">15min</option>
            <option value="2h">2h</option>
            <option value="1d">1d</option>
          </select>
        </label>
        <label>
          Lookback
          <select value={lookback} onChange={(e) => setLookback(e.target.value)}>
            <option value="7">7d</option>
            <option value="14">14d</option>
            <option value="30">30d</option>
          </select>
        </label>
        <button type="button" onClick={() => refetch()}>
          刷新
        </button>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Latest</th>
            <th>1m rows</th>
            <th>Map</th>
            <th>Trend</th>
            <th>Trend last</th>
            <th>Spot</th>
            <th>Multi-leg</th>
            <th>ML last</th>
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((r) => (
              <tr key={r.symbol}>
                <td>
                  <strong>{r.symbol}</strong>
                </td>
                <td>{fmtBarTime(r.latest_bar)}</td>
                <td>{r.bars_1min_rows ?? '—'}</td>
                <td>
                  <Link to={r.map_href || `/trade-map?symbol=${r.symbol}`}>地图</Link>
                </td>
                <td>
                  <StrategyLines block={r.strategies?.trend} />
                </td>
                <td className="muted">
                  <LastStrategyLines block={r.strategies?.trend} />
                </td>
                <td>
                  <StrategyLines block={r.strategies?.spot} />
                </td>
                <td>
                  <StrategyLines block={r.strategies?.multi_leg} />
                </td>
                <td className="muted">
                  <LastStrategyLines block={r.strategies?.multi_leg} />
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={9} className="muted">
                无数据
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="status-line">{status}</p>
    </div>
  );
}