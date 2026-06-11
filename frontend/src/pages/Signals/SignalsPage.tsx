import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { useMemo, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import { usePageVisible, visibleRefetchInterval } from '@/hooks/usePageVisible.ts';
import type { FunnelSnapshot, SignalRow } from '@/api/types.ts';
import { SCOPE_LABELS } from '@/lib/shell.ts';
import styles from './SignalsPage.module.css';

function fmtBarTime(meta: { timestamp?: string } | undefined): string {
  if (!meta?.timestamp) return '—';
  const s = String(meta.timestamp);
  return s.length >= 16 ? s.slice(0, 16).replace('T', ' ') : s;
}

type StrategyBlock = NonNullable<SignalRow['strategies']>[string];

function strategyAccountLayer(strategyId: string): string {
  const sid = String(strategyId || '').toLowerCase();
  if (sid.includes('spot')) return 'spot';
  if (sid === 'chop_grid' || sid === 'trend_scalp') return 'multi_leg';
  return 'trend';
}

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
          <div key={sid} className={styles.strategyLine}>
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

function spotBlockersTitle(block?: StrategyBlock): string {
  const by = block?.by_strategy || {};
  return Object.values(by)
    .flatMap((x) => x.blockers || [])
    .filter(Boolean)
    .join('; ');
}

interface FunnelFlatRow {
  timestamp: string;
  symbol: string;
  strategy: string;
  account_layer: string;
  regime_passed: number;
  regime_denied: number;
  prefilter_passed: number;
  prefilter_denied: number;
  direction: number;
  gate_passed: number;
}

function flattenFunnelRows(rows: FunnelSnapshot[]): FunnelFlatRow[] {
  const flat: FunnelFlatRow[] = [];
  for (const snap of rows || []) {
    const bys = snap.by_strategy || {};
    for (const [strat, st] of Object.entries(bys)) {
      if (!st || typeof st !== 'object') continue;
      flat.push({
        timestamp: String(snap.timestamp || ''),
        symbol: String(snap.symbol || ''),
        strategy: strat,
        account_layer: strategyAccountLayer(strat),
        regime_passed: Number(st.regime_passed) || 0,
        regime_denied: Number(st.regime_denied) || 0,
        prefilter_passed: Number(st.prefilter_passed) || 0,
        prefilter_denied: Number(st.prefilter_denied) || 0,
        direction: Number(st.direction) || 0,
        gate_passed: Number(st.gate_passed) || 0,
      });
    }
  }
  return flat;
}

function FunnelPanel() {
  const [open, setOpen] = useState(false);
  const [layer, setLayer] = useState('');
  const [symbol, setSymbol] = useState('');
  const [strategy, setStrategy] = useState('');

  const funnelQuery = useQuery({
    queryKey: ['funnel', layer, symbol, strategy],
    queryFn: () =>
      apiGet<FunnelSnapshot[]>('/api/trend/funnel', {
        limit: '48',
        ...(symbol ? { symbol } : {}),
        ...(layer ? { account_layer: layer } : {}),
        ...(strategy ? { strategy } : {}),
      }),
    enabled: open,
  });

  const allRows = funnelQuery.data?.data || [];
  const flat = useMemo(() => flattenFunnelRows(allRows).slice(0, 120), [allRows]);

  const symbolOptions = useMemo(() => {
    const set = new Set(allRows.map((r) => r.symbol).filter(Boolean));
    return [...set].sort();
  }, [allRows]);

  const strategyOptions = useMemo(() => {
    const set = new Set<string>();
    for (const snap of allRows) {
      for (const sid of Object.keys(snap.by_strategy || {})) set.add(sid);
    }
    return [...set].sort();
  }, [allRows]);

  return (
    <details
      className={styles.funnelPanel}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>
        <span className={styles.funnelTitle}>策略漏斗（15min · stats_15min）</span>
        <span className={`muted ${styles.funnelHint}`}>按具体策略 · B/A/C 分层 · 点击展开</span>
      </summary>
      <p className={`muted ${styles.funnelNote}`}>
        所有层的策略漏斗（B·Trend / A·Spot / C·Multi-leg）都来自 live_monitor.db.stats_15min；如某策略漏斗为空，确认对应
        runner 已重启并完成至少一次 15min flush（默认 MLBOT_*_FUNNEL_FLUSH_SECONDS=900）。
      </p>
      <div className={styles.funnelFilters}>
        <label>
          账户层
          <select value={layer} onChange={(e) => setLayer(e.target.value)}>
            <option value="">全部</option>
            <option value="trend">{SCOPE_LABELS.trend}</option>
            <option value="spot">{SCOPE_LABELS.spot}</option>
            <option value="multi_leg">{SCOPE_LABELS.multi_leg}</option>
          </select>
        </label>
        <label>
          Symbol
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            <option value="">全部</option>
            {symbolOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label>
          策略
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="">全部</option>
            {strategyOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>时间</th>
              <th>Symbol</th>
              <th>层</th>
              <th>策略</th>
              <th>regime ✓</th>
              <th>regime ✗</th>
              <th>prefilter ✓</th>
              <th>prefilter ✗</th>
              <th>direction</th>
              <th>gate ✓</th>
            </tr>
          </thead>
          <tbody>
            {funnelQuery.isFetching && !flat.length ? (
              <tr>
                <td colSpan={10} className="muted">
                  加载中…
                </td>
              </tr>
            ) : flat.length ? (
              flat.map((r, i) => (
                <tr key={`${r.timestamp}-${r.symbol}-${r.strategy}-${i}`}>
                  <td>{r.timestamp.slice(0, 16)}</td>
                  <td>{r.symbol}</td>
                  <td>{SCOPE_LABELS[r.account_layer] || r.account_layer}</td>
                  <td>{r.strategy}</td>
                  <td>{r.regime_passed}</td>
                  <td>{r.regime_denied}</td>
                  <td>{r.prefilter_passed}</td>
                  <td>{r.prefilter_denied}</td>
                  <td>{r.direction}</td>
                  <td>{r.gate_passed}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={10} className="muted">
                  无 funnel 数据：检查对应 runner（quant-trend / quant-spot / quant-multi-leg）是否在跑，且已写过至少一次
                  stats_15min 快照
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </details>
  );
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
    <div className={styles.page}>
      <div className="toolbar-row">
        <h2>策略信号</h2>
        <label>
          特征周期
          <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            <option value="2h">2h</option>
            <option value="15min">15min</option>
            <option value="1d">1d</option>
          </select>
        </label>
        <label>
          回看
          <select value={lookback} onChange={(e) => setLookback(e.target.value)}>
            <option value="7">7 天</option>
            <option value="14">14 天</option>
            <option value="30">30 天</option>
          </select>
        </label>
        <button type="button" onClick={() => refetch()}>
          刷新
        </button>
      </div>

      <p className={`muted ${styles.hint}`}>
        全 universe 策略信号一览；点击「地图」进入该 symbol 的 K 线交易地图。
      </p>

      <section className="panel">
        <h3>全 Symbol 策略信号概览</h3>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>最新 Bar</th>
                <th>1m 行数</th>
                <th>地图</th>
                <th>{SCOPE_LABELS.trend}</th>
                <th>最近 Trend</th>
                <th>{SCOPE_LABELS.spot}</th>
                <th>{SCOPE_LABELS.multi_leg}</th>
                <th>最近 Multi-leg</th>
              </tr>
            </thead>
            <tbody>
              {rows.length ? (
                rows.map((r) => {
                  const spotTitle = spotBlockersTitle(r.strategies?.spot);
                  return (
                    <tr key={r.symbol}>
                      <td>
                        <strong>{r.symbol}</strong>
                      </td>
                      <td>{fmtBarTime(r.latest_bar)}</td>
                      <td>{r.bars_1min_rows ?? '—'}</td>
                      <td>
                        <Link to={r.map_href || `/trade-map?symbol=${r.symbol}`}>地图</Link>
                      </td>
                      <td className={styles.strategyCell}>
                        <StrategyLines block={r.strategies?.trend} />
                      </td>
                      <td className={`muted ${styles.strategyCell}`}>
                        <LastStrategyLines block={r.strategies?.trend} />
                      </td>
                      <td className={styles.strategyCell} title={spotTitle || undefined}>
                        <StrategyLines block={r.strategies?.spot} />
                      </td>
                      <td className={styles.strategyCell}>
                        <StrategyLines block={r.strategies?.multi_leg} />
                      </td>
                      <td className={`muted ${styles.strategyCell}`}>
                        <LastStrategyLines block={r.strategies?.multi_leg} />
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={9} className="muted">
                    {isFetching ? '加载中…' : '无数据'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <FunnelPanel />

      <p className="status-line">{status}</p>
    </div>
  );
}
