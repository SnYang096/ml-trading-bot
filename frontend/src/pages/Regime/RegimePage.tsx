import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/api/client.ts';
import type { RegimeCockpitData, RegimeOpsRow } from '@/api/types.ts';
import styles from './RegimePage.module.css';

function fmtLc(lc: Record<string, unknown> | undefined): string {
  if (!lc || typeof lc !== 'object') return '—';
  const notes = lc.notes ? String(lc.notes).slice(0, 80) : '';
  if (notes) return notes;
  const ts = lc.timestamp || lc.data_source || '';
  return ts ? String(ts) : '—';
}

function fmtIsoShort(iso: string | undefined): string {
  const s = String(iso || '').trim();
  if (!s) return '';
  const norm = s.includes('T') ? s.replace('T', ' ') : s;
  return norm.length > 19 ? norm.slice(0, 19) : norm;
}

function fmtDriftTime(row: RegimeOpsRow, meta: Record<string, unknown>) {
  const checked = row.drift_checked_at || meta?.drift_generated_at;
  if (checked) return { prefix: '检测', at: fmtIsoShort(String(checked)) };
  const baseline = row.config_reference_at;
  if (baseline) return { prefix: '基准', at: fmtIsoShort(baseline) };
  return null;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

function labelBadgeClass(label: string | undefined): string {
  const l = String(label || '').toLowerCase();
  if (l === 'bull') return styles.badgeBull;
  if (l === 'bear') return styles.badgeBear;
  return styles.badgeNeutral;
}

function alertClass(alert: string | undefined): string {
  if (alert === 'REBALANCE_SUGGEST') return styles.alertSuggest;
  if (alert === 'WATCH') return styles.alertWatch;
  return styles.alertOk;
}

function navSegmentClass(scope: string | undefined): string {
  if (scope === 'spot') return styles.navSpot;
  if (scope === 'rolling') return styles.navRolling;
  if (scope === 'trend') return styles.navTrend;
  return styles.navMultileg;
}

const COMPOSITE_TIERS = [
  {
    key: 'risk_off',
    range: '0 – 3',
    title: 'risk-off（收缩 beta）',
    meaning: '宏观偏弱、深熊或 B 层 bull 少 → 宜降低现货/长期敞口，防守为主',
  },
  {
    key: 'neutral',
    range: '4 – 6',
    title: '中性',
    meaning: '多层信号混杂，无明显牛熊 → 维持均衡 NAV 目标带',
  },
  {
    key: 'risk_on',
    range: '7 – 11',
    title: 'risk-on（偏多 beta）',
    meaning: '宏观偏强、周线上方、趋势 bull 占比高 → 宜提高 A 现货/长期持仓，勿仅靠 B 扛牛市',
  },
] as const;

export function RegimePage() {
  const cockpitQ = useQuery({
    queryKey: ['regime-cockpit'],
    queryFn: () => apiGet<RegimeCockpitData>('/api/regime/cockpit'),
    staleTime: 60_000,
  });

  const data = cockpitQ.data?.data;
  const meta = cockpitQ.data?.meta || {};
  const rows = data?.ops || [];
  const layers = data?.layers || {};
  const alloc = data?.allocation;
  const alert = alloc?.alert || 'OK';
  const scopes = alloc?.scopes || [];
  const lastSched = data?.last_scheduled;
  const compositeLabel = String(data?.composite?.label || alloc?.composite || '').toLowerCase();
  const compositeScore = data?.composite?.total_score;
  const compositeBreakdown = data?.composite?.breakdown || [];

  return (
    <div className={`page ${styles.regimePage}`}>
      <div className={styles.toolbarRow}>
        <h2>Regime Cockpit</h2>
        <button type="button" onClick={() => cockpitQ.refetch()}>
          刷新
        </button>
      </div>
      <p className={`muted ${styles.sectionNote}`}>
        A/B/C 三层 regime 并列展示；调仓告警由独立的 composite 档位 + NAV 目标带驱动（只建议、不自动划转）。
      </p>

      <details className={styles.explainer}>
        <summary>页面说明 · composite 与 A/B/C 的区别</summary>
        <div className={styles.explainerBody}>
          <p>
            <strong>A/B/C 卡片</strong>：各策略自己的 regime（周线成本区 / 2H 牛熊 / chop·动量），语义和时钟不同，
            <strong>不合并</strong>成一个 bull/bear。
          </p>
          <p>
            <strong>composite</strong>：仅用于<strong>调仓建议</strong>的启发式打分（不写入策略 yaml）。
            把 macro、周线、bull_share、chop 等加权求和为 <code>total_score</code>，再映射到档位：
          </p>
          <ul>
            <li>
              <code>total ≤ 3</code> → risk-off：组合宜收缩 beta
            </li>
            <li>
              <code>4 – 6</code> → neutral：均衡配置
            </li>
            <li>
              <code>≥ 7</code> → risk-on：组合宜提高现货/长期 beta（满分约 11）
            </li>
          </ul>
          <p>
            <strong>NAV 目标带</strong>随 composite 档位切换（例如 risk-on 时 A·Spot 目标占比更高）。
            定时任务每 4h 采样一次落盘；本页为打开时的实时快照。
          </p>
        </div>
      </details>

      {cockpitQ.isLoading ? <p className="muted">加载中…</p> : null}
      {cockpitQ.error ? <p className="pnl-neg">{String(cockpitQ.error)}</p> : null}

      {data ? (
        <>
          <div className={`${styles.alertBanner} ${alertClass(alert)}`}>
            <strong>调仓告警：{alert}</strong>
            <span className="muted">
              {' '}
              · composite={data.composite?.label_title || data.composite?.label || '—'}
              {data.feature_bus?.stale ? ' · feature bus STALE' : ''}
              {data.as_of ? ` · bar ${fmtIsoShort(data.as_of)}` : ''}
            </span>
            {alloc?.suggestions?.length ? (
              <ul className={styles.suggestions}>
                {alloc.suggestions.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            ) : null}
            {lastSched?.ts ? (
              <p className={`muted ${styles.kv}`}>
                定时检查 {String(lastSched.ts).slice(0, 19)} · status=
                {lastSched.status || '—'}
                {lastSched.detail?.alert ? ` · 记录 ${lastSched.detail.alert}` : ''}
              </p>
            ) : null}
          </div>

          <section className={styles.compositePanel}>
            <h3 className={styles.layerTitle}>调仓档位 · composite</h3>
            <p className={styles.kv}>
              当前{' '}
              <span className={styles.badge}>
                {data.composite?.label_title || data.composite?.label || '—'}
              </span>
              {compositeScore != null ? (
                <>
                  {' '}
                  · total_score=<strong>{compositeScore}</strong>
                </>
              ) : null}
              {alloc?.composite ? (
                <span className="muted"> · NAV 目标带按 {alloc.composite} 选取</span>
              ) : null}
            </p>
            {compositeBreakdown.length ? (
              <div className={styles.compositeBreakdown}>
                {compositeBreakdown.map((row) => (
                  <span key={String(row.id)} className={styles.compositeChip}>
                    {row.id}: {row.score}×{row.weight}={row.weighted}
                  </span>
                ))}
              </div>
            ) : null}
            <table className={styles.compositeMap}>
              <thead>
                <tr>
                  <th>总分区间</th>
                  <th>档位</th>
                  <th>调仓含义</th>
                </tr>
              </thead>
              <tbody>
                {COMPOSITE_TIERS.map((tier) => (
                  <tr
                    key={tier.key}
                    className={compositeLabel === tier.key ? styles.compositeMapActive : ''}
                  >
                    <td>{tier.range}</td>
                    <td>{tier.title}</td>
                    <td className="muted">{tier.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h3 className={styles.layerTitle}>NAV 占比</h3>
            <p className={`muted ${styles.sectionNote}`}>
              各账户实盘占比 vs 目标带；告警等级 OK / WATCH / REBALANCE_SUGGEST 由偏离程度与 composite 矛盾共同决定。
            </p>
            <div className={styles.navBar}>
              {scopes
                .filter((s) => s.nav_pct != null)
                .map((s) => {
                const pct = s.nav_pct ?? 0;
                const width = Math.max(pct * 100, 4);
                return (
                  <div
                    key={s.scope}
                    className={`${styles.navSegment} ${navSegmentClass(s.scope)}`}
                    style={{ flexGrow: width, flexBasis: `${width}%` }}
                    title={`${s.label} ${fmtPct(s.nav_pct)}`}
                  >
                    {s.label} {fmtPct(s.nav_pct)}
                  </div>
                );
              })}
            </div>
            <div className={`${styles.navLegend} muted`}>
              {scopes.map((s) => (
                <span key={`leg-${s.scope}`}>
                  {s.label}: {fmtPct(s.nav_pct)}
                  {s.band
                    ? ` (目标 ${fmtPct(s.band.target)}，带 ${fmtPct(s.band.min)}–${fmtPct(s.band.max)})`
                    : ''}
                  {s.status && s.status !== 'OK' ? ` [${s.status}]` : ''}
                </span>
              ))}
              <span>总 NAV: {alloc?.total_nav_usdt?.toFixed(0) ?? '—'} USDT</span>
            </div>
          </section>

          <p className={`muted ${styles.sectionNote}`}>
            下方三层为<strong>参考面板</strong>（live feature bus），与 composite 独立；勿用 B 的 bull 直接代表 A 该加仓。
          </p>
          <div className={styles.layerGrid}>
            <article className={styles.layerCard}>
              <h3 className={styles.layerTitle}>A·Beta 慢层</h3>
              <p className={styles.kv}>
                周线 EMA200 位:{' '}
                {layers.a_spot?.weekly_ema_200_position?.toFixed(4) ?? '—'}
              </p>
              <p className={styles.kv}>
                状态: <span className={styles.badge}>{layers.a_spot?.deploy_state}</span>
                {layers.a_spot?.deploy_allowed ? ' · 允许 deploy' : ' · 不新开 deploy'}
              </p>
              <p className={styles.kv}>
                宏观分: {layers.a_spot?.abc_macro_regime_score ?? '—'}{' '}
                {layers.a_spot?.macro_label ? `(${layers.a_spot.macro_label})` : ''}
              </p>
              <p className="muted">{layers.a_spot?.hint}</p>
            </article>

            <article className={styles.layerCard}>
              <h3 className={styles.layerTitle}>B·Swing α</h3>
              <p className={styles.kv}>
                当前:{' '}
                <span
                  className={`${styles.badge} ${labelBadgeClass(layers.b_trend?.current_label)}`}
                >
                  {layers.b_trend?.current_label || '—'}
                </span>
              </p>
              <p className={styles.kv}>
                ADX50: {layers.b_trend?.features?.adx_50 ?? '—'} · EMA1200 pos:{' '}
                {layers.b_trend?.features?.ema_1200_position ?? '—'}
              </p>
              <p className={styles.kv}>
                7d bull_share: {fmtPct(layers.b_trend?.bull_share_7d)} (baseline{' '}
                {fmtPct(layers.b_trend?.baseline_bull_share)})
                {layers.b_trend?.drift_alert ? ' · 漂移' : ''}
              </p>
              {layers.b_trend?.divergence ? (
                <p className="muted">
                  分化: {layers.b_trend.divergence.symbol}=
                  {layers.b_trend.divergence.label}
                </p>
              ) : null}
              <p className="muted">{layers.b_trend?.hint}</p>
            </article>

            <article className={styles.layerCard}>
              <h3 className={styles.layerTitle}>C·Micro α</h3>
              <p className={styles.kv}>
                chop: {String(layers.c_multileg?.chop_grid?.value ?? '—')} →{' '}
                {String(layers.c_multileg?.chop_grid?.state ?? '—')}
              </p>
              <p className={styles.kv}>
                momentum: {String(layers.c_multileg?.trend_scalp?.value ?? '—')} →{' '}
                {String(layers.c_multileg?.trend_scalp?.state ?? '—')}
              </p>
              <p className={styles.kv}>
                router: {layers.c_multileg?.router_hint || '—'}
              </p>
              <p className="muted">{layers.c_multileg?.hint}</p>
            </article>
          </div>
        </>
      ) : null}

      <section className={styles.opsSection}>
        <h3>Regime Ops</h3>
        <p className={`muted ${styles.sectionNote}`}>
          策略配置与离线 drift 监测（底层表保留）
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>账户层</th>
              <th>策略</th>
              <th>配置</th>
              <th>规则数</th>
              <th>允许方向</th>
              <th>最近校准</th>
              <th>Drift</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((r) => {
                const driftSt = r.drift_status || '—';
                const driftCls =
                  driftSt === '漂移' || driftSt === '告警'
                    ? 'pnl-neg'
                    : driftSt === '正常'
                      ? 'pnl-pos'
                      : '';
                const time = fmtDriftTime(r, meta);
                return (
                  <tr key={`${r.strategy}-${r.account_layer}`}>
                    <td>{r.account_layer_title || r.account_layer || '—'}</td>
                    <td>
                      <strong>{r.strategy}</strong>
                    </td>
                    <td className="muted">
                      {r.present ? '✓' : '—'} {r.regime_source || '—'}
                      <br />
                      <span className="muted">{r.regime_path}</span>
                    </td>
                    <td>{r.n_rules ?? 0}</td>
                    <td>{(r.allowed_sides || []).join(', ')}</td>
                    <td>{fmtLc(r.last_calibration as Record<string, unknown>)}</td>
                    <td>
                      <span className={driftCls} title={r.drift_detail}>
                        {driftSt}
                      </span>
                      {time ? (
                        <>
                          <br />
                          <span className="muted">
                            {time.prefix} {time.at}
                          </span>
                        </>
                      ) : null}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={7} className="muted">
                  无数据
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <p className="status-line">
        {data
          ? `${meta.symbol || data.symbol} · ${rows.length} strategies · ${new Date(cockpitQ.dataUpdatedAt || Date.now()).toLocaleTimeString()}`
          : ''}
      </p>
    </div>
  );
}
