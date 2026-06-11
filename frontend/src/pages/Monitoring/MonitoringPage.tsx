import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/api/client.ts';
import type { MonitoringCard, MonitoringDashboard } from '@/api/types.ts';
import styles from './MonitoringPage.module.css';

const CADENCE_LABELS: Record<string, string> = {
  daily: '日更',
  weekly: '周更',
  monthly: '月更',
  quarterly: '季更',
  yearly: '年更',
};

function fmtAge(hours: number | null | undefined): string {
  if (hours == null || Number.isNaN(hours)) return '从未运行';
  if (hours < 48) return `${hours.toFixed(1)} 小时前`;
  return `${(hours / 24).toFixed(1)} 天前`;
}

function statusClass(st: string | undefined): string {
  if (st === 'ALERT') return styles.alert;
  if (st === 'MISSED') return styles.missed;
  return styles.ok;
}

function CadenceCard({ card }: { card: MonitoringCard }) {
  const label = CADENCE_LABELS[card.cadence] || card.cadence;
  const st = card.display_status || '—';
  const wd =
    card.watchdog_any_alert === true ? 'ALERT' : card.watchdog_any_alert === false ? 'OK' : '—';
  let dr = '—';
  if (card.drift_any_alert) dr = 'ALERT';
  else if (card.drift_no_plateaus) dr = '未校准';
  else if (card.drift_any_alert === false) dr = 'OK';

  return (
    <article className={`${styles.card} ${statusClass(st)}`}>
      <div className={styles.cardHead}>
        <span className={styles.cardTitle}>{label}</span>
        <span className={styles.badge}>{st}</span>
      </div>
      <div className={styles.cardBody}>
        <div>
          最近运行：{card.run_ts || '—'}（{fmtAge(card.age_hours)}）
        </div>
        <div>
          watchdog：{wd} · drift：{dr}
        </div>
        <div className="muted">上限 {card.max_age_hours}h 内有效</div>
        {card.output_dir ? (
          <div className={styles.meta}>
            <code>{card.output_dir}</code>
          </div>
        ) : null}
      </div>
    </article>
  );
}

export function MonitoringPage() {
  const { data, isLoading, error, refetch, dataUpdatedAt } = useQuery({
    queryKey: ['monitoring-dashboard'],
    queryFn: () => apiGet<MonitoringDashboard>('/api/monitoring/dashboard'),
  });

  const dashboard = data?.data;
  const summary = dashboard?.summary || {};
  const bannerClass =
    summary.any_alert || summary.any_missed || summary.any_uncalibrated
      ? styles.bannerWarn
      : styles.bannerOk;

  const alertRows: { cadence: string; source?: string; strategy?: string }[] = [];
  for (const [cadence, items] of Object.entries(dashboard?.strategy_alerts || {})) {
    for (const it of items || []) {
      alertRows.push({ cadence, ...it });
    }
  }

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>漂移监控</h2>
        <button type="button" onClick={() => refetch()}>
          刷新
        </button>
      </div>
      {isLoading ? <p className="muted">加载中…</p> : null}
      {error ? <p className="pnl-neg">{String(error)}</p> : null}
      {dashboard ? (
        <>
          <div className={`${styles.banner} ${bannerClass}`}>
            {summary.any_alert ? <strong className="pnl-neg">存在 ALERT</strong> : null}
            {summary.any_missed ? <strong className={styles.missedText}>存在缺勤</strong> : null}
            {summary.any_uncalibrated ? (
              <strong className={styles.uncalText}>plateau 未校准（需 Tier-0）</strong>
            ) : null}
            {!summary.any_alert && !summary.any_missed && !summary.any_uncalibrated ? (
              <strong className="pnl-pos">全部 cadence 正常</strong>
            ) : null}
            {dashboard.index_updated_at ? (
              <span className="muted"> · 索引 {dashboard.index_updated_at}</span>
            ) : null}
          </div>
          <div className={styles.cards}>
            {(dashboard.cards || []).length ? (
              dashboard.cards!.map((c) => <CadenceCard key={c.cadence} card={c} />)
            ) : (
              <p className="muted">无调度记录（远程需 enable systemd timer）</p>
            )}
          </div>
          <section className="panel">
            <h3>策略级 ALERT</h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Cadence</th>
                  <th>Source</th>
                  <th>Strategy</th>
                </tr>
              </thead>
              <tbody>
                {alertRows.length ? (
                  alertRows.map((r, i) => (
                    <tr key={`${r.cadence}-${r.strategy}-${i}`}>
                      <td>{CADENCE_LABELS[r.cadence] || r.cadence}</td>
                      <td>{r.source}</td>
                      <td>
                        <strong>{r.strategy}</strong>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="muted">
                      无策略级 ALERT
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </>
      ) : null}
      <p className="status-line">
        {dataUpdatedAt
          ? `已刷新 ${new Date(dataUpdatedAt).toLocaleTimeString()}`
          : ''}
      </p>
    </div>
  );
}
