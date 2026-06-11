import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/api/client.ts';
import type {
  MonitoringCard,
  MonitoringDashboard,
  MonitoringIssueRow,
} from '@/api/types.ts';
import styles from './MonitoringPage.module.css';

const CADENCE_LABELS: Record<string, string> = {
  daily: '日更',
  weekly: '周更',
  monthly: '月更',
  quarterly: '季更',
  yearly: '年更',
};

const CADENCE_ORDER = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'];

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

function fmtStrategy(strategy: string | undefined): string {
  if (strategy === '_factor_health') return '因子健康 (PSI/IC)';
  return strategy || '—';
}

function sortCards(cards: MonitoringCard[]): MonitoringCard[] {
  return [...cards].sort(
    (a, b) =>
      (CADENCE_ORDER.indexOf(a.cadence) ?? 99) - (CADENCE_ORDER.indexOf(b.cadence) ?? 99),
  );
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
        {(card.alert_details || []).length ? (
          <ul className={styles.detailList}>
            {(card.alert_details || []).map((line) => (
              <li key={line} className={styles.alertLine}>
                {line}
              </li>
            ))}
          </ul>
        ) : null}
        {(card.uncalibrated_details || []).length ? (
          <ul className={styles.detailList}>
            {(card.uncalibrated_details || []).map((line) => (
              <li key={line} className={styles.uncalLine}>
                {line}
              </li>
            ))}
          </ul>
        ) : null}
        {card.output_dir ? (
          <div className={styles.meta}>
            <code>{card.output_dir}</code>
          </div>
        ) : null}
      </div>
    </article>
  );
}

function IssueTable({
  title,
  rows,
  emptyText,
  statusLabel,
  statusClassName,
}: {
  title: string;
  rows: { cadence: string; row: MonitoringIssueRow }[];
  emptyText: string;
  statusLabel: string;
  statusClassName: string;
}) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Cadence</th>
            <th>Source</th>
            <th>Strategy</th>
            <th>状态</th>
            <th>详情</th>
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((r, i) => (
              <tr key={`${r.cadence}-${r.row.strategy}-${i}`}>
                <td>{CADENCE_LABELS[r.cadence] || r.cadence}</td>
                <td>{r.row.source}</td>
                <td>
                  <strong>{fmtStrategy(r.row.strategy)}</strong>
                </td>
                <td className={statusClassName}>{statusLabel}</td>
                <td className={styles.msgCell}>
                  {(r.row.messages || []).length ? (
                    <ul className={styles.detailList}>
                      {(r.row.messages || []).map((m) => (
                        <li key={m}>{m}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={5} className="muted">
                {emptyText}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </section>
  );
}

export function MonitoringPage() {
  const { data, isLoading, error, refetch, dataUpdatedAt } = useQuery({
    queryKey: ['monitoring-dashboard'],
    queryFn: () => apiGet<MonitoringDashboard>('/api/monitoring/dashboard'),
    staleTime: 60_000,
  });

  const dashboard = data?.data;
  const summary = dashboard?.summary || {};
  const bannerClass =
    summary.any_alert || summary.any_missed || summary.any_uncalibrated
      ? styles.bannerWarn
      : styles.bannerOk;

  const alertRows: { cadence: string; row: MonitoringIssueRow }[] = [];
  for (const [cadence, items] of Object.entries(dashboard?.strategy_alerts || {})) {
    for (const row of items || []) {
      alertRows.push({ cadence, row });
    }
  }
  alertRows.sort(
    (a, b) =>
      (CADENCE_ORDER.indexOf(a.cadence) ?? 99) - (CADENCE_ORDER.indexOf(b.cadence) ?? 99),
  );

  const uncalRows: { cadence: string; row: MonitoringIssueRow }[] = [];
  for (const [cadence, items] of Object.entries(dashboard?.strategy_uncalibrated || {})) {
    for (const row of items || []) {
      uncalRows.push({ cadence, row });
    }
  }
  uncalRows.sort(
    (a, b) =>
      (CADENCE_ORDER.indexOf(a.cadence) ?? 99) - (CADENCE_ORDER.indexOf(b.cadence) ?? 99),
  );

  const cards = sortCards(dashboard?.cards || []);

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
            {cards.length ? (
              cards.map((c) => <CadenceCard key={c.cadence} card={c} />)
            ) : (
              <p className="muted">无调度记录（远程需 enable systemd timer）</p>
            )}
          </div>
          <IssueTable
            title="告警详情"
            rows={alertRows}
            emptyText="无 ALERT"
            statusLabel="ALERT"
            statusClassName="pnl-neg"
          />
          <IssueTable
            title="plateau 未校准"
            rows={uncalRows}
            emptyText="无未校准项"
            statusLabel="未校准"
            statusClassName={styles.uncalText}
          />
        </>
      ) : null}
      <p className="status-line">
        {dataUpdatedAt ? `已刷新 ${new Date(dataUpdatedAt).toLocaleTimeString()}` : ''}
      </p>
    </div>
  );
}
