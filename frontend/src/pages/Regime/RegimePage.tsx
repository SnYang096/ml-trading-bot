import { useQuery } from '@tanstack/react-query';
import { apiGet } from '@/api/client.ts';
import type { RegimeOpsRow } from '@/api/types.ts';

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

export function RegimePage() {
  const { data, isLoading, error, refetch, dataUpdatedAt } = useQuery({
    queryKey: ['regime-ops'],
    queryFn: () => apiGet<RegimeOpsRow[]>('/api/trend/regime-ops'),
  });

  const rows = data?.data || [];
  const meta = data?.meta || {};

  return (
    <div className="page">
      <div className="toolbar-row">
        <h2>Regime Ops</h2>
        <button type="button" onClick={() => refetch()}>
          刷新
        </button>
      </div>
      {isLoading ? <p className="muted">加载中…</p> : null}
      {error ? <p className="pnl-neg">{String(error)}</p> : null}
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
      <p className="status-line">
        {rows.length
          ? `${meta.count ?? rows.length} strategies · ${String(meta.strategies_root || '')}${
              meta.drift_report_path
                ? ` · drift ${fmtIsoShort(String(meta.drift_generated_at || '')) || meta.drift_report_path}`
                : ' · 无 drift 报告'
            } · ${new Date(dataUpdatedAt || Date.now()).toLocaleTimeString()}`
          : ''}
      </p>
    </div>
  );
}
