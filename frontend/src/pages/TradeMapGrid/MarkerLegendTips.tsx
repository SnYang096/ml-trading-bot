import styles from './TradeMapGridPage.module.css';

const ROWS: Array<{ shape: string; color: string; label: string; note?: string }> = [
  { shape: '△ / ▽', color: '绿 / 红', label: '开仓 entry', note: '多 / 空' },
  { shape: '●', color: '亮绿', label: '平仓 exit 盈利' },
  { shape: '●', color: '红', label: '平仓 exit 亏损' },
  { shape: '●', color: '深绿', label: '挂单 pending', note: '勾选「含挂单」后显示' },
  { shape: '●', color: '橙', label: 'regime 风控退出', note: 'chop_grid 等' },
  { shape: '■', color: '橙', label: '止盈 TP 成交', note: 'chop_grid 各 leg' },
  { shape: '■', color: '绿', label: '网格挂单 grid', note: 'L2/S1 resting limit' },
  { shape: '◆', color: '绿 / 红', label: '加仓 add' },
];

export function MarkerLegendTips() {
  return (
    <details className={styles.legendPanel}>
      <summary className={styles.legendSummary}>
        <span className={styles.legendTitle}>标记图例</span>
        <span className={styles.legendHint}>形状与颜色 · 策略标签 · 点击展开</span>
      </summary>
      <div className={styles.legendBody}>
        <p className={styles.legendIntro}>
          圆点/方块/三角是<strong>成交与挂单标记</strong>，不是 K 线本身。文字为<strong>策略名</strong>（tpc /
          bpc / chop / spot / scalp），不是账户层 B/A/C。同一根 2h K 线上事件多时会叠成竖列。
        </p>
        <table className={styles.legendTable}>
          <thead>
            <tr>
              <th>形状</th>
              <th>颜色</th>
              <th>含义</th>
              <th>备注</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={`${r.shape}-${r.label}`}>
                <td>{r.shape}</td>
                <td>{r.color}</td>
                <td>{r.label}</td>
                <td className={styles.legendNote}>{r.note || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <ul className={styles.legendList}>
          <li>
            <strong>B·Trend / A·Spot / C·Multi-leg</strong>：只过滤对应层标记，<em>不重载</em> K 线。
          </li>
          <li>
            点击品种名进入<strong>交易地图</strong>可看完整标签（如 <code>tpc:entry</code>、leg 名）与订单详情。
          </li>
          <li>彩色折线为已平仓盈亏连线（盈绿 / 亏红）。</li>
        </ul>
      </div>
    </details>
  );
}
