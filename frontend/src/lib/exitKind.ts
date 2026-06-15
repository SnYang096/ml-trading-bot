/** Closed-trade exit_kind labels + tooltips (trade_links / orders legs view). */

export interface ExitKindMeta {
  label: string;
  tip: string;
}

type ExitKindRule = {
  match: (k: string) => boolean;
  meta: ExitKindMeta;
};

/** Order matters: first match wins. */
const EXIT_KIND_RULES: ExitKindRule[] = [
  {
    match: (k) => k.includes('take_profit') || k === 'tp' || k.includes('grid_tp'),
    meta: {
      label: '止盈',
      tip: '保护止盈单成交（chop_grid 单腿 TP / basket TP / repair 补挂 TP）。网格常态获利出场。',
    },
  },
  {
    match: (k) => k.includes('cross_strategy') || k.includes('foreign_flatten'),
    meta: {
      label: '跨策略平',
      tip: 'C 账户内另一策略发起市价 flatten（常见：chop_grid 清掉 trend_scalp 共用 hedge slot 的残仓）。',
    },
  },
  {
    match: (k) => k.includes('regime_or_risk') || k.includes('regime'),
    meta: {
      label: 'Regime平',
      tip: '行情 Regime 切换或段级风控触发整段/全网格市价退出（regime_exit / regime_or_risk_exit）。',
    },
  },
  {
    match: (k) => k.includes('trailing'),
    meta: {
      label: '移动止损',
      tip: 'B 层趋势 trailing 激活后止损跟随价格上移/下移并触发出场（exit_reason 含 trailing_sl）。',
    },
  },
  {
    match: (k) => k.includes('structural'),
    meta: {
      label: '结构退出',
      tip: 'B 层 bull Regime 下按结构位（如 SR/宽通道）主动离场，而非固定 TP/SL 单。',
    },
  },
  {
    match: (k) => k.includes('stop') || k.endsWith('_sl') || k.includes('_sl_'),
    meta: {
      label: '止损',
      tip: '保护止损 STOP 成交，或 exit_reason 归类为 stop/SL（含 grid_sl / emergency_sl 等）。',
    },
  },
  {
    match: (k) => k.includes('market_exit') || (k.includes('market') && k.includes('exit')),
    meta: {
      label: '市价平',
      tip: '主动市价 reduce-only 平仓：段末清理、late_fixup 补平、dust 残量、手动 flatten 等。',
    },
  },
  {
    match: (k) => k === 'sell',
    meta: {
      label: '卖出',
      tip: 'Spot 层卖出兑现（A 账户现货回合）。',
    },
  },
  {
    match: (k) => k === 'exit',
    meta: {
      label: '平仓',
      tip: '已配对平仓但未归入上述细类（通用 fallback）。',
    },
  },
];

const EMPTY_META: ExitKindMeta = {
  label: '—',
  tip: '尚未平仓，或本地未能识别退出路径。',
};

/** All known kinds for column legend (deduped by label). */
export const EXIT_KIND_GUIDE: ExitKindMeta[] = EXIT_KIND_RULES.map((r) => r.meta);

export function exitKindGuideText(): string {
  const lines = [
    '为何有多种平仓方式？A Spot / B Trend / C Multi-leg 共用此列；不同策略与触发条件写入不同 exit_kind，便于区分止盈、止损、Regime 强平、跨策略清理等。',
    ...EXIT_KIND_GUIDE.map((g) => `${g.label}：${g.tip}`),
  ];
  return lines.join('\n');
}

export function exitKindMeta(kind: unknown): ExitKindMeta {
  const k = String(kind || '').trim().toLowerCase();
  if (!k) return EMPTY_META;
  for (const rule of EXIT_KIND_RULES) {
    if (rule.match(k)) return rule.meta;
  }
  return {
    label: k,
    tip: `未收录的原始 exit_kind：${k}`,
  };
}

export function displayExitKind(kind: unknown): string {
  return exitKindMeta(kind).label;
}

export function exitKindTip(kind: unknown): string {
  return exitKindMeta(kind).tip;
}
