import { describe, expect, it } from 'vitest';
import { displayExitKind, exitKindMeta, exitKindTip } from '@/lib/exitKind.ts';

describe('exitKind', () => {
  it('maps canonical exit kinds to Chinese labels', () => {
    expect(displayExitKind('take_profit')).toBe('止盈');
    expect(displayExitKind('stop_loss')).toBe('止损');
    expect(displayExitKind('market_exit')).toBe('市价平');
    expect(displayExitKind('regime_exit')).toBe('Regime平');
    expect(displayExitKind('regime_or_risk_exit')).toBe('Regime平');
    expect(displayExitKind('cross_strategy_exit')).toBe('跨策略平');
    expect(displayExitKind('trailing_sl')).toBe('移动止损');
    expect(displayExitKind('structural_exit')).toBe('结构退出');
    expect(displayExitKind('sell')).toBe('卖出');
    expect(displayExitKind('exit')).toBe('平仓');
  });

  it('returns tips for hover', () => {
    expect(exitKindTip('take_profit')).toContain('止盈');
    expect(exitKindTip('cross_strategy_exit')).toContain('chop_grid');
    expect(exitKindMeta('')).toEqual({
      label: '—',
      tip: '尚未平仓，或本地未能识别退出路径。',
    });
  });
});
