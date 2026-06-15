/**
 * FeatureMetricsTable 组件测试。
 *
 * 测试范围：
 * - 空数据时显示占位提示
 * - 有数据时渲染指标行
 * - 高亮柱的指标值正确显示
 * - chop_grid 策略显示 regime 行
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders, resetTradeMapStore } from '@/test/utils.tsx';
import { FeatureMetricsTable } from '../components/FeatureMetricsTable.tsx';

// ---------- 测试数据 ----------

const candles = Array.from({ length: 10 }, (_, i) => ({
  time: 1700000000 + i * 7200,
  open: 3000 + i,
  high: 3010 + i,
  low: 2990 + i,
  close: 3005 + i,
  volume: 500,
}));

const overlays = {
  box_pos_60: {
    available: true,
    points: candles.map((c) => ({ time: c.time, value: 0.35 + Math.random() * 0.3 })),
    reference_lines: [
      { label: '>=0.35', value: 0.35, operator: '>=' },
      { label: '<=0.65', value: 0.65, operator: '<=' },
    ],
  },
  bpc_semantic_chop: {
    available: true,
    points: candles.map((c) => ({ time: c.time, value: 0.4 + Math.random() * 0.4 })),
    reference_lines: [{ label: '>=0.50', value: 0.5, operator: '>=' }],
  },
};

describe('FeatureMetricsTable', () => {
  beforeEach(() => {
    resetTradeMapStore();
  });

  it('空指标列但策略有默认指标时仍渲染表格', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="tpc"
        columns={[]}
        candles={candles}
        overlays={{}}
        highlightTime={null}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    // tpc 策略即使 columns 为空也有默认指标行，表格应渲染
    expect(screen.getByText('TPC · 指标矩阵')).toBeDefined();
    expect(screen.getByText('可入场')).toBeDefined();
  });

  it('未知策略无指标列时显示占位提示', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="__unknown_strategy__"
        columns={[]}
        candles={candles}
        overlays={{}}
        highlightTime={null}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    // 无指标列
    expect(screen.getByText(/无指标列/)).toBeDefined();
  });

  it('无 K 线数据时显示占位提示', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="chop_grid"
        columns={['box_pos_60']}
        candles={[]}
        overlays={overlays}
        highlightTime={null}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    expect(screen.getByText(/无 K 线数据/)).toBeDefined();
  });

  it('有数据时渲染标题和指标行', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="chop_grid"
        columns={['box_pos_60']}
        candles={candles}
        overlays={overlays}
        highlightTime={candles[2].time}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    // 标题
    expect(screen.getByText(/指标矩阵/)).toBeDefined();

    // 可入场行存在
    expect(screen.getByText('可入场')).toBeDefined();

    // chop_grid 特有的 regime 行
    expect(screen.getByText('regime滞回')).toBeDefined();
    expect(screen.getByText('regime退出')).toBeDefined();
  });

  it('高亮柱时显示具体数值而非占位符', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="chop_grid"
        columns={['box_pos_60']}
        candles={candles}
        overlays={overlays}
        highlightTime={candles[2].time}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    // 不应显示 "悬停图表查看" 占位
    const hints = screen.queryAllByText(/悬停图表查看/);
    expect(hints.length).toBe(0);
  });

  it('未悬停时显示 — 占位', () => {
    renderWithProviders(
      <FeatureMetricsTable
        strategyId="chop_grid"
        columns={['box_pos_60']}
        candles={candles}
        overlays={overlays}
        highlightTime={null}
        mainChart={null}
        onBarClick={() => {}}
      />,
    );

    // 有占位提示
    expect(screen.getByText(/悬停图表查看/)).toBeDefined();
  });
});
