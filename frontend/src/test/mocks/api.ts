/**
 * API Mock 工具。
 *
 * Vitest 的 vi.mock 会被 hoist，因此 mock 定义必须放在
 * 文件顶层或 vi.hoisted() 回调中。本文件提供一组 helper 类型和
 * 快速 mock 工厂，供各测试文件 import 使用。
 */

import type { BundleData, OrderRow, SymbolRow } from '@/api/types.ts';

// ---------- 类型 ----------
export type ApiResponse<T = unknown> = { data: T; meta?: Record<string, unknown> };

// ---------- 工厂函数 ----------

const EMPTY_BUNDLE_FIELDS: Omit<BundleData, 'ohlcv'> = {
    markers: [],
    trade_links: [],
    overlays: {},
    main_overlays: {},
    chop_grid_overlay: {},
    chop_regime_regions: [],
    strategy_stage_regions: {},
};

/** 创建空的 BundleData */
export function emptyBundleData(): BundleData {
    return {
        ohlcv: { candles: [] },
        ...EMPTY_BUNDLE_FIELDS,
    };
}

/** 创建带 candles 的 BundleData */
export function bundleWithCandles(count: number, baseTime = 1700000000): BundleData {
    const candles = Array.from({ length: count }, (_, i) => ({
        time: baseTime + i * 7200,
        open: 3000 + Math.random() * 100,
        high: 3100 + Math.random() * 100,
        low: 2950 + Math.random() * 100,
        close: 3050 + Math.random() * 100,
        volume: 100 + Math.random() * 900,
    }));
    return {
        ohlcv: { candles },
        ...EMPTY_BUNDLE_FIELDS,
    };
}

/** 创建 symbol 列表 */
export function symbolList(): SymbolRow[] {
    return [
        { symbol: 'ETHUSDT' },
        { symbol: 'BTCUSDT' },
        { symbol: 'HYPEUSDT' },
    ];
}

/** 创建模拟订单 */
export function mockOrders(count: number): OrderRow[] {
    return Array.from({ length: count }, (_, i) => ({
        order_id: `order-${i}`,
        symbol: 'ETHUSDT',
        side: i % 2 === 0 ? 'buy' : 'sell',
        type: 'limit',
        price: String(3000 + i * 10),
        amount: String(1 + i * 0.1),
        status: 'filled',
        create_time: new Date().toISOString(),
        scope: 'trend',
        marker_id: i === 0 ? 'm1' : null,
    })) as unknown as OrderRow[];
}

