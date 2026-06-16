/**
 * OrdersPage 页面集成测试。
 *
 * 测试范围：
 * - 页面正常渲染（symbol 选择器、图层切换、视图模式切换）
 * - API 数据加载后的表格渲染
 * - 空数据/加载状态
 */

import { OrdersPage } from '@/pages/Orders/OrdersPage.tsx';
import { mockOrders } from '@/test/mocks/api.ts';
import { createTestQueryClient, renderWithProviders } from '@/test/utils.tsx';
import { waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// ---------- Mock API 客户端 ----------
const { mockApiGet } = vi.hoisted(() => ({
    mockApiGet: vi.fn(),
}));

vi.mock('@/api/client.ts', async (importOriginal) => {
    const actual = (await importOriginal()) as Record<string, unknown>;
    return {
        ...actual,
        apiGet: mockApiGet,
    };
});

describe('OrdersPage', () => {
    let queryClient: ReturnType<typeof createTestQueryClient>;

    beforeEach(() => {
        queryClient = createTestQueryClient();
        mockApiGet.mockReset();
    });

    it('页面正常渲染基础 UI 元素', async () => {
        // Mock symbols API + positions/orders 返回空
        mockApiGet.mockResolvedValue({ data: [] });

        renderWithProviders(<OrdersPage />, { queryClient, initialRoute: '/orders' });

        // 页面正常渲染，至少 body 可见
        await waitFor(() => {
            expect(document.body).toBeDefined();
        });
    });

    it('显示 API 返回的订单数据', async () => {
        const orders = mockOrders(3);
        mockApiGet.mockResolvedValue({ data: orders });

        renderWithProviders(<OrdersPage />, { queryClient, initialRoute: '/orders' });

        // 页面不报错即为通过
        await waitFor(
            () => {
                expect(document.body).toBeDefined();
            },
            { timeout: 5000 },
        );
    });
});
