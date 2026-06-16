import { ThemeProvider } from '@/context/ThemeContext.tsx';
import { resetHistoryState, useTradeMapStore } from '@/stores/tradeMapStore.ts';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, type RenderOptions } from '@testing-library/react';
import type { ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';

/**
 * 为每个测试创建全新的 QueryClient，避免用例间缓存污染。
 */
export function createTestQueryClient() {
    return new QueryClient({
        defaultOptions: {
            queries: { retry: false, gcTime: 0 },
            mutations: { retry: false },
        },
    });
}

interface AllProvidersProps {
    children: React.ReactNode;
    initialRoute?: string;
    queryClient?: QueryClient;
}

/**
 * 包裹所有必需的 Provider（Router、Query、Theme），
 * 用于 Testing Library 的 render。
 */
export function AllProviders({
    children,
    initialRoute = '/trade-map',
    queryClient,
}: AllProvidersProps) {
    const qc = queryClient || createTestQueryClient();
    return (
        <QueryClientProvider client={qc}>
            <MemoryRouter initialEntries={[initialRoute]}>
                <ThemeProvider>{children}</ThemeProvider>
            </MemoryRouter>
        </QueryClientProvider>
    );
}

/**
 * 带 Provider 的 render 封装。
 * 用法：renderWithProviders(<MyComponent />, { initialRoute: '/orders' })
 */
export function renderWithProviders(
    ui: ReactElement,
    options?: Omit<RenderOptions, 'wrapper'> & {
        initialRoute?: string;
        queryClient?: QueryClient;
    },
) {
    const { initialRoute, queryClient, ...renderOptions } = options || {};
    return render(ui, {
        wrapper: ({ children }) => (
            <AllProviders initialRoute={initialRoute} queryClient={queryClient}>
                {children}
            </AllProviders>
        ),
        ...renderOptions,
    });
}

/**
 * 每个测试用例前调用，将 tradeMap zustand store 重置到初始状态。
 */
export function resetTradeMapStore() {
    resetHistoryState();
    useTradeMapStore.setState({
        symbol: 'ETHUSDT',
        timeframe: '2h',
        layers: {
            trend: true,
            spot: true,
            multiLeg: true,
            pending: false,
            chopGrid: true,
            prefilter: true,
            gate: false,
        },
        selectedFeatureColumns: [],
        availableFeatureColumns: [],
        featureStrategyFocus: '',
        featureSearchQuery: '',
        featureDrawerOpen: false,
        paneVolume: true,
        ordersDockOpen: true,
        mainEma1200: false,
        mainWeeklyEma200: false,
    });
}
