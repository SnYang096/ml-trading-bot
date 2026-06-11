import { lazy, Suspense, type ComponentType } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from '@/components/AppShell/AppShell.tsx';
import { ChunkLoadError } from '@/components/ChunkLoadError.tsx';

function lazyPage(
  loader: () => Promise<{ default: ComponentType<object> }>,
  label: string,
) {
  return lazy(() =>
    loader().catch((err: unknown) => {
      const detail = err instanceof Error ? err.message : String(err);
      const Fallback = () => <ChunkLoadError page={label} detail={detail} />;
      return { default: Fallback };
    }),
  );
}

const TradeMapPage = lazyPage(
  () => import('@/pages/TradeMap/TradeMapPage.tsx').then((m) => ({ default: m.TradeMapPage })),
  '交易地图',
);
const TradeMapGridPage = lazyPage(
  () =>
    import('@/pages/TradeMapGrid/TradeMapGridPage.tsx').then((m) => ({
      default: m.TradeMapGridPage,
    })),
  '多品种地图',
);
const OrdersPage = lazyPage(
  () => import('@/pages/Orders/OrdersPage.tsx').then((m) => ({ default: m.OrdersPage })),
  '订单',
);
const SignalsPage = lazyPage(
  () => import('@/pages/Signals/SignalsPage.tsx').then((m) => ({ default: m.SignalsPage })),
  '策略信号',
);
const AccountPage = lazyPage(
  () => import('@/pages/Account/AccountPage.tsx').then((m) => ({ default: m.AccountPage })),
  '账户总览',
);
const RegimePage = lazyPage(
  () => import('@/pages/Regime/RegimePage.tsx').then((m) => ({ default: m.RegimePage })),
  'Regime',
);
const MonitoringPage = lazyPage(
  () =>
    import('@/pages/Monitoring/MonitoringPage.tsx').then((m) => ({
      default: m.MonitoringPage,
    })),
  '漂移监控',
);

function PageFallback() {
  return (
    <div className="page">
      <p className="muted">加载中…</p>
    </div>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/trade-map" replace />} />
        <Route
          path="/trade-map"
          element={
            <Suspense fallback={<PageFallback />}>
              <TradeMapPage />
            </Suspense>
          }
        />
        <Route
          path="/trade-map-grid"
          element={
            <Suspense fallback={<PageFallback />}>
              <TradeMapGridPage />
            </Suspense>
          }
        />
        <Route
          path="/orders"
          element={
            <Suspense fallback={<PageFallback />}>
              <OrdersPage />
            </Suspense>
          }
        />
        <Route
          path="/signals"
          element={
            <Suspense fallback={<PageFallback />}>
              <SignalsPage />
            </Suspense>
          }
        />
        <Route
          path="/account"
          element={
            <Suspense fallback={<PageFallback />}>
              <AccountPage />
            </Suspense>
          }
        />
        <Route
          path="/regime"
          element={
            <Suspense fallback={<PageFallback />}>
              <RegimePage />
            </Suspense>
          }
        />
        <Route
          path="/monitoring"
          element={
            <Suspense fallback={<PageFallback />}>
              <MonitoringPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/trade-map" replace />} />
      </Route>
    </Routes>
  );
}
