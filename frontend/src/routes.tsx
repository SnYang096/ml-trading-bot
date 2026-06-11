import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from '@/components/AppShell/AppShell.tsx';

const TradeMapPage = lazy(() =>
  import('@/pages/TradeMap/TradeMapPage.tsx').then((m) => ({ default: m.TradeMapPage })),
);
const TradeMapGridPage = lazy(() =>
  import('@/pages/TradeMapGrid/TradeMapGridPage.tsx').then((m) => ({ default: m.TradeMapGridPage })),
);
const OrdersPage = lazy(() =>
  import('@/pages/Orders/OrdersPage.tsx').then((m) => ({ default: m.OrdersPage })),
);
const SignalsPage = lazy(() =>
  import('@/pages/Signals/SignalsPage.tsx').then((m) => ({ default: m.SignalsPage })),
);
const AccountPage = lazy(() =>
  import('@/pages/Account/AccountPage.tsx').then((m) => ({ default: m.AccountPage })),
);
const RegimePage = lazy(() =>
  import('@/pages/Regime/RegimePage.tsx').then((m) => ({ default: m.RegimePage })),
);
const MonitoringPage = lazy(() =>
  import('@/pages/Monitoring/MonitoringPage.tsx').then((m) => ({ default: m.MonitoringPage })),
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
