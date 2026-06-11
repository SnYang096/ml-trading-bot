import { lazy, type ComponentType } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { AccountPage } from '@/pages/Account/AccountPage.tsx';
import { MonitoringPage } from '@/pages/Monitoring/MonitoringPage.tsx';
import { OrdersPage } from '@/pages/Orders/OrdersPage.tsx';
import { RegimePage } from '@/pages/Regime/RegimePage.tsx';
import { SignalsPage } from '@/pages/Signals/SignalsPage.tsx';
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

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/trade-map" replace />} />
        <Route path="/trade-map" element={<TradeMapPage />} />
        <Route path="/trade-map-grid" element={<TradeMapGridPage />} />
        <Route path="/orders" element={<OrdersPage />} />
        <Route path="/signals" element={<SignalsPage />} />
        <Route path="/account" element={<AccountPage />} />
        <Route path="/regime" element={<RegimePage />} />
        <Route path="/monitoring" element={<MonitoringPage />} />
        <Route path="*" element={<Navigate to="/trade-map" replace />} />
      </Route>
    </Routes>
  );
}
