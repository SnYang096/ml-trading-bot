import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from '@/components/AppShell/AppShell.tsx';
import { AccountPage } from '@/pages/Account/AccountPage.tsx';
import { MonitoringPage } from '@/pages/Monitoring/MonitoringPage.tsx';
import { OrdersPage } from '@/pages/Orders/OrdersPage.tsx';
import { RegimePage } from '@/pages/Regime/RegimePage.tsx';
import { SignalsPage } from '@/pages/Signals/SignalsPage.tsx';
import { TradeMapPage } from '@/pages/TradeMap/TradeMapPage.tsx';

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/trade-map" replace />} />
        <Route path="/trade-map" element={<TradeMapPage />} />
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
