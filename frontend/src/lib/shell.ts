export const SYMBOL_KEY = 'mlbot_console_symbol';
export const SCOPES_KEY = 'mlbot_console_scopes';
export const ORDERS_FILTER_KEY = 'mlbot_orders_filter_v3';
export const TRADE_MAP_LAYOUT_KEY = 'mlbot_trade_map_layout_v2';
export const SYMBOL_ALL = '*';

export const SCOPE_LABELS: Record<string, string> = {
  trend: 'B·Trend',
  spot: 'A·Spot',
  multi_leg: 'C·Multi-leg',
};

export const PAGES = [
  { id: 'signals', href: '/signals', label: '策略信号' },
  { id: 'trade-map', href: '/trade-map', label: '交易地图' },
  { id: 'orders', href: '/orders', label: '订单' },
  { id: 'regime', href: '/regime', label: 'Regime' },
  { id: 'monitoring', href: '/monitoring', label: '漂移监控' },
  { id: 'account', href: '/account', label: '账户总览' },
] as const;

export function getSymbol(): string {
  return localStorage.getItem(SYMBOL_KEY) || '';
}

export function isAllSymbols(sym: string): boolean {
  const s = String(sym || '').trim();
  return !s || s === SYMBOL_ALL || s.toUpperCase() === 'ALL';
}

export function setSymbol(sym: string): void {
  if (sym && !isAllSymbols(sym)) localStorage.setItem(SYMBOL_KEY, sym);
}

export interface ScopesState {
  trend: boolean;
  spot: boolean;
  multiLeg: boolean;
  pending: boolean;
}

export function getScopesDefault(): ScopesState | null {
  try {
    return JSON.parse(localStorage.getItem(SCOPES_KEY) || 'null') as ScopesState | null;
  } catch {
    return null;
  }
}

export function setScopesState(state: ScopesState): void {
  localStorage.setItem(SCOPES_KEY, JSON.stringify(state));
}

export function browserLocalUrl(port: string | number, path = ''): string {
  const host = window.location.hostname || '127.0.0.1';
  return `http://${host}:${port}${path}`;
}

export function resolveLinkUrl(link: { id?: string; url?: string }): string {
  if (link?.id === 'grafana') return browserLocalUrl(3000);
  const raw = link?.url || '';
  if (raw.includes('host.docker.internal')) {
    try {
      const u = new URL(raw);
      return browserLocalUrl(u.port || '3000', u.pathname);
    } catch {
      return browserLocalUrl(3000);
    }
  }
  return raw;
}

export function fmtPnl(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}`;
}

export function pnlClass(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return '';
  if (v > 0) return 'pnl-pos';
  if (v < 0) return 'pnl-neg';
  return '';
}
