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
  { id: 'trade-map-grid', href: '/trade-map-grid', label: '多品种地图' },
  { id: 'orders', href: '/orders', label: '订单' },
  { id: 'regime', href: '/regime', label: 'Regime' },
  { id: 'monitoring', href: '/monitoring', label: '漂移监控' },
  { id: 'account', href: '/account', label: '账户总览' },
] as const;

export function getSymbol(): string {
  return localStorage.getItem(SYMBOL_KEY) || '';
}

/** Account / Orders symbol filter: URL param wins; otherwise 全部 (not trade-map localStorage). */
export function resolveConsoleSymbol(searchParam: string | null | undefined): string {
  const fromUrl = String(searchParam ?? '').trim();
  if (fromUrl) return fromUrl;
  return SYMBOL_ALL;
}

export function isAllSymbols(sym: string): boolean {
  const s = String(sym || '').trim();
  return !s || s === SYMBOL_ALL || s.toUpperCase() === 'ALL';
}

/** Show filled size; treat 0 as missing so quantity fallback still works. */
export function displayOrderQty(row: {
  filled_quantity?: number | null;
  quantity?: number | null;
}): string {
  for (const key of ['filled_quantity', 'quantity'] as const) {
    const n = Number(row[key]);
    if (Number.isFinite(n) && n > 0) return String(n);
  }
  return '—';
}

export function displayLinkQty(link: { qty?: number | null }): string {
  const n = Number(link.qty);
  if (Number.isFinite(n) && n > 0) return String(n);
  return '—';
}

/** Limit/fill/trigger price; chop_grid SL/TP algo orders use stop_price, not price. */
export function displayOrderPrice(row: {
  display_price?: number | null;
  average_price?: number | null;
  price?: number | null;
  stop_price?: number | null;
  stop_loss_price?: number | null;
  take_profit_price?: number | null;
  exit_price?: number | null;
}): string {
  for (const key of [
    'display_price',
    'average_price',
    'price',
    'stop_price',
    'stop_loss_price',
    'take_profit_price',
    'exit_price',
  ] as const) {
    const n = Number(row[key]);
    if (Number.isFinite(n) && n > 0) return String(n);
  }
  return '—';
}

type OrderActionRow = {
  side?: string | null;
  position_side?: string | null;
  is_closing?: boolean | null;
  purpose?: string | null;
  order_type?: string | null;
  order_id?: string | null;
  status?: string | null;
};

function orderPositionSide(row: OrderActionRow): 'long' | 'short' | null {
  const ps = String(row.position_side || '').toUpperCase();
  if (ps === 'LONG') return 'long';
  if (ps === 'SHORT') return 'short';
  return null;
}

function orderIsClosing(row: OrderActionRow): boolean {
  if (row.is_closing === true) return true;
  if (row.is_closing === false) return false;
  const purpose = String(row.purpose || row.order_type || '').toLowerCase();
  const oid = String(row.order_id || '').toLowerCase();
  if (
    purpose.includes('take_profit') ||
    purpose.includes('stop_loss') ||
    purpose.includes('market_exit') ||
    purpose.includes('position_exit')
  ) {
    return true;
  }
  if (purpose.includes('entry') || purpose.includes('position_entry') || purpose === 'place') {
    return false;
  }
  if (oid.includes('_tp') || oid.includes('_sl') || oid.endsWith(':exit')) return true;
  const pos = orderPositionSide(row);
  const side = String(row.side || '').toUpperCase();
  if (pos === 'long' && side === 'SELL') return true;
  if (pos === 'short' && side === 'BUY') return true;
  return false;
}

/** Human-readable open/close label (开多/平多/开空/平空). */
export function displayOrderAction(row: OrderActionRow): string {
  const pos = orderPositionSide(row);
  const closing = orderIsClosing(row);
  const st = String(row.status || '').toLowerCase();
  const pending = ['open', 'new', 'pending', 'submitted', 'partially_filled', 'shadow'].includes(st);
  const suffix = pending ? '·挂' : '';
  if (pos === 'long') return `${closing ? '平多' : '开多'}${suffix}`;
  if (pos === 'short') return `${closing ? '平空' : '开空'}${suffix}`;
  const side = String(row.side || '').toUpperCase();
  if (side === 'BUY') return `买入${suffix}`;
  if (side === 'SELL') return `卖出${suffix}`;
  return side || '—';
}

type OrderKindRow = {
  order_type?: string | null;
  purpose?: string | null;
  stop_price?: number | null;
  stop_loss_price?: number | null;
  take_profit_price?: number | null;
};

/** Limit vs conditional (algo) vs market — for CMS order type column. */
export function displayOrderKind(row: OrderKindRow): string {
  const ot = String(row.order_type || row.purpose || '').toLowerCase();
  if (
    ot.includes('stop_market') ||
    ot.includes('take_profit_market') ||
    ot.includes('trailing_stop') ||
    ot.includes('stop') ||
    ot.includes('take_profit')
  ) {
    return '条件单';
  }
  for (const key of ['stop_price', 'stop_loss_price', 'take_profit_price'] as const) {
    const n = Number(row[key]);
    if (Number.isFinite(n) && n > 0) return '条件单';
  }
  if (ot.includes('limit') || ot === 'marketable_limit') return '限价';
  if (ot.includes('market')) return '市价';
  return ot ? ot : '—';
}

export function displayPositionSideLabel(side: unknown): string {
  const s = String(side || '').toLowerCase();
  if (s === 'long') return '做多';
  if (s === 'short') return '做空';
  return s || '—';
}

export function formatUnixTs(ts: unknown): string {
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return '—';
  const ms = n > 1e12 ? n : n * 1000;
  return new Date(ms).toISOString().slice(0, 19).replace('T', ' ');
}

export { displayExitKind, exitKindMeta, exitKindTip, exitKindGuideText } from '@/lib/exitKind.ts';

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
