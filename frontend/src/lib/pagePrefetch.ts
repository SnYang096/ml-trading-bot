/** Warm lazy route chunks on nav hover/focus so clicks feel instant. */
const loaders: Partial<Record<string, () => Promise<unknown>>> = {
  'trade-map': () => import('@/pages/TradeMap/TradeMapPage.tsx'),
  'trade-map-grid': () => import('@/pages/TradeMapGrid/TradeMapGridPage.tsx'),
};

export function prefetchPage(pageId: string): void {
  const load = loaders[pageId];
  if (load) void load();
}
