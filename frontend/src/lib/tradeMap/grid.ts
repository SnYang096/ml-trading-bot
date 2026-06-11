/** Highcap universe — keep in sync with live/highcap/universe.yaml */
export const GRID_SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'BNBUSDT',
  'SOLUSDT',
  'XRPUSDT',
  'HYPEUSDT',
] as const;

/** Compact OHLCV window for multi-symbol grid (days). */
export const GRID_INITIAL_DAYS = 21;

export function gridOhlcvQueryRange(timeframe: string | null | undefined): {
  from: string;
  to: string;
  full_range: string;
} {
  void timeframe;
  const end = new Date();
  const start = new Date(end.getTime() - GRID_INITIAL_DAYS * 86400000);
  return {
    from: start.toISOString(),
    to: end.toISOString(),
    full_range: 'false',
  };
}
