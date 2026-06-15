import type { ChartTheme } from '@/lib/theme.ts';

export function chartLayoutOptions(theme: ChartTheme) {
  return {
    layout: {
      background: { color: theme.bg },
      textColor: theme.text,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: theme.grid },
      horzLines: { color: theme.grid },
    },
    rightPriceScale: {
      borderColor: theme.border,
      scaleMargins: { top: 0.08, bottom: 0.12 },
      minimumWidth: 72,
    },
  };
}

export function candleSeriesOptions(theme: ChartTheme) {
  return {
    upColor: theme.candleUp,
    downColor: theme.candleDown,
    borderVisible: false,
    wickUpColor: theme.candleUp,
    wickDownColor: theme.candleDown,
  };
}
