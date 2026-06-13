import { describe, expect, it } from 'vitest';
import {
  forwardFillOverlayToCandles,
  mergeCandlesByTime,
  overlayAsOfAtCandleTimes,
  overlayValueAtCandle,
  scopesFromLayers,
} from '@/lib/tradeMap/ohlcv.ts';
import {
  barSpacingForCount,
  defaultVisibleBarCount,
  isValidLogicalRange,
  logicalRangeAfterHistoryPrepend,
  sanitizeCandlesForLwc,
  visibleLogicalRange,
} from '@/lib/tradeMap/candles.ts';
import {
  chopGridHysteresisActive,
  chopRegimeExitBarTimes,
  chopRegimeHysteresisOnAtTime,
  chopRegimeSeriesFromOverlay,
  isFeatureBusRegimeExitMarker,
  compactMarkerLabel,
  dedupeMarkersForChart,
  markersForChartDisplay,
  markersToLwc,
  prepareChartMarkers,
  snapMarkersToCandleTimes,
} from '@/lib/tradeMap/markers.ts';
import { mergeTradeLinks, tradeLinksForDisplay, buildTradeLinkLines, resolveTradeLinkEndpoints } from '@/lib/tradeMap/tradeLinks.ts';
import { visibleCandleIndexRange } from '@/lib/tradeMap/chartOverlay.ts';
import { barSecForTimeframe, orderOnBar, orderRowUnixSec } from '@/lib/tradeMap/orderTime.ts';
import {
  chopGridMetricsRowCell,
  chopGridMetricsRowSpecs,
  listStrategiesForLayers,
  resolveMetricsTableColumns,
  setFeatureTaxonomy,
  strategyMetricsRowSpecs,
} from '@/lib/tradeMap/features.ts';
import type { TradeMarker } from '@/lib/tradeMap/types.ts';

const sampleMarkers: TradeMarker[] = [
  { id: 'trend:orders:1', time: 1, scope: 'trend', event: 'entry', side: 'long', status: 'filled', strategy: 'tpc', symbol: 'ETH' },
  { id: 'multi_leg:orders:6', time: 6, scope: 'multi_leg', strategy: 'chop_grid', event: 'tp', side: 'long', status: 'filled', symbol: 'ETH', detail: { leg_label: 'S2_tp' } },
];

describe('tradeMap ohlcv', () => {
  it('scopesFromLayers', () => {
    expect(scopesFromLayers({ trend: true, spot: false, multiLeg: true })).toBe(
      'trend,multi_leg',
    );
  });

  it('sanitizeCandlesForLwc dedupes times', () => {
    const clean = sanitizeCandlesForLwc([
      { time: 100, open: 1, high: 2, low: 0.5, close: 1.5 },
      { time: 100, open: 9, high: 9, low: 9, close: 9 },
      { time: 200, open: 2, high: 3, low: 1, close: 2.5 },
    ]);
    expect(clean).toHaveLength(2);
    expect(clean[1].time).toBe(200);
  });

  it('mergeCandlesByTime keeps volume', () => {
    const merged = mergeCandlesByTime(
      [{ time: 100, open: 1, high: 2, low: 0.5, close: 1, volume: 42 }],
      [{ time: 100, open: 9, high: 9, low: 9, close: 9 }],
    );
    expect(merged[0].volume).toBe(42);
  });

  it('barSpacingForCount scales down for long history', () => {
    expect(barSpacingForCount(800)).toBeLessThan(barSpacingForCount(50));
  });

  it('defaultVisibleBarCount caps bars', () => {
    expect(defaultVisibleBarCount(5000)).toBeLessThan(5000);
  });

  it('logicalRangeAfterHistoryPrepend rejects range before setData length', () => {
    const snap = { from: 10, to: 50 };
    expect(logicalRangeAfterHistoryPrepend(snap, 360, 100)).toBeNull();
    expect(logicalRangeAfterHistoryPrepend(snap, 360, 460)).toEqual({ from: 370, to: 410 });
  });

  it('isValidLogicalRange guards out-of-bounds viewport', () => {
    expect(isValidLogicalRange({ from: 370, to: 410 }, 100)).toBe(false);
    expect(isValidLogicalRange({ from: 370, to: 410 }, 460)).toBe(true);
  });

  it('visibleLogicalRange handles single-bar series', () => {
    const lr = visibleLogicalRange(1);
    expect(lr).toEqual({ from: 0, to: 0.5 });
    expect(isValidLogicalRange(lr, 1)).toBe(true);
  });
});

describe('tradeMap markers', () => {
  it('markersToLwc hides text for mini grid', () => {
    const lwc = markersToLwc(sampleMarkers, null, { showText: false });
    expect(lwc.every((m) => m.text === '')).toBe(true);
  });

  it('markersToLwc produces shapes', () => {
    const lwc = markersToLwc(sampleMarkers, null);
    expect(lwc).toHaveLength(2);
    expect(lwc[1].shape).toBe('square');
  });

  it('markersForChartDisplay keeps selected id', () => {
    const scoped = markersForChartDisplay(sampleMarkers, 'tpc', 'multi_leg:orders:6');
    expect(scoped.some((m) => m.id === 'multi_leg:orders:6')).toBe(true);
  });

  it('snapMarkersToCandleTimes pins off-bar order times to nearest bar', () => {
    const candles = [
      { time: 200, open: 1, high: 2, low: 1, close: 1.5 },
      { time: 300, open: 1.5, high: 2, low: 1.4, close: 1.8 },
    ];
    const markers = [
      {
        id: 'trend:positions:p1:entry',
        time: 250,
        scope: 'trend',
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
      },
    ] as TradeMarker[];
    const snapped = snapMarkersToCandleTimes(markers, candles);
    expect(snapped[0].time).toBe(200);
    expect(snapped[0].detail?.order_time).toBe(250);
    const prepared = prepareChartMarkers(markers, candles, {}, { trend: true, spot: false, multiLeg: false }, 'tpc');
    expect(prepared[0].time).toBe(200);
  });

  it('isFeatureBusRegimeExitMarker detects feature_bus hysteresis exits', () => {
    expect(
      isFeatureBusRegimeExitMarker({
        id: 'multi_leg:regime_exit:BNB:1',
        detail: { source: 'feature_bus_hysteresis' },
      } as TradeMarker),
    ).toBe(true);
    expect(
      isFeatureBusRegimeExitMarker({
        id: 'multi_leg:orders:1',
        detail: { source: 'feature_bus_hysteresis' },
      } as TradeMarker),
    ).toBe(false);
  });
});

describe('tradeMap trade links', () => {
  it('tradeLinksForDisplay filters closed links by layer and focus', () => {
    const links = [
      {
        strategy: 'chop_grid',
        status: 'closed',
        entry_time: 1,
        entry_price: 100,
        exit_time: 2,
        exit_price: 101,
        exit_marker_id: 'a',
      },
      {
        strategy: 'tpc',
        status: 'open',
        entry_time: 1,
        entry_price: 100,
        exit_time: 2,
        exit_price: 101,
        exit_marker_id: 'b',
      },
    ];
    const shown = tradeLinksForDisplay(links, {
      trend: true,
      spot: true,
      multiLeg: true,
      pending: false,
      chopGrid: true,
      prefilter: true,
      gate: false,
    }, 'chop_grid');
    expect(shown).toHaveLength(1);
    expect(shown[0].strategy).toBe('chop_grid');
  });

  it('mergeTradeLinks dedupes by key', () => {
    const a = {
      strategy: 'chop_grid',
      entry_time: 1,
      entry_price: 1,
      exit_time: 2,
      exit_price: 2,
    };
    expect(mergeTradeLinks([a], [a])).toHaveLength(1);
  });

  it('compactMarkerLabel uses strategy slug not account layer', () => {
    expect(compactMarkerLabel({ strategy: 'tpc', scope: 'trend' } as TradeMarker)).toBe('tpc');
    expect(compactMarkerLabel({ strategy: 'chop_grid', scope: 'multi_leg' } as TradeMarker)).toBe(
      'chop',
    );
    expect(
      compactMarkerLabel({ strategy: 'spot_accum_simple', scope: 'spot' } as TradeMarker),
    ).toBe('spot');
  });

  it('prepareChartMarkers filters scopes and pending client-side', () => {
    const raw = [
      { id: '1', time: 100, scope: 'trend', strategy: 'tpc', status: 'filled' },
      { id: '2', time: 200, scope: 'spot', strategy: 'spot_accum_simple', status: 'filled' },
      { id: '3', time: 300, scope: 'multi_leg', strategy: 'chop_grid', status: 'pending' },
    ] as TradeMarker[];
    const candles = [
      { time: 100, open: 1, high: 1, low: 1, close: 1 },
      { time: 200, open: 1, high: 1, low: 1, close: 1 },
      { time: 300, open: 1, high: 1, low: 1, close: 1 },
    ];
    const trendOnly = prepareChartMarkers(raw, candles, null, {
      trend: true,
      spot: false,
      multiLeg: false,
      pending: true,
    }, '');
    expect(trendOnly.map((m) => m.id)).toEqual(['1']);
    const noPending = prepareChartMarkers(raw, candles, null, {
      trend: true,
      spot: true,
      multiLeg: true,
      pending: false,
    }, '');
    expect(noPending.map((m) => m.id)).toEqual(['1', '2']);
  });

  it('prepareChartMarkers filters by strategy focus and still applies scope layers', () => {
    const raw = [
      { id: '1', time: 100, scope: 'trend', strategy: 'tpc', status: 'filled' },
      { id: '2', time: 200, scope: 'multi_leg', strategy: 'trend_scalp', status: 'filled' },
      { id: '3', time: 300, scope: 'multi_leg', strategy: 'chop_grid', status: 'filled' },
    ] as TradeMarker[];
    const candles = [
      { time: 100, open: 1, high: 1, low: 1, close: 1 },
      { time: 200, open: 1, high: 1, low: 1, close: 1 },
      { time: 300, open: 1, high: 1, low: 1, close: 1 },
    ];
    const trendScalpOnly = prepareChartMarkers(raw, candles, null, {
      trend: true,
      spot: false,
      multiLeg: true,
      pending: false,
    }, 'trend_scalp');
    expect(trendScalpOnly.map((m) => m.id)).toEqual(['2']);
  });

  it('dedupeMarkersForChart keeps positions over orders on same bar', () => {
    const markers = [
      {
        id: 'trend:positions:p1:entry',
        time: 1000,
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
        scope: 'trend',
      },
      {
        id: 'trend:orders:o1',
        time: 1000,
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
        scope: 'trend',
      },
      {
        id: 'trend:orders:o2',
        time: 1000,
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
        scope: 'trend',
      },
    ] as TradeMarker[];
    const deduped = dedupeMarkersForChart(markers);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].id).toBe('trend:positions:p1:entry');
  });

  it('buildTradeLinkLines snaps endpoints onto loaded candle bars', () => {
    const candles = [
      { time: 1000, open: 1, high: 1, low: 1, close: 1 },
      { time: 8200, open: 1, high: 1, low: 1, close: 1 },
    ];
    const markers = [
      {
        id: 'trend:positions:p1:entry',
        time: 1500,
        symbol: 'XRP',
        scope: 'trend',
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
        price: 1.54,
      },
      {
        id: 'trend:positions:p1:exit',
        time: 7800,
        symbol: 'XRP',
        scope: 'trend',
        strategy: 'tpc',
        event: 'exit',
        side: 'long',
        price: 1.42,
      },
    ];
    const links = [
      {
        strategy: 'tpc',
        status: 'closed',
        entry_time: 1500,
        entry_price: 1.54,
        exit_time: 7800,
        exit_price: 1.42,
        entry_marker_id: 'trend:positions:p1:entry',
        exit_marker_id: 'trend:positions:p1:exit',
      },
    ];
    const layers = {
      trend: true,
      spot: true,
      multiLeg: true,
      pending: false,
      chopGrid: true,
      prefilter: true,
      gate: false,
    };
    const resolved = resolveTradeLinkEndpoints(links[0], markers, candles, '2h');
    expect(resolved?.entry.time).toBe(1000);
    expect(resolved?.exit.time).toBe(8200);
    expect(resolved?.entry.value).toBe(1);
    expect(resolved?.exit.value).toBe(1);
    const lines = buildTradeLinkLines(links, candles, layers, 'tpc', '2h', markers);
    expect(lines).toHaveLength(1);
    expect(lines[0].points).toEqual([
      { time: 1000, value: 1 },
      { time: 8200, value: 1 },
    ]);
  });

  it('buildTradeLinkLines keeps same-bar links vertical', () => {
    const candles = [{ time: 1000, open: 1, high: 1, low: 1, close: 1 }];
    const links = [
      {
        strategy: 'tpc',
        status: 'closed',
        entry_time: 1000,
        entry_price: 1.5,
        exit_time: 1000,
        exit_price: 1.4,
        entry_marker_id: 'trend:positions:p1:entry',
        exit_marker_id: 'trend:positions:p1:exit',
      },
    ];
    const markers = [
      {
        id: 'trend:positions:p1:entry',
        time: 1000,
        symbol: 'XRP',
        scope: 'trend',
        strategy: 'tpc',
        event: 'entry',
        side: 'long',
        price: 1.5,
      },
      {
        id: 'trend:positions:p1:exit',
        time: 1000,
        symbol: 'XRP',
        scope: 'trend',
        strategy: 'tpc',
        event: 'exit',
        side: 'long',
        price: 1.4,
      },
    ];
    const layers = {
      trend: true,
      spot: false,
      multiLeg: false,
      pending: false,
      chopGrid: false,
      prefilter: false,
      gate: false,
    };
    const lines = buildTradeLinkLines(links, candles, layers, 'tpc', '2h', markers);
    expect(lines[0].points[0].time).toBe(lines[0].points[1].time);
    expect(lines[0].points[0].value).toBe(1);
    expect(lines[0].points[1].value).toBe(1);
  });
});

describe('tradeMap chop regime hysteresis', () => {
  const candles = [{ time: 1000 }, { time: 2000 }, { time: 3000 }, { time: 4000 }];
  const overlaysStale = {
    bpc_semantic_chop: {
      points: [
        { time: 1000, value: 0.55 },
        { time: 2000, value: 0.55 },
      ],
      reference_lines: [{ y: 0.5, operator: '>=' }, { y: 0.32, operator: '<' }],
    },
  };

  it('chopGridHysteresisActive matches finite series', () => {
    const vals = [0.2, 0.55, 0.45, 0.25, 0.35];
    expect(chopGridHysteresisActive(vals, 0.5, 0.32)).toEqual([
      false,
      true,
      true,
      false,
      false,
    ]);
  });

  it('stale chop overlay exits after last feature point', () => {
    const { vals, chopOn } = chopRegimeSeriesFromOverlay(candles, overlaysStale);
    expect(vals).toEqual([0.55, 0.55, null, null]);
    expect(chopOn).toEqual([true, true, false, false]);
    expect([...chopRegimeExitBarTimes(candles, overlaysStale)]).toEqual([3000]);
    expect(chopRegimeHysteresisOnAtTime(candles, overlaysStale, 2000)).toBe(true);
    expect(chopRegimeHysteresisOnAtTime(candles, overlaysStale, 3000)).toBe(false);
    expect(chopRegimeHysteresisOnAtTime(candles, overlaysStale, 4000)).toBe(false);

    const rows = chopGridMetricsRowSpecs(['bpc_semantic_chop'], overlaysStale);
    const chopRow = rows.find((r) => r.column === 'bpc_semantic_chop');
    expect(chopRow).toBeTruthy();
    expect(chopGridMetricsRowCell(chopRow!, overlaysStale, 2000, candles)).toMatchObject({
      value: '0.550',
      pass: true,
    });
    expect(chopGridMetricsRowCell(chopRow!, overlaysStale, 4000, candles)).toMatchObject({
      value: '—',
      pass: null,
    });

    const ff = forwardFillOverlayToCandles(
      overlaysStale.bpc_semantic_chop.points,
      candles,
    );
    const asof = overlayAsOfAtCandleTimes(
      overlaysStale.bpc_semantic_chop.points,
      candles,
    );
    expect(ff[ff.length - 1].value).toBe(0.55);
    expect(asof[asof.length - 1].value).toBeNull();
  });
});

describe('visibleCandleIndexRange', () => {
  const candles = Array.from({ length: 200 }, (_, i) => ({
    time: 1_700_000_000 + i * 7200,
    open: 1,
    high: 2,
    low: 0.5,
    close: 1.5,
  }));

  it('defaults to tail window when logical range is null', () => {
    const { from, to } = visibleCandleIndexRange(candles, null, 80);
    expect(to).toBe(199);
    expect(from).toBe(120);
  });

  it('keeps tail when logical window is wider than cap', () => {
    const { from, to } = visibleCandleIndexRange(candles, { from: 0, to: 199 }, 80);
    expect(to).toBe(199);
    expect(from).toBe(120);
  });

  it('follows narrow logical window', () => {
    const { from, to } = visibleCandleIndexRange(candles, { from: 10, to: 20 }, 80);
    expect(from).toBe(10);
    expect(to).toBe(20);
  });
});

describe('listStrategiesForLayers', () => {
  it('falls back to KNOWN_STRATEGIES when taxonomy is not loaded', () => {
    setFeatureTaxonomy(null);
    const multiOnly = listStrategiesForLayers({
      trend: false,
      spot: false,
      multiLeg: true,
    });
    expect(multiOnly.map((s) => s.id)).toEqual(['chop_grid', 'trend_scalp']);
  });
});

describe('metrics table columns', () => {
  it('defaults TPC rows without selected columns or bus catalog', () => {
    setFeatureTaxonomy(null);
    const cols = resolveMetricsTableColumns('tpc', [], []);
    expect(cols).toEqual([
      'tpc_pullback_depth',
      'ema_1200_position',
      'macd_atr',
      'tpc_semantic_chop',
      'tpc_vol_pullback_confirm',
    ]);
    const rows = strategyMetricsRowSpecs('tpc', [], {});
    expect(rows.map((r) => r.column)).toEqual(cols);
  });

  it('defaults chop_grid metrics rows without selection', () => {
    setFeatureTaxonomy(null);
    const rows = chopGridMetricsRowSpecs([], {});
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.some((r) => r.column === 'bpc_semantic_chop')).toBe(true);
  });
});

describe('overlayValueAtCandle', () => {
  it('forward-fills sparse feature points onto chart bars', () => {
    const candles = [
      { time: 1000, open: 1, high: 1, low: 1, close: 1 },
      { time: 2000, open: 1, high: 1, low: 1, close: 1 },
      { time: 3000, open: 1, high: 1, low: 1, close: 1 },
    ];
    const points = [{ time: 1000, value: 0.42 }];
    expect(overlayValueAtCandle(points, candles, 2000)).toBe(0.42);
    expect(forwardFillOverlayToCandles(points, candles)).toHaveLength(3);
  });
});

describe('orderTime', () => {
  it('parses ISO order timestamps to unix seconds', () => {
    const t = orderRowUnixSec({
      order_id: '1',
      symbol: 'ETH',
      scope: 'trend',
      filled_at: '2024-05-20T12:00:00Z',
    });
    expect(t).toBe(Math.floor(Date.parse('2024-05-20T12:00:00Z') / 1000));
  });

  it('matches order to chart bar time', () => {
    const bar = 1_715_000_000;
    const row = {
      order_id: '1',
      symbol: 'ETH',
      scope: 'trend',
      time: bar + 100,
    };
    expect(orderOnBar(row, bar, barSecForTimeframe('1h'))).toBe(true);
  });
});
