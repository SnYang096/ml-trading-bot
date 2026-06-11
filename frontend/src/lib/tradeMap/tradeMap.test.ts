import { describe, expect, it } from 'vitest';
import {
  mergeCandlesByTime,
  scopesFromLayers,
} from '@/lib/tradeMap/ohlcv.ts';
import {
  barSpacingForCount,
  defaultVisibleBarCount,
  sanitizeCandlesForLwc,
} from '@/lib/tradeMap/candles.ts';
import {
  chopGridHysteresisActive,
  chopRegimeExitBarTimes,
  chopRegimeHysteresisOnAtTime,
  chopRegimeSeriesFromOverlay,
  isFeatureBusRegimeExitMarker,
  markersForChartDisplay,
  markersToLwc,
} from '@/lib/tradeMap/markers.ts';
import { mergeTradeLinks, tradeLinksForDisplay } from '@/lib/tradeMap/tradeLinks.ts';
import { forwardFillOverlayToCandles, overlayAsOfAtCandleTimes } from '@/lib/tradeMap/ohlcv.ts';
import {
  chopGridMetricsRowCell,
  chopGridMetricsRowSpecs,
  listStrategiesForLayers,
  setFeatureTaxonomy,
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
