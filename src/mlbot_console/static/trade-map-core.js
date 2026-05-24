/**
 * Pure helpers for Trade Map (testable in Node and browser).
 */
(function (root) {
  const ENTRY_SHAPES = { long: "arrowUp", short: "arrowDown" };
  const EXIT_SHAPE = "circle";

  function scopesFromLayers(layers) {
    const parts = [];
    if (layers.trend) parts.push("trend");
    if (layers.spot) parts.push("spot");
    if (layers.multiLeg) parts.push("multi_leg");
    return parts.join(",") || "trend,spot";
  }

  function markerRole(marker) {
    const ev = String(marker.event || "").toLowerCase();
    if (ev === "tp") return "tp";
    if (ev === "grid") return "grid";
    if (ev === "exit") return "exit";
    return "entry";
  }

  function markerShape(marker) {
    const role = markerRole(marker);
    if (role === "tp" || role === "exit") return EXIT_SHAPE;
    if (role === "grid") return "square";
    const side = (marker.side || "long").toLowerCase();
    if (marker.is_add && side === "long") return "diamond";
    if (marker.is_add && side === "short") return "diamond";
    return ENTRY_SHAPES[side] || "arrowUp";
  }

  function markerColor(marker) {
    const pending = (marker.status || "filled").toLowerCase() === "pending";
    if (pending) return "#888888";
    const role = markerRole(marker);
    const side = (marker.side || "long").toLowerCase();
    const pnl = marker.pnl_usdt;
    if (role === "exit" && pnl != null) {
      return pnl >= 0 ? "#26a69a" : "#ef5350";
    }
    if (role === "tp") return marker.color || "#73BF69";
    if (role === "grid") return marker.color || "#73BF69";
    if (role === "entry") {
      return side === "long" ? "#2e7d32" : "#c62828";
    }
    return marker.color || "#3274D9";
  }

  function timeframeToleranceSec(timeframe) {
    const tf = String(timeframe || "2h").toLowerCase();
    if (tf === "1min") return 90;
    if (tf === "15min") return 900;
    if (tf === "1d") return 86400;
    if (tf === "1w") return 604800;
    return 7200;
  }

  /**
   * Initial bundle OHLCV query for a timeframe.
   * 1d/1w: Vision macro full history (no from/to; backend full_range).
   */
  function ohlcvInitialQueryRange(timeframe) {
    const tf = String(timeframe || "2h");
    if (tf === "1d" || tf === "1w") {
      return { full_range: "true" };
    }
    const days = tradeMapInitialDays(tf);
    const end = new Date();
    const start = new Date(end.getTime() - days * 86400000);
    return {
      from: start.toISOString(),
      to: end.toISOString(),
      full_range: "true",
    };
  }

  /** Default OHLCV window (days) — keep in sync with TRADE_MAP_INITIAL_DAYS. */
  function tradeMapInitialDays(timeframe) {
    const tf = String(timeframe || "2h");
    const map = {
      "15min": 14,
      "2h": 60,
      "120T": 60,
      "1d": 120,
      "1w": 365,
    };
    return map[tf] ?? 60;
  }

  /** One pan-left prefetch chunk (days). */
  function tradeMapHistoryChunkDays(timeframe) {
    const tf = String(timeframe || "2h");
    const map = {
      "15min": 7,
      "2h": 30,
      "120T": 30,
      "1d": 90,
      "1w": 180,
    };
    return map[tf] ?? 30;
  }

  function barDurationSec(timeframe) {
    const tf = String(timeframe || "2h").toLowerCase();
    if (tf === "15min") return 900;
    if (tf === "1d") return 86400;
    if (tf === "1w") return 604800;
    return 7200;
  }

  function mergeCandlesByTime(existing, incoming) {
    const byTime = new Map();
    for (const c of existing || []) {
      if (c && c.time != null) byTime.set(Number(c.time), c);
    }
    for (const c of incoming || []) {
      if (c && c.time != null) byTime.set(Number(c.time), c);
    }
    return [...byTime.values()].sort((a, b) => a.time - b.time);
  }

  /** Drop feature points outside the loaded OHLCV window (poll/history merge safety). */
  function clipOverlayPointsToCandles(points, candles) {
    if (!points?.length || !candles?.length) return points || [];
    const tMin = Number(candles[0].time);
    const tMax = Number(candles[candles.length - 1].time);
    if (!Number.isFinite(tMin) || !Number.isFinite(tMax)) return points;
    return points.filter((p) => {
      const t = Number(p?.time);
      return Number.isFinite(t) && t >= tMin && t <= tMax;
    });
  }

  /** One timeline entry per OHLCV bar (whitespace where feature is missing). */
  function alignSeriesToCandleTimes(points, candles) {
    if (!candles?.length) return points || [];
    const byTime = new Map();
    for (const p of points || []) {
      const t = Number(p?.time);
      if (!Number.isFinite(t)) continue;
      byTime.set(t, p?.value);
    }
    const out = [];
    for (const c of candles) {
      const t = Number(c?.time);
      if (!Number.isFinite(t)) continue;
      if (!byTime.has(t)) {
        out.push({ time: t });
        continue;
      }
      const v = byTime.get(t);
      if (v == null || (typeof v === "number" && v !== v)) {
        out.push({ time: t });
      } else {
        out.push({ time: t, value: Number(v) });
      }
    }
    return out;
  }

  function filterSubchartColumns(columns, layers, strategyFocus) {
    const focus = strategyFocus ? String(strategyFocus).trim() : "";
    return (columns || []).filter((col) => {
      const meta = lookupFeatureMeta(col);
      if (!isLayerEnabled(meta.account_layer, layers)) return false;
      if (focus && meta.strategy !== focus) return false;
      return true;
    });
  }

  function isoFromUnixSec(sec) {
    return new Date(Number(sec) * 1000).toISOString();
  }

  function stageRegionsQueryParam(prefilter, gate) {
    const parts = [];
    if (prefilter) parts.push("prefilter");
    if (gate) parts.push("gate");
    return parts.length ? parts.join(",") : "";
  }

  function mainOverlaysQueryParam(ema1200, weeklyEma200) {
    const parts = [];
    if (ema1200) parts.push("ema_1200");
    if (weeklyEma200) parts.push("weekly_ema_200");
    return parts.length ? parts.join(",") : "";
  }

  function findMarkerByTime(markers, clickTime, toleranceSec) {
    const t = Number(clickTime);
    if (!Number.isFinite(t)) return null;
    const tol = Number(toleranceSec) || 3600;
    let best = null;
    let bestDist = Infinity;
    for (const m of markers || []) {
      const mt = Number(m.time);
      if (!Number.isFinite(mt)) continue;
      const dist = Math.abs(mt - t);
      if (dist <= tol && dist < bestDist) {
        bestDist = dist;
        best = m;
      }
    }
    return best;
  }

  function chopGridMarkerDisplayText(m, pending) {
    const strat = (m.strategy || "").toLowerCase();
    if (strat !== "chop_grid") return "";
    const leg = String((m.detail && m.detail.leg_label) || "").trim().toUpperCase();
    if (!leg) return "";
    const ev = String(m.event || "").toLowerCase();
    if (ev === "tp") return leg.endsWith("_TP") ? leg : `${leg}_TP`;
    if (ev === "entry" && !pending) return `${leg} 成交`;
    if (pending || ev === "grid") return `${leg} 挂单`;
    return leg;
  }

  function chopGridLegSide(legLabel) {
    const leg = String(legLabel || "").toUpperCase();
    const m = leg.match(/(?:^|_)([LS])(\d+)/);
    if (!m) return null;
    return m[1] === "L" ? "long" : "short";
  }

  /** Long grid labels below price line; short above; long TP above; short TP below. */
  function chopGridLabelAnchor(side, kind) {
    if (kind === "center") return "below";
    const isLong = String(side || "").toLowerCase() === "long";
    const isTp = kind === "tp";
    if (isTp) return isLong ? "above" : "below";
    return isLong ? "below" : "above";
  }

  function chopSegmentedLinePoints(regions, price, barSec) {
    const px = Number(price);
    if (!Number.isFinite(px) || !regions?.length) return [];
    const gap = Math.max(1, Number(barSec) || 7200);
    const pts = [];
    const sorted = [...regions].sort((a, b) => Number(a.start) - Number(b.start));
    for (const r of sorted) {
      const start = Number(r.start);
      const end = Number(r.end);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      pts.push({ time: start, value: px });
      pts.push({ time: end, value: px });
      pts.push({ time: end + gap, value: NaN });
    }
    return pts;
  }

  function markersToLwc(markers, selectedId) {
    return (markers || []).map((m) => {
      const role = markerRole(m);
      const pending = (m.status || "filled").toLowerCase() === "pending";
      const selected = selectedId && m.id === selectedId;
      const strat = (m.strategy || m.scope || "").toLowerCase();
      const chopText = chopGridMarkerDisplayText(m, pending);
      const leg =
        (m.detail && (m.detail.leg_label || m.detail.leg_id)) || "";
      let legToken = "";
      if (leg) {
        const parts = String(leg).split("_").filter(Boolean);
        legToken = parts[parts.length - 1] || "";
        if (legToken.toLowerCase() === String(m.event || "").toLowerCase()) {
          legToken = parts[parts.length - 2] || "";
        }
      }
      const legTag = legToken ? `:${legToken}` : "";
      const baseText = chopText
        ? chopText
        : `${strat}:${m.event}${legTag}${pending ? ":pending" : ""}`;
      let aboveBar = role === "exit" || role === "tp";
      if (strat === "chop_grid") {
        const legSide = chopGridLegSide(
          (m.detail && m.detail.leg_label) || (m.detail && m.detail.leg_id) || legToken
        );
        if (role === "tp") {
          if (legSide === "short") aboveBar = false;
          else if (legSide === "long") aboveBar = true;
        } else if (role === "entry" && !pending) {
          // Filled grid entries: short (S1/S2) above bar, long below.
          if (legSide === "short") aboveBar = true;
          else if (legSide === "long") aboveBar = false;
        }
      }
      return {
        time: m.time,
        position: aboveBar ? "aboveBar" : "belowBar",
        color: selected ? "#ffeb3b" : markerColor(m),
        shape: pending ? "circle" : markerShape(m),
        text: selected ? `★ ${baseText}` : baseText,
        id: m.id,
      };
    });
  }

  function scrollIndexForTime(candles, targetTime) {
    const t = Number(targetTime);
    if (!Array.isArray(candles) || !candles.length || !Number.isFinite(t)) {
      return -1;
    }
    let idx = candles.findIndex((c) => Number(c.time) >= t);
    if (idx < 0) idx = candles.length - 1;
    return idx;
  }

  function formatEligibility(elig) {
    if (!elig) return "—";
    const lines = [
      `can_buy: ${elig.can_buy}`,
      `weekly_ema_200_position: ${elig.weekly_ema_200_position ?? "n/a"}`,
      `blockers: ${(elig.blockers || []).join(", ") || "none"}`,
    ];
    return lines.join("\n");
  }

  function browserLocalUrl(port, path) {
    const host =
      (typeof globalThis !== "undefined" &&
        globalThis.location &&
        globalThis.location.hostname) ||
      "127.0.0.1";
    return `http://${host}:${port}${path || ""}`;
  }

  function resolveLinkUrl(link) {
    if (link && link.id === "grafana") {
      return browserLocalUrl(3000);
    }
    return (link && link.url) || "";
  }

  const SUBCHART_COLORS = [
    "#ffeb3b",
    "#58a6ff",
    "#f78166",
    "#7ee787",
    "#d2a8ff",
    "#ffa657",
  ];

  function subchartColor(index) {
    return SUBCHART_COLORS[Math.abs(index) % SUBCHART_COLORS.length];
  }

  function featureColumnsParam(selected) {
    return (selected || []).filter(Boolean).join(",");
  }

  function parseStoredLayout(raw) {
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (_) {
      return null;
    }
  }

  function filterFeatureColumns(columns, query) {
    const q = String(query || "")
      .trim()
      .toLowerCase();
    if (!q) return columns || [];
    return (columns || []).filter((c) => String(c).toLowerCase().includes(q));
  }

  /** Strategies selectable in the feature drawer for the enabled account layers. */
  function listStrategiesForLayers(layers) {
    if (featureTaxonomy && featureTaxonomy.strategies) {
      return featureTaxonomy.strategies.filter((s) =>
        isLayerEnabled(s.account_layer, layers)
      );
    }
    return [];
  }

  /**
   * Picker columns: optional single-strategy focus; otherwise enabled layers only (no shared flood).
   */
  function filterColumnsForFeaturePicker(columns, layers, strategyFocus) {
    const focus = strategyFocus ? String(strategyFocus).trim() : "";
    return (columns || []).filter((col) => {
      const m = lookupFeatureMeta(col);
      if (focus) {
        return m.strategy === focus;
      }
      if (m.account_layer === "shared" || m.strategy === "shared") {
        return false;
      }
      return isLayerEnabled(m.account_layer, layers);
    });
  }

  /** Drop selected columns whose owning account_layer is now disabled (keeps shared cols). */
  function filterSelectedFeaturesByLayers(columns, layers) {
    return (columns || []).filter((col) => {
      const m = lookupFeatureMeta(col);
      const layer = m && m.account_layer ? m.account_layer : "shared";
      if (layer === "shared") return true;
      return isLayerEnabled(layer, layers);
    });
  }

  function inferStrategyFocusFromLayers(layers) {
    const enabled = ACCOUNT_LAYER_ORDER.filter((id) => isLayerEnabled(id, layers));
    if (enabled.length !== 1) return null;
    if (enabled[0] === "multi_leg") return "chop_grid";
    if (enabled[0] === "spot") return "spot_accum_simple";
    return null;
  }

  function strategyFocusLabel(strategyId) {
    if (!strategyId) return "全部（当前账户层）";
    return strategyMeta(strategyId).title || strategyId;
  }

  const ACCOUNT_LAYER_ORDER = ["trend", "spot", "multi_leg", "shared"];
  const STAGE_ORDER = [
    "regime",
    "prefilter",
    "direction",
    "gate",
    "entry",
    "evidence",
    "execution",
  ];

  const ACCOUNT_LAYER_META = {
    trend: { id: "trend", title: "B·Trend", layerKey: "trend" },
    spot: { id: "spot", title: "A·Spot", layerKey: "spot" },
    multi_leg: { id: "multi_leg", title: "C·Multi-leg", layerKey: "multiLeg" },
    shared: { id: "shared", title: "未归类", layerKey: null },
  };

  let featureTaxonomy = null;

  function setFeatureTaxonomy(taxonomy) {
    featureTaxonomy = taxonomy && typeof taxonomy === "object" ? taxonomy : null;
  }

  function taxonomyIndex() {
    return (featureTaxonomy && featureTaxonomy.index) || {};
  }

  function layerKeyForAccount(accountLayer) {
    const m = ACCOUNT_LAYER_META[accountLayer];
    return m ? m.layerKey : null;
  }

  function isLayerEnabled(accountLayer, layers) {
    const key = layerKeyForAccount(accountLayer);
    if (!key || !layers) return true;
    return layers[key] !== false;
  }

  /** Account layer when strategy slug is unknown. */
  function classifyFeatureColumn(column) {
    return inferStrategyIdFromColumn(column).account_layer;
  }

  /** Map feature column name -> { strategy, account_layer } when not in YAML index. */
  function inferStrategyIdFromColumn(column) {
    const lc = String(column || "").toLowerCase();
    if (
      lc.includes("weekly_ema") ||
      lc.startsWith("spot_") ||
      lc.includes("can_buy") ||
      lc.includes("spot_accum")
    ) {
      return { strategy: "spot_accum_simple", account_layer: "spot" };
    }
    if (lc.startsWith("tpc_")) return { strategy: "tpc", account_layer: "trend" };
    if (lc.startsWith("fer_")) return { strategy: "fer", account_layer: "trend" };
    if (lc.startsWith("me_")) return { strategy: "me", account_layer: "trend" };
    if (lc.startsWith("srb_")) return { strategy: "srb", account_layer: "trend" };
    if (lc.startsWith("bpc_") && !lc.includes("chop")) {
      return { strategy: "bpc", account_layer: "trend" };
    }
    if (
      lc.startsWith("chop_") ||
      (lc.includes("semantic_chop") && !lc.startsWith("tpc_")) ||
      lc.includes("grid") ||
      lc.includes("vol_clustering") ||
      lc.startsWith("box_pos_60") ||
      lc.startsWith("box_stability_60") ||
      lc.startsWith("box_width_pct_60") ||
      lc.startsWith("box_touches_")
    ) {
      return { strategy: "chop_grid", account_layer: "multi_leg" };
    }
    if (
      lc.startsWith("vpin") ||
      lc === "trend_confidence" ||
      lc.startsWith("trend_confidence")
    ) {
      return { strategy: "trend_scalp", account_layer: "multi_leg" };
    }
    if (
      lc.includes("trend_div") ||
      lc.startsWith("ema_1200") ||
      lc.startsWith("macd_atr") ||
      lc.startsWith("box_pos_120") ||
      lc.startsWith("box_breakout")
    ) {
      return { strategy: "tpc", account_layer: "trend" };
    }
    return { strategy: "shared", account_layer: "shared" };
  }

  function lookupFeatureMeta(column) {
    const col = String(column || "");
    const idx = taxonomyIndex();
    const hits = idx[col] || (col.endsWith("_f") ? idx[col.slice(0, -2)] : null);
    if (hits && hits.length) return hits[0];
    const inferred = inferStrategyIdFromColumn(col);
    const layer = inferred.account_layer;
    const layerMeta = ACCOUNT_LAYER_META[layer] || ACCOUNT_LAYER_META.shared;
    const sm = strategyMeta(inferred.strategy);
    return {
      column: col,
      account_layer: layer,
      account_layer_title: layerMeta.title,
      strategy: inferred.strategy,
      strategy_title: sm.title || inferred.strategy,
      stage: "other",
      stage_title: "其他",
    };
  }

  function strategyMeta(strategyId) {
    if (featureTaxonomy && featureTaxonomy.strategies) {
      const hit = featureTaxonomy.strategies.find((s) => s.id === strategyId);
      if (hit) {
        return {
          id: hit.id,
          title: hit.title,
          layerKey: layerKeyForAccount(hit.account_layer),
          account_layer: hit.account_layer,
        };
      }
    }
    const layer = strategyId === "shared" ? "shared" : classifyFeatureColumn(strategyId);
    const lm = ACCOUNT_LAYER_META[layer] || ACCOUNT_LAYER_META.shared;
    return { id: strategyId, title: strategyId, layerKey: lm.layerKey, account_layer: layer };
  }

  function _bucketColumnsByTaxonomy(columns) {
    /** @type {Record<string, Record<string, Record<string, string[]>>>} */
    const tree = {};
    for (const col of columns || []) {
      const meta = lookupFeatureMeta(col);
      const layer = meta.account_layer || "shared";
      const strat = meta.strategy || "unknown";
      const stage = meta.stage || "other";
      if (!tree[layer]) tree[layer] = {};
      if (!tree[layer][strat]) tree[layer][strat] = {};
      if (!tree[layer][strat][stage]) tree[layer][strat][stage] = [];
      tree[layer][strat][stage].push(String(col));
    }
    for (const layer of Object.keys(tree)) {
      for (const strat of Object.keys(tree[layer])) {
        for (const stage of Object.keys(tree[layer][strat])) {
          tree[layer][strat][stage].sort();
        }
      }
    }
    return tree;
  }

  function _strategyOrderForLayer(layerId) {
    if (featureTaxonomy && featureTaxonomy.strategies) {
      return featureTaxonomy.strategies
        .filter((s) => s.account_layer === layerId)
        .map((s) => s.id);
    }
    return [];
  }

  function groupFeatureColumnsByStrategy(columns, layers) {
    const tree = _bucketColumnsByTaxonomy(columns);
    const out = [];
    for (const layerId of ACCOUNT_LAYER_ORDER) {
      if (!isLayerEnabled(layerId, layers)) continue;
      const layerNode = tree[layerId];
      if (!layerNode) continue;
      const layerTitle = (ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
      const stratOrder = _strategyOrderForLayer(layerId);
      const stratIds = [
        ...stratOrder.filter((id) => layerNode[id]),
        ...Object.keys(layerNode).filter((id) => !stratOrder.includes(id)),
      ];
      for (const stratId of stratIds) {
        const stageNode = layerNode[stratId];
        if (!stageNode) continue;
        const stratTitle =
          (lookupFeatureMeta(stageNode[Object.keys(stageNode)[0]][0]) || {}).strategy_title ||
          stratId;
        const stages = featureTaxonomy && featureTaxonomy.stage_order
          ? featureTaxonomy.stage_order
          : STAGE_ORDER;
        for (const stage of [...stages, "other"]) {
          const cols = stageNode[stage];
          if (!cols || !cols.length) continue;
          const stageTitle =
            (featureTaxonomy &&
              featureTaxonomy.stage_labels &&
              featureTaxonomy.stage_labels[stage]) ||
            stage;
          out.push([
            `${layerTitle} › ${stratTitle} › ${stageTitle}`,
            cols,
            { layer: layerId, strategy: stratId, stage },
          ]);
        }
      }
    }
    const sharedNode = tree.shared;
    if (sharedNode && isLayerEnabled("shared", layers)) {
      for (const stratId of Object.keys(sharedNode)) {
        for (const stage of Object.keys(sharedNode[stratId])) {
          const cols = sharedNode[stratId][stage];
          if (!cols.length) continue;
          out.push([`未归类 › ${stage}`, cols, { layer: "shared", strategy: stratId, stage }]);
        }
      }
    }
    return out;
  }

  function groupFeatureColumns(columns) {
    return groupFeatureColumnsByStrategy(columns, {
      trend: true,
      spot: true,
      multiLeg: true,
    });
  }

  function orderFeaturePaneItems(columns, layers, strategyFocus) {
    const focus = strategyFocus ? String(strategyFocus).trim() : "";
    const tree = _bucketColumnsByTaxonomy(columns);
    const items = [];
    let firstLayer = true;
    for (const layerId of ACCOUNT_LAYER_ORDER) {
      if (!isLayerEnabled(layerId, layers)) continue;
      const layerNode = tree[layerId];
      if (!layerNode) continue;
      if (!firstLayer) items.push({ type: "gap", id: `gap-layer-${layerId}` });
      firstLayer = false;
      const layerTitle = (ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
      items.push({
        type: "header",
        strategy: layerId,
        title: layerTitle,
        headerKind: "layer",
      });
      const stratOrder = _strategyOrderForLayer(layerId);
      const stratIds = [
        ...stratOrder.filter((id) => layerNode[id]),
        ...Object.keys(layerNode).filter((id) => !stratOrder.includes(id)),
      ];
      let firstStrat = true;
      for (const stratId of stratIds) {
        if (focus && stratId !== focus) continue;
        const stageNode = layerNode[stratId];
        if (!stageNode) continue;
        if (!firstStrat) items.push({ type: "gap", id: `gap-strat-${layerId}-${stratId}` });
        firstStrat = false;
        const sample = stageNode[Object.keys(stageNode)[0]][0];
        const sm = lookupFeatureMeta(sample);
        items.push({
          type: "header",
          strategy: stratId,
          title: sm.strategy_title || stratId,
          headerKind: "strategy",
          accountLayer: layerId,
        });
        const stages = featureTaxonomy && featureTaxonomy.stage_order
          ? featureTaxonomy.stage_order
          : STAGE_ORDER;
        for (const stage of [...stages, "other"]) {
          const cols = stageNode[stage];
          if (!cols || !cols.length) continue;
          const stageTitle =
            (featureTaxonomy &&
              featureTaxonomy.stage_labels &&
              featureTaxonomy.stage_labels[stage]) ||
            stage;
          items.push({
            type: "header",
            strategy: stratId,
            title: stageTitle,
            headerKind: "stage",
            accountLayer: layerId,
            stage,
          });
          for (const col of cols) {
            items.push({
              type: "feature",
              column: col,
              strategy: stratId,
              accountLayer: layerId,
              stage,
            });
          }
        }
      }
    }
    return items;
  }

  function presetColumnsForStrategy(strategyId, available, maxCols) {
    const sid = String(strategyId || "").toLowerCase();
    const avail = new Set(available || []);
    const picks = [];
    if (sid === "chop_grid") {
      for (const c of [
        "bpc_semantic_chop",
        "box_pos_60",
        "box_stability_60",
        "box_width_pct_60",
      ]) {
        if (avail.has(c) && !picks.includes(c)) picks.push(c);
        if (picks.length >= maxCols) return picks;
      }
    }
    if (featureTaxonomy && featureTaxonomy.strategies) {
      const strat = featureTaxonomy.strategies.find((s) => s.id === sid);
      if (strat) {
        for (const stage of ["regime", "prefilter", "direction", "gate", "entry", "evidence"]) {
          for (const c of (strat.stages && strat.stages[stage]) || []) {
            if (avail.has(c) && !picks.includes(c)) picks.push(c);
            if (picks.length >= maxCols) return picks;
          }
        }
      }
    }
    return picks;
  }

  function presetColumnsForAccountLayer(layerId, available, maxCols) {
    const avail = new Set(available || []);
    const picks = [];
    if (layerId === "multi_leg") {
      return presetColumnsForStrategy("chop_grid", available, maxCols);
    }
    if (featureTaxonomy && featureTaxonomy.strategies) {
      for (const s of featureTaxonomy.strategies) {
        if (s.account_layer !== layerId) continue;
        for (const stage of ["regime", "prefilter", "gate", "entry"]) {
          for (const c of (s.stages && s.stages[stage]) || []) {
            if (avail.has(c) && !picks.includes(c)) picks.push(c);
            if (picks.length >= maxCols) return picks;
          }
        }
      }
    }
    return picks;
  }

  const DEFAULT_VISIBLE_BARS = 320;

  /** How many bars to show by default (tail window); avoids fitContent squashing 2k+ bars to 0px. */
  function defaultVisibleBarCount(barCount, cap) {
    const n = Math.max(0, Number(barCount) || 0);
    if (n <= 0) return 0;
    const limit = Number(cap) > 0 ? Number(cap) : DEFAULT_VISIBLE_BARS;
    return Math.min(n, Math.max(30, limit));
  }

  function visibleLogicalRange(barCount, visibleBars) {
    const n = Math.max(0, Number(barCount) || 0);
    if (n <= 0) return null;
    const vis = defaultVisibleBarCount(n, visibleBars);
    return { from: Math.max(0, n - vis), to: n - 1 };
  }

  function clampCandleOhlc(open, high, low, close) {
    let o = open;
    let h = high;
    let l = low;
    let c = close;
    if (!Number.isFinite(o)) o = c;
    if (!Number.isFinite(h)) h = Math.max(o, c);
    if (!Number.isFinite(l)) l = Math.min(o, c);
    if (l < 0) l = Math.min(o, c);
    if (h < l) {
      const t = h;
      h = l;
      l = t;
    }
    const ref = Math.max(Math.abs(c), Math.abs(o), 1);
    const wickCap = Math.max(ref * 0.35, 5);
    if (h > c + wickCap * 8) h = Math.max(o, c);
    if (l < c - wickCap * 8) l = Math.min(o, c);
    if (h < Math.max(o, c)) h = Math.max(o, c);
    if (l > Math.min(o, c)) l = Math.min(o, c);
    return { open: o, high: h, low: l, close: c };
  }

  function sanitizeCandlesForLwc(candles) {
    if (!Array.isArray(candles) || !candles.length) return [];
    const out = [];
    let lastT = null;
    for (const raw of candles) {
      const time = Number(raw?.time);
      const close = Number(raw?.close);
      if (!Number.isFinite(time) || !Number.isFinite(close) || close <= 0) continue;
      if (lastT != null && time <= lastT) continue;
      lastT = time;
      const ohlc = clampCandleOhlc(
        Number(raw?.open),
        Number(raw?.high),
        Number(raw?.low),
        close
      );
      const c = { time, ...ohlc };
      if (raw?.volume != null && Number.isFinite(Number(raw.volume))) {
        c.volume = Number(raw.volume);
      }
      out.push(c);
    }
    return out;
  }

  /** Min/max price for bars in the visible logical index window (for autoscale). */
  function priceRangeForVisibleCandles(candles, logicalRange) {
    if (!Array.isArray(candles) || !candles.length || !logicalRange) return null;
    const from = Math.max(0, Math.floor(Number(logicalRange.from)));
    const to = Math.min(candles.length - 1, Math.ceil(Number(logicalRange.to)));
    if (to < from) return null;
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = from; i <= to; i++) {
      const c = candles[i];
      if (!c) continue;
      const lo = Number(c.low);
      const hi = Number(c.high);
      if (Number.isFinite(lo)) minV = Math.min(minV, lo);
      if (Number.isFinite(hi)) maxV = Math.max(maxV, hi);
    }
    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return null;
    const span = Math.max(maxV - minV, maxV * 0.0005);
    const pad = Math.max(span * 0.06, maxV * 0.001);
    return { minValue: minV - pad, maxValue: maxV + pad };
  }

  /** Visible-window range, else full series (never return null when candles exist). */
  function priceRangeForChartAutoscale(candles, logicalRange) {
    if (!Array.isArray(candles) || !candles.length) return null;
    const vis = priceRangeForVisibleCandles(candles, logicalRange);
    if (vis) return vis;
    return priceRangeForVisibleCandles(candles, {
      from: 0,
      to: candles.length - 1,
    });
  }

  /** Expand OHLC autoscale to include main-chart overlay values in the visible window. */
  function expandPriceRangeForOverlays(baseRange, candles, logicalRange, overlayDataByKey) {
    if (!baseRange || !overlayDataByKey || typeof overlayDataByKey.forEach !== "function") {
      return baseRange;
    }
    let minV = Number(baseRange.minValue);
    let maxV = Number(baseRange.maxValue);
    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return baseRange;
    const fromIdx =
      logicalRange && Number.isFinite(Number(logicalRange.from))
        ? Math.max(0, Math.floor(Number(logicalRange.from)))
        : 0;
    const toIdx =
      logicalRange && Number.isFinite(Number(logicalRange.to))
        ? Math.min(candles.length - 1, Math.ceil(Number(logicalRange.to)))
        : candles.length - 1;
    const tMin = Number(candles[fromIdx]?.time);
    const tMax = Number(candles[toIdx]?.time);
    overlayDataByKey.forEach((pts) => {
      for (const p of pts || []) {
        const t = Number(p.time);
        const v = Number(p.value);
        if (!Number.isFinite(v)) continue;
        if (Number.isFinite(tMin) && Number.isFinite(tMax) && (t < tMin || t > tMax)) continue;
        minV = Math.min(minV, v);
        maxV = Math.max(maxV, v);
      }
    });
    const pad = Math.max((maxV - minV) * 0.02, 1e-6);
    return { minValue: minV - pad, maxValue: maxV + pad };
  }

  /** Bar spacing in px for the *visible* window (not full history length). */
  function barSpacingForCount(barCount) {
    const n = Math.max(0, Number(barCount) || 0);
    if (n > 600) return 3;
    if (n > 300) return 4;
    if (n > 120) return 5;
    if (n > 50) return 6;
    return 8;
  }

  const FEATURE_PRESETS = {
    default: ["weekly_ema_200_position", "ema_1200_position"],
    trend: ["ema_1200_position", "tpc_pullback_depth", "tpc_semantic_chop", "bpc_pullback_depth"],
    spot: ["weekly_ema_200_position"],
    multi_leg: ["bpc_semantic_chop", "box_pos_60", "box_stability_60"],
  };

  root.MLBotTradeMapCore = {
    scopesFromLayers,
    markersToLwc,
    findMarkerByTime,
    timeframeToleranceSec,
    ohlcvInitialQueryRange,
    tradeMapInitialDays,
    tradeMapHistoryChunkDays,
    barDurationSec,
    mergeCandlesByTime,
    clipOverlayPointsToCandles,
    alignSeriesToCandleTimes,
    filterSubchartColumns,
    isoFromUnixSec,
    mainOverlaysQueryParam,
    stageRegionsQueryParam,
    scrollIndexForTime,
    markerColor,
    markerShape,
    formatEligibility,
    browserLocalUrl,
    resolveLinkUrl,
    subchartColor,
    featureColumnsParam,
    parseStoredLayout,
    filterFeatureColumns,
    filterColumnsForFeaturePicker,
    filterSelectedFeaturesByLayers,
    listStrategiesForLayers,
    inferStrategyFocusFromLayers,
    strategyFocusLabel,
    setFeatureTaxonomy,
    lookupFeatureMeta,
    classifyFeatureColumn,
    strategyMeta,
    groupFeatureColumns,
    groupFeatureColumnsByStrategy,
    orderFeaturePaneItems,
    presetColumnsForStrategy,
    presetColumnsForAccountLayer,
    inferStrategyIdFromColumn,
    defaultVisibleBarCount,
    visibleLogicalRange,
    sanitizeCandlesForLwc,
    clampCandleOhlc,
    priceRangeForVisibleCandles,
    priceRangeForChartAutoscale,
    expandPriceRangeForOverlays,
    barSpacingForCount,
    chopSegmentedLinePoints,
    chopGridLabelAnchor,
    chopGridMarkerDisplayText,
    DEFAULT_VISIBLE_BARS,
    FEATURE_PRESETS,
    ACCOUNT_LAYER_ORDER,
    STAGE_ORDER,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
