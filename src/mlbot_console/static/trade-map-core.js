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

  function markerShape(marker, isExit) {
    if (isExit) return EXIT_SHAPE;
    const side = (marker.side || "long").toLowerCase();
    if (marker.is_add && side === "long") return "diamond";
    if (marker.is_add && side === "short") return "diamond";
    return ENTRY_SHAPES[side] || "arrowUp";
  }

  function markerColor(marker, isExit) {
    const pending = (marker.status || "filled").toLowerCase() === "pending";
    if (pending) return "#888888";
    const side = (marker.side || "long").toLowerCase();
    const pnl = marker.pnl_usdt;
    if (isExit && pnl != null) {
      return pnl >= 0 ? "#26a69a" : "#ef5350";
    }
    if (!isExit) {
      return side === "long" ? "#2e7d32" : "#c62828";
    }
    return marker.color || "#3274D9";
  }

  function timeframeToleranceSec(timeframe) {
    const tf = String(timeframe || "2h").toLowerCase();
    if (tf === "1min") return 90;
    if (tf === "15min") return 900;
    if (tf === "1d") return 86400;
    return 7200;
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

  function markersToLwc(markers, selectedId) {
    return (markers || []).map((m) => {
      const isExit = m.event === "exit";
      const pending = (m.status || "filled").toLowerCase() === "pending";
      const selected = selectedId && m.id === selectedId;
      const baseText = `${m.scope}:${m.event}${pending ? ":pending" : ""}`;
      return {
        time: m.time,
        position: isExit ? "aboveBar" : "belowBar",
        color: selected ? "#ffeb3b" : markerColor(m, isExit),
        shape: pending ? "circle" : markerShape(m, isExit),
        text: selected ? `★ ${baseText}` : baseText,
        id: m.id,
        _raw: m,
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

  const ACCOUNT_LAYER_ORDER = ["trend", "spot", "multi_leg", "shared"];
  const STAGE_ORDER = [
    "prefilter",
    "direction",
    "gate",
    "entry",
    "evidence",
    "regime",
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

  /** Heuristic fallback when column is not in archetype taxonomy index. */
  function classifyFeatureColumn(column) {
    const lc = String(column || "").toLowerCase();
    if (
      lc.includes("weekly_ema") ||
      lc.startsWith("spot_") ||
      lc.includes("can_buy") ||
      lc.includes("spot_accum")
    ) {
      return "spot";
    }
    if (
      lc.startsWith("tpc_") ||
      lc.startsWith("fer_") ||
      lc.startsWith("me_") ||
      lc.startsWith("srb_") ||
      (lc.startsWith("bpc_") && !lc.includes("chop")) ||
      lc.includes("trend_div") ||
      lc.includes("trend_") ||
      lc.startsWith("ema_1200") ||
      lc.startsWith("macd_atr") ||
      lc.startsWith("box_pos_120") ||
      lc.startsWith("box_breakout")
    ) {
      return "trend";
    }
    if (
      lc.startsWith("chop_") ||
      lc.includes("semantic_chop") ||
      lc.startsWith("vpin") ||
      lc.includes("grid") ||
      lc.includes("vol_clustering") ||
      lc === "trend_confidence" ||
      lc.startsWith("box_pos_60")
    ) {
      return "multi_leg";
    }
    return "shared";
  }

  function lookupFeatureMeta(column) {
    const col = String(column || "");
    const idx = taxonomyIndex();
    const hits = idx[col] || (col.endsWith("_f") ? idx[col.slice(0, -2)] : null);
    if (hits && hits.length) return hits[0];
    const layer = classifyFeatureColumn(col);
    const layerMeta = ACCOUNT_LAYER_META[layer] || ACCOUNT_LAYER_META.shared;
    return {
      column: col,
      account_layer: layer,
      account_layer_title: layerMeta.title,
      strategy: layer === "shared" ? "shared" : "unknown",
      strategy_title: "未在 archetype 登记",
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

  function orderFeaturePaneItems(columns, layers) {
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

  function presetColumnsForAccountLayer(layerId, available, maxCols) {
    const avail = new Set(available || []);
    const picks = [];
    if (featureTaxonomy && featureTaxonomy.strategies) {
      for (const s of featureTaxonomy.strategies) {
        if (s.account_layer !== layerId) continue;
        for (const stage of ["prefilter", "regime", "gate", "entry"]) {
          for (const c of (s.stages && s.stages[stage]) || []) {
            if (avail.has(c) && !picks.includes(c)) picks.push(c);
            if (picks.length >= maxCols) return picks;
          }
        }
      }
    }
    return picks;
  }

  /** Tighter spacing => more bars visible on screen. */
  function barSpacingForCount(barCount) {
    const n = Math.max(0, Number(barCount) || 0);
    if (n > 800) return 1;
    if (n > 400) return 2;
    if (n > 200) return 3;
    if (n > 80) return 4;
    if (n > 30) return 5;
    return 6;
  }

  const FEATURE_PRESETS = {
    default: ["weekly_ema_200_position"],
    trend: ["tpc_pullback_depth", "tpc_semantic_chop", "bpc_pullback_depth"],
    spot: ["weekly_ema_200_position"],
    multi_leg: ["bpc_semantic_chop", "box_pos_60", "trend_confidence"],
  };

  root.MLBotTradeMapCore = {
    scopesFromLayers,
    markersToLwc,
    findMarkerByTime,
    timeframeToleranceSec,
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
    setFeatureTaxonomy,
    lookupFeatureMeta,
    classifyFeatureColumn,
    strategyMeta,
    groupFeatureColumns,
    groupFeatureColumnsByStrategy,
    orderFeaturePaneItems,
    presetColumnsForAccountLayer,
    barSpacingForCount,
    FEATURE_PRESETS,
    ACCOUNT_LAYER_ORDER,
    STAGE_ORDER,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
