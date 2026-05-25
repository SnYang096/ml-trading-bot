/**
 * Trade Map pure helpers — 40-features. Attaches to MLBotTradeMapCore (load in numeric order).
 */
(function (root) {
  const Core = (root.MLBotTradeMapCore = root.MLBotTradeMapCore || {});
    function isTrendScalpOtherPaneColumn(column, meta) {
      const stage = meta.stage || "other";
      if (stage !== "other") return false;
      const strat = meta.strategy || inferStrategyIdFromColumn(column).strategy;
      if (strat !== "trend_scalp") return false;
      const lc = String(column || "").toLowerCase();
      if (
        lc.includes("semantic_chop") ||
        lc.startsWith("box_") ||
        lc.includes("grid") ||
        lc.includes("scalp")
      ) {
        return false;
      }
      return true;
    }

    function filterSubchartColumns(columns, layers, strategyFocus) {
      const focus = strategyFocus ? String(strategyFocus).trim() : "";
      return (columns || []).filter((col) => {
        const meta = lookupFeatureMeta(col);
        if (!isLayerEnabled(meta.account_layer, layers)) return false;
        if (focus && meta.strategy !== focus) return false;
        if (isTrendScalpOtherPaneColumn(col, meta) && focus !== "trend_scalp") {
          return false;
        }
        return true;
      });
    }

    /**
     * Columns to render in feature subcharts: honor strategy focus, fall back to preset
     * when the current selection belongs to another strategy.
     */
    function resolveSubchartColumns(
      selectedColumns,
      availableColumns,
      layers,
      strategyFocus,
      maxCols
    ) {
      const max = Math.max(1, Number(maxCols) || 6);
      const selected = (selectedColumns || []).slice(0, max);
      const focus = strategyFocus ? String(strategyFocus).trim() : "";
      const filtered = filterSubchartColumns(selected, layers, focus);
      if (!focus) {
        return filtered.length
          ? filtered
          : filterSubchartColumns(selected, layers, null).slice(0, max);
      }
      const hasForeign = selected.some(
        (col) => lookupFeatureMeta(col).strategy !== focus
      );
      if (!hasForeign && filtered.length) return filtered;
      const pool = filterColumnsForFeaturePicker(availableColumns, layers, null);
      const preset = presetColumnsForStrategy(focus, pool, max);
      return preset.length ? preset : filtered;
    }

    function knownStrategyRecord(strategyId) {
      const sid = String(strategyId || "").trim().toLowerCase();
      if (!sid) return null;
      if (featureTaxonomy && featureTaxonomy.live_strategies) {
        const liveHit = featureTaxonomy.live_strategies.find(
          (s) => String(s.id).toLowerCase() === sid
        );
        if (liveHit) return liveHit;
      }
      if (featureTaxonomy && featureTaxonomy.strategies) {
        const hit = featureTaxonomy.strategies.find(
          (s) => String(s.id).toLowerCase() === sid
        );
        if (hit) return hit;
      }
      return (
        Core.KNOWN_STRATEGIES.find((s) => String(s.id).toLowerCase() === sid) || null
      );
    }

    function liveStrategyRecords() {
      const tax = featureTaxonomy;
      if (tax && Array.isArray(tax.live_strategies) && tax.live_strategies.length) {
        return tax.live_strategies;
      }
      const ids = tax && tax.live_strategy_ids;
      if (!Array.isArray(ids) || !ids.length) return [];
      const out = [];
      const seen = new Set();
      for (const rawId of ids) {
        const sid = String(rawId || "").trim().toLowerCase();
        if (!sid || seen.has(sid)) continue;
        const meta = knownStrategyRecord(sid);
        if (meta) {
          seen.add(sid);
          out.push(meta);
        }
      }
      return out;
    }

    function listStrategiesForLayers(layers) {
      const out = [];
      const seen = new Set();
      for (const meta of liveStrategyRecords()) {
        const sid = String(meta.id || "").trim().toLowerCase();
        if (!sid || seen.has(sid)) continue;
        if (!isLayerEnabled(meta.account_layer, layers)) continue;
        seen.add(sid);
        const layer = meta.account_layer;
        out.push({
          id: meta.id || sid,
          account_layer: layer,
          account_layer_title:
            meta.account_layer_title ||
            (Core.ACCOUNT_LAYER_META[layer] || {}).title ||
            layer,
          title: meta.title || sid,
          stages: meta.stages || {},
        });
      }
      return out.sort((a, b) => String(a.id).localeCompare(String(b.id)));
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
      const enabled = Core.ACCOUNT_LAYER_ORDER.filter((id) => isLayerEnabled(id, layers));
      if (enabled.length !== 1) return null;
      if (enabled[0] === "multi_leg") return "chop_grid";
      if (enabled[0] === "spot") return "spot_accum_simple";
      return null;
    }

    function strategyFocusLabel(strategyId) {
      if (!strategyId) return "全部（当前账户层）";
      return strategyMeta(strategyId).title || strategyId;
    }
    let featureTaxonomy = null;

    function setFeatureTaxonomy(taxonomy) {
      featureTaxonomy = taxonomy && typeof taxonomy === "object" ? taxonomy : null;
    }

    function getFeatureTaxonomy() {
      return featureTaxonomy;
    }

    function taxonomyIndex() {
      return (featureTaxonomy && featureTaxonomy.index) || {};
    }

    function layerKeyForAccount(accountLayer) {
      const m = Core.ACCOUNT_LAYER_META[accountLayer];
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
      if (lc === "chop_grid" || lc === "trend_scalp") {
        return {
          strategy: lc,
          account_layer: "multi_leg",
        };
      }
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
      const layerMeta = Core.ACCOUNT_LAYER_META[layer] || Core.ACCOUNT_LAYER_META.shared;
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
      const hit = knownStrategyRecord(strategyId);
      if (hit) {
        return {
          id: hit.id,
          title: hit.title,
          layerKey: layerKeyForAccount(hit.account_layer),
          account_layer: hit.account_layer,
        };
      }
      const layer = strategyId === "shared" ? "shared" : classifyFeatureColumn(strategyId);
      const lm = Core.ACCOUNT_LAYER_META[layer] || Core.ACCOUNT_LAYER_META.shared;
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
      for (const layerId of Core.ACCOUNT_LAYER_ORDER) {
        if (!isLayerEnabled(layerId, layers)) continue;
        const layerNode = tree[layerId];
        if (!layerNode) continue;
        const layerTitle = (Core.ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
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
            : Core.STAGE_ORDER;
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
      for (const layerId of Core.ACCOUNT_LAYER_ORDER) {
        if (!isLayerEnabled(layerId, layers)) continue;
        const layerNode = tree[layerId];
        if (!layerNode) continue;
        if (!firstLayer) items.push({ type: "gap", id: `gap-layer-${layerId}` });
        firstLayer = false;
        const layerTitle = (Core.ACCOUNT_LAYER_META[layerId] || {}).title || layerId;
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
            : Core.STAGE_ORDER;
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
        // Match prefilter.yaml: regime.entry_feature + rules (box_pos band).
        // regime.box_prefilter (stability/width/touches) is optional reference only.
        for (const c of ["bpc_semantic_chop", "box_pos_60"]) {
          if (avail.has(c) && !picks.includes(c)) picks.push(c);
          if (picks.length >= maxCols) return picks;
        }
      }
      if (sid === "trend_scalp") {
        for (const c of ["trend_confidence", "bpc_semantic_chop"]) {
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

  Object.assign(Core, {
    isTrendScalpOtherPaneColumn,
    filterSubchartColumns,
    resolveSubchartColumns,
    listStrategiesForLayers,
    filterColumnsForFeaturePicker,
    filterSelectedFeaturesByLayers,
    inferStrategyFocusFromLayers,
    strategyFocusLabel,
    setFeatureTaxonomy,
    getFeatureTaxonomy,
    lookupFeatureMeta,
    classifyFeatureColumn,
    strategyMeta,
    groupFeatureColumns,
    groupFeatureColumnsByStrategy,
    orderFeaturePaneItems,
    presetColumnsForStrategy,
    presetColumnsForAccountLayer,
    inferStrategyIdFromColumn,
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
