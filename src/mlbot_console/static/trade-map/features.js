/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function toggleFeatureDrawer(forceOpen) {
  const drawer = document.getElementById("featureDrawer");
  const backdrop = document.getElementById("featureDrawerBackdrop");
  const btn = document.getElementById("featurePanelBtn");
  if (!drawer || !btn) return;
  const open = forceOpen ?? drawer.classList.contains("hidden");
  drawer.classList.toggle("hidden", !open);
  if (backdrop) backdrop.classList.toggle("hidden", !open);
  document.body.classList.toggle("feature-drawer-open", open);
  btn.setAttribute("aria-expanded", open ? "true" : "false");
}

function pickerSourceColumns() {
  return Core.filterColumnsForFeaturePicker(
    S.availableFeatureColumns,
    layersState(),
    S.featureStrategyFocus
  );
}

function strategyPickerEmptyHint(layers, strategies) {
  const hasLayer = layers.trend || layers.spot || layers.multiLeg;
  if (!hasLayer) return "请先勾选账户层";
  if (strategies.length) return "";
  const liveIds = (Core.getFeatureTaxonomy() || {}).live_strategy_ids;
  if (Array.isArray(liveIds) && liveIds.length) {
    return "当前账户层下无实盘策略";
  }
  return "实盘策略未加载（检查宪法 YAML）";
}

function syncFeatureStrategySelectOptions() {
  const layers = layersState();
  const strategies = Core.listStrategiesForLayers(layers);
  const prev = S.featureStrategyFocus || "";
  const options = strategies.length
    ? [
        `<option value="">全部（不推荐，附图会混杂）</option>`,
        ...strategies.map(
          (s) =>
            `<option value="${escHtml(s.id)}"${s.id === prev ? " selected" : ""}>${escHtml(s.title || s.id)}</option>`
        ),
      ]
    : [`<option value="">${escHtml(strategyPickerEmptyHint(layers, strategies))}</option>`];
  const html = options.join("");
  for (const id of ["featureStrategySelect", "mapStrategySelect"]) {
    const sel = document.getElementById(id);
    if (sel) sel.innerHTML = html;
  }
  renderMapStrategyChips();
}

function applyPresetForStrategy(strategyId) {
  const sid = String(strategyId || "").trim();
  if (!sid) return;
  const pool = Core.filterColumnsForFeaturePicker(
    S.availableFeatureColumns,
    layersState(),
    null
  );
  const picks = Core.presetColumnsForStrategy(sid, pool, S.MAX_FEATURE_SUBCHARTS);
  setSelectedFeatures(picks, { refresh: false });
}

/** Toolbar chips / dropdown: switch strategy focus, preset columns, refresh subcharts. */
function switchMapStrategy(strategyId) {
  const sid =
    strategyId != null && String(strategyId).trim()
      ? String(strategyId).trim()
      : null;
  setFeatureStrategyFocus(sid, { refreshPicker: true, refreshSubcharts: false });
  if (sid) applyPresetForStrategy(sid);
  if (S.lastCandles?.length) {
    syncSubcharts(S.lastCandles, S.lastOverlays || {});
  }
  saveLayout();
  refreshBundle({ mode: "full" }).catch((e) => setStatus(String(e)));
}

function renderMapStrategyChips() {
  const host = document.getElementById("mapStrategyChips");
  if (!host) return;
  const strategies = Core.listStrategiesForLayers(layersState());
  const focus = S.featureStrategyFocus || "";
  if (!strategies.length) {
    const hint = strategyPickerEmptyHint(layersState(), strategies);
    host.innerHTML = `<span class="muted map-strategy-hint">${escHtml(hint)}</span>`;
    return;
  }
  const buttons = [
    `<button type="button" class="map-strategy-chip${focus === "" ? " active" : ""}" data-strategy="">全部</button>`,
    ...strategies.map(
      (s) =>
        `<button type="button" class="map-strategy-chip${
          focus === s.id ? " active" : ""
        }" data-strategy="${escHtml(s.id)}">${escHtml(s.title || s.id)}</button>`
    ),
  ];
  host.innerHTML = buttons.join("");
  host.querySelectorAll(".map-strategy-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sid = btn.getAttribute("data-strategy") || "";
      const sel = document.getElementById("mapStrategySelect");
      if (sel) sel.value = sid;
      switchMapStrategy(sid || null);
    });
  });
}

/** After account-layer toggles: pick default strategy + preset feature columns. */
function applyLayerStrategyDefaults() {
  const layers = layersState();
  const strategies = Core.listStrategiesForLayers(layers);
  const inferred = Core.inferStrategyFocusFromLayers(layers);
  const focus = S.featureStrategyFocus || "";
  const allowed = new Set(strategies.map((s) => s.id));

  if (!strategies.length) {
    setFeatureStrategyFocus(null, { refreshPicker: true, refreshSubcharts: false });
  } else if (inferred && (!focus || !allowed.has(focus))) {
    setFeatureStrategyFocus(inferred, { refreshPicker: true, refreshSubcharts: false });
    applyPresetForStrategy(inferred);
  } else if (focus && !allowed.has(focus)) {
    setFeatureStrategyFocus(inferred || null, {
      refreshPicker: true,
      refreshSubcharts: false,
    });
    if (inferred) applyPresetForStrategy(inferred);
  } else {
    renderMapStrategyChips();
    syncFeatureStrategySelectOptions();
  }
}

function setFeatureStrategyFocus(strategyId, { refreshPicker = true, refreshSubcharts = true } = {}) {
  S.featureStrategyFocus =
    strategyId != null && String(strategyId).trim()
      ? String(strategyId).trim()
      : null;
  syncFeatureStrategySelectOptions();
  if (refreshPicker) renderFeaturePicker();
  saveLayout();
  if (refreshSubcharts && S.lastCandles?.length) {
    syncSubcharts(S.lastCandles, S.lastOverlays || {});
  }
}

function setSelectedFeatures(cols, { refresh = true } = {}) {
  S.selectedFeatureColumns = [...new Set(cols.filter(Boolean))];
  renderFeaturePicker();
  saveLayout();
  if (refresh) refreshBundle().catch((e) => setStatus(String(e)));
}

function renderSelectedChips() {
  const el = document.getElementById("featureSelectedChips");
  if (!S.selectedFeatureColumns.length) {
    el.innerHTML = '<span class="muted">点击下方列名添加；或点「推荐」</span>';
    return;
  }
  el.innerHTML = S.selectedFeatureColumns
    .map((col) => {
      const m = Core.lookupFeatureMeta(col);
      const tag = `${m.account_layer_title || ""} › ${m.strategy_title || ""} › ${m.stage_title || ""}`;
      return `<span class="feature-chip"><span class="feature-chip-strategy" data-strategy="${escHtml(m.account_layer || "")}">${escHtml(tag)}</span>${escHtml(col)}<button type="button" data-remove-col="${escHtml(col)}" aria-label="移除">×</button></span>`;
    })
    .join("");
  el.querySelectorAll("[data-remove-col]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const col = btn.getAttribute("data-remove-col");
      setSelectedFeatures(S.selectedFeatureColumns.filter((c) => c !== col));
    });
  });
}

function renderFeaturePicker() {
  const list = document.getElementById("featureColumnList");
  const hint = document.getElementById("featurePickerHint");
  renderSelectedChips();
  if (!S.availableFeatureColumns.length) {
    list.innerHTML = '<p class="muted">当前周期无 features Parquet</p>';
    hint.textContent = "0";
    return;
  }
  const pool = pickerSourceColumns();
  const focusLabel = Core.strategyFocusLabel(S.featureStrategyFocus);
  hint.textContent = S.featureStrategyFocus
    ? `${S.selectedFeatureColumns.length}/${pool.length} · ${focusLabel}`
    : `${S.selectedFeatureColumns.length}/${pool.length}`;
  const filtered = Core.filterFeatureColumns(pool, S.featureSearchQuery);
  if (!filtered.length) {
    list.innerHTML = S.featureStrategyFocus
      ? `<p class="muted">当前策略「${escHtml(focusLabel)}」无匹配列；可改策略筛选或搜索</p>`
      : '<p class="muted">无匹配列</p>';
    return;
  }
  const groups = Core.groupFeatureColumnsByStrategy(filtered, layersState());
  list.innerHTML = groups
    .map(([title, cols, meta]) => {
      const items = cols
        .map((col) => {
          const on = S.selectedFeatureColumns.includes(col);
          const m = Core.lookupFeatureMeta(col);
          return `<label class="feature-item${on ? " selected" : ""}" data-account-layer="${escHtml(m.account_layer || "")}" data-stage="${escHtml(m.stage || "")}">
            <input type="checkbox" data-feature-col="${escHtml(col)}" ${on ? "checked" : ""} />
            <span>${escHtml(col)}</span>
          </label>`;
        })
        .join("");
      const dataAttrs = meta
        ? ` data-account-layer="${escHtml(meta.layer || "")}" data-strategy="${escHtml(meta.strategy || "")}" data-stage="${escHtml(meta.stage || "")}"`
        : "";
      return `<section class="feature-group"${dataAttrs}><h4 class="feature-group-title">${escHtml(title)} <span class="strategy-hint">(${cols.length})</span></h4><div class="feature-grid">${items}</div></section>`;
    })
    .join("");
  list.querySelectorAll("input[data-feature-col]").forEach((inp) => {
    inp.addEventListener("change", () => {
      const col = inp.getAttribute("data-feature-col");
      let next = [...S.selectedFeatureColumns];
      if (inp.checked) {
        if (!next.includes(col)) next.push(col);
      } else {
        next = next.filter((c) => c !== col);
      }
      setSelectedFeatures(next);
    });
  });
}

function bindFeaturePanel() {
  const btn = document.getElementById("featurePanelBtn");
  const drawer = document.getElementById("featureDrawer");
  const backdrop = document.getElementById("featureDrawerBackdrop");
  const closeBtn = document.getElementById("featureDrawerClose");
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    toggleFeatureDrawer(drawer.classList.contains("hidden"));
  });
  if (closeBtn) {
    closeBtn.addEventListener("click", () => toggleFeatureDrawer(false));
  }
  if (backdrop) {
    backdrop.addEventListener("click", () => toggleFeatureDrawer(false));
  }
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && drawer && !drawer.classList.contains("hidden")) {
      toggleFeatureDrawer(false);
    }
  });
  drawer.addEventListener("click", (ev) => ev.stopPropagation());
  document.getElementById("featureSearch").addEventListener("input", (ev) => {
    S.featureSearchQuery = ev.target.value;
    renderFeaturePicker();
  });
  const stratSel = document.getElementById("featureStrategySelect");
  if (stratSel) {
    stratSel.addEventListener("change", () => {
      switchMapStrategy(stratSel.value || null);
    });
  }
  drawer.querySelectorAll("[data-feature-action]").forEach((el) => {
    el.addEventListener("click", () => {
      const action = el.getAttribute("data-feature-action");
      if (action === "clear") {
        setSelectedFeatures([]);
        return;
      }
      if (action === "preset-default" || action.startsWith("preset-")) {
        const key =
          action === "preset-default" ? "default" : action.replace("preset-", "");
        const strategyPresets = new Set([
          "tpc",
          "bpc",
          "me",
          "srb",
          "chop_grid",
          "trend_scalp",
          "spot_accum_simple",
        ]);
        if (strategyPresets.has(key)) {
          switchMapStrategy(key);
          return;
        }
        const pool = pickerSourceColumns();
        let picks = Core.presetColumnsForAccountLayer(key, pool, S.MAX_FEATURE_SUBCHARTS);
        if (!picks.length) {
          const preset = Core.FEATURE_PRESETS[key] || Core.FEATURE_PRESETS.default;
          for (const name of preset) {
            if (pool.includes(name)) picks.push(name);
          }
        }
        if (key === "default" || key === "spot") {
          for (const c of pool) {
            if (String(c).toLowerCase().includes("weekly_ema") && !picks.includes(c)) {
              picks.push(c);
            }
          }
        }
        if (!picks.length && pool.length) {
          picks.push(pool[0]);
        }
        if (action.startsWith("preset-") && action !== "preset-default") {
          const merged = [...S.selectedFeatureColumns];
          for (const c of picks) {
            if (!merged.includes(c)) merged.push(c);
          }
          setSelectedFeatures(merged.slice(0, S.MAX_FEATURE_SUBCHARTS));
        } else {
          setSelectedFeatures(picks.slice(0, S.MAX_FEATURE_SUBCHARTS));
        }
      }
    });
  });
}

async function loadFeatureTaxonomy() {
  try {
    const { data } = await Shell.api("/api/bus/features/taxonomy");
    Core.setFeatureTaxonomy(data || null);
  } catch (_) {
  }
  syncFeatureStrategySelectOptions();
}

async function loadFeatureColumns() {
  const symbol = document.getElementById("symbolSelect").value;
  const timeframe = document.getElementById("timeframeSelect").value;
  try {
    const { data } = await Shell.api(
      `/api/bus/features/columns?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`
    );
    S.availableFeatureColumns = data.columns || [];
    Core.setFeatureTaxonomy(data.taxonomy || null);
    const defaults = data.defaults || [];
    S.selectedFeatureColumns = S.selectedFeatureColumns.filter((c) =>
      S.availableFeatureColumns.includes(c)
    );
    if (!S.selectedFeatureColumns.length && defaults.length) {
      S.selectedFeatureColumns = [...defaults];
    }
    if (!S.selectedFeatureColumns.length && S.availableFeatureColumns.length) {
      S.selectedFeatureColumns = [S.availableFeatureColumns[0]];
    }
  } catch (_) {
    S.availableFeatureColumns = [];
  }
  applyLayerStrategyDefaults();
  renderFeaturePicker();
  saveLayout();
}

async function showMarkerDetail(markerId) {
  const panel = document.getElementById("detailPanel");
  const body = document.getElementById("detailBody");
  panel.classList.remove("hidden");
  const raw = S.markerById.get(markerId);
  body.textContent = JSON.stringify(raw || { id: markerId }, null, 2);
  try {
    const { data } = await Shell.api(
      `/api/trade-map/marker-detail?marker_id=${encodeURIComponent(markerId)}`
    );
    body.textContent = JSON.stringify({ marker: raw, db: data }, null, 2);
  } catch (e) {
    body.textContent += `\n\n(DB lookup failed: ${e})`;
  }
}


