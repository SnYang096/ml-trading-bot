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

function syncFeatureStrategySelectOptions() {
  const sel = document.getElementById("featureStrategySelect");
  if (!sel) return;
  const layers = layersState();
  const strategies = Core.listStrategiesForLayers(layers);
  const prev = S.featureStrategyFocus || "";
  const options = [
    `<option value="">全部（当前账户层）</option>`,
    ...strategies.map(
      (s) =>
        `<option value="${escHtml(s.id)}"${s.id === prev ? " selected" : ""}>${escHtml(s.title || s.id)}</option>`
    ),
  ];
  sel.innerHTML = options.join("");
}

function setFeatureStrategyFocus(strategyId, { refreshPicker = true } = {}) {
  S.featureStrategyFocus =
    strategyId != null && String(strategyId).trim()
      ? String(strategyId).trim()
      : null;
  syncFeatureStrategySelectOptions();
  if (refreshPicker) renderFeaturePicker();
  saveLayout();
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
      const v = stratSel.value;
      setFeatureStrategyFocus(v || null);
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
          setFeatureStrategyFocus(key, { refreshPicker: false });
        }
        const pool = pickerSourceColumns();
        let picks = [];
        if (strategyPresets.has(key)) {
          picks = Core.presetColumnsForStrategy(key, pool, S.MAX_FEATURE_SUBCHARTS);
        } else {
          picks = Core.presetColumnsForAccountLayer(key, pool, S.MAX_FEATURE_SUBCHARTS);
        }
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
  if (!S.featureStrategyFocus) {
    const inferred = Core.inferStrategyFocusFromLayers(layersState());
    if (inferred) S.featureStrategyFocus = inferred;
  }
  syncFeatureStrategySelectOptions();
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


