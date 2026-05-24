/** Trade Map module — loaded via script tag (shared global functions). */
var S = globalThis.MLBotTradeMapPage;
var Core = globalThis.MLBotTradeMapCore;
var Shell = globalThis.MLBotConsole;

function bindControls() {
  const rerun = (opts = {}) =>
    refreshBundle({ mode: "full", ...opts }).catch((e) => setStatus(String(e)));
  const rerunAll = async () => {
    S.chartFitPending = true;
    resetOhlcvLoadedRange();
    resetMarkerQueryRange();
    saveLayout();
    await loadFeatureColumns();
    await rerun();
  };
  document.getElementById("refreshBtn").addEventListener("click", () => {
    S.chartFitPending = true;
    rerunAll();
  });
  const resetChartRangeIds = new Set(["symbolSelect", "timeframeSelect"]);
  [
    "symbolSelect",
    "timeframeSelect",
    "mainEma1200",
    "mainWeeklyEma200",
    "layerPrefilter",
    "layerGate",
    "layerTrend",
    "layerSpot",
    "layerMultiLeg",
    "layerPending",
    "layerChopGrid",
    "paneVolume",
  ].forEach((id) =>
    document.getElementById(id).addEventListener("change", () => {
      if (id === "paneVolume") {
        saveLayout();
        rerun();
        return;
      }
      if (resetChartRangeIds.has(id)) {
        if (id === "symbolSelect") {
          Shell.setSymbol(document.getElementById("symbolSelect").value);
        }
        resetOhlcvLoadedRange();
        resetMarkerQueryRange();
        S.chartFitPending = true;
        if (id.startsWith("layer")) {
          const inferred = Core.inferStrategyFocusFromLayers(layersState());
          if (inferred) setFeatureStrategyFocus(inferred, { refreshPicker: true });
          else syncFeatureStrategySelectOptions();
          renderFeaturePicker();
        }
        if (S.ordersDockOpen) refreshOrdersList().catch(() => { });
        rerunAll();
        return;
      }
      if (id.startsWith("layer")) {
        const layers = layersState();
        const accountLayerIds = ["layerTrend", "layerSpot", "layerMultiLeg"];
        if (accountLayerIds.includes(id)) {
          const filtered = Core.filterSelectedFeaturesByLayers(
            S.selectedFeatureColumns,
            layers
          );
          if (filtered.length !== S.selectedFeatureColumns.length) {
            S.selectedFeatureColumns = filtered;
            saveLayout();
          }
        }
        const inferred = Core.inferStrategyFocusFromLayers(layers);
        if (inferred) setFeatureStrategyFocus(inferred, { refreshPicker: true });
        else syncFeatureStrategySelectOptions();
        renderFeaturePicker();
      }
      if (S.ordersDockOpen) refreshOrdersList().catch(() => { });
      rerun();
    })
  );
  document.getElementById("detailCloseBtn").addEventListener("click", () => {
    document.getElementById("detailPanel").classList.add("hidden");
  });
  document.getElementById("ordersDockToggle").addEventListener("click", () => {
    toggleOrdersDock();
  });
  ["hideExpired", "hideCanceled"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      Shell.saveOrdersFilter(Shell.ordersFilterFromControls());
      saveLayout();
      if (S.ordersDockOpen) refreshOrdersList().catch((e) => setStatus(String(e)));
    });
  });
  Shell.bindSymbolPersist("symbolSelect");
}

(async () => {
  try {
    Shell.initAppNav("trade-map");
    applyScopesFromStorage();
    applyLayoutToControls(loadLayout());
    initMainChart();
    bindFeaturePanel();
    bindControls();
    await Shell.loadExtLinks();
    await Shell.loadSymbols("symbolSelect");
    Shell.bindOrdersFilterSync(() => {
      if (S.ordersDockOpen) refreshOrdersList().catch(() => { });
    });
    const pageUrl = new URL(window.location.href);
    const symParam = pageUrl.searchParams.get("symbol");
    if (symParam) {
      const sel = document.getElementById("symbolSelect");
      if ([...sel.options].some((o) => o.value === symParam)) {
        sel.value = symParam;
        Shell.setSymbol(symParam);
      }
    }
    await loadFeatureColumns();
    tickClock();
    if (S.clockTimer) clearInterval(S.clockTimer);
    S.clockTimer = setInterval(tickClock, 1000);
    await refreshBundle();
    startPoll();
  } catch (e) {
    setStatus(`启动失败: ${e}`);
    console.error(e);
  }
})();

