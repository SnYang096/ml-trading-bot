/**
 * Trade Map pure helpers — 00-constants. Attaches to MLBotTradeMapCore (load in numeric order).
 */
(function (root) {
  const Core = (root.MLBotTradeMapCore = root.MLBotTradeMapCore || {});
    const ENTRY_SHAPES = { long: "arrowUp", short: "arrowDown" };
    const EXIT_SHAPE = "circle";
    /** Filled take-profit markers (chop_grid legs); pending TP stays gray circle. */
    const TP_MARKER_COLOR = "#E8B923";
    const TP_MARKER_SHAPE = "square";
    const SUBCHART_COLORS = [
      "#ffeb3b",
      "#58a6ff",
      "#f78166",
      "#7ee787",
      "#d2a8ff",
      "#ffa657",
    ];
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
    const DEFAULT_VISIBLE_BARS = 320;
    const FEATURE_PRESETS = {
      default: ["weekly_ema_200_position", "ema_1200_position"],
      trend: ["ema_1200_position", "tpc_pullback_depth", "tpc_semantic_chop", "bpc_pullback_depth"],
      spot: ["weekly_ema_200_position"],
      multi_leg: ["bpc_semantic_chop", "box_pos_60", "box_stability_60"],
    };


  Object.assign(Core, {
    ENTRY_SHAPES,
    EXIT_SHAPE,
    TP_MARKER_COLOR,
    TP_MARKER_SHAPE,
    ACCOUNT_LAYER_ORDER,
    STAGE_ORDER,
    ACCOUNT_LAYER_META,
    FEATURE_PRESETS,
    DEFAULT_VISIBLE_BARS,
    SUBCHART_COLORS,
  });

})(typeof globalThis !== "undefined" ? globalThis : window);
