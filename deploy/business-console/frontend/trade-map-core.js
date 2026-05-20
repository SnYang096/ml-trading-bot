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

  function markersToLwc(markers) {
    return (markers || []).map((m) => {
      const isExit = m.event === "exit";
      const pending = (m.status || "filled").toLowerCase() === "pending";
      return {
        time: m.time,
        position: isExit ? "aboveBar" : "belowBar",
        color: markerColor(m, isExit),
        shape: pending ? "circle" : markerShape(m, isExit),
        text: `${m.scope}:${m.event}${pending ? ":pending" : ""}`,
        id: m.id,
        _raw: m,
      };
    });
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

  function groupFeatureColumns(columns) {
    const groups = { 推荐: [], vpin: [], ema: [], 其他: [] };
    for (const col of columns || []) {
      const c = String(col);
      const lc = c.toLowerCase();
      if (lc.includes("weekly_ema") || lc.includes("regime")) {
        groups.推荐.push(c);
      } else if (lc.startsWith("vpin")) {
        groups.vpin.push(c);
      } else if (lc.includes("ema") || lc.includes("macd") || lc.includes("rsi")) {
        groups.ema.push(c);
      } else {
        groups.其他.push(c);
      }
    }
    return Object.entries(groups).filter(([, items]) => items.length > 0);
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
    default: ["weekly_ema_200_position", "weekly_ema_200_position_f"],
  };

  root.MLBotTradeMapCore = {
    scopesFromLayers,
    markersToLwc,
    markerColor,
    markerShape,
    formatEligibility,
    browserLocalUrl,
    resolveLinkUrl,
    subchartColor,
    featureColumnsParam,
    parseStoredLayout,
    filterFeatureColumns,
    groupFeatureColumns,
    barSpacingForCount,
    FEATURE_PRESETS,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
