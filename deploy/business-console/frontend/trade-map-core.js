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

  root.MLBotTradeMapCore = {
    scopesFromLayers,
    markersToLwc,
    markerColor,
    markerShape,
    formatEligibility,
    browserLocalUrl,
    resolveLinkUrl,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
