/**
 * Trade Map pure helpers — 50-misc. Attaches to MLBotTradeMapCore (load in numeric order).
 */
(function (root) {
  const Core = (root.MLBotTradeMapCore = root.MLBotTradeMapCore || {});
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

    function subchartColor(index) {
      const colors = Core.SUBCHART_COLORS || [];
      return colors[Math.abs(index) % colors.length];
    }

    function filterFeatureColumns(columns, query) {
      const q = String(query || "")
        .trim()
        .toLowerCase();
      if (!q) return columns || [];
      return (columns || []).filter((c) => String(c).toLowerCase().includes(q));
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

  Object.assign(Core, {
    stageRegionsQueryParam,
    mainOverlaysQueryParam,
    formatEligibility,
    browserLocalUrl,
    resolveLinkUrl,
    subchartColor,
    featureColumnsParam,
    parseStoredLayout,
    filterFeatureColumns,
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
