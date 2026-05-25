/**
 * Trade Map pure helpers — 20-markers. Attaches to MLBotTradeMapCore (load in numeric order).
 */
(function (root) {
  const Core = (root.MLBotTradeMapCore = root.MLBotTradeMapCore || {});
    function markerRole(marker) {
      const ev = String(marker.event || "").toLowerCase();
      if (ev === "tp") return "tp";
      if (ev === "grid") return "grid";
      if (ev === "exit") return "exit";
      return "entry";
    }

    function markerShape(marker) {
      const role = markerRole(marker);
      const pending = (marker.status || "filled").toLowerCase() === "pending";
      if (role === "tp") return pending ? "circle" : Core.TP_MARKER_SHAPE;
      if (role === "exit") return Core.EXIT_SHAPE;
      if (role === "grid") return "square";
      const side = (marker.side || "long").toLowerCase();
      if (marker.is_add && side === "long") return "diamond";
      if (marker.is_add && side === "short") return "diamond";
      return Core.ENTRY_SHAPES[side] || "arrowUp";
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
      if (role === "tp") return Core.TP_MARKER_COLOR;
      if (role === "grid") return marker.color || "#73BF69";
      if (role === "entry") {
        return side === "long" ? "#2e7d32" : "#c62828";
      }
      return marker.color || "#3274D9";
    }
    function filterMarkersByStrategy(markers, strategyFocus) {
      const focus = strategyFocus ? String(strategyFocus).trim().toLowerCase() : "";
      if (!focus) return markers || [];
      return (markers || []).filter(
        (m) => String(m.strategy || "").toLowerCase() === focus
      );
    }

    /** Chart markers: strategy filter + keep selected id visible for highlight/scroll. */
    function markersForChartDisplay(allMarkers, strategyFocus, selectedMarkerId) {
      const scoped = filterMarkersByStrategy(allMarkers || [], strategyFocus);
      const sel = selectedMarkerId ? String(selectedMarkerId).trim() : "";
      if (!sel) return scoped;
      if (scoped.some((m) => m.id === sel)) return scoped;
      const hit = (allMarkers || []).find((m) => m.id === sel);
      if (!hit) return scoped;
      return [...scoped, hit].sort((a, b) => Number(a.time) - Number(b.time));
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
        let position = aboveBar ? "aboveBar" : "belowBar";
        if (strat === "chop_grid") {
          const legSide = chopGridLegSide(
            (m.detail && m.detail.leg_label) || (m.detail && m.detail.leg_id) || legToken
          );
          // chop_grid stacks: S entry + L TP collide aboveBar; L entry + S TP
          // collide belowBar. Anchor filled entries on the leg's home side and
          // route TPs through inBar so they sit on the candle body instead of
          // piling onto the opposite-side entry.
          if (role === "tp") {
            position = "inBar";
          } else if (role === "entry" && !pending) {
            if (legSide === "short") position = "aboveBar";
            else if (legSide === "long") position = "belowBar";
          } else {
            position = aboveBar ? "aboveBar" : "belowBar";
          }
        }
        const isTp = role === "tp";
        const highlightSelected = selected && !isTp;
        return {
          time: m.time,
          position,
          color: highlightSelected ? "#ffeb3b" : markerColor(m),
          shape: markerShape(m),
          text: highlightSelected ? `★ ${baseText}` : baseText,
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

  Object.assign(Core, {
    markerRole,
    markerShape,
    markerColor,
    filterMarkersByStrategy,
    markersForChartDisplay,
    findMarkerByTime,
    chopGridMarkerDisplayText,
    chopGridLegSide,
    chopGridLabelAnchor,
    chopSegmentedLinePoints,
    markersToLwc,
    scrollIndexForTime,
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
