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
      const regimeExit =
        role === "exit" &&
        marker.detail &&
        String(marker.detail.exit_kind || "").toLowerCase() === "regime_or_risk_exit";
      if (regimeExit) return "circle";
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
      if (
        role === "exit" &&
        marker.detail &&
        String(marker.detail.exit_kind || "").toLowerCase() === "regime_or_risk_exit"
      ) {
        return marker.color || "#ff7043";
      }
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

    /** Prefer markers pinned to the clicked bar (avoids jumping to another bar's marker). */
    function findMarkerOnBar(markers, clickTime, toleranceSec) {
      const t = Number(clickTime);
      if (!Number.isFinite(t)) return null;
      const onBar = (markers || []).filter((m) => Number(m.time) === t);
      if (!onBar.length) return findMarkerByTime(markers, clickTime, toleranceSec);
      const regime = onBar.filter(
        (m) =>
          String(m.event || "").toLowerCase() === "exit" &&
          m.detail &&
          String(m.detail.exit_kind || "").toLowerCase() === "regime_or_risk_exit"
      );
      if (regime.length) return regime[regime.length - 1];
      return onBar[onBar.length - 1];
    }

    function isFeatureBusRegimeExitMarker(m) {
      if (!m) return false;
      const id = String(m.id || "");
      if (!id.startsWith("multi_leg:regime_exit:")) return false;
      return String(m.detail?.source || "") === "feature_bus_hysteresis";
    }
    function chopRegimeThresholdsFromOverlay(overlay) {
      let entryMin = 0.5;
      let exitBelow = 0.32;
      const refs = overlay?.reference_lines || [];
      for (const r of refs) {
        const y = Number(r.y != null ? r.y : r.value);
        if (!Number.isFinite(y)) continue;
        const op = String(r.operator || "");
        if (op.includes(">=")) entryMin = y;
        else if (op === "<" || (op.includes("<") && !op.includes("="))) exitBelow = y;
      }
      return { entryMin, exitBelow };
    }

    function chopGridHysteresisActive(values, entryMin, exitBelow) {
      let active = false;
      const out = [];
      for (const val of values) {
        if (val == null || !Number.isFinite(val)) {
          out.push(active);
          continue;
        }
        if (!active) active = val >= entryMin;
        else if (val < exitBelow) active = false;
        out.push(active);
      }
      return out;
    }

    /** Synthetic regime exits from bpc_semantic_chop overlay (matches live hysteresis flatten). */
    function synthesizeChopRegimeExitMarkers(candles, overlays) {
      const ol = overlays?.bpc_semantic_chop;
      if (!ol || !Array.isArray(candles) || !candles.length) return [];
      const filled = Core.forwardFillOverlayToCandles(ol.points || [], candles);
      if (!filled.length) return [];
      const { entryMin, exitBelow } = chopRegimeThresholdsFromOverlay(ol);
      const vals = filled.map((p) =>
        p.value == null || !Number.isFinite(Number(p.value)) ? null : Number(p.value)
      );
      const chopOn = chopGridHysteresisActive(vals, entryMin, exitBelow);
      const sym = String(
        (candles[0] && candles[0].symbol) || "BNBUSDT"
      ).toUpperCase();
      const markers = [];
      for (let i = 1; i < chopOn.length; i++) {
        if (!(chopOn[i - 1] && !chopOn[i])) continue;
        const val = vals[i];
        if (val == null || val >= exitBelow) continue;
        const t = Number(candles[i].time);
        if (!Number.isFinite(t)) continue;
        markers.push({
          id: `multi_leg:regime_exit:${sym}:${t}`,
          time: t,
          symbol: sym,
          scope: "multi_leg",
          strategy: "chop_grid",
          event: "exit",
          side: "long",
          status: "filled",
          color: "#ff7043",
          detail: {
            exit_kind: "regime_or_risk_exit",
            exit_reason: "regime_or_risk_exit",
            chop: val,
            entry_chop_min: entryMin,
            exit_chop_below: exitBelow,
            source: "overlay_hysteresis",
          },
        });
      }
      return markers;
    }

    /** Bar times (unix sec) where chop regime hysteresis turns off — for metrics table headers. */
    function chopRegimeExitBarTimes(candles, overlays) {
      const times = new Set();
      for (const m of synthesizeChopRegimeExitMarkers(candles, overlays)) {
        const t = Number(m.time);
        if (Number.isFinite(t)) times.add(t);
      }
      return times;
    }

    function mergeRegimeExitMarkers(markers, regimeExits) {
      const base = markers || [];
      const adds = regimeExits || [];
      if (!adds.length) return base;
      const chopExitTimes = new Set();
      for (const m of base) {
        if (String(m.strategy || "").toLowerCase() !== "chop_grid") continue;
        if (String(m.event || "").toLowerCase() !== "exit") continue;
        const t = Number(m.time);
        if (Number.isFinite(t)) chopExitTimes.add(t);
      }
      const seen = new Set(base.map((m) => String(m.id || "")));
      const out = base.slice();
      for (const m of adds) {
        const id = String(m.id || "");
        if (seen.has(id)) continue;
        const t = Number(m.time);
        if (!Number.isFinite(t)) continue;
        let dup = false;
        for (const et of chopExitTimes) {
          if (Math.abs(t - et) <= 1) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        seen.add(id);
        chopExitTimes.add(t);
        out.push(m);
      }
      return out.sort((a, b) => Number(a.time) - Number(b.time));
    }

    function chopGridMarkerDisplayText(m, pending) {
      const strat = (m.strategy || "").toLowerCase();
      if (strat !== "chop_grid") return "";
      const ev = String(m.event || "").toLowerCase();
      if (
        ev === "exit" &&
        m.detail &&
        String(m.detail.exit_kind || "").toLowerCase() === "regime_or_risk_exit"
      ) {
        return "regime退出";
      }
      const leg = String((m.detail && m.detail.leg_label) || "").trim().toUpperCase();
      if (!leg) return "";
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
          const regimeExit =
            role === "exit" &&
            m.detail &&
            String(m.detail.exit_kind || "").toLowerCase() === "regime_or_risk_exit";
          if (regimeExit) {
            position = "inBar";
          } else {
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
    findMarkerOnBar,
    isFeatureBusRegimeExitMarker,
    chopGridMarkerDisplayText,
    chopRegimeThresholdsFromOverlay,
    chopGridHysteresisActive,
    synthesizeChopRegimeExitMarkers,
    chopRegimeExitBarTimes,
    mergeRegimeExitMarkers,
    chopGridLegSide,
    chopGridLabelAnchor,
    chopSegmentedLinePoints,
    markersToLwc,
    scrollIndexForTime,
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
