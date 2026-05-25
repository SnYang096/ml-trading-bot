/**
 * Trade Map core logic lives in trade-map/core/*.js (load in numeric order).
 * This stub remains so old bookmarks to /static/trade-map-core.js still resolve.
 */
(function (root) {
  if (typeof console !== "undefined" && console.warn) {
    console.warn(
      "trade-map-core.js is split into /static/trade-map/core/00-constants.js … 50-misc.js; include those scripts in HTML."
    );
  }
  root.MLBotTradeMapCore = root.MLBotTradeMapCore || {};
})(typeof globalThis !== "undefined" ? globalThis : window);
