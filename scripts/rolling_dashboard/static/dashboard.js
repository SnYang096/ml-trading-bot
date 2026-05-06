(function () {
  var chipsEl = document.getElementById("layer-chips");
  var dashCards = document.getElementById("dash-cards");
  var strategyTabsEl = document.getElementById("strategy-tabs");
  if (!dashCards) return;

  var stats = {};
  var activeStrategyTab = "__all__";
  var cardPage =
    (typeof window.__DASHBOARD_CARD_PAGE__ === "number" &&
      window.__DASHBOARD_CARD_PAGE__) ||
    parseInt(dashCards.getAttribute("data-card-page") || "80", 10) ||
    80;

  function getScope() {
    return dashCards.getAttribute("data-dashboard-scope") || "all";
  }

  function wantAdoptButtons() {
    return dashCards.getAttribute("data-adopt-buttons") === "1";
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getStrategyTab() {
    return activeStrategyTab;
  }

  function filterQsExtra() {
    var sp = new URLSearchParams(location.search);
    var qv = sp.get("q");
    var st = getStrategyTab();
    var parts = [];
    if (st !== "__all__") parts.push("strategy=" + encodeURIComponent(st));
    if (qv) parts.push("q=" + encodeURIComponent(qv));
    if (!parts.length) return "";
    return "&" + parts.join("&");
  }

  function fetchCards(kind, reset, append) {
    var listId =
      kind === "rolling" ? "ledger-list-rolling" : "ledger-list-flat";
    var hintId =
      kind === "rolling" ? "rolling-load-hint" : "flat-load-hint";
    var btnId =
      kind === "rolling" ? "btn-more-rolling" : "btn-more-flat";
    var listEl = document.getElementById(listId);
    var hintEl = document.getElementById(hintId);
    var btn = document.getElementById(btnId);
    if (!listEl) return Promise.resolve();

    var offset = 0;
    if (append) {
      offset =
        parseInt(listEl.getAttribute("data-next-offset") || "0", 10) || 0;
    }
    if (reset) {
      listEl.innerHTML = '<p class="muted">加载中…</p>';
      offset = 0;
      listEl.removeAttribute("data-next-offset");
      listEl.removeAttribute("data-total");
    }

    var url =
      "/api/dashboard-cards.html?kind=" +
      encodeURIComponent(kind) +
      "&offset=" +
      offset +
      "&limit=" +
      cardPage +
      filterQsExtra();
    if (kind === "flat" && wantAdoptButtons()) {
      url += "&adopt_buttons=1";
    }

    return fetch(url)
      .then(function (r) {
        var total = parseInt(r.headers.get("X-Total-Count") || "0", 10);
        var nextOff = parseInt(r.headers.get("X-Next-Offset") || "0", 10);
        return r.text().then(function (html) {
          return { html: html, total: total, nextOff: nextOff };
        });
      })
      .then(function (o) {
        if (reset) listEl.innerHTML = o.html;
        else listEl.innerHTML += o.html;
        listEl.setAttribute("data-next-offset", String(o.nextOff));
        listEl.setAttribute("data-total", String(o.total));
        if (hintEl) {
          if (o.total === 0) {
            hintEl.textContent = "";
            hintEl.hidden = true;
          } else {
            hintEl.hidden = false;
            var shown = Math.min(o.nextOff, o.total);
            hintEl.textContent =
              "已加载 " + shown + " / " + o.total + " 条（分页加载）";
          }
        }
        if (btn) btn.hidden = o.nextOff >= o.total || o.total === 0;
      })
      .catch(function () {
        if (reset)
          listEl.innerHTML =
            '<p class="err">加载失败（网络或服务错误）</p>';
      });
  }

  function reloadBothSections() {
    var sc = getScope();
    if (sc === "research") {
      return fetchCards("rolling", true, false);
    }
    if (sc === "prod") {
      return fetchCards("flat", true, false);
    }
    return Promise.all([
      fetchCards("rolling", true, false),
      fetchCards("flat", true, false),
    ]);
  }

  function buildQueryString() {
    var sp = new URLSearchParams(location.search);
    var qv = sp.get("q");
    var st = getStrategyTab();
    var parts = [];
    if (st !== "__all__") parts.push("strategy=" + encodeURIComponent(st));
    if (qv) parts.push("q=" + encodeURIComponent(qv));
    if (!parts.length) return "";
    return "?" + parts.join("&");
  }

  function syncHistory() {
    try {
      history.replaceState(null, "", location.pathname + buildQueryString());
    } catch (e) {}
  }

  function renderChips(tabKey) {
    if (!chipsEl) return;
    var s = stats[tabKey] || stats["__all__"] || {};
    var pipe = s.pipeline || {};
    var mode = s.mode || {};
    var rkd = s.run_kind || {};
    var parts = [];
    Object.keys(pipe)
      .sort(function (a, b) {
        return pipe[b] - pipe[a];
      })
      .forEach(function (k) {
        parts.push(
          '<span class="chip">' + esc(k) + " ×" + pipe[k] + "</span>"
        );
      });
    Object.keys(mode)
      .sort(function (a, b) {
        return mode[b] - mode[a];
      })
      .forEach(function (k) {
        parts.push(
          '<span class="chip dim">mode:' +
            esc(k) +
            " ×" +
            mode[k] +
            "</span>"
        );
      });
    Object.keys(rkd)
      .sort(function (a, b) {
        return rkd[b] - rkd[a];
      })
      .forEach(function (k) {
        parts.push(
          '<span class="chip dim">kind:' +
            esc(k) +
            " ×" +
            rkd[k] +
            "</span>"
        );
      });
    chipsEl.innerHTML = parts.length
      ? parts.join("")
      : '<span class="muted">无统计</span>';
  }

  function setStrategyTab(tabKey) {
    activeStrategyTab = tabKey || "__all__";
    if (strategyTabsEl)
      strategyTabsEl.setAttribute("data-active-tab", activeStrategyTab);
    document.querySelectorAll(".tab-btn").forEach(function (btn) {
      var t = btn.getAttribute("data-tab") || "__all__";
      btn.setAttribute(
        "aria-selected",
        t === activeStrategyTab ? "true" : "false"
      );
    });
    var fStrat = document.querySelector('form.search input[name="strategy"]');
    if (fStrat)
      fStrat.value =
        activeStrategyTab === "__all__" ? "" : activeStrategyTab;
    syncHistory();
    reloadBothSections().then(function () {
      renderChips(activeStrategyTab);
    });
  }

  document.querySelectorAll(".tab-btn").forEach(function (btn) {
    btn.addEventListener("click", function (ev) {
      setStrategyTab(ev.currentTarget.getAttribute("data-tab") || "__all__");
    });
  });

  var bmR = document.getElementById("btn-more-rolling");
  if (bmR)
    bmR.addEventListener("click", function () {
      fetchCards("rolling", false, true);
    });
  var bmF = document.getElementById("btn-more-flat");
  if (bmF)
    bmF.addEventListener("click", function () {
      fetchCards("flat", false, true);
    });

  function applyPartitionView(mode) {
    var dw = document.getElementById("dash-wrap");
    var sr = document.getElementById("sec-rolling");
    var sh = document.getElementById("sec-history");
    var m = mode || "all";
    if (dw) dw.setAttribute("data-dash-view", m);
    if (sr && sh) {
      if (m === "all") {
        sr.style.removeProperty("display");
        sh.style.removeProperty("display");
      } else if (m === "rolling") {
        sr.style.removeProperty("display");
        sh.style.setProperty("display", "none", "important");
      } else if (m === "history") {
        sr.style.setProperty("display", "none", "important");
        sh.style.removeProperty("display");
      }
    }
    document.querySelectorAll(".view-seg-btn").forEach(function (b) {
      var bv = b.getAttribute("data-view") || "all";
      var on = bv === m;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.classList.toggle("view-seg-active", on);
    });
  }

  var dashWrap = document.getElementById("dash-wrap");
  if (dashWrap) {
    dashWrap.addEventListener("click", function (ev) {
      var t =
        ev.target && ev.target.closest
          ? ev.target.closest(".view-seg-btn")
          : null;
      if (!t || !dashWrap.contains(t)) return;
      ev.preventDefault();
      var v = t.getAttribute("data-view") || "all";
      applyPartitionView(v);
      try {
        if (v === "history") {
          var he = document.getElementById("sec-history");
          if (he) he.scrollIntoView({ behavior: "smooth", block: "start" });
        } else if (v === "rolling") {
          var re = document.getElementById("sec-rolling");
          if (re) re.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } catch (e) {}
    });
  }

  var spInit = new URLSearchParams(location.search);
  if (spInit.get("strategy")) activeStrategyTab = spInit.get("strategy");
  else {
    var ib = dashCards.getAttribute("data-initial-tab");
    if (ib) activeStrategyTab = ib;
  }

  document.querySelectorAll(".tab-btn").forEach(function (btn) {
    var t = btn.getAttribute("data-tab") || "__all__";
    btn.setAttribute(
      "aria-selected",
      t === activeStrategyTab ? "true" : "false"
    );
  });
  var fStrat0 = document.querySelector('form.search input[name="strategy"]');
  if (fStrat0)
    fStrat0.value =
      activeStrategyTab === "__all__" ? "" : activeStrategyTab;
  if (strategyTabsEl)
    strategyTabsEl.setAttribute("data-active-tab", activeStrategyTab);

  if (
    document.getElementById("sec-rolling") &&
    document.getElementById("sec-history")
  ) {
    applyPartitionView("all");
  }

  dashCards.addEventListener("click", function (e) {
    // 卡片快捷链（continuous / stitched / JSON 等）强制新标签打开；保留 Ctrl/Cmd/中键由浏览器默认处理。
    var ql =
      e.target.closest && e.target.closest(".quick-links a[href]");
    if (ql) {
      var qh = ql.getAttribute("href") || "";
      if (
        qh &&
        qh.charAt(0) !== "#" &&
        e.button === 0 &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.shiftKey &&
        !e.altKey
      ) {
        e.preventDefault();
        window.open(qh, "_blank", "noopener,noreferrer");
      }
      return;
    }
    if (e.target.classList.contains("btn-adopt-cmd")) {
      var cardA = e.target.closest(".ledger-card");
      if (!cardA) return;
      var strat = cardA.getAttribute("data-strategy") || "";
      var tsEl = cardA.querySelector(".ledger-ts");
      var ts = tsEl ? (tsEl.textContent || "").trim() : "";
      if (!strat || !ts) {
        alert("无法解析策略或时间戳");
        return;
      }
      var cmd = "mlbot pipeline adopt --strategy " + strat + " " + ts;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(cmd).then(
          function () {
            alert("已复制：\n" + cmd);
          },
          function () {
            window.prompt("请手动复制：", cmd);
          }
        );
      } else {
        window.prompt("请手动复制：", cmd);
      }
      return;
    }
    if (e.target.classList.contains("btn-deploy-hint")) {
      alert(
        "Deploy：请查看该批次目录下 report.json 中的 deploy_gate / 指标；具体上线步骤以团队规范为准。"
      );
      return;
    }
    if (!e.target.classList.contains("btn-del")) return;
    var card = e.target.closest(".ledger-card");
    if (!card) return;
    var rel = card.getAttribute("data-ledger-rel");
    if (!rel || !confirm("确定删除整个目录？\n" + rel)) return;
    fetch("/api/delete-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ledger: rel }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        if (j && j.ok) card.remove();
        else alert(j && j.error ? j.error : "删除失败");
      })
      .catch(function () {
        alert("删除请求失败");
      });
  });

  var btnStale = document.getElementById("btn-del-stale");
  if (btnStale)
    btnStale.addEventListener("click", function () {
      var qs = location.search || "";
      fetch("/api/incomplete-rolling.json" + qs)
        .then(function (r) {
          return r.json();
        })
        .then(function (j) {
          var ps = j.paths || [];
          if (!ps.length) {
            alert(
              "当前筛选下没有「待清理」rolling（无有效 stitched_summary.json）"
            );
            return;
          }
          if (
            !confirm(
              "将删除 " +
                ps.length +
                " 个 rolling 目录（无有效 stitched_summary：缺失/空/坏 JSON）。不可恢复，确定？"
            )
          )
            return;
          fetch("/api/bulk-delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: ps }),
          })
            .then(function (r) {
              return r.json();
            })
            .then(function (resp) {
              var del = resp.deleted || [];
              var err = resp.errors || [];
              if (del.length > 0) {
                if (err.length > 0)
                  alert(
                    "已删除 " +
                      del.length +
                      " 个；另有失败 " +
                      err.length +
                      " 个：\n" +
                      JSON.stringify(err)
                  );
                location.reload();
              } else {
                alert(
                  err.length
                    ? "未能删除：\n" + JSON.stringify(err)
                    : "未能删除任何目录"
                );
              }
            })
            .catch(function () {
              alert("批量删除请求失败");
            });
        })
        .catch(function () {
          alert("请求失败");
        });
    });

  var btnFlat = document.getElementById("btn-del-flat");
  if (btnFlat)
    btnFlat.addEventListener("click", function () {
      var qs = location.search || "";
      fetch("/api/flat-run-paths.json" + qs)
        .then(function (r) {
          return r.json();
        })
        .then(function (j) {
          var ps = j.paths || [];
          if (!ps.length) {
            alert("当前筛选下没有 History（单次）批次");
            return;
          }
          if (
            !confirm(
              "将删除 " +
                ps.length +
                " 个 History（单次）目录（当前策略/搜索筛选下）。不可恢复，确定？"
            )
          )
            return;
          fetch("/api/bulk-delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: ps }),
          })
            .then(function (r) {
              return r.json();
            })
            .then(function (resp) {
              var del = resp.deleted || [];
              var err = resp.errors || [];
              if (del.length > 0) {
                if (err.length > 0)
                  alert(
                    "已删除 " +
                      del.length +
                      " 个；另有失败 " +
                      err.length +
                      " 个：\n" +
                      JSON.stringify(err)
                  );
                location.reload();
              } else {
                alert(
                  err.length
                    ? "未能删除：\n" + JSON.stringify(err)
                    : "未能删除任何目录"
                );
              }
            })
            .catch(function () {
              alert("批量删除请求失败");
            });
        })
        .catch(function () {
          alert("请求失败");
        });
    });

  reloadBothSections().then(function () {
    renderChips(activeStrategyTab);
  });

  fetch("/api/dashboard-stats.json" + (location.search || ""))
    .then(function (r) {
      return r.json();
    })
    .then(function (j) {
      stats = j || {};
      renderChips(getStrategyTab());
    })
    .catch(function () {
      stats = {};
      renderChips(getStrategyTab());
    });
})();
