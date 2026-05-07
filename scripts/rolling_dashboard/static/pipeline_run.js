(function () {
  var cardRoot = document.getElementById("strategy-run-cards");
  var btnAll = document.getElementById("btn-run-all-strategies");
  var pollTimer = null;
  /** @type {Record<string, Object>} */
  var activeMonitors = {};

  function findCardByStrategy(slug) {
    if (!cardRoot) return null;
    var nodes = cardRoot.querySelectorAll("[data-strategy]");
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].getAttribute("data-strategy") === slug) return nodes[i];
    }
    return null;
  }

  function pcmPanelRefs() {
    return {
      wrap: document.getElementById("pcm-progress-wrap"),
      st: document.getElementById("pcm-status"),
      fill: document.getElementById("pcm-progress-fill"),
      label: document.getElementById("pcm-progress-label"),
      logNote: document.getElementById("pcm-log-note"),
      logA: document.getElementById("pcm-log-link"),
      resWrap: document.getElementById("pcm-result-links"),
      resUl: document.getElementById("pcm-result-ul"),
      btnRun: btnAll,
      key: "pcm",
    };
  }

  function makeStrategyProgressPanel() {
    var wrap = document.createElement("div");
    wrap.className = "pipeline-card-progress";
    wrap.hidden = true;

    var st = document.createElement("p");
    st.className = "pipeline-runner-status pipeline-card-status";
    st.setAttribute("aria-live", "polite");

    var pw = document.createElement("div");
    pw.className = "pipeline-progress-wrap";
    var track = document.createElement("div");
    track.className = "pipeline-progress-track";
    var fill = document.createElement("div");
    fill.className = "pipeline-progress-fill";
    fill.style.width = "0%";
    track.appendChild(fill);
    pw.appendChild(track);
    var label = document.createElement("p");
    label.className = "pipeline-progress-meta muted";

    var logNote = document.createElement("p");
    logNote.className = "muted pipeline-run-log-note";
    logNote.hidden = true;
    var logA = document.createElement("a");
    logA.target = "_blank";
    logA.rel = "noopener noreferrer";
    logNote.appendChild(document.createTextNode("完整日志（排障）："));
    logNote.appendChild(logA);

    var resWrap = document.createElement("div");
    resWrap.className = "pipeline-result-links";
    resWrap.hidden = true;
    var resHint = document.createElement("p");
    resHint.className = "muted";
    resHint.style.margin = "0 0 0.35rem";
    resHint.style.fontSize = "0.84rem";
    resHint.textContent = "完成后跳转";
    var resUl = document.createElement("ul");
    resUl.className = "pipeline-result-ul";
    resWrap.appendChild(resHint);
    resWrap.appendChild(resUl);

    wrap.appendChild(st);
    wrap.appendChild(pw);
    wrap.appendChild(label);
    wrap.appendChild(logNote);
    wrap.appendChild(resWrap);

    return {
      wrap: wrap,
      st: st,
      fill: fill,
      label: label,
      logNote: logNote,
      logA: logA,
      resWrap: resWrap,
      resUl: resUl,
      btnRun: null,
      key: null,
    };
  }

  function applyProgress(panel, p) {
    if (!panel || !p || !panel.fill || !panel.label) return;
    var pct = typeof p.pct === "number" ? p.pct : 0;
    pct = Math.max(0, Math.min(100, pct));
    panel.fill.style.width = pct + "%";
    panel.fill.classList.toggle("indeterminate", !!p.indeterminate);
    panel.label.textContent = p.label || "";
  }

  function applyJobPanel(panel, j) {
    if (!panel || !j) return;
    if (panel.wrap) panel.wrap.hidden = false;
    if (panel.logNote && panel.logA && j.log_path) {
      panel.logNote.hidden = false;
      panel.logA.href = j.log_url || "/" + String(j.log_path).replace(/\\/g, "/");
      panel.logA.textContent = j.log_path;
    }
    if (j.progress) applyProgress(panel, j.progress);
    if (panel.st) {
      var st = j.status || "?";
      var line =
        "任务 " +
        j.id +
        " · " +
        st +
        (j.config_path ? " · " + j.config_path : "");
      if (j.returncode != null && st !== "running")
        line += " · exit " + j.returncode;
      if (j.error && (st === "failed" || st === "interrupted"))
        line += " · " + j.error;
      panel.st.textContent = line;
    }
    if (j.status === "running") return;

    delete activeMonitors[j.id];
    if (panel.btnRun) panel.btnRun.disabled = false;

    if (j.progress) applyProgress(panel, j.progress);
    if (panel.resWrap && panel.resUl && j.result_links && j.result_links.length) {
      panel.resWrap.hidden = false;
      panel.resUl.innerHTML = j.result_links
        .map(function (L) {
          return (
            '<li><a href="' +
            String(L.href).replace(/"/g, "&quot;") +
            '">' +
            String(L.label).replace(/</g, "&lt;") +
            "</a></li>"
          );
        })
        .join("");
    } else if (panel.resWrap) {
      panel.resWrap.hidden = true;
    }

    if (Object.keys(activeMonitors).length === 0 && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function pollTick() {
    var ids = Object.keys(activeMonitors);
    ids.forEach(function (id) {
      fetch("/api/pipeline-run/status?id=" + encodeURIComponent(id))
        .then(function (r) {
          return r.json();
        })
        .then(function (o) {
          if (o && o.ok && o.job && activeMonitors[id])
            applyJobPanel(activeMonitors[id], o.job);
        })
        .catch(function () {});
    });
  }

  function ensurePoll() {
    if (pollTimer) return;
    pollTimer = setInterval(pollTick, 1200);
    pollTick();
  }

  function attachMonitor(jobId, panel) {
    activeMonitors[jobId] = panel;
    ensurePoll();
  }

  function clearMonitorsForPanel(panel) {
    if (!panel) return;
    Object.keys(activeMonitors).forEach(function (id) {
      if (activeMonitors[id] === panel) delete activeMonitors[id];
    });
  }

  function postPayload(body, panel) {
    if (!panel) return;
    clearMonitorsForPanel(panel);
    if (panel.btnRun) panel.btnRun.disabled = true;
    if (panel.resWrap) panel.resWrap.hidden = true;
    if (panel.resUl) panel.resUl.innerHTML = "";
    if (panel.logNote) panel.logNote.hidden = true;
    if (panel.label) panel.label.textContent = "";
    if (panel.fill) {
      panel.fill.style.width = "2%";
      panel.fill.classList.add("indeterminate");
    }
    if (panel.st) panel.st.textContent = "提交中…";
    if (panel.wrap) panel.wrap.hidden = false;

    fetch("/api/pipeline-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        return r.json().then(function (j) {
          return { r: r, j: j };
        });
      })
      .then(function (o) {
        var r = o.r;
        var j = o.j;
        if (r.status === 503 && j && j.error === "pipeline_run_disabled") {
          if (panel.st) {
            panel.st.textContent =
              "管线运行已禁用（ROLLING_DASHBOARD_PIPELINE_RUN）";
          }
          if (panel.btnRun) panel.btnRun.disabled = false;
          return;
        }
        if (r.status === 403) {
          if (panel.st) {
            panel.st.textContent =
              "拒绝：仅从本机 loopback 可启动（或 ROLLING_DASHBOARD_PIPELINE_REMOTE=1）";
          }
          if (panel.btnRun) panel.btnRun.disabled = false;
          return;
        }
        if (!j || !j.ok || !j.job) {
          if (panel.st)
            panel.st.textContent =
              "启动失败：" + (j && j.error ? j.error : r.status);
          if (panel.btnRun) panel.btnRun.disabled = false;
          return;
        }
        attachMonitor(j.job.id, panel);
        applyJobPanel(panel, j.job);
      })
      .catch(function () {
        if (panel.st) panel.st.textContent = "请求失败（网络或服务错误）";
        if (panel.btnRun) panel.btnRun.disabled = false;
      });
  }

  if (btnAll) {
    btnAll.addEventListener("click", function () {
      if (
        !confirm(
          "将执行 PCM 多策略编排（--all + pcm_orchestrate 配置），耗时很长。确定？"
        )
      )
        return;
      postPayload({ run_all: true }, pcmPanelRefs());
    });
  }

  if (!cardRoot) return;

  function strategySlugFromRel(rel) {
    var parts = String(rel || "").replace(/\\/g, "/").split("/");
    if (
      parts.length >= 4 &&
      parts[0] === "config" &&
      parts[1] === "strategies"
    ) {
      return parts[2];
    }
    return "（其它）";
  }

  function displayRelUnderStrategy(rel, slug) {
    var prefix = "config/strategies/" + slug + "/research/";
    var r = String(rel || "").replace(/\\/g, "/");
    if (r.indexOf(prefix) === 0) return r.slice(prefix.length);
    var fallback = "config/strategies/" + slug + "/";
    if (r.indexOf(fallback) === 0) return r.slice(fallback.length);
    return r;
  }

  function slugToDomId(slug) {
    return String(slug || "x")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .replace(/^_+|_+$/g, "") || "other";
  }

  function buildStrategyCard(slug, list) {
    var section = document.createElement("section");
    section.className = "pipeline-run-card pipeline-strategy-card";
    section.setAttribute("data-strategy", slug);

    var h2 = document.createElement("h2");
    h2.className = "pipeline-run-card-title pipeline-strategy-card-title";
    h2.textContent = slug;
    section.appendChild(h2);

    var hint = document.createElement("p");
    hint.className = "muted";
    hint.style.fontSize = "0.84rem";
    hint.style.margin = "0 0 0.5rem";
    hint.style.lineHeight = "1.45";
    hint.textContent =
      "本卡仅含 config/strategies/" +
      slug +
      "/research/ 下 " +
      list.length +
      " 个管线配置；选项值为完整路径。";
    section.appendChild(hint);

    var field = document.createElement("div");
    field.className = "pipeline-run-field";

    var selId = "pipeline-select-" + slugToDomId(slug);
    var label = document.createElement("label");
    label.setAttribute("for", selId);
    label.textContent = "研究配置（YAML）";

    var sel = document.createElement("select");
    sel.id = selId;
    sel.setAttribute("aria-label", slug + " 策略 yaml");

    list.forEach(function (c) {
      var opt = document.createElement("option");
      opt.value = c.rel_path;
      opt.textContent = displayRelUnderStrategy(c.rel_path, slug);
      sel.appendChild(opt);
    });

    field.appendChild(label);
    field.appendChild(sel);
    section.appendChild(field);

    var actions = document.createElement("div");
    actions.className = "pipeline-run-actions";

    var btnRun = document.createElement("button");
    btnRun.type = "button";
    btnRun.className = "btn-pipeline-primary";
    btnRun.textContent = "运行所选配置";

    var prog = makeStrategyProgressPanel();
    prog.btnRun = btnRun;
    prog.key = slug;
    section._pipePanel = prog;

    btnRun.addEventListener("click", function () {
      var rel = (sel.value || "").trim();
      if (!rel) return;
      postPayload({ config_path: rel }, prog);
    });

    actions.appendChild(btnRun);
    section.appendChild(actions);
    section.appendChild(prog.wrap);

    return section;
  }

  function restoreRunningJobs() {
    fetch("/api/pipeline-run/jobs?running_only=1")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        var jobs = (data && data.jobs) || [];
        var byKey = {};
        jobs.forEach(function (job) {
          if (job.status !== "running") return;
          var k = job.run_all ? "__pcm__" : job.strategy;
          var prev = byKey[k];
          if (!prev || String(job.started_at) > String(prev.started_at))
            byKey[k] = job;
        });
        Object.keys(byKey).forEach(function (k) {
          var job = byKey[k];
          if (job.run_all) {
            var pcm = pcmPanelRefs();
            if (!pcm.wrap) return;
            clearMonitorsForPanel(pcm);
            pcm.wrap.hidden = false;
            attachMonitor(job.id, pcm);
            applyJobPanel(pcm, job);
            if (btnAll) btnAll.disabled = true;
          } else {
            var card = findCardByStrategy(job.strategy);
            if (!card || !card._pipePanel) return;
            var pg = card._pipePanel;
            clearMonitorsForPanel(pg);
            pg.wrap.hidden = false;
            attachMonitor(job.id, pg);
            applyJobPanel(pg, job);
            if (pg.btnRun) pg.btnRun.disabled = true;
          }
        });
      })
      .catch(function () {});
  }

  fetch("/api/bpc-research-configs.json")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      var cfgs = (data && data.configs) || [];
      cardRoot.innerHTML = "";

      if (!cfgs.length) {
        var empty = document.createElement("section");
        empty.className = "pipeline-run-card";
        empty.innerHTML =
          "<h2 class=\"pipeline-run-card-title\">无配置</h2><p class=\"muted\" style=\"font-size:0.84rem;margin:0\">未发现 research 管线 yaml，请检查 <code>config/strategies/&lt;slug&gt;/research/</code></p>";
        cardRoot.appendChild(empty);
        restoreRunningJobs();
        return;
      }

      var bySlug = {};
      cfgs.forEach(function (c) {
        var s = strategySlugFromRel(c.rel_path);
        if (!bySlug[s]) bySlug[s] = [];
        bySlug[s].push(c);
      });
      var slugs = Object.keys(bySlug).sort(function (a, b) {
        return a.localeCompare(b, "en");
      });

      slugs.forEach(function (slug) {
        var list = bySlug[slug].slice().sort(function (a, b) {
          return String(a.rel_path).localeCompare(String(b.rel_path), "en");
        });
        cardRoot.appendChild(buildStrategyCard(slug, list));
      });
      restoreRunningJobs();
    })
    .catch(function () {
      cardRoot.innerHTML = "";
      var err = document.createElement("section");
      err.className = "pipeline-run-card";
      err.innerHTML =
        "<h2 class=\"pipeline-run-card-title\">加载失败</h2><p class=\"muted\" style=\"font-size:0.84rem;margin:0\">无法拉取配置列表</p>";
      cardRoot.appendChild(err);
    });
})();
