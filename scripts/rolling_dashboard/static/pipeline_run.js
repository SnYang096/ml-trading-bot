(function () {
  var cardRoot = document.getElementById("strategy-run-cards");
  var btnAll = document.getElementById("btn-run-all-strategies");
  var elStatus = document.getElementById("pipeline-status");
  var elFill = document.getElementById("pipeline-progress-fill");
  var elProgLabel = document.getElementById("pipeline-progress-label");
  var elLogNote = document.getElementById("pipeline-log-note");
  var elLogA = document.getElementById("pipeline-log-link");
  var elResWrap = document.getElementById("pipeline-result-links");
  var elResUl = document.getElementById("pipeline-result-ul");
  var pollTimer = null;
  var jobId = null;
  var runButtons = [];

  if (!elStatus) return;

  if (btnAll) {
    btnAll.addEventListener("click", function () {
      if (
        !confirm(
          "将执行 PCM 多策略编排（--all + pcm_orchestrate 配置），耗时很长。确定？"
        )
      )
        return;
      postPayload({ run_all: true });
    });
  }

  if (!cardRoot) return;

  function setBusy(busy) {
    var v = !!busy;
    runButtons.forEach(function (b) {
      b.disabled = v;
    });
    if (btnAll) btnAll.disabled = v;
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function applyProgress(p) {
    if (!p || !elFill || !elProgLabel) return;
    var pct = typeof p.pct === "number" ? p.pct : 0;
    pct = Math.max(0, Math.min(100, pct));
    elFill.style.width = pct + "%";
    elFill.classList.toggle("indeterminate", !!p.indeterminate);
    elProgLabel.textContent = p.label || "";
  }

  function applyJob(j) {
    if (!j) return;
    if (elLogNote && elLogA && j.log_path) {
      elLogNote.hidden = false;
      elLogA.href = j.log_url || "/" + String(j.log_path).replace(/\\/g, "/");
      elLogA.textContent = j.log_path;
    }
    if (j.progress) applyProgress(j.progress);
    if (elStatus) {
      var st = j.status || "?";
      var line =
        "任务 " +
        j.id +
        " · " +
        st +
        (j.config_path ? " · " + j.config_path : "");
      if (j.returncode != null && st !== "running")
        line += " · exit " + j.returncode;
      if (j.error && st === "failed") line += " · " + j.error;
      elStatus.textContent = line;
    }
    if (j.status === "running") return;
    stopPoll();
    jobId = null;
    setBusy(false);
    if (j.progress) applyProgress(j.progress);
    if (elResWrap && elResUl && j.result_links && j.result_links.length) {
      elResWrap.hidden = false;
      elResUl.innerHTML = j.result_links
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
    } else if (elResWrap) {
      elResWrap.hidden = true;
    }
  }

  function pollOnce() {
    if (!jobId) return;
    fetch("/api/pipeline-run/status?id=" + encodeURIComponent(jobId))
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        if (j && j.ok && j.job) applyJob(j.job);
      })
      .catch(function () {});
  }

  function startPoll() {
    stopPoll();
    pollOnce();
    pollTimer = setInterval(pollOnce, 1200);
  }

  function postPayload(body) {
    if (!elStatus) return;
    setBusy(true);
    if (elResWrap) elResWrap.hidden = true;
    if (elResUl) elResUl.innerHTML = "";
    if (elLogNote) elLogNote.hidden = true;
    if (elProgLabel) elProgLabel.textContent = "";
    if (elFill) {
      elFill.style.width = "2%";
      elFill.classList.add("indeterminate");
    }
    elStatus.textContent = "提交中…";
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
          elStatus.textContent =
            "管线运行已禁用（ROLLING_DASHBOARD_PIPELINE_RUN）";
          setBusy(false);
          return;
        }
        if (r.status === 403) {
          elStatus.textContent =
            "拒绝：仅从本机 loopback 可启动（或 ROLLING_DASHBOARD_PIPELINE_REMOTE=1）";
          setBusy(false);
          return;
        }
        if (r.status === 409 && j && j.error === "busy") {
          elStatus.textContent =
            "已有任务在运行" +
            (j.active && j.active.id ? "（id=" + j.active.id + "）" : "");
          setBusy(false);
          return;
        }
        if (!j || !j.ok || !j.job) {
          elStatus.textContent =
            "启动失败：" + (j && j.error ? j.error : r.status);
          setBusy(false);
          return;
        }
        jobId = j.job.id;
        applyJob(j.job);
        startPoll();
      })
      .catch(function () {
        elStatus.textContent = "请求失败（网络或服务错误）";
        setBusy(false);
      });
  }

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
    btnRun.addEventListener("click", function () {
      var rel = (sel.value || "").trim();
      if (!rel) return;
      postPayload({ config_path: rel });
    });

    runButtons.push(btnRun);
    actions.appendChild(btnRun);
    section.appendChild(actions);

    return section;
  }

  fetch("/api/bpc-research-configs.json")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      var cfgs = (data && data.configs) || [];
      cardRoot.innerHTML = "";
      runButtons = [];

      if (!cfgs.length) {
        var empty = document.createElement("section");
        empty.className = "pipeline-run-card";
        empty.innerHTML =
          "<h2 class=\"pipeline-run-card-title\">无配置</h2><p class=\"muted\" style=\"font-size:0.84rem;margin:0\">未发现 research 管线 yaml，请检查 <code>config/strategies/&lt;slug&gt;/research/</code></p>";
        cardRoot.appendChild(empty);
        return;
      }

      var bySlug = {};
      cfgs.forEach(function (c) {
        var slug = strategySlugFromRel(c.rel_path);
        if (!bySlug[slug]) bySlug[slug] = [];
        bySlug[slug].push(c);
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
    })
    .catch(function () {
      cardRoot.innerHTML = "";
      runButtons = [];
      var err = document.createElement("section");
      err.className = "pipeline-run-card";
      err.innerHTML =
        "<h2 class=\"pipeline-run-card-title\">加载失败</h2><p class=\"muted\" style=\"font-size:0.84rem;margin:0\">无法拉取配置列表</p>";
      cardRoot.appendChild(err);
    });

})();
