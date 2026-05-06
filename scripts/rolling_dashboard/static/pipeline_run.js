(function () {
  var sel = document.getElementById("bpc-config-select");
  var btnRun = document.getElementById("btn-run-bpc-config");
  var btnAll = document.getElementById("btn-run-all-strategies");
  var elSkip = document.getElementById("pipeline-skip-shap");
  var elStatus = document.getElementById("pipeline-status");
  var elFill = document.getElementById("pipeline-progress-fill");
  var elProgLabel = document.getElementById("pipeline-progress-label");
  var elLogNote = document.getElementById("pipeline-log-note");
  var elLogA = document.getElementById("pipeline-log-link");
  var elResWrap = document.getElementById("pipeline-result-links");
  var elResUl = document.getElementById("pipeline-result-ul");
  var pollTimer = null;
  var jobId = null;

  if (!sel || !btnRun) return;

  function setBusy(busy) {
    btnRun.disabled = !!busy;
    if (btnAll) btnAll.disabled = !!busy;
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

  fetch("/api/bpc-research-configs.json")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      var cfgs = (data && data.configs) || [];
      sel.innerHTML = "";
      if (!cfgs.length) {
        sel.innerHTML =
          '<option value="">（未发现 yaml，请检查 config/strategies/bpc/research/）</option>';
        btnRun.disabled = true;
        return;
      }
      cfgs.forEach(function (c) {
        var opt = document.createElement("option");
        opt.value = c.name;
        opt.textContent = c.name + " → " + c.rel_path;
        sel.appendChild(opt);
      });
    })
    .catch(function () {
      sel.innerHTML = '<option value="">加载配置列表失败</option>';
      btnRun.disabled = true;
    });

  btnRun.addEventListener("click", function () {
    var name = (sel.value || "").trim();
    if (!name) return;
    postPayload({
      strategy: "bpc",
      bpc_research_config: name,
      skip_shap: elSkip ? !!elSkip.checked : true,
    });
  });

  if (btnAll)
    btnAll.addEventListener("click", function () {
      if (
        !confirm(
          "将执行 --all（全部策略），耗时很长且无单一 BPC yaml。确定？"
        )
      )
        return;
      postPayload({ run_all: true, skip_shap: elSkip ? !!elSkip.checked : true });
    });
})();
