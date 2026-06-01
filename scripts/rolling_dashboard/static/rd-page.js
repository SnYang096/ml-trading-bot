/**
 * Local R&D experiments — config/experiments/ (rolling_dashboard /rd).
 */

let allRows = [];
let selectedId = null;

async function api(path, options) {
  const r = await fetch(path, options);
  const text = await r.text();
  let j;
  try {
    j = JSON.parse(text);
  } catch (_) {
    throw new Error(r.ok ? `Invalid JSON from ${path}` : `${r.status} ${path}`);
  }
  if (!j.ok) {
    throw new Error(j.error?.message || j.detail || r.statusText || "API error");
  }
  return j;
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function verdictPill(verdict) {
  if (!verdict) return '<span class="rd-pill rd-pill-muted">—</span>';
  const cls =
    verdict === "promote"
      ? "rd-pill-promote"
      : verdict === "reject"
        ? "rd-pill-reject"
        : verdict === "park"
          ? "rd-pill-park"
          : "rd-pill-muted";
  return `<span class="rd-pill ${cls}">${esc(verdict)}</span>`;
}

function filteredRows() {
  const strat = document.getElementById("strategyFilter").value;
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  const decisionOnly = document.getElementById("decisionOnly").checked;
  return allRows.filter((r) => {
    if (strat && (r.strategy || "") !== strat) return false;
    if (decisionOnly && !r.has_decision) return false;
    if (q) {
      const hay = [r.id, r.topic, r.strategy, r.hypothesis, r.decision_title]
        .join(" ")
        .toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return r.category !== "special" || !q;
  });
}

function renderTable() {
  const rows = filteredRows();
  const tbody = document.getElementById("rdBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="muted">无匹配实验</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const active = r.id === selectedId ? " rd-row-active" : "";
      const title = r.decision_title || r.hypothesis || r.topic || r.id;
      return `<tr class="rd-row${active}" data-id="${esc(r.id)}">
        <td class="muted">${esc(r.date || "—")}</td>
        <td><strong>${esc(r.strategy || "—")}</strong></td>
        <td>
          <div class="rd-id">${esc(r.id)}</div>
          <div class="rd-sub">${esc(title.slice(0, 120))}</div>
        </td>
        <td>${r.has_decision ? verdictPill(r.verdict) : '<span class="muted">无</span>'}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".rd-row").forEach((tr) => {
    tr.addEventListener("click", () => {
      selectedId = tr.getAttribute("data-id");
      renderTable();
      loadDetail(selectedId).catch((e) => {
        document.getElementById("detailPane").innerHTML =
          `<p class="pnl-neg">${esc(String(e))}</p>`;
      });
    });
  });
}

function renderMarkdownBlock(title, text, path) {
  if (!text) return "";
  const preview = text.length > 4000 ? text.slice(0, 4000) + "\n\n…" : text;
  const file = path ? path.split("/").pop() : "";
  return `<section class="rd-detail-section">
    <h3>${esc(title)} ${file ? `<a class="rd-raw-link" href="#" data-file="${esc(file)}">全文</a>` : ""}</h3>
    <pre class="rd-md-preview">${esc(preview)}</pre>
  </section>`;
}

function renderLinks(links) {
  if (!links || !links.length) return "";
  return `<section class="rd-detail-section">
    <h3>results/ 产物链接</h3>
    <ul class="rd-links">${links.map((p) => `<li><code>${esc(p)}</code></li>`).join("")}</ul>
    <p class="muted rd-sub">相对仓库根；若路径在 results/ 下可经本服务 static 打开。</p>
  </section>`;
}

function renderYamls(snippets) {
  const keys = Object.keys(snippets || {});
  if (!keys.length) return "";
  return keys
    .map(
      (k) => `<section class="rd-detail-section">
      <h3><code>${esc(k)}</code></h3>
      <pre class="rd-md-preview">${esc((snippets[k] || "").slice(0, 2000))}</pre>
    </section>`
    )
    .join("");
}

async function loadDetail(id) {
  const pane = document.getElementById("detailPane");
  pane.innerHTML = '<p class="muted">加载详情…</p>';
  const { data } = await api(`/api/rd/experiment/${encodeURIComponent(id)}`);
  pane.innerHTML = `
    <header class="rd-detail-header">
      <h2>${esc(data.id)}</h2>
      <p class="muted">${esc(data.strategy || "")} · ${esc(data.date || "")} · ${esc(data.topic || "")}</p>
      ${data.hypothesis ? `<p>${esc(data.hypothesis)}</p>` : ""}
      ${data.has_decision ? `<p>决策：${verdictPill(data.verdict)} ${esc(data.decision_title || "")}</p>` : ""}
    </header>
    ${renderLinks(data.results_links)}
    ${renderMarkdownBlock("README", data.readme_text, data.readme_path)}
    ${data.decision_text ? renderMarkdownBlock("DECISION", data.decision_text, data.decision_path) : ""}
    ${renderYamls(data.yaml_snippets)}
    <section class="rd-detail-section muted">
      <p>物料目录：<code>${esc(data.dir)}</code></p>
      ${data.rd_loop_yaml ? `<p>rd_loop：<code>${esc(data.rd_loop_yaml)}</code></p>` : ""}
    </section>
  `;
  pane.querySelectorAll(".rd-raw-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const file = a.getAttribute("data-file");
      if (!file) return;
      api(`/api/rd/experiment/${encodeURIComponent(id)}/raw/${encodeURIComponent(file)}`)
        .then(({ data: raw }) => {
          const w = window.open("", "_blank");
          if (w) {
            w.document.write(`<pre>${esc(raw.content)}</pre>`);
            w.document.title = file;
          }
        })
        .catch((e) => alert(String(e)));
    });
  });
}

function populateStrategyFilter(strategies) {
  const sel = document.getElementById("strategyFilter");
  const current = sel.value;
  sel.innerHTML =
    '<option value="">全部</option>' +
    (strategies || []).map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  if (current) sel.value = current;
}

async function refresh() {
  document.getElementById("statusLine").textContent = "加载中…";
  await api("/api/rd/refresh", { method: "POST" });
  const { data, meta } = await api("/api/rd/experiments");
  allRows = data || [];
  populateStrategyFilter(meta.strategies || []);
  renderTable();
  document.getElementById("statusLine").textContent =
    `${meta.count ?? allRows.length} experiments · ${new Date().toLocaleTimeString()}`;
  if (selectedId && allRows.some((r) => r.id === selectedId)) {
    await loadDetail(selectedId);
  }
}

document.getElementById("refreshBtn").addEventListener("click", () =>
  refresh().catch((e) => {
    document.getElementById("statusLine").textContent = String(e);
  })
);
document.getElementById("strategyFilter").addEventListener("change", renderTable);
document.getElementById("searchInput").addEventListener("input", renderTable);
document.getElementById("decisionOnly").addEventListener("change", renderTable);

refresh().catch((e) => {
  document.getElementById("statusLine").textContent = String(e);
});
