/* PHC-X dashboard client (vanilla JS, no build step). */
"use strict";

const $ = (id) => document.getElementById(id);
const LEVEL_COLORS = {
  LOW: "#37d399", MODERATE: "#f4c542", HIGH: "#ff9248", CRITICAL: "#ff5c72",
};

function apiBase() {
  const v = ($("api-base").value || "").trim().replace(/\/+$/, "");
  return v || "http://127.0.0.1:8000";
}

async function api(path, opts = {}) {
  const res = await fetch(apiBase() + path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  });
  (Array.isArray(children) ? children : [children]).forEach((c) => c && node.appendChild(c));
  return node;
}

/* ------------------------------------------------------------------ health */
async function refreshHealth() {
  const badge = $("health-badge");
  try {
    const h = await api("/health");
    badge.textContent = `online / ${h.n_requests} req`;
    badge.className = "badge ok";
    $("model-badge").textContent = `model: ${h.model}`;
    $("footer-version").textContent = `API v${h.version}`;
    loadPolicy();
    loadMetrics();
    loadResults();
  } catch (e) {
    badge.textContent = "offline";
    badge.className = "badge err";
    $("model-badge").textContent = "start phc serve";
  }
}

/* ----------------------------------------------------------------- history */
function turnRow(role = "user", content = "") {
  const sel = el("select", {}, []);
  ["user", "assistant", "system"].forEach((r) => {
    const o = el("option", { value: r, text: r });
    if (r === role) o.selected = true;
    sel.appendChild(o);
  });
  const input = el("input", { value: content, placeholder: "turn content…", spellcheck: "false" });
  const del = el("button", { class: "ghost small", text: "×" });
  del.onclick = () => row.remove();
  const row = el("div", { class: "turn" }, [sel, input, del]);
  return row;
}

function readHistory() {
  return [...document.querySelectorAll("#history .turn")].map((row) => ({
    role: row.querySelector("select").value,
    content: row.querySelector("input").value,
  })).filter((t) => t.content.trim());
}

/* --------------------------------------------------------------- analyze */
async function analyze() {
  const prompt = $("prompt").value.trim();
  const button = $("btn-analyze");
  const meta = $("analyze-meta");
  if (!prompt) {
    meta.textContent = "Add a prompt to begin";
    meta.className = "action-meta error";
    $("prompt").focus();
    return;
  }
  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing...";
  meta.textContent = "Tracing prompt, history, and evidence...";
  meta.className = "action-meta";
  try {
    const payload = await api("/v1/analyze", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        conversation_history: readHistory(),
        evidence: $("evidence").value.split("\n").map((s) => s.trim()).filter(Boolean),
        conversation_id: "dashboard",
        prefer_clarify: $("prefer-clarify").checked,
      }),
    });
    renderResult(payload);
    $("analyze-meta").textContent =
      `request ${payload.request_id} · ${payload.latency_ms} ms · model ${payload.model}`;
    loadMetrics();
  } catch (e) {
    meta.textContent = `error: ${e.message}`;
    meta.className = "action-meta error";
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Analyze risk";
  }
}

function renderResult(p) {
  $("result").classList.remove("hidden");
  const score = p.risk_score ?? 0;
  const level = p.risk_level || "LOW";
  const color = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;
  const g = $("gauge");
  g.style.setProperty("--p", `${(score * 360).toFixed(1)}deg`);
  g.style.setProperty("--c", color);
  $("gauge-score").textContent = score.toFixed(2);
  $("gauge-level").textContent = level;

  const dec = p.decision || {};
  const action = $("action");
  action.textContent = dec.action || "–";
  action.style.color = color;
  action.style.borderColor = color;
  $("reason").textContent = dec.reason || "–";
  $("latency").textContent = `${p.latency_ms ?? "–"} ms`;

  // factors
  const factors = $("factors");
  factors.innerHTML = "";
  const ex = p.explanation || {};
  $("explain-method").textContent = ex.method ? `via ${ex.method}` : "";
  const list = ex.top_factors || p.factors || [];
  const max = Math.max(0.001, ...list.map((f) => Math.abs(f.contribution)));
  list.forEach((f) => {
    const w = (Math.abs(f.contribution) / max) * 50;
    const bar = el("div", { class: "bar" });
    const fill = el("i", { class: f.contribution >= 0 ? "pos" : "neg" });
    fill.style.width = `${w}%`;
    bar.appendChild(fill);
    factors.appendChild(el("div", { class: "factor" }, [
      el("span", { class: "name", text: f.feature, title: f.label || f.feature }),
      bar,
      el("span", { class: "val", text: `${f.contribution >= 0 ? "+" : ""}${f.contribution.toFixed(3)}` }),
    ]));
  });
  if (!list.length) factors.appendChild(el("p", { class: "hint", text: "no significant factors" }));

  // signals
  const sig = $("signals");
  sig.innerHTML = "";
  (p.engine_reasons || []).forEach((r) => sig.appendChild(el("div", { class: "signal", text: `⚙ ${r}` })));
  (p.provenance_warnings || []).forEach((r) => sig.appendChild(el("div", { class: "signal", text: `⚑ ${r}` })));
  const ev = p.evidence || {};
  if (ev && Object.keys(ev).length) {
    sig.appendChild(el("div", { class: "signal",
      text: `evidence: coverage ${(ev.coverage_score ?? 0).toFixed(2)}, ` +
            `supporting ${ev.supporting_passages ?? 0}, contradictory ${ev.contradictory_passages ?? 0}` }));
  }
  if (!(p.engine_reasons || []).length && !(p.provenance_warnings || []).length) {
    sig.appendChild(el("div", { class: "signal", text: "no engine warnings" }));
  }
}

/* ----------------------------------------------------------------- policy */
async function loadPolicy() {
  try {
    const pol = await api("/v1/policy");
    const t = pol.thresholds || {};
    setSlider("t-low", "v-low", t.low ?? 0.25);
    setSlider("t-moderate", "v-moderate", t.moderate ?? 0.5);
    setSlider("t-high", "v-high", t.high ?? 0.75);
    $("pref").value = pol.user_preference || "balanced";
  } catch (e) { /* offline */ }
}

function setSlider(id, vid, value) {
  $(id).value = value;
  $(vid).textContent = Number(value).toFixed(2);
}

async function savePolicy() {
  const low = parseFloat($("t-low").value);
  const moderate = parseFloat($("t-moderate").value);
  const high = parseFloat($("t-high").value);
  if (!(low <= moderate && moderate <= high)) {
    $("policy-meta").textContent = "need low ≤ moderate ≤ high";
    return;
  }
  try {
    await api("/v1/policy", {
      method: "PUT",
      body: JSON.stringify({
        thresholds: { low, moderate, high },
        user_preference: $("pref").value,
      }),
    });
    $("policy-meta").textContent = "policy applied";
  } catch (e) {
    $("policy-meta").textContent = `error: ${e.message}`;
  }
}

/* ---------------------------------------------------------------- metrics */
async function loadMetrics() {
  try {
    const m = await api("/v1/metrics");
    $("m-n").textContent = m.n_requests ?? 0;
    $("m-risk").textContent = m.mean_risk ?? "–";
    $("m-labeled").textContent = m.labeled ?? 0;
    const dist = $("actions-dist");
    dist.innerHTML = "";
    const entries = Object.entries(m.action_distribution || {});
    const total = entries.reduce((s, [, n]) => s + n, 0) || 1;
    const colors = {
      GENERATE: "#37d399", GENERATE_WITH_EVIDENCE: "#8bd450",
      RETRIEVE: "#f4c542", CLARIFY: "#ff9248",
      CLARIFY_OR_ABSTAIN: "#ff5c72", ABSTAIN: "#ff5c72",
    };
    entries.forEach(([action, n]) => {
      const row = el("div", { class: "dist-row" });
      const bar = el("div", { class: "bar" });
      const fill = el("i", {});
      fill.style.width = `${(n / total) * 100}%`;
      fill.style.background = colors[action] || "#5aa9ff";
      bar.appendChild(fill);
      row.appendChild(el("span", { text: action }));
      row.appendChild(bar);
      row.appendChild(el("span", { text: `${n} (${((n / total) * 100).toFixed(0)}%)` }));
      dist.appendChild(row);
    });
    if (!entries.length) dist.appendChild(el("p", { class: "hint", text: "no decisions logged yet" }));

    // gates from the latest benchmark result (if present)
    const box = $("gate-box");
    if (m.benchmark_gates) {
      const g = m.benchmark_gates;
      const ok = g.safety && g.coverage_ok;
      box.className = `gates ${ok ? "pass" : "fail"}`;
      box.innerHTML = `<b>gates: ${ok ? "PASS" : "FAIL"}</b> · safety=${g.safety} · ` +
        `coverage=${(g.coverage * 100).toFixed(1)}% · positives caught ` +
        `${(g.positives_caught * 100).toFixed(1)}% · generated risk ` +
        `${(g.risk_generated * 100).toFixed(1)}% vs base ${(g.risk_overall * 100).toFixed(1)}%`;
      box.classList.remove("hidden");
    } else {
      box.classList.add("hidden");
    }
  } catch (e) { /* offline */ }
}

/* ---------------------------------------------------------------- results */
async function loadResults() {
  try {
    const list = await api("/v1/results");
    const box = $("result-files");
    box.innerHTML = "";
    (list.files || []).forEach((f) => {
      const chip = el("span", { class: "chip", text: f });
      chip.onclick = () => {
        [...box.children].forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        showResult(f);
      };
      box.appendChild(chip);
    });
  } catch (e) { /* offline */ }
}

function kv(label, value) {
  const b = el("div", {});
  b.appendChild(el("b", { text: String(value) }));
  b.appendChild(el("span", { text: label }));
  return b;
}

async function showResult(name) {
  const view = $("result-view");
  view.innerHTML = "<p class='hint'>loading…</p>";
  try {
    const data = await api(`/v1/results/${encodeURIComponent(name)}`);
    const wrap = el("div", {});
    const grid = el("div", { class: "kv" });
    const test = data.test && (data.test.calibrated || data.test);
    if (test && test.auroc !== undefined) {
      grid.appendChild(kv("AUROC", test.auroc.toFixed(3)));
      grid.appendChild(kv("AUPRC", test.auprc.toFixed(3)));
      grid.appendChild(kv("F1", test.f1.toFixed(3)));
      grid.appendChild(kv("ECE", test.ece.toFixed(3)));
      grid.appendChild(kv("precision", test.precision.toFixed(3)));
      grid.appendChild(kv("recall", test.recall.toFixed(3)));
    }
    if (data.gates) {
      const g = data.gates;
      grid.appendChild(kv("gates", data.gate_status || (g.safety ? "PASS" : "FAIL")));
      grid.appendChild(kv("coverage", `${(g.coverage * 100).toFixed(1)}%`));
      grid.appendChild(kv("positives caught", `${(g.positives_caught * 100).toFixed(1)}%`));
      grid.appendChild(kv("generated risk", `${(g.risk_generated * 100).toFixed(1)}%`));
    }
    if (data.runs) {
      Object.entries(data.runs).forEach(([k, r]) => {
        if (r.test) grid.appendChild(kv(k, r.test.auroc.toFixed(3)));
      });
    }
    if (data.action_distribution) {
      Object.entries(data.action_distribution).forEach(([k, v]) => {
        if (v.n) grid.appendChild(kv(k, `${v.n} · pos ${(v.hallucination_rate * 100).toFixed(0)}%`));
      });
    }
    if (data.test && data.test.per_category) {
      Object.entries(data.test.per_category).forEach(([k, v]) => {
        grid.appendChild(kv(k, `AUROC ${v.auroc.toFixed(2)}`));
      });
    }
    wrap.appendChild(grid);
    const pre = el("pre", { text: JSON.stringify(data, null, 2) });
    wrap.appendChild(pre);
    view.innerHTML = "";
    view.appendChild(wrap);
  } catch (e) {
    view.innerHTML = `<p class="hint">error: ${e.message}</p>`;
  }
}

/* ------------------------------------------------------------------- init */
function init() {
  if (location.protocol.startsWith("http")) {
    $("api-base").value = location.origin;
  }
  $("history").appendChild(turnRow("user", ""));
  $("history").appendChild(turnRow("assistant", ""));
  $("btn-add-turn").onclick = () => $("history").appendChild(turnRow("user", ""));
  $("btn-analyze").onclick = analyze;
  $("btn-policy").onclick = savePolicy;
  $("btn-metrics").onclick = loadMetrics;
  $("btn-refresh").onclick = refreshHealth;
  $("btn-reset").onclick = async () => { await api("/v1/reset", { method: "POST" }); loadMetrics(); };
  $("prompt").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });
  ["t-low", "t-moderate", "t-high"].forEach((id) => {
    $(id).addEventListener("input", (e) => {
      $(id.replace("t-", "v-")).textContent = Number(e.target.value).toFixed(2);
    });
  });
  $("prompt").value = "What is the current accuracy of Orion-A on the held-out set?";
  refreshHealth();
  setInterval(refreshHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);
