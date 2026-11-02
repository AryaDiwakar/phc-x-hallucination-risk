/* PHC-X content script: pre-generation risk gate for chat composers.
 *
 * Intercepts Enter / send-button submissions, asks the local PHC-X service
 * for a pre-generation risk score (conversation history + provenance aware),
 * and shows an inline panel with the recommended intervention.  Blocking
 * actions (CLARIFY / ABSTAIN) require an explicit "send anyway". */
"use strict";

const API_DEFAULT = "http://127.0.0.1:8000";
let CFG = { enabled: true, blockOnCritical: false };
let bypassUntil = 0;

chrome.storage.local.get({ enabled: true, blockOnCritical: false, apiBase: API_DEFAULT })
  .then((s) => { CFG = s; });
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) CFG.enabled = changes.enabled.newValue;
  if (changes.blockOnCritical) CFG.blockOnCritical = changes.blockOnCritical.newValue;
  if (changes.apiBase) CFG.apiBase = changes.apiBase.newValue;
});

/* ------------------------------------------------------------------ utils */
function isComposer(el) {
  if (!el) return false;
  if (el.tagName === "TEXTAREA") return true;
  if (el.isContentEditable) return true;
  return false;
}

function composerText(el) {
  if (!el) return "";
  return (el.tagName === "TEXTAREA" || el.tagName === "INPUT")
    ? el.value
    : (el.innerText || el.textContent || "");
}

function clearComposer(el) {
  if (!el) return;
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    el.value = "";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  } else {
    el.innerHTML = "";
    el.dispatchEvent(new InputEvent("input", { bubbles: true }));
  }
}

function setComposerText(el, text) {
  if (!el) return;
  el.focus();
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
    const proto = el.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(el, text);
    el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
  } else {
    el.textContent = text;
    el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
  }
}

const HISTORY_SELECTORS = [
  // ChatGPT marks each turn with this role attribute. Prefer it over generic
  // message selectors so the full transcript is captured exactly once.
  "[data-message-author-role='user'], [data-message-author-role='assistant']",
  "[data-role='user'], [data-role='assistant']",
  "[class*='message'][data-author]",
  ".message",
  "[class*='turn']",
  // exact class-token match, so it cannot over-match the way a substring
  // selector would (".msg" is a substring of ".message")
  "[class~='msg']",
];

function collectHistory() {
  for (const sel of HISTORY_SELECTORS) {
    const nodes = [...document.querySelectorAll(sel)];
    if (nodes.length < 2) continue;
    const turns = [];
    for (const n of nodes) {
      const text = (n.innerText || "").trim();
      if (!text) continue;
      let role = "assistant";
      const r =
        n.getAttribute("data-message-author-role") ||
        n.getAttribute("data-role") ||
        n.getAttribute("data-author") ||
        "";
      const low = r.toLowerCase();
      if (low.includes("user") || low.includes("human") || low.includes("you")) role = "user";
      else if (low.includes("assistant") || low.includes("bot") || low.includes("gpt") ||
               low.includes("claude") || low.includes("model")) role = "assistant";
      else if (/^you|user/i.test(n.className || "")) role = "user";
      // Do not truncate by turn count or character count: PHC-X's consistency
      // and contamination checks need the whole conversation for context.
      turns.push({ role, content: text });
    }
    if (turns.length >= 2) return turns;
  }
  return [];
}

function sendMessage(msg) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(msg, (resp) => {
        if (chrome.runtime.lastError) resolve({ ok: false, error: chrome.runtime.lastError.message });
        else resolve(resp || { ok: false, error: "no response" });
      });
    } catch (e) {
      resolve({ ok: false, error: String(e) });
    }
  });
}

/* ------------------------------------------------------------------- panel */
let panel = null;

function ensurePanel(anchor) {
  if (panel && panel.isConnected) return panel;
  panel = document.createElement("div");
  panel.className = "phcx-panel";
  document.body.appendChild(panel);
  positionPanel(anchor);
  return panel;
}

function positionPanel(anchor) {
  if (!panel || !anchor) return;
  const r = anchor.getBoundingClientRect();
  const top = Math.max(8, r.top + window.scrollY - 8);
  panel.style.top = `${top}px`;
  const left = Math.min(
    Math.max(8, r.left + window.scrollX),
    window.scrollX + document.documentElement.clientWidth - 340,
  );
  panel.style.left = `${left}px`;
}

function hidePanel(delay = 0) {
  const p = panel;
  if (!p) return;
  setTimeout(() => {
    p.classList.remove("phcx-show");
    setTimeout(() => p.remove(), 250);
    panel = null;
  }, delay);
}

const LEVEL_COLORS = {
  LOW: "#37d399", MODERATE: "#f4c542", HIGH: "#ff9248", CRITICAL: "#ff5c72",
};

function renderLoading(anchor) {
  const p = ensurePanel(anchor);
  p.innerHTML =
    `<div class="phcx-head"><span class="phcx-logo">PHC-X</span>` +
    `<span class="phcx-loading">checking prompt…</span></div>`;
  p.classList.add("phcx-show");
}

function renderResult(res, anchor, onProceed) {
  const p = ensurePanel(anchor);
  const score = res.risk_score ?? 0;
  const level = res.risk_level || "LOW";
  const color = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;
  const dec = res.decision || {};
  // `blockOnCritical` is the user's "pause on clarify/abstain" preference:
  // when it is off the guard still shows the panel but never blocks the send.
  const blocking = CFG.blockOnCritical !== false &&
    ["CLARIFY", "CLARIFY_OR_ABSTAIN", "ABSTAIN"].includes(dec.action);
  const factors = (res.explanation && res.explanation.top_factors || []).slice(0, 3);
  const reasons = (res.engine_reasons || []).slice(0, 3);

  p.innerHTML = "";
  const head = document.createElement("div");
  head.className = "phcx-head";
  head.innerHTML =
    `<span class="phcx-logo">PHC-X</span>` +
    `<span class="phcx-score" style="color:${color}">${score.toFixed(2)} ${level}</span>` +
    `<span class="phcx-action" style="border-color:${color};color:${color}">${dec.action || ""}</span>`;
  p.appendChild(head);

  if (dec.reason) {
    const reason = document.createElement("div");
    reason.className = "phcx-reason";
    reason.textContent = dec.reason;
    p.appendChild(reason);
  }

  if (res.risk_adjustment && Number.isFinite(res.model_risk_score) &&
      Math.abs(res.model_risk_score - score) > 0.001) {
    const adjustment = document.createElement("div");
    adjustment.className = "phcx-reason";
    adjustment.textContent =
      `Raw model estimate ${res.model_risk_score.toFixed(2)}; ` +
      `source-aware adjustment: ${res.risk_adjustment}.`;
    p.appendChild(adjustment);
  }
  if (res.risk_method && res.risk_method.startsWith("heuristic_rf_ensemble")) {
    const method = document.createElement("div");
    method.className = "phcx-reason";
    method.textContent = `Experimental ensemble: 85% transparent heuristic + 15% trained model (raw model estimate ${Number(res.model_risk_score || 0).toFixed(2)}).`;
    p.appendChild(method);
  }

  if (res.context_support && res.context_support.supported) {
    const grounded = document.createElement("div");
    grounded.className = "phcx-reason";
    grounded.textContent = "Directly grounded in an earlier user-provided turn (not independently verified).";
    p.appendChild(grounded);
  }

  if (factors.length) {
    const ul = document.createElement("div");
    ul.className = "phcx-factors";
    factors.forEach((f) => {
      const row = document.createElement("div");
      row.innerHTML =
        `<span>${f.feature}</span>` +
        `<b style="color:${f.contribution >= 0 ? "#ff5c72" : "#37d399"}">` +
        `${f.contribution >= 0 ? "+" : ""}${f.contribution.toFixed(3)}</b>`;
      ul.appendChild(row);
    });
    p.appendChild(ul);
  }

  if (reasons.length) {
    const r = document.createElement("div");
    r.className = "phcx-reasons";
    r.textContent = reasons.join(" · ");
    p.appendChild(r);
  }

  const buttons = document.createElement("div");
  buttons.className = "phcx-buttons";

  const improve = document.createElement("button");
  improve.textContent = "Improve prompt";
  improve.onclick = async () => {
    improve.disabled = true;
    improve.textContent = "improving…";
    p.dataset.phcxEnhancing = "true";
    const resp = await sendMessage({
      type: "phc-enhance",
      payload: {
        prompt: composerText(anchor).trim(),
        conversation_history: collectHistory(),
        evidence: [],
        conversation_id: location.hostname,
      },
    });
    if (!resp.ok) {
      p.dataset.phcxEnhancing = "false";
      improve.disabled = false;
      improve.textContent = "Improve prompt";
      const message = document.createElement("div");
      message.className = "phcx-reason";
      message.textContent = `Could not create a rewrite: ${resp.error || "service error"}`;
      p.appendChild(message);
      return;
    }
    renderEnhancement(anchor, resp.result);
  };
  buttons.appendChild(improve);

  if (blocking) {
    // intervention gate: nothing is sent until the user explicitly overrides
    const proceed = document.createElement("button");
    proceed.className = "phcx-primary";
    proceed.textContent = "send anyway";
    proceed.onclick = () => { hidePanel(); onProceed(); };
    buttons.appendChild(proceed);
    const dismiss = document.createElement("button");
    dismiss.textContent = "edit prompt";
    dismiss.onclick = () => { hidePanel(); anchor && anchor.focus && anchor.focus(); };
    buttons.appendChild(dismiss);
    p.appendChild(buttons);
  } else {
    // informative only: auto-continue after a short glance
    const proceed = document.createElement("button");
    proceed.className = "phcx-primary";
    proceed.textContent = "send now";
    proceed.onclick = () => { hidePanel(); onProceed(); };
    buttons.appendChild(proceed);
    const dismiss = document.createElement("button");
    dismiss.textContent = "dismiss";
    dismiss.onclick = () => { hidePanel(); onProceed(); };
    buttons.appendChild(dismiss);
    p.appendChild(buttons);
    setTimeout(() => {
      if (panel === p && p.dataset.phcxEnhancing !== "true") {
        hidePanel(); onProceed();
      }
    }, 2600);
  }
  p.classList.add("phcx-show");
}

function renderEnhancement(anchor, result) {
  const p = ensurePanel(anchor);
  p.dataset.phcxEnhancing = "false";
  p.innerHTML = "";
  const head = document.createElement("div");
  head.className = "phcx-head";
  const title = document.createElement("span");
  title.className = "phcx-logo";
  title.textContent = "PHC-X / rewrite";
  head.appendChild(title);
  p.appendChild(head);

  const summary = document.createElement("div");
  summary.className = "phcx-reason";
  const before = Number(result.original_risk || 0).toFixed(2);
  const after = Number(result.estimated_risk || 0).toFixed(2);
  summary.textContent = result.risk_reduced
    ? `Estimated risk: ${before} → ${after}. Review the rewrite before sending.`
    : `No lower-risk wording was found (estimate ${before} → ${after}). This rewrite adds grounding instructions; it cannot create missing evidence.`;
  if (result.enhancement_method) {
    summary.textContent += ` Method: ${result.enhancement_method}.`;
  }
  p.appendChild(summary);

  const preview = document.createElement("div");
  preview.className = "phcx-rewrite";
  preview.textContent = result.enhanced_prompt || "No rewrite returned.";
  p.appendChild(preview);

  const buttons = document.createElement("div");
  buttons.className = "phcx-buttons";
  const useDraft = document.createElement("button");
  useDraft.className = "phcx-primary";
  useDraft.textContent = "Use draft";
  useDraft.onclick = () => {
    setComposerText(anchor, result.enhanced_prompt || "");
    hidePanel();
  };
  buttons.appendChild(useDraft);
  const cancel = document.createElement("button");
  cancel.textContent = "Keep original";
  cancel.onclick = () => hidePanel();
  buttons.appendChild(cancel);
  p.appendChild(buttons);
  p.classList.add("phcx-show");
}

function renderError(anchor, message, proceed) {
  // fail open: the guard must never brick the user's chat
  const p = ensurePanel(anchor);
  p.innerHTML =
    `<div class="phcx-head"><span class="phcx-logo">PHC-X</span>` +
    `<span class="phcx-score" style="color:#8b97ab">service offline</span></div>` +
    `<div class="phcx-reason">${message}</div>`;
  const buttons = document.createElement("div");
  buttons.className = "phcx-buttons";
  const proceedBtn = document.createElement("button");
  proceedBtn.className = "phcx-primary";
  proceedBtn.textContent = "send anyway";
  proceedBtn.onclick = () => { hidePanel(); proceed(); };
  buttons.appendChild(proceedBtn);
  const editBtn = document.createElement("button");
  editBtn.textContent = "dismiss";
  editBtn.onclick = () => { hidePanel(); };
  buttons.appendChild(editBtn);
  p.appendChild(buttons);
  p.classList.add("phcx-show");
  setTimeout(() => { if (panel === p) { hidePanel(); proceed(); } }, 3000);
}

/* ------------------------------------------------------------------ guard */
async function guard(text, anchor, proceed) {
  renderLoading(anchor);
  const resp = await sendMessage({
    type: "phc-analyze",
    payload: {
      prompt: text,
      conversation_history: collectHistory(),
      evidence: [],
      conversation_id: location.hostname,
    },
  });
  if (!resp.ok) {
    if (resp.skipped) { hidePanel(); proceed(); return; }
    renderError(anchor, `${resp.error || "request failed"} — start it with: phc serve`, proceed);
    return;
  }
  renderResult(resp.result, anchor, proceed);
}

/* -------------------------------------------------------- event intercept */
function withBypass(fn) {
  bypassUntil = Date.now() + 4000;
  try { fn(); } finally { setTimeout(() => { bypassUntil = 0; }, 500); }
}

function resynthesizeEnter(el, original) {
  const init = {
    key: "Enter", code: "Enter", keyCode: 13, which: 13,
    bubbles: true, cancelable: true, composed: true,
  };
  withBypass(() => {
    ["keydown", "keypress", "keyup"].forEach((type) => {
      el.dispatchEvent(new KeyboardEvent(type, init));
    });
  });
}

function resynthesizeClick(btn) {
  withBypass(() => btn.click());
}

function isSendButton(btn) {
  if (!btn || btn.nodeType !== 1) return false;
  const label = (btn.innerText || btn.getAttribute("aria-label") || btn.title || "").trim().toLowerCase();
  if (!label) return false;
  if (btn.closest(".phcx-panel")) return false;
  return ["send", "send message", "submit", "arrowup", "↑"].some((k) => label === k || label.includes(k));
}

document.addEventListener("keydown", (e) => {
  if (!CFG.enabled || Date.now() < bypassUntil) return;
  if (e.key !== "Enter" || e.shiftKey || e.ctrlKey || e.metaKey || e.isComposing) return;
  if (!isComposer(e.target)) return;
  const text = composerText(e.target).trim();
  if (!text) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  const anchor = e.target;
  guard(text, anchor, () => resynthesizeEnter(anchor, e));
}, true);

document.addEventListener("click", (e) => {
  if (!CFG.enabled || Date.now() < bypassUntil) return;
  const btn = e.target.closest && e.target.closest("button, [role='button'], [aria-label*='end']");
  if (!isSendButton(btn)) return;
  const composer = document.querySelector("textarea:focus, [contenteditable='true']:focus")
    || document.querySelector("textarea, [contenteditable='true']");
  const text = composerText(composer).trim();
  if (!text) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  const anchor = composer || btn;
  guard(text, anchor, () => resynthesizeClick(btn));
}, true);

/* keep the panel anchored while typing/scrolling */
document.addEventListener("focusin", (e) => {
  if (panel && isComposer(e.target)) positionPanel(e.target);
});
