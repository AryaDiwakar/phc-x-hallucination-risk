/* PHC-X content script: pre-generation risk gate for chat composers.
 *
 * Intercepts Enter / send-button submissions, asks the local PHC-X service
 * for a pre-generation risk score (conversation history + provenance aware),
 * and shows an inline panel with the recommended intervention.  Blocking
 * actions (CLARIFY / ABSTAIN) require an explicit "send anyway". */
"use strict";

const API_DEFAULT = "http://127.0.0.1:8000";
let CFG = { enabled: true, blockOnCritical: true };
let bypassUntil = 0;

chrome.storage.local.get({ enabled: true, blockOnCritical: true, apiBase: API_DEFAULT })
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

const HISTORY_SELECTORS = [
  "[data-message-author-role]",
  "[data-role='user'], [data-role='assistant']",
  "[class*='message'][data-author]",
  ".message",
  "[class*='turn']",
  // exact class-token match, so it cannot over-match the way a substring
  // selector would (".msg" is a substring of ".message")
  "[class~='msg']",
];

function collectHistory(limit = 8) {
  for (const sel of HISTORY_SELECTORS) {
    const nodes = [...document.querySelectorAll(sel)];
    if (nodes.length < 2) continue;
    const turns = [];
    for (const n of nodes.slice(-limit)) {
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
      if (text.length > 1200) continue;
      turns.push({ role, content: text.slice(0, 1200) });
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
      if (panel === p) { hidePanel(); onProceed(); }
    }, 2600);
  }
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
