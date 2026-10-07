/* PHC-X extension background service worker.
 * All API traffic flows through here so https pages can still reach the
 * local http://127.0.0.1 service (extension host permissions). */
"use strict";

const DEFAULTS = {
  apiBase: "http://127.0.0.1:8000",
  enabled: true,
  blockOnCritical: true,
  showBadge: true,
};

async function settings() {
  const stored = await chrome.storage.local.get(Object.keys(DEFAULTS));
  return { ...DEFAULTS, ...stored };
}

async function analyze(payload, cfg) {
  const res = await fetch(cfg.apiBase.replace(/\/+$/, "") + "/v1/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function updateBadge(result, cfg) {
  if (!cfg.showBadge || !chrome.action) return;
  const pct = Math.round((result.risk_score || 0) * 100);
  const color =
    result.risk_level === "CRITICAL" ? "#ff5c72" :
    result.risk_level === "HIGH" ? "#ff9248" :
    result.risk_level === "MODERATE" ? "#f4c542" : "#37d399";
  chrome.action.setBadgeBackgroundColor({ color });
  chrome.action.setBadgeText({ text: String(Math.min(pct, 99)) });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "phc-analyze") {
    (async () => {
      try {
        const cfg = await settings();
        if (!cfg.enabled) {
          sendResponse({ ok: false, skipped: true });
          return;
        }
        const result = await analyze(msg.payload, cfg);
        updateBadge(result, cfg);
        sendResponse({ ok: true, result, settings: cfg });
      } catch (err) {
        sendResponse({ ok: false, error: String(err && err.message || err) });
      }
    })();
    return true; // async response
  }
  if (msg && msg.type === "phc-health") {
    (async () => {
      try {
        const cfg = await settings();
        const res = await fetch(cfg.apiBase.replace(/\/+$/, "") + "/health");
        sendResponse({ ok: res.ok, health: await res.json(), settings: cfg });
      } catch (err) {
        sendResponse({ ok: false, error: String(err && err.message || err) });
      }
    })();
    return true;
  }
  if (msg && msg.type === "phc-save-settings") {
    chrome.storage.local.set(msg.settings || {}).then(() => sendResponse({ ok: true }));
    return true;
  }
  return false;
});
