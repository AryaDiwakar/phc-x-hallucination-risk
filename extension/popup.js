/* PHC-X popup: settings + service health. */
"use strict";

const DEFAULTS = {
  apiBase: "http://127.0.0.1:8000",
  enabled: true,
  blockOnCritical: true,
  showBadge: true,
};

function $(id) { return document.getElementById(id); }

async function load() {
  const s = await chrome.storage.local.get(Object.keys(DEFAULTS));
  const cfg = { ...DEFAULTS, ...s };
  $("api-base").value = cfg.apiBase;
  $("enabled").checked = cfg.enabled;
  $("block").checked = cfg.blockOnCritical;
  $("badge").checked = cfg.showBadge;
  return cfg;
}

function readForm() {
  return {
    apiBase: ($("api-base").value || DEFAULTS.apiBase).trim().replace(/\/+$/, ""),
    enabled: $("enabled").checked,
    blockOnCritical: $("block").checked,
    showBadge: $("badge").checked,
  };
}

async function save() {
  const settings = readForm();
  await chrome.runtime.sendMessage({ type: "phc-save-settings", settings });
  const status = $("status");
  status.textContent = "settings saved";
  status.className = "ok";
  setTimeout(() => { status.textContent = "–"; status.className = ""; }, 1500);
}

async function checkHealth() {
  const status = $("status");
  status.textContent = "checking…";
  status.className = "";
  const resp = await chrome.runtime.sendMessage({ type: "phc-health" });
  if (resp && resp.ok) {
    status.textContent = "service online";
    status.className = "ok";
    $("model").textContent =
      `model: ${resp.health.model} · uptime ${resp.health.uptime_s}s · ` +
      `${resp.health.n_requests} requests`;
  } else {
    status.textContent = `offline: ${(resp && resp.error) || "no response"}`;
    status.className = "err";
    $("model").textContent = "start it with: phc serve";
  }
}

async function openDashboard() {
  const cfg = readForm();
  await chrome.tabs.create({ url: `${cfg.apiBase}/dashboard/` });
}

document.addEventListener("DOMContentLoaded", async () => {
  await load();
  $("save").onclick = save;
  $("health").onclick = checkHealth;
  $("dash").onclick = openDashboard;
  checkHealth();
});
