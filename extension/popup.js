// The popup is deliberately thin: read the active tab's URL, hand it to the
// service worker, show what comes back. All the real work (cookies, the token
// fetch, the POST to the app) is in background.js, which outlives this popup.

const btn = document.getElementById("sync");
const msg = document.getElementById("msg");

function show(text, cls) {
  msg.textContent = text;
  msg.className = "msg " + (cls || "");
}

btn.addEventListener("click", async () => {
  btn.disabled = true;
  show("Syncing…");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.url) {
      show("No active tab.", "err");
      btn.disabled = false;
      return;
    }
    const res = await chrome.runtime.sendMessage({ type: "SYNC", url: tab.url });
    if (res && res.ok) {
      show(`Synced — league ${res.league}, team ${res.team}. Switch to the `
        + `Draft Helper.`, "ok");
    } else {
      show((res && res.error) || "Sync failed.", "err");
      btn.disabled = false;
    }
  } catch (e) {
    show("Sync failed: " + e.message, "err");
    btn.disabled = false;
  }
});
