// Content script injected ONLY into the Draft Helper's own pages (see the
// `matches` in manifest.json). Its whole job is to let the page know the
// extension is installed, so onboarding can gate on it.
//
// Two channels, because a page can read either without knowing the extension
// exists first:
//   - a DOM marker on <html>, readable synchronously on load
//   - a postMessage, for a page that would rather listen than poll
//
// It announces presence and version. It does NOT expose any capability the
// page could use to read cookies or act on ESPN -- all of that stays behind
// the user clicking the extension's own popup. The page can detect us; it
// cannot drive us.

const VERSION = chrome.runtime.getManifest().version;

document.documentElement.dataset.draftHelperExtension = VERSION;

window.postMessage(
  { source: "draft-helper-extension", installed: true, version: VERSION },
  window.location.origin,
);

// Answer a late-loading page that asks after we've already announced.
window.addEventListener("message", (e) => {
  if (e.source !== window) return;
  if (e.data && e.data.source === "draft-helper-page" && e.data.type === "ping") {
    window.postMessage(
      { source: "draft-helper-extension", installed: true, version: VERSION },
      window.location.origin,
    );
  }
});
