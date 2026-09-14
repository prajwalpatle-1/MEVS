// Minimal background service worker for MV3
chrome.runtime.onInstalled.addListener(() => {
  console.log('MEVS extension installed');
});
// Allows users to open the side panel by clicking the extension icon
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error(error));