// src/theme.js
export function getSavedTheme() {
  return localStorage.getItem("theme") || "system";
}
export function computeIsDark(mode = getSavedTheme()) {
  return mode === "dark" ||
    (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
}
export function applyTheme(mode = getSavedTheme()) {
  const dark = computeIsDark(mode);
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.dataset.theme = dark ? "dark" : "light";
}
export function startThemeListener() {
  applyTheme();
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const onMedia = () => getSavedTheme() === "system" && applyTheme("system");
  media.addEventListener("change", onMedia);
  window.addEventListener("storage", (e) => {
    if (e.key === "theme") applyTheme(e.newValue || "system");
  });
  window.addEventListener("theme:change", () => applyTheme());
}
