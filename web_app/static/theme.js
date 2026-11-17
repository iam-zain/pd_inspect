/* ──────────────────────────────────────────────────────────────────────────
   Theme switcher: System / Light / Dark.

   Loaded in <head> BEFORE the stylesheets render content, so the stored theme
   is applied without a flash of the wrong colours. The choice is kept in
   localStorage under "pdinspect-theme"; "system" follows the OS setting.
   ────────────────────────────────────────────────────────────────────────── */
(function () {
  var KEY = "pdinspect-theme";
  var VALID = ["system", "light", "dark"];

  function stored() {
    try {
      var v = window.localStorage.getItem(KEY);
      return VALID.indexOf(v) >= 0 ? v : "system";
    } catch (e) {
      return "system"; // private mode / storage blocked
    }
  }

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    // Keep native controls (scrollbars, date pickers) in step with the theme
    var dark = theme === "dark" ||
      (theme === "system" && window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
  }

  function setTheme(theme) {
    if (VALID.indexOf(theme) < 0) theme = "system";
    try {
      window.localStorage.setItem(KEY, theme);
    } catch (e) {
      /* ignore: the theme still applies for this page view */
    }
    apply(theme);
    syncButtons(theme);
  }

  function syncButtons(theme) {
    var btns = document.querySelectorAll(".pd-theme-switch button[data-theme-value]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute("aria-pressed",
        btns[i].getAttribute("data-theme-value") === theme ? "true" : "false");
    }
  }

  apply(stored()); // before first paint

  window.setTheme = setTheme;

  document.addEventListener("DOMContentLoaded", function () {
    syncButtons(stored());
    var btns = document.querySelectorAll(".pd-theme-switch button[data-theme-value]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener("click", function () {
        setTheme(this.getAttribute("data-theme-value"));
      });
    }
  });

  // Follow the OS while "system" is selected
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () {
      if (stored() === "system") apply("system");
    };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }
})();
