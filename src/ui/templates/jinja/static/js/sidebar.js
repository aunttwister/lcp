// ── Usage sidebar refresh (spin → check mark) ──────────────────────────
// Clicking the ↻ on the Usage nav item re-scrapes the provider cache in the
// background. The arrow spins while the request is in flight, then morphs
// into a ✓ (or ! on failure) with a short fade/scale transition — never an
// instant swap.
var SB_REFRESH_MIN_SPIN = 700; // ms the ↻ keeps spinning before morphing
var _sbUsageRefreshing = false;

function sbUsageRefresh(ev) {
  ev.stopPropagation();
  var btn = document.getElementById('sbUsageRefreshBtn');
  if (!btn || _sbUsageRefreshing) return;
  _sbUsageRefreshing = true;
  btn.classList.remove('is-done', 'is-error');
  btn.classList.add('is-spinning');

  var started = Date.now();

  function settle(ok) {
    // Keep the arrow going around for at least SB_REFRESH_MIN_SPIN so the
    // motion actually reads, then transition into the check mark.
    var wait = Math.max(0, SB_REFRESH_MIN_SPIN - (Date.now() - started));
    setTimeout(function () {
      btn.classList.remove('is-spinning');
      btn.classList.add('is-done');
      btn.classList.toggle('is-error', !ok);
      setSbRefreshIcon(btn, ok ? '✓' : '!');
      setTimeout(function () {
        btn.classList.remove('is-done', 'is-error');
        setSbRefreshIcon(btn, '↻', false);
        _sbUsageRefreshing = false;
      }, 1400);
    }, wait);
  }

  fetch('/api/settings/cache/refresh', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      settle(!!d.ok);
      if (!d.ok && d.error) alert('Refresh failed: ' + d.error);
    })
    .catch(function () { settle(false); });
}

// Swap the icon glyph with a short fade/scale-in (animate) or instantly (not).
function setSbRefreshIcon(btn, glyph, animate) {
  var icon = btn.querySelector('.sb-usage-refresh-icon');
  if (!icon) return;
  icon.textContent = glyph;
  if (animate === false) {
    icon.style.transition = 'none';
    icon.style.opacity = '1';
    icon.style.transform = 'scale(1)';
    return;
  }
  // Start hidden + small, force a reflow, then animate in so the check mark
  // fades/scales in rather than appearing instantly.
  icon.style.transition = 'none';
  icon.style.opacity = '0';
  icon.style.transform = 'scale(0.5)';
  void icon.offsetWidth;
  icon.style.transition = 'opacity 0.28s ease, transform 0.28s ease';
  icon.style.opacity = '1';
  icon.style.transform = 'scale(1)';
}

// Sidebar toggle — shared across all pages.
function toggleSidebar() {
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebarOverlay');
  if (window.innerWidth <= 768) {
    sb.classList.toggle('open');
    ov.classList.toggle('show');
  } else {
    sb.classList.toggle('collapsed');
    localStorage.setItem('lcp-sidebar', sb.classList.contains('collapsed') ? 'collapsed' : 'pinned');
  }
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}

// ── Usage submenu collapse ──────────────────────────────────────────────
// Persists collapsed state per user; auto-expands when on the /usage page.
// Expanding/collapsing animates by measuring the content height and
// animating max-height (0 ⇄ natural-height px).

function animateUsageSubmenu(menu, collapsed) {
  if (!menu) return;
  if (collapsed) {
    // Animate from the current measured height down to 0.
    var h = menu.scrollHeight;
    menu.style.maxHeight = h + 'px';
    void menu.offsetHeight; // force reflow so the start value registers
    menu.style.maxHeight = '0px';
  } else {
    // Animate from 0 up to the natural height, then unlock.
    menu.style.maxHeight = 'none';
    var h = menu.scrollHeight;
    menu.style.maxHeight = '0px';
    void menu.offsetHeight;
    menu.style.maxHeight = h + 'px';
    menu.addEventListener('transitionend', function handler() {
      menu.style.maxHeight = 'none';
      menu.removeEventListener('transitionend', handler);
    });
  }
}

function toggleUsageSubmenu() {
  var menu = document.getElementById('usageSubmenu');
  var chev = document.getElementById('usageChevron');
  if (!menu) return;
  var collapsed = menu.classList.toggle('collapsed');
  if (chev) chev.classList.toggle('collapsed', collapsed);
  localStorage.setItem('lcp-usage-submenu', collapsed ? 'collapsed' : 'open');
  animateUsageSubmenu(menu, collapsed);
}

function initUsageSubmenu() {
  var menu = document.getElementById('usageSubmenu');
  var chev = document.getElementById('usageChevron');
  if (!menu || !chev) return;
  // Auto-expand on the usage page; otherwise honour the saved preference.
  var onUsage = document.querySelector('.sb-usage-nav a.active') !== null;
  var saved = localStorage.getItem('lcp-usage-submenu');
  var collapsed = onUsage ? false : saved === 'collapsed';
  menu.classList.toggle('collapsed', collapsed);
  chev.classList.toggle('collapsed', collapsed);
  // No animation on first paint: jump straight to the right height.
  menu.style.maxHeight = collapsed ? '0px' : 'none';
}

(function () {
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {
    document.getElementById('sidebar').classList.add('collapsed');
  }
  initUsageSubmenu();
})();

// Alert badge polling — updates active alert count on all pages.
(function() {
  function pollAlertBadge() {
    var badge = document.getElementById('sidebarAlertBadge');
    if (!badge) return;
    try {
      fetch('/api/alerts/active').then(function(r) { return r.json(); }).then(function(d) {
        var count = (d.alerts || []).length;
        if (count > 0) {
          badge.textContent = count;
          badge.style.display = 'inline-block';
        } else {
          badge.style.display = 'none';
        }
      });
    } catch(e) {}
  }
  pollAlertBadge();
  setInterval(pollAlertBadge, 15000);
})();
// The Usage submenu (provider links + usage details) is rendered by
// plugin-status.js, which runs after sidebar.js in base.html.
