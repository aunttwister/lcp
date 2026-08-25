// ── Sidebar cache tools (refresh / clear) ──────────────────────────────
function toggleSbCacheMenu(ev) {
  ev.stopPropagation();
  var menu = document.getElementById('sbCacheMenu');
  if (!menu) return;
  menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

function closeSbCacheMenu() {
  var menu = document.getElementById('sbCacheMenu');
  if (menu) menu.style.display = 'none';
}

document.addEventListener('click', function (ev) {
  var wrap = document.getElementById('sbCacheWrap');
  if (wrap && !wrap.contains(ev.target)) closeSbCacheMenu();
});

function sbCacheFlash(msg) {
  var btn = document.getElementById('sbCacheBtn');
  if (!btn) return;
  btn.textContent = msg;
  setTimeout(function () { if (btn) btn.textContent = '↻'; }, 1200);
}

function sbCacheRefresh(ev) {
  ev.stopPropagation();
  closeSbCacheMenu();
  fetch('/api/settings/cache/refresh', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      sbCacheFlash(d.ok ? '✓' : '!');
      if (!d.ok && d.error) alert('Refresh failed: ' + d.error);
    })
    .catch(function (e) { sbCacheFlash('!'); });
}

function sbCacheClear(ev) {
  ev.stopPropagation();
  if (!confirm('Clear the cached provider data? The background worker will re-scrape shortly.')) return;
  closeSbCacheMenu();
  fetch('/api/settings/cache/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      sbCacheFlash(d.ok ? '✓' : '!');
      if (!d.ok && d.error) alert('Clear failed: ' + d.error);
    })
    .catch(function (e) { sbCacheFlash('!'); });
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
