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
function toggleUsageSubmenu() {
  var menu = document.getElementById('usageSubmenu');
  var chev = document.getElementById('usageChevron');
  if (!menu) return;
  var collapsed = menu.classList.toggle('collapsed');
  if (chev) chev.classList.toggle('collapsed', collapsed);
  localStorage.setItem('lcp-usage-submenu', collapsed ? 'collapsed' : 'open');
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
