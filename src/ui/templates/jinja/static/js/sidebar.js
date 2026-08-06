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

(function () {
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {
    document.getElementById('sidebar').classList.add('collapsed');
  }
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
