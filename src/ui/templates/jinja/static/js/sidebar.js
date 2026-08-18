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

// Usage submenu — provider links deep-link into the /usage page tabs.
(function() {
  var KEY = 'lcp-prov-usage';
  var toggle = document.getElementById('usageSubmenuToggle');
  var menu = document.getElementById('usageSubmenu');
  if (!toggle || !menu) return;

  var provs = (typeof configuredProviders !== 'undefined' && Array.isArray(configuredProviders)) ? configuredProviders : [];
  if (!provs.length) {
    toggle.style.display = 'none';
    menu.style.display = 'none';
    return;
  }

  menu.innerHTML = provs.map(function(p) {
    return '<a href="/usage#' + encodeURIComponent(p) + '">' + p + '</a>';
  }).join('');

  function syncActive() {
    var hashProv = (window.location.hash || '').replace(/^#/, '');
    menu.querySelectorAll('a').forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '/usage#' + hashProv);
    });
  }

  function apply(state) {
    var collapsed = state === 'collapsed';
    toggle.classList.toggle('collapsed', collapsed);
    menu.classList.toggle('collapsed', collapsed);
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }

  apply(localStorage.getItem(KEY) || 'expanded');
  syncActive();

  toggle.addEventListener('click', function () {
    var next = menu.classList.contains('collapsed') ? 'expanded' : 'collapsed';
    localStorage.setItem(KEY, next);
    apply(next);
  });
  toggle.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggle.click();
    }
  });
  window.addEventListener('hashchange', syncActive);
})();
