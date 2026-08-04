// shadcn-style expandable card stacks.
// Provides: toggleStack(id), expandStack(id), collapseStack(id)
(function () {
  'use strict';

  var ACTIVE = null;

  window.toggleStack = function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    if (el.classList.contains('expanded')) collapseStack(id);
    else expandStack(id);
  };

  window.expandStack = function (id) {
    if (ACTIVE && ACTIVE !== id) collapseStack(ACTIVE);
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.add('expanded');
    el.setAttribute('aria-expanded', 'true');
    ACTIVE = id;
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  window.collapseStack = function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('expanded');
    el.setAttribute('aria-expanded', 'false');
    if (ACTIVE === id) ACTIVE = null;
  };

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && ACTIVE) collapseStack(ACTIVE);
  });
})();
