// Provider usage details in the sidebar Usage submenu.
// Requires: `monthly` and `configuredProviders` globals injected server-side.
// Each configured provider renders as a link into its /usage tab, with a
// usage detail line (monthly %, credits, available balance, local tokens).

function formatTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function loadPluginStatus() {
  var menu = document.getElementById('usageSubmenu');
  if (!menu) return;

  Promise.all([
    fetch('/api/cost-plugins/usage').then(function (r) { return r.json(); }).catch(function () { return { plugin_usage: {} }; }),
    fetch('/api/cost-plugins/balances').then(function (r) { return r.json(); }).catch(function () { return { plugin_balances: {} }; }),
    fetch('/api/cost-plugins/summary').then(function (r) { return r.json(); }).catch(function () { return { plugin_summaries: {} }; }),
    fetch('/api/cost-plugins/subscriptions').then(function (r) { return r.json(); }).catch(function () { return { plugin_subscriptions: {} }; })
  ]).then(function (results) {
    var allUsage = results[0].plugin_usage || {};
    var balances = results[1].plugin_balances || {};
    var pluginSummaries = results[2].plugin_summaries || {};
    var subscriptions = results[3].plugin_subscriptions || {};

    var allProviders = Object.keys(allUsage).concat(Object.keys(balances));
    var provs = allProviders.filter(function (v, i, a) { return a.indexOf(v) === i; })
      .filter(function (v) { return configuredProviders.indexOf(v) !== -1; });

    if (!provs.length) {
      menu.style.display = 'none';
      return;
    }

    menu.innerHTML = provs.map(function (prov) {
      var bal = balances[prov];
      var usg = allUsage[prov] || [];
      var totalTokens = usg.reduce(function (s, r) { return s + r.prompt_tokens + r.completion_tokens; }, 0);

      // Fallback to gateway DB when plugin has no usage data
      if (totalTokens === 0 && monthly[prov]) {
        totalTokens = monthly[prov].tokens || 0;
      }
      var sum = pluginSummaries[prov];

      var detail = '';
      if (prov === 'opencode') {
        var sub = subscriptions[prov];
        var ocbal = bal && bal.balance != null ? bal.balance : null;
        var parts = [];
        if (ocbal != null) parts.push('credits: $' + ocbal.toFixed(2));
        if (sub && sub.monthly_pct != null) parts.push('monthly: ' + sub.monthly_pct.toFixed(0) + '%');
        detail = parts.join(' · ');
      } else if (prov === 'commandcode') {
        var ccsub = subscriptions[prov];
        var ccparts = [];
        if (ccsub && ccsub.monthly_credits_remaining != null) {
          ccparts.push('credits: $' + ccsub.monthly_credits_remaining.toFixed(2));
        }
        if (ccsub && ccsub.monthly_pct != null && ccsub.monthly_pct > 0) {
          ccparts.push('monthly: ' + ccsub.monthly_pct.toFixed(0) + '%');
        }
        if (ccparts.length === 0 && ccsub && ccsub.five_hour_pct != null) {
          ccparts.push('5h: ' + ccsub.five_hour_pct.toFixed(0) + '%');
        }
        detail = ccparts.join(' · ');
      } else if (prov === 'deepseek' && sum && sum.balance) {
        detail = 'available balance: $' + sum.balance.available.toFixed(2);
      } else if (bal && bal.balance !== null && bal.balance !== undefined) {
        var currency = bal.currency || 'USD';
        detail = currency + ' ' + bal.balance.toFixed(2) + ' balance';
      } else if (prov === 'llamacpp') {
        detail = formatTokens(totalTokens) + ' tokens · local';
      } else {
        var m = monthly[prov] || {};
        var mr = m.reqs || 0;
        var mt = m.tokens || 0;
        if (mr > 0) detail = mr + ' req · ' + formatTokens(mt) + ' tok this month';
      }

      return '<a href="/usage#' + encodeURIComponent(prov) + '">' +
        '<span class="sb-usage-name">' + prov + '</span>' +
        (detail ? '<span class="sb-usage-detail">' + detail + '</span>' : '') +
        '</a>';
    }).join('');

    syncUsageActive();
  });
}

function syncUsageActive() {
  var menu = document.getElementById('usageSubmenu');
  if (!menu) return;
  var hashProv = (window.location.hash || '').replace(/^#/, '');
  menu.querySelectorAll('a').forEach(function (a) {
    a.classList.toggle('active', a.getAttribute('href') === '/usage#' + hashProv);
  });
}

// Load on page load and every 60 seconds
loadPluginStatus();
setInterval(loadPluginStatus, 60000);
window.addEventListener('hashchange', syncUsageActive);
