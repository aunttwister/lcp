// Provider plugin status rows in the sidebar.
// Requires: `monthly` and `configuredProviders` globals injected server-side.

function formatTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function loadPluginStatus() {
  var provRows = document.getElementById('providerPluginRows');
  if (!provRows) return;

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
    var uniqueProvs = allProviders.filter(function (v, i, a) { return a.indexOf(v) === i; }).filter(function (v) { return configuredProviders.indexOf(v) !== -1; });

    if (uniqueProvs.length === 0) {
      provRows.innerHTML = '<div class="sb-provider-empty">No plugins active</div>';
    } else {
      var rows = '';
      uniqueProvs.forEach(function (prov) {
        var bal = balances[prov];
        var usg = allUsage[prov] || [];
        var totalCost = usg.reduce(function (s, r) { return s + r.cost; }, 0);
        var totalTokens = usg.reduce(function (s, r) { return s + r.prompt_tokens + r.completion_tokens; }, 0);

        // Fallback to gateway DB when plugin has no usage data
        if (totalCost === 0 && monthly[prov]) {
          totalCost = monthly[prov].cost || 0;
        }
        if (totalTokens === 0 && monthly[prov]) {
          totalTokens = monthly[prov].tokens || 0;
        }
        var sum = pluginSummaries[prov];

        var detailLine = '';
        if (prov === 'opencode') {
          var om = (sum && sum.monthly) ? sum.monthly : (monthly[prov] || {});
          detailLine = '<span class="sb-provider-detail">' +
            'month: ' + formatTokens(om.tokens || totalTokens) + ' tok';
          if (om.cost) detailLine += ' \u00b7 $' + om.cost.toFixed(4);
          var sub = subscriptions[prov];
          if (sub) {
            detailLine += '<br><span class="sb-sub-detail">';
            var parts = [];
            if (sub.rolling_pct != null) {
              var resetMin = Math.floor(sub.rolling_reset_sec / 60);
              var r = '5h: ' + sub.rolling_pct.toFixed(0) + '% used';
              if (resetMin > 0) r += ' \u00b7 resets in ' + resetMin + 'm';
              parts.push(r);
            }
            if (sub.weekly_pct != null) {
              var wkResetHr = Math.floor(sub.weekly_reset_sec / 3600);
              var w = 'week: ' + sub.weekly_pct.toFixed(0) + '% used';
              if (wkResetHr > 0) w += ' \u00b7 resets in ' + wkResetHr + 'h';
              parts.push(w);
            }
            if (sub.monthly_pct != null) {
              var moResetDay = Math.floor(sub.monthly_reset_sec / 86400);
              var m = 'month: ' + sub.monthly_pct.toFixed(0) + '% used';
              if (moResetDay > 0) m += ' \u00b7 resets in ' + moResetDay + 'd';
              parts.push(m);
            }
            detailLine += parts.join('<br>');
            detailLine += '</span>';
          }
          detailLine += '</span>';
        } else if (prov === 'deepseek' && sum && sum.balance) {
          var cur = sum.balance.currency || 'USD';
          detailLine = '<span class="sb-provider-detail">' + cur + ' ' + sum.balance.available.toFixed(2) + ' available';
          if (sum.balance.spent !== null) detailLine += ' \u00b7 spent ' + cur + ' ' + sum.balance.spent.toFixed(2);
          detailLine += '</span>';
        } else if (bal && bal.balance !== null && bal.balance !== undefined) {
          var currency = bal.currency || 'USD';
          detailLine = '<span class="sb-provider-detail">' + currency + ' ' + bal.balance.toFixed(2) + ' balance</span>';
        } else if (prov === 'llamacpp') {
          detailLine = '<span class="sb-provider-detail">' + formatTokens(totalTokens) + ' tokens \u00b7 local</span>';
        } else {
          var m = monthly[prov] || {};
          var mr = m.reqs || 0;
          var mt = m.tokens || 0;
          if (mr > 0) detailLine = '<span class="sb-provider-detail">' + mr + ' req \u00b7 ' + formatTokens(mt) + ' tok this month</span>';
        }

        rows += '<div class="sb-provider-row">' +
          '<div class="sb-provider-top">' +
          '<span class="sb-provider-name">' + prov + '</span>' +
          '<span class="sb-provider-cost">$' + totalCost.toFixed(4) + '</span>' +
          '</div>' +
          (detailLine ? detailLine : '') +
          '</div>';
      });
      provRows.innerHTML = rows;
    }
  });
}

// Load on page load and every 60 seconds
loadPluginStatus();
setInterval(loadPluginStatus, 60000);
