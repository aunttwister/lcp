/* ── Sidebar toggle ── */
function toggleSidebar() {
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebarOverlay');
  if (window.innerWidth <= 768) {
    // mobile: overlay mode
    sb.classList.toggle('open');
    ov.classList.toggle('show');
  } else {
    // desktop: push mode
    sb.classList.toggle('collapsed');
    localStorage.setItem('lcp-sidebar', sb.classList.contains('collapsed') ? 'collapsed' : 'pinned');
  }
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}
// restore state
(function() {
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {
    document.getElementById('sidebar').classList.add('collapsed');
  }
})();

/* ── Copy URL helper ── */
function copyUrl(url) {
  navigator.clipboard.writeText(url).then(function() {
    // brief visual feedback
  });
}

var ppData = {pp_data_json};
var pmData = {pm_data_json};
var currentView = 'aggregate';
var costChart = null, latChart = null;
var profileColors = {
  'l2': {bg: 'hsla(142.1, 70.6%, 45.3%, 0.4)', border: 'hsl(142.1, 70.6%, 45.3%)'},
  'l1': {bg: 'hsla(190, 80%, 50%, 0.4)', border: 'hsl(190, 80%, 50%)'},
  'career': {bg: 'hsla(43.3, 96.4%, 56.3%, 0.4)', border: 'hsl(43.3, 96.4%, 56.3%)'},
  'cron': {bg: 'hsla(0, 83.2%, 60.2%, 0.4)', border: 'hsl(0, 83.2%, 60.2%)'}
};

function buildCharts(view) {
  if (costChart) { costChart.destroy(); costChart = null; }
  if (latChart) { latChart.destroy(); latChart = null; }

  var darkOpts = {
    responsive: true,
    plugins: { legend: { labels: { color: '#a1a1aa' } } },
    scales: {
      x: { ticks: { color: '#a1a1aa', maxTicksLimit: 10 } },
      y: { ticks: { color: '#a1a1aa' } }
    }
  };

  var costDatasets, latDatasets, labels;

  if (view === 'aggregate') {
    labels = {ts_dates_json};
    costDatasets = [{
      label: 'Daily Cost ($)',
      data: {ts_costs_json},
      backgroundColor: 'hsla(142.1, 70.6%, 45.3%, 0.4)',
      borderColor: 'hsl(142.1, 70.6%, 45.3%)',
      borderWidth: 1
    }];
    latDatasets = [{
      label: 'Avg Latency (ms)',
      data: {ts_lats_json},
      borderColor: 'hsl(190, 80%, 50%)',
      backgroundColor: 'hsla(190, 80%, 50%, 0.1)',
      fill: true, tension: 0.3
    }];
  } else if (view === 'per-profile') {
    labels = ppData.dates;
    costDatasets = [];
    latDatasets = [];
    var profs = Object.keys(ppData.profiles);
    for (var i = 0; i < profs.length; i++) {
      var p = profs[i];
      var c = profileColors[p] || {bg: 'hsla(0,0%,50%,0.4)', border: 'hsl(0,0%,50%)'};
      costDatasets.push({
        label: p.toUpperCase() + ' Cost ($)',
        data: ppData.profiles[p].costs,
        backgroundColor: c.bg,
        borderColor: c.border,
        borderWidth: 1
      });
      latDatasets.push({
        label: p.toUpperCase() + ' Latency (ms)',
        data: ppData.profiles[p].lats,
        borderColor: c.border,
        backgroundColor: c.bg,
        fill: false, tension: 0.3
      });
    }
  } else if (view === 'per-model') {
    labels = pmData.dates;
    costDatasets = [];
    latDatasets = [];
    var modelHues = ['142.1', '190', '43.3', '0', '280', '30'];
    var models = Object.keys(pmData.models);
    for (var i = 0; i < models.length; i++) {
      var m = models[i];
      var hue = modelHues[i % modelHues.length];
      costDatasets.push({
        label: m + ' Cost ($)',
        data: pmData.models[m].costs,
        backgroundColor: 'hsla(' + hue + ', 70%, 50%, 0.4)',
        borderColor: 'hsl(' + hue + ', 70%, 50%)',
        borderWidth: 1
      });
      latDatasets.push({
        label: m + ' Latency (ms)',
        data: pmData.models[m].lats,
        borderColor: 'hsl(' + hue + ', 70%, 50%)',
        backgroundColor: 'hsla(' + hue + ', 70%, 50%, 0.1)',
        fill: false, tension: 0.3
      });
    }
  }

  var ctx1 = document.getElementById('costChart');
  if (ctx1) {
    costChart = new Chart(ctx1, {
      type: 'bar',
      data: { labels: labels, datasets: costDatasets },
      options: darkOpts
    });
  }
  var ctx2 = document.getElementById('latencyChart');
  if (ctx2) {
    latChart = new Chart(ctx2, {
      type: 'line',
      data: { labels: labels, datasets: latDatasets },
      options: darkOpts
    });
  }
}

function switchView(view) {
  currentView = view;
  document.getElementById('btnAgg').className = view === 'aggregate' ? 'view-toggle active' : 'view-toggle';
  document.getElementById('btnPP').className = view === 'per-profile' ? 'view-toggle active' : 'view-toggle';
  document.getElementById('btnPM').className = view === 'per-model' ? 'view-toggle active' : 'view-toggle';
  buildCharts(view);
}

buildCharts('aggregate');

/* ── Provider Management ── */
var provData = {};
var provPresets = {};
var _provTestPassed = false;
var _dirtyChains = {};
var _activeTab = '';
window._hostUrl = '{host_url}';

function api(method, url, body) {
  var opts = { method: method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) { return r.json(); });
}

/* ── Modal Open / Close ── */
function openProviderModal() {
  // Collapse sidebar
  var sidebar = document.getElementById('sidebar');
  if (sidebar && !sidebar.classList.contains('collapsed')) {
    sidebar.classList.add('collapsed');
  }
  document.getElementById('provModal').classList.add('open');
  _activeTab = '';
  loadProviders();
}

function closeProviderModal() {
  document.getElementById('provModal').classList.remove('open');
  // Restore sidebar
  var sidebar = document.getElementById('sidebar');
  if (sidebar && sidebar.classList.contains('collapsed')) {
    sidebar.classList.remove('collapsed');
  }
}

/* ── Tab Switching ── */
function switchTab(profile) {
  _activeTab = profile;
  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
  document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
  var btn = document.getElementById('tab-'+profile);
  var panel = document.getElementById('panel-'+profile);
  if (btn) btn.classList.add('active');
  if (panel) panel.classList.add('active');
  document.getElementById('tabDropdown').value = profile;
}

/* ── Load & Render ── */
function loadProviders() {
  api('GET', '/api/providers').then(function(d) {
    provData = d;
    _dirtyChains = JSON.parse(JSON.stringify(d.profile_chains || {}));
    renderTabsAndPanels();
    renderProvList();
  });
  api('GET', '/api/providers/presets').then(function(d) {
    provPresets = d.presets;
    var sel = document.getElementById('provPreset');
    sel.innerHTML = '<option value="">-- Custom --</option>';
    Object.keys(provPresets).forEach(function(k) {
      sel.innerHTML += '<option value="'+k+'">'+k+'</option>';
    });
  });
}

function renderTabsAndPanels() {
  var chains = provData.profile_chains || {};
  var profiles = Object.keys(chains);

  // Tabs
  var tabsHtml = '';
  profiles.forEach(function(p) {
    tabsHtml += '<button class="tab-btn" id="tab-'+p+'" onclick="switchTab(\\''+p+'\\')">'+p.toUpperCase()+'</button>';
  });
  document.getElementById('modalTabs').innerHTML = tabsHtml;

  // Mobile dropdown
  var dd = document.getElementById('tabDropdown');
  dd.innerHTML = profiles.map(function(p) { return '<option value="'+p+'">'+p.toUpperCase()+'</option>'; }).join('');

  // Panels
  var panelsHtml = '';
  profiles.forEach(function(pname) {
    var chain = _dirtyChains[pname] || chains[pname] || [];
    panelsHtml += '<div class="tab-panel" id="panel-'+pname+'">';
    panelsHtml += '<div class="section-label" style="display:flex;justify-content:space-between;align-items:center">';
    panelsHtml += '<span>Fallback Chain — '+pname.toUpperCase()+'</span>';
    panelsHtml += '<button class="btn-sm btn-primary" onclick="addChainItem(\\''+pname+'\\')">+ Add Step</button>';
    panelsHtml += '</div>';
    if (chain.length === 0) {
      panelsHtml += '<div class="empty" style="padding:1rem;font-size:0.75rem">No providers in chain. Add one.</div>';
    } else {
      panelsHtml += '<ul class="chain-list" id="chain-'+pname+'">';
      chain.forEach(function(c, i) {
        panelsHtml += '<li class="chain-item" data-idx="'+i+'">';
        panelsHtml += '<span class="drag-handle">⋮⋮</span>';
        panelsHtml += '<select class="chain-prov-select" onchange="updateChainItem(\\''+pname+'\\','+i+', this.value, null)" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
        Object.keys(provData.providers || {}).forEach(function(pn) {
          var sel = c.provider === pn ? ' selected' : '';
          panelsHtml += '<option value="'+pn+'"'+sel+'>'+pn+'</option>';
        });
        panelsHtml += '</select>';
        panelsHtml += '<select class="chain-model-select" onchange="updateChainItem(\\''+pname+'\\','+i+', null, this.value)" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
        var models = (provData.providers || {})[c.provider]?.models || [];
        models.forEach(function(m) {
          var sel = c.model === m ? ' selected' : '';
          panelsHtml += '<option value="'+m+'"'+sel+'>'+m+'</option>';
        });
        panelsHtml += '</select>';
        panelsHtml += '<button class="btn-sm" style="margin-left:auto" onclick="removeChainItem(\\''+pname+'\\','+i+')">✕</button>';
        panelsHtml += '</li>';
      });
      panelsHtml += '</ul>';
    }
    panelsHtml += '</div>';
  });
  document.getElementById('modalBody').innerHTML = panelsHtml;

  // Init SortableJS
  profiles.forEach(function(pname) {
    var listEl = document.getElementById('chain-'+pname);
    if (listEl) {
      new Sortable(listEl, {
        animation: 150, handle: '.drag-handle',
        onEnd: function() { rebuildDirtyChain(pname); }
      });
    }
  });

  // Select first tab
  if (profiles.length > 0 && !_activeTab) switchTab(profiles[0]);
  else if (_activeTab) switchTab(_activeTab);
}

function rebuildDirtyChain(profile) {
  var items = document.querySelectorAll('#chain-'+profile+' .chain-item');
  var chain = [];
  items.forEach(function(item) {
    var pSel = item.querySelector('.chain-prov-select');
    var mSel = item.querySelector('.chain-model-select');
    chain.push({provider: pSel.value, model: mSel.value});
  });
  _dirtyChains[profile] = chain;
}

function updateChainItem(profile, idx, newProvider, newModel) {
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  if (newProvider !== null && newProvider !== undefined) _dirtyChains[profile][idx].provider = newProvider;
  if (newModel !== null && newModel !== undefined) _dirtyChains[profile][idx].model = newModel;
}

function addChainItem(profile) {
  var provs = Object.keys(provData.providers || {});
  if (provs.length === 0) { alert('Add a provider first'); return; }
  if (!_dirtyChains[profile]) _dirtyChains[profile] = [];
  var p = provs[0];
  var m = (provData.providers[p]?.models || [])[0] || 'default';
  _dirtyChains[profile].push({provider: p, model: m});
  renderTabsAndPanels();
  switchTab(profile);
}

function removeChainItem(profile, idx) {
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  _dirtyChains[profile].splice(idx, 1);
  renderTabsAndPanels();
  switchTab(profile);
}

function saveAllChains() {
  var statusEl = document.getElementById('saveStatus');
  statusEl.textContent = 'Saving...';
  statusEl.style.color = 'hsl(var(--amber-fg))';
  var promises = [];
  Object.keys(_dirtyChains).forEach(function(profile) {
    promises.push(api('PUT', '/api/chains/'+profile, {chain: _dirtyChains[profile]}));
  });
  Promise.all(promises).then(function(results) {
    var allOk = results.every(function(r) { return r.ok; });
    if (allOk) {
      provData.profile_chains = JSON.parse(JSON.stringify(_dirtyChains));
      statusEl.textContent = 'All chains saved';
      statusEl.style.color = 'hsl(var(--green-fg))';
      setTimeout(function() { statusEl.textContent = ''; }, 2000);
    } else {
      statusEl.textContent = 'Some saves failed';
      statusEl.style.color = 'hsl(var(--red-fg))';
    }
  }).catch(function(e) {
    statusEl.textContent = 'Error: '+e;
    statusEl.style.color = 'hsl(var(--red-fg))';
  });
}

/* ── Provider CRUD ── */
function renderProvList() {
  var el = document.getElementById('provList');
  var provs = provData.providers || {};
  var names = Object.keys(provs);
  if (names.length === 0) { el.innerHTML = '<div class="empty">No providers. Add one below.</div>'; return; }
  var h = '';
  names.forEach(function(n) {
    var p = provs[n];
    h += '<div class="prov-item">';
    h += '<div><div class="prov-name">'+n+'</div><div class="prov-detail">'+p.api_base+'</div></div>';
    h += '<div class="prov-actions">';
    h += '<button class="btn-sm" onclick="editProvider(\\''+n+'\\')">Edit</button>';
    h += '<button class="btn-sm btn-danger" onclick="deleteProvider(\\''+n+'\\')">Del</button>';
    h += '</div></div>';
  });
  el.innerHTML = h;
}

function showAddProvForm() {
  document.getElementById('provForm').style.display = 'block';
  document.getElementById('provName').value = '';
  document.getElementById('provUrl').value = '';
  document.getElementById('provKeyEnv').value = '';
  document.getElementById('provModels').value = '';
  document.getElementById('testResult').style.display = 'none';
  document.getElementById('btnSave').disabled = true;
  _provTestPassed = false;
}

function hideAddProvForm() {
  document.getElementById('provForm').style.display = 'none';
  _provTestPassed = false;
  document.getElementById('btnSave').disabled = true;
}

function loadPreset() {
  var key = document.getElementById('provPreset').value;
  if (!key || !provPresets[key]) return;
  var p = provPresets[key];
  document.getElementById('provName').value = key;
  document.getElementById('provUrl').value = p.api_base;
  document.getElementById('provModels').value = (p.models||[]).join(', ');
  document.getElementById('provKeyEnv').value = 'LCP_'+key.toUpperCase()+'_API_KEY';
}

function editProvider(name) {
  var p = provData.providers[name];
  if (!p) return;
  showAddProvForm();
  document.getElementById('provName').value = name;
  document.getElementById('provUrl').value = p.api_base || '';
  document.getElementById('provKeyEnv').value = p.api_key_env || '';
  document.getElementById('provModels').value = (p.models||[]).join(', ');
  // For editing existing, test not required but available
  _provTestPassed = true;
  document.getElementById('btnSave').disabled = false;
  document.getElementById('testResult').style.display = 'none';
}

function testProvider() {
  var url = document.getElementById('provUrl').value.trim();
  var keyEnv = document.getElementById('provKeyEnv').value.trim();
  var models = document.getElementById('provModels').value.split(',')[0].trim();
  var resultEl = document.getElementById('testResult');
  var btnSave = document.getElementById('btnSave');
  var btnTest = document.getElementById('btnTest');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.background = 'hsl(var(--secondary))';
  resultEl.style.color = 'hsl(var(--foreground))';
  btnTest.disabled = true;
  var apiKey = prompt('Enter API key for test (not stored):');
  if (!apiKey) { resultEl.textContent = 'Test cancelled'; btnTest.disabled = false; return; }
  api('POST', '/api/providers/test', {api_base:url, api_key:apiKey, model:models}).then(function(d) {
    btnTest.disabled = false;
    if (d.ok) {
      resultEl.innerHTML = 'Connection OK - model: '+d.model+' (HTTP '+d.status+')';
      resultEl.style.background = 'hsl(var(--green-bg))';
      resultEl.style.color = 'hsl(var(--green-fg))';
      _provTestPassed = true;
      btnSave.disabled = false;
    } else {
      resultEl.innerHTML = 'FAILED: '+(d.error||'HTTP '+d.status);
      resultEl.style.background = 'hsl(var(--red-bg))';
      resultEl.style.color = 'hsl(var(--red-fg))';
      _provTestPassed = false;
      btnSave.disabled = true;
    }
  }).catch(function(e) {
    btnTest.disabled = false;
    resultEl.textContent = 'Network error: '+e;
    resultEl.style.background = 'hsl(var(--red-bg))';
    resultEl.style.color = 'hsl(var(--red-fg))';
    _provTestPassed = false;
    btnSave.disabled = true;
  });
}

function saveProvider() {
  var name = document.getElementById('provName').value.trim();
  var url = document.getElementById('provUrl').value.trim();
  var keyEnv = document.getElementById('provKeyEnv').value.trim();
  var models = document.getElementById('provModels').value.split(',').map(function(s){return s.trim()}).filter(Boolean);
  if (!name) { alert('Provider name required'); return; }

  // For NEW providers (not editing existing), require test to pass
  var isNew = !provData.providers || !provData.providers[name];
  if (isNew && !_provTestPassed) {
    alert('You must test the connection successfully before saving a new provider.');
    return;
  }

  api('POST', '/api/providers', {name:name, api_base:url, api_key_env:keyEnv, models:models}).then(function(d) {
    if (d.ok) {
      hideAddProvForm();
      _provTestPassed = false;
      loadProviders();
    } else {
      alert('Error: '+JSON.stringify(d));
    }
  });
}

function deleteProvider(name) {
  if (!confirm('Delete provider '+name+'? This removes it from all chains.')) return;
  api('DELETE', '/api/providers/'+name).then(function(d) {
    if (d.ok) loadProviders();
    else alert('Error: '+JSON.stringify(d));
  });
}

loadProviders();

/* ── Provider Health Detail Modal ── */
function showPhDetail(event, el) {
  event.stopPropagation();
  var data = JSON.parse(el.dataset.detail);
  var statusClass = data.status === 'healthy' ? 'dot-healthy' : data.status === 'degraded' ? 'dot-degraded' : 'dot-dead';
  var statusColor = data.status === 'healthy' ? 'var(--green-fg)' : data.status === 'degraded' ? 'var(--amber-fg)' : 'var(--red-fg)';
  document.getElementById('phModalTitle').innerHTML = '<span class="ph-dot '+statusClass+'" style="display:inline-block;vertical-align:middle;margin-right:0.375rem"></span>'+data.name+' <span class="ph-profile">'+data.profile+'</span>';
  document.getElementById('phModalBody').innerHTML =
    '<div class="phm-label">Status</div><div style="color:'+statusColor+';font-weight:600">'+data.status.toUpperCase()+'</div>'+
    '<div class="phm-label">Base URL</div><div class="phm-value">'+data.url+'</div>'+
    '<div class="phm-label">Consecutive Failures</div><div class="phm-value">'+data.failures+data.tripped+'</div>'+
    '<div class="phm-label">Last Success</div><div class="phm-value">'+data.last_success+'</div>'+
    '<div class="phm-label">Last Failure</div><div class="phm-value">'+data.last_failure+'</div>';
  document.getElementById('phDetailModal').classList.add('open');
}
function closePhDetail(event) {
  if (event && event.target !== document.getElementById('phDetailModal')) return;
  document.getElementById('phDetailModal').classList.remove('open');
}

/* ── Sidebar Tree ── */
function toggleSbFolder(el) {
  el.classList.toggle('open');
  el.parentElement.classList.toggle('open');
}
function toggleProvider(event, el) {
  event.stopPropagation();
  toggleSbFolder(el);
  document.querySelectorAll('.sb-provider.selected').forEach(function(s) { s.classList.remove('selected'); });
  el.classList.add('selected');
}
function editProviderFromSidebar(event, el) {
  event.stopPropagation();
  toggleSbFolder(el);  // expand to show models too
  document.querySelectorAll('.sb-provider.selected').forEach(function(s) { s.classList.remove('selected'); });
  el.classList.add('selected');
  var ds = el.dataset;
  var h = ds.status;
  var dotClass = h === 'healthy' ? 'dot-healthy' : h === 'degraded' ? 'dot-degraded' : 'dot-dead';
  var statusColor = h === 'healthy' ? 'var(--green-fg)' : h === 'degraded' ? 'var(--amber-fg)' : 'var(--red-fg)';
  document.getElementById('pemTitle').innerHTML = '<span class="ph-dot '+dotClass+'" style="display:inline-block;margin-right:6px"></span>'+ds.provider;
  document.getElementById('pemStatus').innerHTML = '<span style="color:'+statusColor+';font-weight:700">'+h.toUpperCase()+'</span> <span class="ph-profile" style="font-size:0.625rem">'+ds.profile+'</span>';
  document.getElementById('pemUrl').value = ds.url;
  document.getElementById('pemKeyEnv').value = ds.keyenv || '';
  document.getElementById('pemModels').value = ds.models || '';
  document.getElementById('pemTestResult').style.display = 'none';
  document.getElementById('provEditModal').classList.add('open');
}
function closeProvEditModal() {
  document.getElementById('provEditModal').classList.remove('open');
}
var _pemTestPassed = true;
function testPemProvider() {
  var url = document.getElementById('pemUrl').value.trim();
  var models = document.getElementById('pemModels').value.split(',')[0].trim();
  var resultEl = document.getElementById('pemTestResult');
  var btn = document.getElementById('pemTestBtn');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.background = 'hsl(var(--secondary))';
  resultEl.style.color = 'hsl(var(--foreground))';
  btn.disabled = true;
  var apiKey = prompt('Enter API key for test (not stored):');
  if (!apiKey) { resultEl.textContent = 'Test cancelled'; btn.disabled = false; return; }
  api('POST', '/api/providers/test', {api_base:url, api_key:apiKey, model:models}).then(function(d) {
    btn.disabled = false;
    if (d.ok) {
      resultEl.innerHTML = 'Connection OK - model: '+d.model+' (HTTP '+d.status+')';
      resultEl.style.background = 'hsl(var(--green-bg))';
      resultEl.style.color = 'hsl(var(--green-fg))';
      _pemTestPassed = true;
    } else {
      resultEl.innerHTML = 'FAILED: '+(d.error||'HTTP '+d.status);
      resultEl.style.background = 'hsl(var(--red-bg))';
      resultEl.style.color = 'hsl(var(--red-fg))';
      _pemTestPassed = false;
    }
  }).catch(function(e) {
    btn.disabled = false;
    resultEl.textContent = 'Network error: '+e;
    resultEl.style.background = 'hsl(var(--red-bg))';
    resultEl.style.color = 'hsl(var(--red-fg))';
    _pemTestPassed = false;
  });
}
function savePemProvider() {
  var titleEl = document.getElementById('pemTitle');
  var name = titleEl.textContent.replace(/^\\s+/, '').split(' ').pop() || '';
  var url = document.getElementById('pemUrl').value.trim();
  var keyEnv = document.getElementById('pemKeyEnv').value.trim();
  var models = document.getElementById('pemModels').value.split(',').map(function(s){return s.trim()}).filter(Boolean);
  if (!name) { alert('Provider name missing'); return; }
  api('POST', '/api/providers', {name:name, api_base:url, api_key_env:keyEnv, models:models}).then(function(d) {
    if (d.ok) {
      document.getElementById('pemTestResult').style.display = 'block';
      document.getElementById('pemTestResult').textContent = 'Saved. Refresh to see changes.';
      document.getElementById('pemTestResult').style.background = 'hsl(var(--green-bg))';
      document.getElementById('pemTestResult').style.color = 'hsl(var(--green-fg))';
    } else {
      alert('Error: '+JSON.stringify(d));
    }
  });
}
// Default expand only top-level profile folders (not providers)
(function() {
  document.querySelectorAll('.sidebar-nav > .sb-tree > .sb-folder').forEach(function(f) { toggleSbFolder(f); });
})();

/* ── Profiles & Keys ── */
function addProfile() {
  var name = prompt('New profile name (lowercase, e.g. "admin"):');
  if (!name) return;
  name = name.trim().toLowerCase();
  if (!/^[a-z0-9_-]+$/.test(name)) { alert('Invalid name. Use lowercase letters, numbers, hyphens, underscores.'); return; }
  api('POST', '/api/profiles', {name: name}).then(function(d) {
    if (d.ok) { location.reload(); }
    else { alert('Error: '+JSON.stringify(d)); }
  });
}
function copyUrl(url) {
  navigator.clipboard.writeText(url).then(function() {
    // brief flash
  }).catch(function() {
    prompt('Copy this URL:', url);
  });
}
/* ── Profile Config Modal ── */
var _pcmProfile = null;

function openProfileConfig(event, profile) {
  event.stopPropagation();
  _pcmProfile = profile;

  // Expand the sidebar folder for this profile
  var folder = document.querySelector('.sb-profile-row[data-profile="' + profile + '"]');
  if (folder && !folder.classList.contains('open')) {
    toggleSbFolder(folder);
  }

  document.getElementById('pcmTitle').textContent = 'Profile: ' + profile.toUpperCase();
  document.getElementById('pcmUrl').textContent = window._hostUrl + '/' + profile + '/chat/completions';
  document.getElementById('pcmCurl').textContent = 'curl -H "Authorization: Bearer lcp_YOUR_KEY" ' + window._hostUrl + '/' + profile + '/chat/completions';
  switchPcmTab('pcm-apikeys');
  loadProfileKeys(profile);
  loadProfileChain(profile);
  var modal = document.getElementById('profileConfigModal');
  if (modal) { modal.classList.add('open'); }
}

function closeProfileConfig() {
  document.getElementById('profileConfigModal').classList.remove('open');
}

function switchPcmTab(tabId) {
  document.querySelectorAll('#pcmTabs .modal-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.tab === tabId);
  });
  document.querySelectorAll('#pcmBody .tab-panel').forEach(function(p) {
    p.classList.toggle('active', p.id === 'panel-' + tabId);
  });
}

function loadProfileKeys(profile) {
  api('GET', '/api/keys').then(function(d) {
    var el = document.getElementById('pcmKeysList');
    var keys = (d.keys || []).filter(function(k) { return k.profile === profile; });
    if (keys.length === 0) { el.innerHTML = '<div class="empty">No API keys for this profile.</div>'; return; }
    var h = '';
    keys.forEach(function(k) {
      h += '<div class="prov-item">';
      h += '<div><div class="prov-name">'+k.label+'</div><div class="prov-detail">Created: '+(k.created||'').slice(0,10)+(k.last_used ? ' | Last used: '+k.last_used.slice(0,10) : '')+'</div></div>';
      h += '<div class="prov-actions">';
      h += '<button class="btn-sm btn-danger" onclick="deleteProfileKey(\\''+k.id+'\\')">Revoke</button>';
      h += '</div></div>';
    });
    el.innerHTML = h;
  });
}

function generateProfileKey() {
  if (!_pcmProfile) return;
  var label = prompt('Label (e.g. "L2 agent"):', 'Key for ' + _pcmProfile);
  if (!label) return;
  api('POST', '/api/keys', {profile: _pcmProfile, label: label}).then(function(d) {
    if (d.ok) {
      var copied = false;
      try { navigator.clipboard.writeText(d.key); copied = true; } catch(e) {}
      alert('API Key (copied to clipboard):\\n\\n'+d.key+'\\n\\n'+(copied ? 'Copied! Save it now — it won\\'t be shown again.' : 'SAVE THIS KEY — it won\\'t be shown again.'));
      loadProfileKeys(_pcmProfile);
    } else {
      alert('Error: '+JSON.stringify(d));
    }
  });
}

function deleteProfileKey(id) {
  if (!confirm('Revoke this API key? It will stop working immediately.')) return;
  api('DELETE', '/api/keys/'+id).then(function(d) {
    if (d.ok) loadProfileKeys(_pcmProfile);
    else alert('Error: '+JSON.stringify(d));
  });
}

function copyProfileUrl() {
  var url = document.getElementById('pcmUrl').textContent;
  navigator.clipboard.writeText(url).then(function() {
    // brief flash
  }).catch(function() {
    prompt('Copy this URL:', url);
  });
}

function loadProfileChain(profile) {
  // Requires provData to be loaded (loaded via loadProviders on page init)
  var chain = (provData.profile_chains || {})[profile] || [];
  var providers = provData.providers || {};
  var el = document.getElementById('pcmChainList');
  if (!el) return;
  if (chain.length === 0) {
    el.innerHTML = '<div class="empty">No providers in this profile\\'s chain. <br><br><button class="btn-sm btn-primary" onclick="addProviderToChain(\\''+profile+'\\')">+ Add Provider</button></div>';
    return;
  }
  var h = '<div class="pcm-chain-header"><span class="phm-label">Fallback Chain</span><span style="font-size:0.625rem;color:hsl(var(--muted-foreground))">first → last (falls back on failure)</span></div>';
  chain.forEach(function(step, i) {
    var pn = step.provider;
    var bu = step.base_url || '';
    var pinfo = providers[pn] || {};
    var models = (pinfo.models || []).join(', ') || 'default';
    var order = i + 1;
    h += '<div class="pcm-chain-row" data-idx="'+i+'">';
    h += '  <span class="pcm-chain-order">'+order+'</span>';
    h += '  <span class="ph-dot dot-healthy" style="display:inline-block;flex-shrink:0"></span>';
    h += '  <span class="pcm-chain-prov">'+pn+'</span>';
    h += '  <span class="pcm-chain-models">'+models+'</span>';
    h += '  <button class="pcm-chain-edit-btn" onclick="editChainProvider(event, \\''+profile+'\\', \\''+pn+'\\', \\''+bu.replace(/'/g, "\\\\'")+'\\', \\''+(pinfo.api_key_env||'').replace(/'/g, "\\\\'")+'\\', \\''+models.replace(/'/g, "\\\\'")+'\\')" title="Edit provider">⚙</button>';
    h += '  <button class="pcm-chain-remove-btn" onclick="removeChainProvider(\\''+profile+'\\', '+i+')" title="Remove from chain">✕</button>';
    h += '</div>';
  });
  h += '<div style="margin-top:0.75rem;display:flex;gap:0.375rem">';
  h += '  <button class="btn-sm btn-primary" onclick="addProviderToChain(\\''+profile+'\\')">+ Add to Chain</button>';
  h += '  <button class="btn-sm btn-success" onclick="saveProfileChain(\\''+profile+'\\')">Save Chain</button>';
  h += '</div>';
  el.innerHTML = h;
}

function editChainProvider(event, profile, pn, url, keyenv, models) {
  event.stopPropagation();
  document.getElementById('pemTitle').innerHTML = '<span class="ph-dot dot-healthy" style="display:inline-block;margin-right:6px"></span>'+pn;
  document.getElementById('pemStatus').innerHTML = '<span class="ph-profile" style="font-size:0.8125rem">in profile: '+profile.toUpperCase()+'</span>';
  document.getElementById('pemUrl').value = url || '';
  document.getElementById('pemKeyEnv').value = keyenv || '';
  document.getElementById('pemModels').value = models || '';
  document.getElementById('pemTestResult').style.display = 'none';
  document.getElementById('provEditModal').classList.add('open');
}

function addProviderToChain(profile) {
  var provs = Object.keys(provData.providers || {});
  if (provs.length === 0) { alert('No providers defined. Add a provider first via Configuration → Providers.'); return; }
  var pn = prompt('Provider to add to chain ('+provs.join(', ')+'):', provs[0]);
  if (!pn || !provData.providers[pn]) return;
  var bu = provData.providers[pn].api_base || '';
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  _dirtyChains[profile].push({provider: pn, base_url: bu, model: (provData.providers[pn].models || [])[0] || 'default'});
  provData.profile_chains[profile] = JSON.parse(JSON.stringify(_dirtyChains[profile]));
  loadProfileChain(profile);
  updateSidebarChains();
}

function removeChainProvider(profile, idx) {
  if (!_dirtyChains[profile]) _dirtyChains[profile] = JSON.parse(JSON.stringify(provData.profile_chains[profile] || []));
  _dirtyChains[profile].splice(idx, 1);
  provData.profile_chains[profile] = JSON.parse(JSON.stringify(_dirtyChains[profile]));
  loadProfileChain(profile);
  updateSidebarChains();
}

function saveProfileChain(profile) {
  if (!_dirtyChains[profile]) return;
  api('PUT', '/api/chains/'+profile, {chain: _dirtyChains[profile]}).then(function(r) {
    if (r.ok) {
      provData.profile_chains[profile] = JSON.parse(JSON.stringify(_dirtyChains[profile]));
      loadProfileChain(profile);
      updateSidebarChains();
    } else {
      alert('Failed to save chain: ' + JSON.stringify(r));
    }
  });
}

function updateSidebarChains() {
  // Refresh sidebar by reloading page (simple approach)
  // In a more advanced version, we'd re-render the sidebar DOM
  var reloadBtn = document.createElement('div');
  reloadBtn.style.cssText = 'position:fixed;bottom:1rem;right:1rem;background:hsl(var(--primary));color:hsl(var(--primary-foreground));padding:0.5rem 1rem;border-radius:var(--radius);font-size:0.75rem;cursor:pointer;z-index:9999;box-shadow:0 2px 8px rgba(0,0,0,0.3)';
  reloadBtn.textContent = '↻ Reload to see sidebar changes';
  reloadBtn.onclick = function() { location.reload(); };
  document.body.appendChild(reloadBtn);
  setTimeout(function() { if (reloadBtn.parentNode) reloadBtn.remove(); }, 8000);
}

// Profile config modal tab clicks
document.addEventListener('DOMContentLoaded', function() {
  // Profile config modal tabs
  var pcmTabs = document.getElementById('pcmTabs');
  if (pcmTabs) {
    pcmTabs.addEventListener('click', function(e) {
      var tab = e.target.closest('.modal-tab');
      if (tab && tab.dataset.tab) switchPcmTab(tab.dataset.tab);
    });
  }
  // Backdrop click to close profile config modal
  var pcm = document.getElementById('profileConfigModal');
  if (pcm) {
    pcm.addEventListener('click', function(e) {
      if (e.target === pcm) closeProfileConfig();
    });
  }
  // Backdrop click to close keys modal
  var km = document.getElementById('keysModal');
  if (km) {
    km.addEventListener('click', function(e) {
      if (e.target === km) closeKeysModal();
    });
  }
});

function openKeysModal() {
  loadKeys();
  document.getElementById('keysModal').classList.add('open');
}
function closeKeysModal() {
  document.getElementById('keysModal').classList.remove('open');
}
function loadKeys() {
  api('GET', '/api/keys').then(function(d) {
    var el = document.getElementById('keysList');
    var keys = d.keys || [];
    if (keys.length === 0) { el.innerHTML = '<div class="empty">No API keys yet.</div>'; return; }
    var h = '';
    keys.forEach(function(k) {
      h += '<div class="prov-item">';
      h += '<div><div class="prov-name">'+k.label+'</div><div class="prov-detail">Profile: '+k.profile+' | Created: '+(k.created||'').slice(0,10)+'</div></div>';
      h += '<div class="prov-actions">';
      h += '<button class="btn-sm btn-danger" onclick="deleteKey(\\''+k.id+'\\')">Revoke</button>';
      h += '</div></div>';
    });
    el.innerHTML = h;
  });
}
function generateKey() {
  var profiles = Object.keys(provData.profile_chains || {});
  if (profiles.length === 0) { alert('No profiles exist. Create one first.'); return; }
  var profile = prompt('Profile for this key ('+profiles.join(', ')+'):', profiles[0]);
  if (!profile) return;
  var label = prompt('Label (e.g. "L2 agent"):', 'Key for '+profile);
  if (!label) return;
  api('POST', '/api/keys', {profile: profile, label: label}).then(function(d) {
    if (d.ok) {
      var copied = false;
      try { navigator.clipboard.writeText(d.key); copied = true; } catch(e) {}
      alert('API Key (copied to clipboard):\\n\\n'+d.key+'\\n\\n'+(copied ? 'Copied! Save it now — it won\\'t be shown again.' : 'SAVE THIS KEY — it won\\'t be shown again.'));
      loadKeys();
    } else {
      alert('Error: '+JSON.stringify(d));
    }
  });
}
function deleteKey(id) {
  if (!confirm('Revoke this API key? It will stop working immediately.')) return;
  api('DELETE', '/api/keys/'+id).then(function(d) {
    if (d.ok) loadKeys();
    else alert('Error: '+JSON.stringify(d));
  });
}