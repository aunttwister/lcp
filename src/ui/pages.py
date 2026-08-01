"""Standalone page-rendering functions (HTML templates).

Each function returns a complete HTML page as a string.
Called from endpoint mixins in src.server.endpoints.
"""

import json
from datetime import date as _date
from pathlib import Path
from sqlalchemy import func


def render_providers_page(config, engine=None) -> str:
    """Render the Providers management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    prov_rows = ""
    for name, pdata in config.providers.items():
        models = ", ".join(pdata.get("models", []))
        prov_rows += (
            f'<tr>'
            f'<td><b>{name}</b></td>'
            f'<td class="mono" style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{pdata.get("api_base", "—")}</td>'
            f'<td>{pdata.get("api_key_env", "—")}</td>'
            f'<td>{models or "—"}</td>'
            f'<td><button class="btn-sm" onclick="editProvider(\'{name}\')">Edit</button> '
            f'<button class="btn-sm btn-danger" onclick="deleteProvider(\'{name}\')">Del</button></td>'
            f'</tr>'
        )
    if not prov_rows:
        prov_rows = '<tr><td colspan="5" class="empty">No providers configured.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LCP — Providers</title>
<style>{css}</style>
</head>
<body>
{render_sidebar_html(config, "providers")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Providers</h1>
<p class="subtitle">Manage LLM API providers and their models</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="showAddProvForm()">+ Add Provider</button>
  <select id="provPreset" onchange="loadPreset()" style="padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.75rem">
    <option value="">-- Quick-add preset --</option>
  </select>
</div>

<div class="prov-form" id="provForm" style="display:none">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
    <div>
      <label>Provider Name</label>
      <input id="provName" placeholder="e.g. openai">
    </div>
    <div>
      <label>API Base URL</label>
      <input id="provUrl" placeholder="https://api.openai.com/v1">
    </div>
    <div>
      <label>API Key Env Var</label>
      <input id="provKeyEnv" placeholder="OPENAI_API_KEY">
    </div>
    <div>
      <label>Models (comma-separated)</label>
      <input id="provModels" placeholder="gpt-4o, gpt-4o-mini">
    </div>
  </div>
  <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
    <button class="btn-sm btn-success" id="provTestBtn" onclick="testProvider()">Test Connection</button>
    <button class="btn-sm btn-primary" id="provSaveBtn" onclick="saveProvider()" disabled>Save Provider</button>
    <button class="btn-sm" onclick="hideAddProvForm()">Cancel</button>
  </div>
  <div id="testResult" class="test-result" style="display:none"></div>
</div>

<div class="table-wrap" style="margin-top:0.75rem">
<table>
<thead><tr>
  <th>Name</th><th>Base URL</th><th>Key Env</th><th>Models</th><th>Actions</th>
</tr></thead>
<tbody id="providersBody">{prov_rows}</tbody>
</table>
</div>
</div>

<script>
function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function loadPresets() {{
  api('GET', '/api/providers/presets').then(function(d) {{
    var sel = document.getElementById('provPreset');
    Object.keys(d.presets || {{}}).forEach(function(k) {{
      sel.innerHTML += '<option value="' + k + '">' + k + '</option>';
    }});
  }});
}}

function loadPreset() {{
  var key = document.getElementById('provPreset').value;
  if (!key) return;
  showAddProvForm();
  api('GET', '/api/providers/presets').then(function(d) {{
    var p = (d.presets || {{}})[key];
    if (!p) return;
    document.getElementById('provName').value = key;
    document.getElementById('provUrl').value = p.api_base || '';
    document.getElementById('provModels').value = (p.models || []).join(', ');
    document.getElementById('provKeyEnv').value = 'LCP_' + key.toUpperCase() + '_API_KEY';
    document.getElementById('provSaveBtn').disabled = false;
  }});
}}

function showAddProvForm() {{
  document.getElementById('provForm').style.display = 'block';
  document.getElementById('provName').value = '';
  document.getElementById('provUrl').value = '';
  document.getElementById('provKeyEnv').value = '';
  document.getElementById('provModels').value = '';
  document.getElementById('testResult').style.display = 'none';
  document.getElementById('provSaveBtn').disabled = true;
}}

function hideAddProvForm() {{
  document.getElementById('provForm').style.display = 'none';
}}

function testProvider() {{
  var resultEl = document.getElementById('testResult');
  resultEl.style.display = 'block';
  resultEl.textContent = 'Testing...';
  resultEl.style.color = 'hsl(var(--amber-fg))';
  api('POST', '/api/providers/test', {{
    api_base: document.getElementById('provUrl').value,
    api_key: '',
    model: (document.getElementById('provModels').value || 'default').split(',')[0].trim()
  }}).then(function(d) {{
    if (d.ok) {{
      resultEl.textContent = 'Connected — model: ' + d.model;
      resultEl.style.color = 'hsl(var(--green-fg))';
      document.getElementById('provSaveBtn').disabled = false;
    }} else {{
      resultEl.textContent = 'Failed: ' + (d.error || 'unknown');
      resultEl.style.color = 'hsl(var(--red-fg))';
    }}
  }});
}}

function editProvider(name) {{
  api('GET', '/api/providers').then(function(d) {{
    var p = (d.providers || {{}})[name];
    if (!p) return;
    showAddProvForm();
    document.getElementById('provName').value = name;
    document.getElementById('provUrl').value = p.api_base || '';
    document.getElementById('provKeyEnv').value = p.api_key_env || '';
    document.getElementById('provModels').value = (p.models || []).join(', ');
    document.getElementById('provSaveBtn').disabled = false;
  }});
}}

function saveProvider() {{
  var name = document.getElementById('provName').value.trim();
  if (!name) {{ alert('Provider name is required'); return; }}
  var body = {{
    name: name,
    api_base: document.getElementById('provUrl').value.trim(),
    api_key_env: document.getElementById('provKeyEnv').value.trim(),
    models: document.getElementById('provModels').value.split(',').map(function(s) {{ return s.trim(); }}).filter(Boolean)
  }};
  api('POST', '/api/providers', body).then(function(d) {{
    if (d.ok) {{ location.reload(); }}
    else {{ alert('Error: ' + (d.error || 'unknown')); }}
  }});
}}

function deleteProvider(name) {{
  if (!confirm('Delete provider "' + name + '"? This removes it from all profile chains.')) return;
  api('DELETE', '/api/providers/' + name).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

loadPresets();
</script>
""" + render_sidebar_plugin_js(config, engine) + """
</body>
</html>"""


def render_profiles_page(config, engine=None) -> str:
    """Render the Profiles management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    profile_rows = ""
    for pname, pcfg in config.profiles.items():
        chain = pcfg.get("chain", [])
        steps = " → ".join(f"{s['provider']}/{s['model']}" for s in chain) if chain else "—"
        forbidden = ", ".join(pcfg.get("forbidden_tools", []) or []) or "none"
        auth_required = pcfg.get("auth_required", True)
        auth_badge = "key" if auth_required else "public"
        profile_rows += (
            f'<tr>'
            f'<td><b>{pname.upper()}</b></td>'
            f'<td style="font-size:0.6875rem">{auth_badge}</td>'
            f'<td style="font-size:0.75rem">{steps}</td>'
            f'<td style="font-size:0.6875rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{forbidden}">{forbidden}</td>'
            f'<td class="mono" style="font-size:0.6875rem">/{pname}/chat/completions</td>'
            f'<td><button class="btn-sm" onclick="editProfile(\'{pname}\')">Edit</button> '
            f'<button class="btn-sm btn-danger" onclick="deleteProfile(\'{pname}\')">Del</button></td>'
            f'</tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LCP — Profiles</title>
<style>{css}</style>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
</head>
<body>
{render_sidebar_html(config, "profiles")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Profiles</h1>
<p class="subtitle">Manage routing profiles, fallback chains, and tool restrictions</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="addProfile()">+ Add Profile</button>
</div>

<div class="table-wrap">
<table>
<thead><tr>
  <th>Profile</th><th>Auth</th><th>Chain</th><th>Blocked Tools</th><th>Gateway URL</th><th>Actions</th>
</tr></thead>
<tbody id="profilesBody">{profile_rows}</tbody>
</table>
</div>
</div>

<!-- Profile Edit Modal -->
<div class="modal-overlay" id="profileEditModal">
<div class="modal" style="width:min(600px,95vw)">
  <div class="modal-header">
    <h2 id="pemProfileTitle">Edit Profile</h2>
    <button class="modal-close" onclick="closeProfileEdit()">✕</button>
  </div>
  <div class="modal-body">
    <div style="margin-bottom:0.75rem">
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Access Control</label>
      <label style="display:flex;align-items:center;gap:0.5rem;cursor:pointer;font-size:0.8125rem">
        <input type="checkbox" id="pemAuthRequired" onchange="this.nextElementSibling.textContent = this.checked ? 'API key required' : 'No key required (public)'">
        <span>API key required</span>
      </label>
    </div>
    <div style="margin-bottom:0.75rem">
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Blocked Tools</label>
      <div id="pemToolsList" style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-bottom:0.375rem"></div>
      <div style="display:flex;gap:0.375rem">
        <input id="pemNewTool" placeholder="tool_name" style="flex:1;padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.75rem">
        <button class="btn-sm btn-primary" onclick="addBlockedTool()">+ Add</button>
      </div>
    </div>
    <div>
      <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Fallback Chain (drag to reorder)</label>
      <ul class="chain-list" id="pemChainList"></ul>
      <button class="btn-sm btn-primary" onclick="addChainStep()" style="margin-top:0.25rem">+ Add Step</button>
    </div>
  </div>
  <div class="modal-footer">
    <span id="pemSaveStatus" style="font-size:0.6875rem;color:hsl(var(--muted-foreground));margin-right:auto"></span>
    <button class="btn-sm btn-primary" onclick="saveProfileEdit()">Save</button>
    <button class="btn-sm" onclick="closeProfileEdit()">Cancel</button>
  </div>
</div>
</div>

<script>
function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function addProfile() {{
  var name = prompt('Profile name (lowercase, e.g. "admin"):');
  if (!name) return;
  api('POST', '/api/profiles', {{ name: name }}).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

function deleteProfile(name) {{
  if (!confirm('Delete profile "' + name + '"? This cannot be undone.')) return;
  api('DELETE', '/api/profiles/' + name).then(function(d) {{
    if (d.ok) location.reload();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

var _editProfileName = '';
var _editProfileTools = [];
var _editProfileChain = [];
var _allProviders = {{}};
var _allChains = {{}};

function editProfile(name) {{
  _editProfileName = name;
  api('GET', '/api/providers').then(function(d) {{
    _allProviders = d.providers || {{}};
    _allChains = d.profile_chains || {{}};
    _editProfileChain = JSON.parse(JSON.stringify(_allChains[name] || []));
    api('GET', '/api/profiles').then(function(pd) {{
      var prof = (pd.profiles || {{}})[name] || {{}};
      _editProfileTools = (prof.forbidden || []).slice();
      document.getElementById('pemProfileTitle').textContent = 'Edit Profile: ' + name.toUpperCase();
      var authReq = prof.auth_required !== false; // default true
      document.getElementById('pemAuthRequired').checked = authReq;
      document.getElementById('pemAuthRequired').nextElementSibling.textContent = authReq ? 'API key required' : 'No key required (public)';
      renderPemTools();
      renderPemChain();
      document.getElementById('profileEditModal').classList.add('open');
    }});
  }});
}}

function closeProfileEdit() {{
  document.getElementById('profileEditModal').classList.remove('open');
}}

function renderPemTools() {{
  var html = '';
  _editProfileTools.forEach(function(t, i) {{
    html += '<span style="display:inline-flex;align-items:center;gap:0.1875rem;padding:0.125rem 0.5rem;background:hsl(var(--red-bg));color:hsl(var(--red-fg));border-radius:9999px;font-size:0.6875rem;font-weight:600">' +
      t +
      '<button onclick="removeBlockedTool(' + i + ')" style="background:none;border:none;color:inherit;cursor:pointer;font-size:0.75rem;padding:0;line-height:1">✕</button>' +
    '</span>';
  }});
  if (!html) html = '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">no tools blocked</span>';
  document.getElementById('pemToolsList').innerHTML = html;
}}

function addBlockedTool() {{
  var t = document.getElementById('pemNewTool').value.trim();
  if (!t) return;
  if (_editProfileTools.indexOf(t) >= 0) return;
  _editProfileTools.push(t);
  document.getElementById('pemNewTool').value = '';
  renderPemTools();
}}

function removeBlockedTool(idx) {{
  _editProfileTools.splice(idx, 1);
  renderPemTools();
}}

function renderPemChain() {{
  var provNames = Object.keys(_allProviders);
  var html = '';
  if (_editProfileChain.length === 0) {{
    html = '<li class="empty" style="padding:0.5rem;font-size:0.6875rem">No providers in chain</li>';
  }} else {{
    _editProfileChain.forEach(function(s, i) {{
      html += '<li class="chain-item" data-idx="' + i + '">';
      html += '<span class="drag-handle">⋮⋮</span>';
      html += '<select onchange="_editProfileChain[' + i + '].provider = this.value; renderPemChain();" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
      provNames.forEach(function(pn) {{
        var sel = s.provider === pn ? ' selected' : '';
        html += '<option value="' + pn + '"' + sel + '>' + pn + '</option>';
      }});
      html += '</select>';
      var models = _allProviders[s.provider]?.models || [];
      html += '<select onchange="_editProfileChain[' + i + '].model = this.value" style="background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:4px;color:hsl(var(--foreground));padding:0.125rem 0.375rem;font-size:0.6875rem">';
      models.forEach(function(m) {{
        var sel = s.model === m ? ' selected' : '';
        html += '<option value="' + m + '"' + sel + '>' + m + '</option>';
      }});
      html += '</select>';
      html += '<button class="btn-sm" style="margin-left:auto" onclick="_editProfileChain.splice(' + i + ',1);renderPemChain();">✕</button>';
      html += '</li>';
    }});
  }}
  document.getElementById('pemChainList').innerHTML = html;
  var listEl = document.getElementById('pemChainList');
  if (listEl && typeof Sortable !== 'undefined') {{
    new Sortable(listEl, {{
      animation: 150, handle: '.drag-handle',
      onEnd: function() {{
        var items = document.querySelectorAll('#pemChainList .chain-item');
        var newChain = [];
        items.forEach(function(item) {{
          var idx = parseInt(item.getAttribute('data-idx'));
          newChain.push(_editProfileChain[idx]);
        }});
        _editProfileChain = newChain;
        renderPemChain();
      }}
    }});
  }}
}}

function addChainStep() {{
  var provs = Object.keys(_allProviders);
  if (provs.length === 0) {{ alert('Add a provider first'); return; }}
  var p = provs[0];
  var m = (_allProviders[p]?.models || [])[0] || 'default';
  _editProfileChain.push({{provider: p, model: m}});
  renderPemChain();
}}

function saveProfileEdit() {{
  var statusEl = document.getElementById('pemSaveStatus');
  statusEl.textContent = 'Saving...';
  statusEl.style.color = 'hsl(var(--amber-fg))';
  var chain = _editProfileChain.map(function(s) {{
    var bu = '';
    var old = _allChains[_editProfileName] || [];
    old.forEach(function(o) {{
      if (o.provider === s.provider && o.model === s.model) bu = o.base_url || '';
    }});
    return {{provider: s.provider, model: s.model, base_url: bu}};
  }});
  // Save chain first, then profile — sequentially to avoid race on config file
  api('PUT', '/api/chains/' + _editProfileName, {{chain: chain}}).then(function(chainResult) {{
    return api('PUT', '/api/profiles/' + _editProfileName, {{
      forbidden_tools: _editProfileTools,
      auth_required: document.getElementById('pemAuthRequired').checked
    }});
  }}).then(function(profResult) {{
    if (profResult.ok) {{
      statusEl.textContent = 'Saved';
      statusEl.style.color = 'hsl(var(--green-fg))';
      setTimeout(function() {{ location.reload(); }}, 800);
    }} else {{
      statusEl.textContent = 'Save failed';
      statusEl.style.color = 'hsl(var(--red-fg))';
    }}
  }}).catch(function(e) {{
    statusEl.textContent = 'Save failed: ' + (e.message || e);
    statusEl.style.color = 'hsl(var(--red-fg))';
  }});
}}

</script>
""" + render_sidebar_plugin_js(config, engine) + """
</body>
</html>"""


def render_keys_page(config, engine) -> str:
    """Render the API Keys management page."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LCP — API Keys</title>
<style>{css}</style>
</head>
<body>
{render_sidebar_html(config, "keys")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>API Keys</h1>
<p class="subtitle">Manage virtual keys for API access</p>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem">
  <button class="btn-sm btn-primary" onclick="showCreateKeyModal()">+ Create Key</button>
  <input type="text" id="keySearch" placeholder="Search keys..." oninput="filterKeys()"
    style="padding:0.3rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;width:200px">
</div>

<div class="table-wrap">
<table id="keysTable">
<thead><tr>
  <th>Name</th><th>Prefix</th><th>Profiles</th><th>Spend</th><th>Limit</th><th>Status</th><th>Created</th><th>Actions</th>
</tr></thead>
<tbody id="keysBody"><tr><td colspan="8" class="empty">Loading...</td></tr></tbody>
</table>
</div>
</div>

<!-- Create Key Modal -->
<div class="modal-overlay" id="createKeyModal">
<div class="modal" style="width:min(500px,95vw)">
  <div class="modal-header">
    <h2>Create API Key</h2>
    <button class="modal-close" onclick="closeCreateKeyModal()">✕</button>
  </div>
  <div class="modal-body">
    <div style="display:grid;gap:0.75rem">
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Key Name</label>
        <input id="newKeyName" placeholder="e.g. Production Key" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
      </div>
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Allowed Profiles</label>
        <div id="newKeyProfilesPills" style="display:flex;flex-wrap:wrap;gap:0.375rem"></div>
        <div style="font-size:0.625rem;color:hsl(var(--muted-foreground));margin-top:0.25rem">Click to toggle · none selected = all profiles</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Spend Limit ($, 0=unlimited)</label>
          <input id="newKeyLimit" type="number" step="0.01" min="0" value="0" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
        <div>
          <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Expires At (optional)</label>
          <input id="newKeyExpires" type="date" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem">
        </div>
      </div>
    </div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm btn-primary" onclick="createKey()">Create</button>
    <button class="btn-sm" onclick="closeCreateKeyModal()">Cancel</button>
  </div>
</div>
</div>

<!-- Show Key Modal -->
<div class="modal-overlay" id="showKeyModal">
<div class="modal" style="width:min(450px,95vw)">
  <div class="modal-header">
    <h2>Key Created</h2>
    <button class="modal-close" onclick="closeShowKeyModal()">✕</button>
  </div>
  <div class="modal-body">
    <div class="phm-label">API Key (copy now — won't be shown again)</div>
    <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem">
      <code id="shownKey" style="flex:1;background:hsl(var(--secondary));padding:0.5rem;border-radius:var(--radius);font-size:0.8125rem;word-break:break-all;user-select:all"></code>
      <button class="btn-sm btn-primary" onclick="copyShownKey()">Copy</button>
    </div>
    <div id="shownKeyInfo" style="margin-top:0.5rem;font-size:0.75rem;color:hsl(var(--muted-foreground))"></div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm" onclick="closeShowKeyModal();loadKeys()">Done</button>
  </div>
</div>
</div>

<script>
var _hostUrl = window.location.origin;

function api(method, url, body) {{
  var opts = {{method:method, headers:{{'Content-Type':'application/json'}}}};
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function(r) {{ return r.json(); }});
}}

function loadKeys() {{
  api('GET', '/api/keys').then(function(d) {{
    var keys = d.keys || [];
    var html = '';
    if (keys.length === 0) {{
      html = '<tr><td colspan="8" class="empty">No keys yet. Create one above.</td></tr>';
    }} else {{
      keys.forEach(function(k) {{
        var spend = '$' + (k.total_spend || 0).toFixed(4);
        var limit = k.spend_limit > 0 ? '$' + k.spend_limit.toFixed(2) : '∞';
        var statusBadge = k.status === 'active'
          ? '<span class="badge badge-success">active</span>'
          : '<span class="badge badge-error">revoked</span>';
        html += '<tr data-search="' + (k.name||'') + ' ' + (k.key_prefix||'') + '">' +
          '<td>' + (k.name || '—') + '</td>' +
          '<td class="mono">' + (k.key_prefix || '—') + '</td>' +
          '<td>' + (k.allowed_profiles || 'all') + '</td>' +
          '<td class="cost mono">' + spend + '</td>' +
          '<td class="cost mono">' + limit + '</td>' +
          '<td>' + statusBadge + '</td>' +
          '<td class="mono" style="font-size:0.6875rem">' + (k.created_at||'').slice(0,10) + '</td>' +
          '<td>' +
            (k.status === 'active'
              ? '<button class="btn-sm" onclick="rotateKey(' + k.id + ')" title="Rotate">↻</button> '
                + '<button class="btn-sm btn-danger" onclick="revokeKey(' + k.id + ')" title="Revoke">✕</button>'
              : '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">' + (k.revoked_at||'').slice(0,10) + '</span>'
            ) +
          '</td>' +
        '</tr>';
      }});
    }}
    document.getElementById('keysBody').innerHTML = html;
  }});
}}

function showCreateKeyModal() {{ 
  document.getElementById('createKeyModal').classList.add('open');
  _selectedProfiles = [];
  loadProfilesForKeys();
}}
function closeCreateKeyModal() {{ 
  document.getElementById('createKeyModal').classList.remove('open');
  _selectedProfiles = [];
}}
function closeShowKeyModal() {{ document.getElementById('showKeyModal').classList.remove('open'); }}

function loadProfilesForKeys() {{
  api('GET', '/api/profiles').then(function(d) {{
    var profiles = Object.keys(d.profiles || {{}});
    var html = '';
    profiles.forEach(function(p) {{
      html += '<span class="profile-pill" data-profile="' + p + '" onclick="toggleProfilePill(this)" style="padding:0.25rem 0.625rem;border-radius:9999px;font-size:0.75rem;font-weight:600;cursor:pointer;border:1px solid hsl(var(--card-border));background:hsl(var(--secondary)/0.3);color:hsl(var(--muted-foreground));transition:all 0.15s;user-select:none">' + p.toUpperCase() + '</span>';
    }});
    if (!html) html = '<span style="font-size:0.6875rem;color:hsl(var(--muted-foreground))">No profiles configured</span>';
    document.getElementById('newKeyProfilesPills').innerHTML = html;
  }});
}}

var _selectedProfiles = [];

function toggleProfilePill(el) {{
  var p = el.getAttribute('data-profile');
  var idx = _selectedProfiles.indexOf(p);
  if (idx >= 0) {{
    _selectedProfiles.splice(idx, 1);
    el.style.background = 'hsl(var(--secondary)/0.3)';
    el.style.color = 'hsl(var(--muted-foreground))';
    el.style.borderColor = 'hsl(var(--card-border))';
  }} else {{
    _selectedProfiles.push(p);
    el.style.background = 'hsl(var(--primary))';
    el.style.color = 'hsl(var(--primary-foreground))';
    el.style.borderColor = 'hsl(var(--primary))';
  }}
}}

function createKey() {{
  var name = document.getElementById('newKeyName').value.trim();
  var limit = parseFloat(document.getElementById('newKeyLimit').value) || 0;
  var expires = document.getElementById('newKeyExpires').value;
  api('POST', '/api/keys', {{
    name: name || 'API Key',
    allowed_profiles: _selectedProfiles.join(','),
    spend_limit: limit,
    expires_at: expires ? expires + 'T00:00:00' : ''
  }}).then(function(d) {{
    if (d.key) {{
      document.getElementById('shownKey').textContent = d.key;
      document.getElementById('shownKeyInfo').innerHTML =
        'Name: <b>' + d.name + '</b> · ID: ' + d.id + '<br>' +
        'Profiles: ' + (d.allowed_profiles || 'all') + ' · Limit: $' + (d.spend_limit || 0);
      closeCreateKeyModal();
      document.getElementById('showKeyModal').classList.add('open');
    }} else {{
      alert('Error: ' + (d.error || 'unknown'));
    }}
  }});
}}

function copyShownKey() {{
  navigator.clipboard.writeText(document.getElementById('shownKey').textContent).then(function() {{
    alert('Key copied to clipboard');
  }});
}}

function rotateKey(id) {{
  if (!confirm('Rotate this key? The old key will be revoked and a new one generated.')) return;
  api('POST', '/api/keys/' + id + '/rotate').then(function(d) {{
    if (d.key) {{
      document.getElementById('shownKey').textContent = d.key;
      document.getElementById('shownKeyInfo').innerHTML =
        'Rotated from ID: ' + d.old_id + ' → <b>' + d.id + '</b><br>' +
        'Name: <b>' + d.name + '</b>';
      document.getElementById('showKeyModal').classList.add('open');
    }} else {{
      alert('Error: ' + (d.error || 'unknown'));
    }}
  }});
}}

function revokeKey(id) {{
  if (!confirm('Revoke this key? It will no longer work for API calls.')) return;
  api('DELETE', '/api/keys/' + id).then(function(d) {{
    if (d.ok) loadKeys();
    else alert('Error: ' + (d.error || 'unknown'));
  }});
}}

function filterKeys() {{
  var q = document.getElementById('keySearch').value.toLowerCase();
  document.querySelectorAll('#keysBody tr').forEach(function(tr) {{
    var txt = (tr.getAttribute('data-search') || '').toLowerCase();
    tr.style.display = txt.includes(q) ? '' : 'none';
  }});
}}

function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    document.getElementById('sidebarOverlay').classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}

loadKeys();
</script>
""" + render_sidebar_plugin_js(config, engine) + """
</body>
</html>"""


def render_sidebar_html(config, active_page: str = "") -> str:
    """Render the sidebar navigation for standalone pages."""
    dash_active = ' class="active"' if active_page == "dashboard" else ""
    keys_active = ' class="active"' if active_page == "keys" else ""
    providers_active = ' class="active"' if active_page == "providers" else ""
    profiles_active = ' class="active"' if active_page == "profiles" else ""
    usage_active = ' class="active"' if active_page == "usage" else ""

    sidebar = (
        '<aside class="sidebar" id="sidebar">\n'
        '  <div class="sidebar-brand">LCP</div>\n'
        '  <nav class="sidebar-nav">\n'
        f'    <a href="/dashboard"{dash_active}>Dashboard</a>\n'
        f'    <a href="/keys"{keys_active}>API Keys</a>\n'
        f'    <a href="/providers"{providers_active}>Providers</a>\n'
        f'    <a href="/usage"{usage_active}>Usage</a>\n'
        '    <div class="sb-provider-rows" id="providerPluginRows"></div>\n'
        f'    <a href="/profiles"{profiles_active}>Profiles</a>\n'
        '    <div class="nav-label">Profiles</div>\n'
    )
    for p in config.profiles.keys():
        sidebar += f'    <a href="/{p}/dashboard">{p.upper()}</a>\n'
    sidebar += (
        '  </nav>\n</aside>'
    )
    return sidebar


def render_sidebar_plugin_js(config, engine=None) -> str:
    """Shared JS for sidebar plugin status rows (loadPluginStatus, toggle, etc.).

    Injects ``monthly`` gateway data and ``configuredProviders`` lists so
    that the sidebar plugin rows can fall back to the gateway DB when the
    provider plugin has no local data (e.g. OpenCode DB not on server).

    Requires ``engine`` (SQLAlchemy) to query the gateway DB for monthly
    totals.  When ``engine`` is None the monthly fallback is silently empty.
    """
    from datetime import date as _date
    from sqlalchemy import func

    # Monthly data from gateway DB
    _first_of_month = _date.today().replace(day=1).isoformat()
    monthly_data = {}
    if engine is not None:
        try:
            from ..api.models import get_session, Request as RequestModel
            with get_session(engine) as _s:
                _monthly_rows = _s.query(
                    RequestModel.provider,
                    func.count(RequestModel.id).label("m_reqs"),
                    func.coalesce(func.sum(RequestModel.completion_tokens + RequestModel.prompt_tokens), 0).label("m_tokens"),
                    func.coalesce(func.sum(RequestModel.cost), 0).label("m_cost"),
                ).filter(
                    RequestModel.timestamp >= _first_of_month,
                    RequestModel.success == 1,
                ).group_by(RequestModel.provider).all()
                for r in _monthly_rows:
                    monthly_data[r.provider] = {
                        "reqs": int(r.m_reqs),
                        "tokens": int(r.m_tokens),
                        "cost": float(r.m_cost),
                    }
        except Exception:
            pass

    monthly_json = json.dumps(monthly_data)
    configured_json = json.dumps(sorted(config.providers.keys()))

    return f"""<script>
function toggleSidebar() {{
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebarOverlay');
  if (window.innerWidth <= 768) {{
    sb.classList.toggle('open');
    ov.classList.toggle('show');
  }} else {{
    sb.classList.toggle('collapsed');
    localStorage.setItem('lcp-sidebar', sb.classList.contains('collapsed') ? 'collapsed' : 'pinned');
  }}
}}
function closeSidebar() {{
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}}
(function() {{
  if (window.innerWidth <= 768) return;
  if (localStorage.getItem('lcp-sidebar') === 'collapsed') {{
    document.getElementById('sidebar').classList.add('collapsed');
  }}
}})();

function formatTokens(n) {{
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return String(n);
}}

var monthly = {monthly_json};
var configuredProviders = {configured_json};

function loadPluginStatus() {{
  var provRows = document.getElementById('providerPluginRows');
  if (!provRows) return;

  Promise.all([
    fetch('/api/cost-plugins/usage').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_usage:{{}}}}}}),
    fetch('/api/cost-plugins/balances').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_balances:{{}}}}}}),
    fetch('/api/cost-plugins/summary').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_summaries:{{}}}}}}),
    fetch('/api/cost-plugins/subscriptions').then(function(r){{return r.json()}}).catch(function(){{return {{plugin_subscriptions:{{}}}}}})
  ]).then(function(results) {{
    var allUsage = results[0].plugin_usage || {{}};
    var balances = results[1].plugin_balances || {{}};
    var pluginSummaries = results[2].plugin_summaries || {{}};
    var subscriptions = results[3].plugin_subscriptions || {{}};

    var allProviders = Object.keys(allUsage).concat(Object.keys(balances));
    var uniqueProvs = allProviders.filter(function(v,i,a){{return a.indexOf(v)===i}}).filter(function(v){{return configuredProviders.indexOf(v) !== -1}});

    if (uniqueProvs.length === 0) {{
      provRows.innerHTML = '<div class="sb-provider-empty">No plugins active</div>';
    }} else {{
      var rows = '';
      uniqueProvs.forEach(function(prov) {{
        var bal = balances[prov];
        var usg = allUsage[prov] || [];
        var totalCost = usg.reduce(function(s,r){{return s + r.cost}}, 0);
        var totalTokens = usg.reduce(function(s,r){{return s + r.prompt_tokens + r.completion_tokens}}, 0);
        // Fallback to gateway DB when plugin has no usage data
        if (totalCost === 0 && monthly[prov]) {{
          totalCost = monthly[prov].cost || 0;
        }}
        if (totalTokens === 0 && monthly[prov]) {{
          totalTokens = monthly[prov].tokens || 0;
        }}
        var sum = pluginSummaries[prov];

        var detailLine = '';
        if (prov === 'opencode') {{
          var om = (sum && sum.monthly) ? sum.monthly : (monthly[prov] || {{}});
          detailLine = '<span class="sb-provider-detail">' +
            'month: ' + formatTokens(om.tokens || totalTokens) + ' tok';
          if (om.cost) detailLine += ' \\u00b7 $' + om.cost.toFixed(4);
          // Subscription data from OpenCode web API
          var sub = subscriptions[prov];
          if (sub) {{
            detailLine += '<br><span class="sb-sub-detail">';
            var parts = [];
            if (sub.rolling_pct != null) {{
              var resetMin = Math.floor(sub.rolling_reset_sec / 60);
              var r = '5h: ' + sub.rolling_pct.toFixed(0) + '% used';
              if (resetMin > 0) r += ' \\u00b7 resets in ' + resetMin + 'm';
              parts.push(r);
            }}
            if (sub.weekly_pct != null) {{
              var wkResetHr = Math.floor(sub.weekly_reset_sec / 3600);
              var w = 'week: ' + sub.weekly_pct.toFixed(0) + '% used';
              if (wkResetHr > 0) w += ' \\u00b7 resets in ' + wkResetHr + 'h';
              parts.push(w);
            }}
            if (sub.monthly_pct != null) {{
              var moResetDay = Math.floor(sub.monthly_reset_sec / 86400);
              var m = 'month: ' + sub.monthly_pct.toFixed(0) + '% used';
              if (moResetDay > 0) m += ' \\u00b7 resets in ' + moResetDay + 'd';
              parts.push(m);
            }}
            detailLine += parts.join('<br>');
            detailLine += '</span>';
          }}
          detailLine += '</span>';
        }} else if (prov === 'deepseek' && sum && sum.balance) {{
          var cur = sum.balance.currency || 'USD';
          detailLine = '<span class="sb-provider-detail">' + cur + ' ' + sum.balance.available.toFixed(2) + ' available';
          if (sum.balance.spent !== null) detailLine += ' \\u00b7 spent ' + cur + ' ' + sum.balance.spent.toFixed(2);
          detailLine += '</span>';
        }} else if (bal && bal.balance !== null && bal.balance !== undefined) {{
          var currency = bal.currency || 'USD';
          detailLine = '<span class="sb-provider-detail">' + currency + ' ' + bal.balance.toFixed(2) + ' balance</span>';
        }} else if (prov === 'llamacpp') {{
          detailLine = '<span class="sb-provider-detail">' + formatTokens(totalTokens) + ' tokens \\u00b7 local</span>';
        }} else {{
          var m = monthly[prov] || {{}};
          var mr = m.reqs || 0;
          var mt = m.tokens || 0;
          if (mr > 0) detailLine = '<span class="sb-provider-detail">' + mr + ' req \\u00b7 ' + formatTokens(mt) + ' tok this month</span>';
        }}

        rows += '<div class="sb-provider-row">' +
          '<div class="sb-provider-top">' +
          '<span class="sb-provider-name">' + prov + '</span>' +
          '<span class="sb-provider-cost">$' + totalCost.toFixed(4) + '</span>' +
          '</div>' +
          (detailLine ? detailLine : '') +
          '</div>';
      }});
      provRows.innerHTML = rows;
    }}
  }});
}}

loadPluginStatus();
setInterval(loadPluginStatus, 60000);
</script>"""


def render_usage_page(config, engine=None) -> str:
    """Render the Usage & Spending page with per-provider stats."""
    from pathlib import Path
    _templates_dir = Path(__file__).parent / "templates"
    css = ""
    try:
        css = (_templates_dir / "dashboard.css").read_text()
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LCP — Usage</title>
<style>{css}
.page-section {{ margin-bottom: 2rem; }}
.page-section h2 {{ font-size: 1.1rem; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; }}
.stat-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }}
.stat-card {{ background: hsl(var(--card)); border: 1px solid hsl(var(--card-border)); border-radius: var(--radius); padding: 1rem; }}
.stat-card .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem; }}
.stat-card .value {{ font-size: 1.4rem; font-weight: 700; }}
.stat-card .sub {{ font-size: 0.7rem; color: hsl(var(--muted-foreground)); margin-top: 0.15rem; }}
.chart-wrap {{ background: hsl(var(--card)); border: 1px solid hsl(var(--card-border)); border-radius: var(--radius); padding: 1rem; margin-bottom: 1rem; }}
.chart-wrap canvas {{ max-height: 250px; }}
.breakdown-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
.breakdown-table th {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid hsl(var(--card-border)); color: hsl(var(--muted-foreground)); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; }}
.breakdown-table td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid hsl(var(--card-border) / 0.4); }}
.breakdown-table tr:last-child td {{ border-bottom: none; }}
.breakdown-table .cost-col {{ text-align: right; font-variant-numeric: tabular-nums; }}
.progress-bar {{ height: 6px; background: hsl(var(--secondary)); border-radius: 3px; overflow: hidden; margin-top: 0.35rem; }}
.progress-fill {{ height: 100%; border-radius: 3px; transition: width 0.5s; }}
.progress-blue {{ background: hsl(var(--blue-fg, 217 91% 60%)); }}
.usage-bars {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }}
.usage-bar-item {{ background: hsl(var(--card)); border: 1px solid hsl(var(--card-border)); border-radius: var(--radius); padding: 0.85rem 1rem; }}
.usage-bar-item .bar-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }}
.usage-bar-item .bar-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: hsl(var(--muted-foreground)); }}
.usage-bar-item .bar-pct {{ font-size: 1.1rem; font-weight: 700; }}
.usage-bar-item .bar-track {{ height: 8px; background: hsl(var(--secondary)); border-radius: 4px; overflow: hidden; margin-bottom: 0.35rem; }}
.usage-bar-item .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.6s ease; }}
.usage-bar-item .bar-fill.orange {{ background: linear-gradient(90deg, #f97316, #f97316); }}
.usage-bar-item .bar-fill.yellow {{ background: linear-gradient(90deg, #eab308, #eab308); }}
.usage-bar-item .bar-fill.green {{ background: linear-gradient(90deg, #22c55e, #22c55e); }}
.usage-bar-item .bar-fill.red {{ background: linear-gradient(90deg, #ef4444, #ef4444); }}
.usage-bar-item .bar-sub {{ display: flex; justify-content: space-between; font-size: 0.65rem; color: hsl(var(--muted-foreground)); }}
.progress-green {{ background: hsl(var(--green-fg)); }}
.tab-bar {{ display: flex; gap: 0; margin-bottom: 1rem; border-bottom: 2px solid hsl(var(--card-border)); }}
.tab-btn {{ padding: 0.5rem 1rem; font-size: 0.8rem; background: none; border: none; color: hsl(var(--muted-foreground)); border-bottom: 2px solid transparent; margin-bottom: -2px; cursor: pointer; transition: color 0.2s, border-color 0.2s; }}
.tab-btn:hover {{ color: hsl(var(--foreground)); }}
.tab-btn.active {{ color: hsl(var(--foreground)); border-bottom-color: hsl(var(--foreground)); }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}
.empty-state {{ text-align: center; padding: 1.5rem; color: hsl(var(--muted-foreground)); font-size: 0.85rem; }}
.date-filters {{ display: flex; gap: 0.35rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }}
.date-pill {{ padding: 0.3rem 0.7rem; font-size: 0.7rem; background: hsl(var(--secondary)); border: 1px solid hsl(var(--card-border)); border-radius: var(--radius); color: hsl(var(--muted-foreground)); cursor: pointer; transition: all 0.15s; }}
.date-pill:hover {{ color: hsl(var(--foreground)); border-color: hsl(var(--muted-foreground)); }}
.date-pill.active {{ background: hsl(var(--foreground)); color: hsl(var(--background)); border-color: hsl(var(--foreground)); }}
.date-pill-input {{ padding: 0.25rem 0.4rem; font-size: 0.7rem; background: hsl(var(--card)); border: 1px solid hsl(var(--card-border)); border-radius: var(--radius); color: hsl(var(--foreground)); font-family: inherit; max-width: 130px; }}
.balance-bar {{ height: 8px; background: hsl(var(--secondary)); border-radius: 4px; overflow: hidden; margin-top: 0.35rem; }}
.balance-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
</style>
</head>
<body>
{render_sidebar_html(config, "usage")}
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">☰</button>
<div class="main-content">
<h1>Usage &amp; Spending</h1>
<p class="subtitle">Per-provider cost breakdowns, balance, and trends</p>

<div class="tab-bar" id="providerTabs"></div>

<div id="providerPanels"></div>

<p style="margin-top:2rem;font-size:0.75rem;color:hsl(var(--muted-foreground))">
  <a href="/health">/health</a> · <a href="/metrics">/metrics</a> · <a href="/export">/export</a>
  · <a href="/providers">Providers</a> · <a href="/dashboard">Dashboard</a>
</p>
<p style="font-size:0.7rem;color:hsl(var(--muted-foreground))">Generated <span id="genTime"></span> · Refresh to update</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
// ── State ──
var configuredProviders = {json.dumps([p for p in config.providers.keys()])};
var monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
document.getElementById('genTime').textContent = new Date().toISOString().replace('T',' ').slice(0,19) + ' UTC';

function formatTokens(n) {{
    if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
    return String(n);
}}

function monthPct() {{
    var now = new Date();
    var start = new Date(now.getFullYear(), now.getMonth(), 1);
    var end = new Date(now.getFullYear(), now.getMonth() + 1, 1);
    return ((now - start) / (end - start)) * 100;
}}

function renderDailyChart(canvasId, daily) {{
    var ctx = document.getElementById(canvasId);
    if (!ctx) return;
    var labels = daily.map(function(d) {{ var p = d.date.split('-'); return p[2] + ' ' + (monthNames[parseInt(p[1])-1] || ''); }});
    var costs = daily.map(function(d) {{ return d.cost; }});
    window._charts = window._charts || {{}};
    window._charts[canvasId] = new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [{{
                label: 'Daily Cost (USD)',
                data: costs,
                backgroundColor: 'hsl(217 91% 60% / 0.6)',
                borderColor: 'hsl(217 91% 60%)',
                borderWidth: 1,
                borderRadius: 3,
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ ticks: {{ color: 'hsl(var(--muted-foreground))', maxTicksLimit: 15, font: {{ size: 10 }} }}, grid: {{ display: false }} }},
                y: {{ ticks: {{ color: 'hsl(var(--muted-foreground))', callback: function(v) {{ return '$' + v.toFixed(2); }}, font: {{ size: 10 }} }}, grid: {{ color: 'hsl(var(--card-border) / 0.3)' }} }}
            }}
        }}
    }});
}}

function renderBreakdownTable(containerId, data, labelKey) {{
    var el = document.getElementById(containerId);
    if (!el) return;
    var keys = Object.keys(data).sort(function(a,b) {{ return data[b].cost - data[a].cost; }});
    if (keys.length === 0) {{
        el.innerHTML = '<div class="empty-state">No data for this period</div>';
        return;
    }}
    var maxCost = data[keys[0]].cost || 1;
    var html = '<table class="breakdown-table"><thead><tr><th>' + labelKey + '</th><th class="cost-col">Cost</th><th class="cost-col">Requests</th><th class="cost-col">Tokens</th><th>Share</th></tr></thead><tbody>';
    keys.forEach(function(k) {{
        var d = data[k];
        var pct = maxCost > 0 ? ((d.cost / maxCost) * 100) : 0;
        var barColor = maxCost > 0 && d.cost === maxCost ? 'progress-blue' : 'progress-green';
        html += '<tr>' +
            '<td><b>' + k + '</b></td>' +
            '<td class="cost-col">$' + d.cost.toFixed(4) + '</td>' +
            '<td class="cost-col">' + d.requests + '</td>' +
            '<td class="cost-col">' + formatTokens(d.prompt_tokens + d.completion_tokens) + '</td>' +
            '<td style="width:120px"><div class="progress-bar"><div class="progress-fill ' + barColor + '" style="width:' + Math.round(pct) + '%"></div></div></td>' +
            '</tr>';
    }});
    html += '</tbody></table>';
    el.innerHTML = html;
}}

function loadStats(provider) {{
    return fetch('/api/usage/stats?provider=' + encodeURIComponent(provider) + '&days=30')
        .then(function(r) {{ return r.json(); }});
}}

function loadSummaries() {{
    return fetch('/api/cost-plugins/summary')
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{ return d.plugin_summaries || {{}}; }});
}}

function loadSubscriptions() {{
    return fetch('/api/cost-plugins/subscriptions')
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{ return d.plugin_subscriptions || {{}}; }});
}}

function buildPage(providers) {{
    return Promise.all([
        Promise.all(providers.map(loadStats)),
        loadSummaries(),
        loadSubscriptions()
    ]).then(function(results) {{
        var statsList = results[0];
        var summaries = results[1];
        var subscriptions = results[2];

        // ── Tab bar ──
        var tabBar = document.getElementById('providerTabs');
        var panelsDiv = document.getElementById('providerPanels');
        var tabsHtml = '';
        var panelsHtml = '';

        providers.forEach(function(prov, i) {{
            var activeClass = i === 0 ? ' active' : '';
            tabsHtml += '<button class="tab-btn' + activeClass + '" onclick="switchTab(\\'' + prov + '\\')">' + prov + '</button>';

            var stats = statsList[i] || {{daily:[], by_model:{{}}, by_profile:{{}}, totals:{{cost:0, requests:0}}}};
            var sum = summaries[prov];
            var panelId = 'panel-' + prov;
            var dailyChartId = 'chart-daily-' + prov;
            var modelTableId = 'table-model-' + prov;
            var profileTableId = 'table-profile-' + prov;

            panelsHtml += '<div class="tab-panel' + activeClass + '" id="' + panelId + '">';

            // ── Provider-specific cards ──
            if (prov === 'deepseek') {{
                var bal = (sum && sum.balance) ? sum.balance : {{}};
                var available = bal.available || 0;
                var spent = bal.spent;
                var topped = bal.topped_up || 0;
                var granted = bal.total_granted || 0;
                var totalEver = topped + granted;
                var balanceUsedPct = totalEver > 0 ? Math.min(100, ((totalEver - available) / totalEver) * 100) : 0;
                function balanceBarClass(pct) {{ return pct >= 90 ? 'red' : pct >= 70 ? 'orange' : pct >= 40 ? 'yellow' : 'green'; }}

                // Cache stats
                var cache = stats.cache || {{hit_tokens:0, miss_tokens:0, savings:0}};
                var cacheTotal = cache.hit_tokens + cache.miss_tokens;
                var cacheRate = cacheTotal > 0 ? (cache.hit_tokens / cacheTotal * 100) : 0;

                // Model entries from stats
                var modelEntries = [];
                var modelKeys = Object.keys(stats.by_model || {{}});
                for (var mi = 0; mi < modelKeys.length; mi++) {{
                    if (modelKeys[mi] !== 'unknown') {{
                        modelEntries.push({{key: modelKeys[mi], data: stats.by_model[modelKeys[mi]]}});
                    }}
                }}

                // Weekly + Today from daily array
                var dailyArr = stats.daily || [];
                var todayCost = 0, todayReqs = 0, weeklyCost = 0, weeklyReqs = 0;
                var todayStr = new Date().toISOString().slice(0,10);
                if (dailyArr.length > 0) {{
                    var lastDay = dailyArr[dailyArr.length - 1];
                    if (lastDay.date === todayStr) {{ todayCost = lastDay.cost; todayReqs = lastDay.requests; }}
                    var wStart = Math.max(0, dailyArr.length - 7);
                    for (var wi = wStart; wi < dailyArr.length; wi++) {{
                        weeklyCost += dailyArr[wi].cost;
                        weeklyReqs += dailyArr[wi].requests;
                    }}
                }}

                // ── Row 1: Balance ──
                panelsHtml += '<div class="stat-cards">' +
                    '<div class="stat-card"><div class="label">Available Balance</div><div class="value">$' + available.toFixed(2) + '</div><div class="sub">' + (bal.currency || 'USD') + '</div></div>' +
                    '<div class="stat-card"><div class="label">Total Spent (API)</div><div class="value">' + (spent != null ? '$' + spent.toFixed(2) : '—') + '</div><div class="sub">topped $' + topped.toFixed(2) + ' + granted $' + granted.toFixed(2) + '</div></div>' +
                    '<div class="stat-card"><div class="label">Balance Used</div><div class="value">' + balanceUsedPct.toFixed(1) + '%</div><div class="sub">of $' + totalEver.toFixed(2) + ' total</div><div class="balance-bar"><div class="balance-bar-fill ' + balanceBarClass(balanceUsedPct) + '" style="width:' + balanceUsedPct + '%"></div></div></div>' +
                    '</div>';

                // ── Row 2: Cache + Models ──
                panelsHtml += '<div class="stat-cards">';
                if (cacheTotal > 0) {{
                    panelsHtml += '<div class="stat-card"><div class="label">Cache Hit Rate</div><div class="value">' + cacheRate.toFixed(1) + '%</div><div class="sub">' + formatTokens(cache.hit_tokens) + ' hit / ' + formatTokens(cache.miss_tokens) + ' miss</div></div>' +
                        '<div class="stat-card"><div class="label">Cache Savings</div><div class="value">$' + cache.savings.toFixed(4) + '</div><div class="sub">from cache-hit discount</div></div>';
                }}
                for (var mi = 0; mi < modelEntries.length; mi++) {{
                    var me = modelEntries[mi];
                    if (me.data.requests > 0) {{
                        panelsHtml += '<div class="stat-card"><div class="label">' + me.key + '</div><div class="value">$' + me.data.cost.toFixed(4) + '</div><div class="sub">' + me.data.requests + ' req · ' + formatTokens(me.data.prompt_tokens + me.data.completion_tokens) + ' tok</div></div>';
                    }}
                }}
                panelsHtml += '</div>';

                // ── Row 3: Gateway Spend / Weekly / Today ──
                panelsHtml += '<div class="stat-cards">' +
                    '<div class="stat-card"><div class="label">Gateway Spend (30d)</div><div class="value">$' + stats.totals.cost.toFixed(4) + '</div><div class="sub">' + stats.totals.requests + ' requests</div></div>' +
                    '<div class="stat-card"><div class="label">Rolling Weekly</div><div class="value">$' + weeklyCost.toFixed(4) + '</div><div class="sub">' + weeklyReqs + ' requests</div></div>' +
                    '<div class="stat-card"><div class="label">Today</div><div class="value">$' + todayCost.toFixed(4) + '</div><div class="sub">' + todayReqs + ' requests</div></div>' +
                    '</div>';
            }} else if (prov === 'opencode') {{
                var monthly = (sum && sum.monthly) ? sum.monthly : {{tokens:0, cost:0, requests:0}};
                var weekly = (sum && sum.weekly) ? sum.weekly : {{tokens:0, cost:0, requests:0}};
                var daily = (sum && sum.daily) ? sum.daily : {{tokens:0, cost:0, requests:0}};
                // Fallback to gateway DB when OpenCode local DB is unavailable
                var gwMonthly = stats.totals || {{cost:0, requests:0}};
                if (monthly.tokens === 0 && monthly.requests === 0) {{
                    monthly.requests = gwMonthly.requests || 0;
                    monthly.cost = (monthly.cost || 0) || (gwMonthly.cost || 0);
                }}
                // Subscription usage from OpenCode web API - progress bars
                var sub = subscriptions['opencode'] || {{}};
                function barColor(pct) {{ return pct >= 90 ? 'red' : pct >= 70 ? 'orange' : pct >= 40 ? 'yellow' : 'green'; }}
                function fmtCountdown(sec) {{
                    if (!sec || sec <= 0) return '';
                    var d = Math.floor(sec / 86400);
                    var h = Math.floor((sec % 86400) / 3600);
                    var m = Math.floor((sec % 3600) / 60);
                    var parts = [];
                    if (d > 0) parts.push(d + 'd');
                    if (h > 0) parts.push(h + 'h');
                    if (m > 0 || parts.length === 0) parts.push(m + 'm');
                    return parts.join(' ');
                }}
                var barsHtml = '';
                if (sub.rolling_pct != null) {{
                    barsHtml += '<div class="usage-bar-item"><div class="bar-header"><span class="bar-label">5h Rolling</span><span class="bar-pct">' + sub.rolling_pct.toFixed(0) + '%</span></div><div class="bar-track"><div class="bar-fill ' + barColor(sub.rolling_pct) + '" style="width:' + sub.rolling_pct + '%"></div></div><div class="bar-sub"><span>' + fmtCountdown(sub.rolling_reset_sec) + '</span><span>' + (sub.rolling_reset_at || '') + '</span></div></div>';
                }}
                if (sub.weekly_pct != null) {{
                    barsHtml += '<div class="usage-bar-item"><div class="bar-header"><span class="bar-label">Weekly</span><span class="bar-pct">' + sub.weekly_pct.toFixed(0) + '%</span></div><div class="bar-track"><div class="bar-fill ' + barColor(sub.weekly_pct) + '" style="width:' + sub.weekly_pct + '%"></div></div><div class="bar-sub"><span>' + fmtCountdown(sub.weekly_reset_sec) + '</span><span>' + (sub.weekly_reset_at || '') + '</span></div></div>';
                }}
                if (sub.monthly_pct != null) {{
                    barsHtml += '<div class="usage-bar-item"><div class="bar-header"><span class="bar-label">Monthly</span><span class="bar-pct">' + sub.monthly_pct.toFixed(0) + '%</span></div><div class="bar-track"><div class="bar-fill ' + barColor(sub.monthly_pct) + '" style="width:' + sub.monthly_pct + '%"></div></div><div class="bar-sub"><span>' + fmtCountdown(sub.monthly_reset_sec) + '</span><span>' + (sub.monthly_reset_at || '') + '</span></div></div>';
                }}
                if (barsHtml) {{
                    panelsHtml += '<div class="usage-bars">' + barsHtml + '</div>';
                }}
                panelsHtml += '<div class="stat-cards">' +
                    '<div class="stat-card"><div class="label">Gateway Spend (30d)</div><div class="value">$' + monthly.cost.toFixed(4) + '</div><div class="sub">' + monthly.requests + ' requests</div></div>' +
                    '<div class="stat-card"><div class="label">Rolling Weekly</div><div class="value">' + formatTokens(weekly.tokens) + ' tok</div><div class="sub">$' + weekly.cost.toFixed(4) + ' · ' + weekly.requests + ' requests</div></div>' +
                    '<div class="stat-card"><div class="label">Today</div><div class="value">' + formatTokens(daily.tokens) + ' tok</div><div class="sub">$' + daily.cost.toFixed(4) + ' · ' + daily.requests + ' requests</div></div>' +
                    '</div>';
            }} else {{
                panelsHtml += '<div class="stat-cards">' +
                    '<div class="stat-card"><div class="label">Gateway Spend (30d)</div><div class="value">$' + stats.totals.cost.toFixed(4) + '</div><div class="sub">' + stats.totals.requests + ' requests</div></div>' +
                    '</div>';
            }}

            // ── Date filter pills (DeepSeek only) ──
            if (prov === 'deepseek') {{
                panelsHtml += '<div class="date-filters" id="dateFilt-' + prov + '">' +
                    '<button class="date-pill active" data-range="7" onclick="applyDateRange(this,\\'' + prov + '\\',7)">7 days</button>' +
                    '<button class="date-pill" data-range="30" onclick="applyDateRange(this,\\'' + prov + '\\',30)">30 days</button>' +
                    '<button class="date-pill" data-range="month" onclick="applyDateRange(this,\\'' + prov + '\\',\\'month\\')">This month</button>' +
                    '<button class="date-pill" data-range="last" onclick="applyDateRange(this,\\'' + prov + '\\',\\'last\\')">Last month</button>' +
                    '<input class="date-pill-input" type="date" id="dsFrom-' + prov + '" title="Start">' +
                    '<input class="date-pill-input" type="date" id="dsTo-' + prov + '" title="End">' +
                    '<button class="date-pill" data-range="custom" onclick="applyDateRange(this,\\'' + prov + '\\',\\'custom\\')">Apply</button>' +
                    '</div>';
            }}

            // ── Daily chart ──
            panelsHtml += '<div class="chart-wrap"><h3 style="font-size:0.85rem;margin-bottom:0.5rem" id="chartTitle-' + prov + '">Daily Spending (7 days)</h3><div style="height:250px"><canvas id="' + dailyChartId + '"></canvas></div></div>';

            // ── Breakdown tables ──
            panelsHtml += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">' +
                '<div class="chart-wrap"><h3 style="font-size:0.85rem;margin-bottom:0.5rem">By Model</h3><div id="' + modelTableId + '"></div></div>' +
                '<div class="chart-wrap"><h3 style="font-size:0.85rem;margin-bottom:0.5rem">By Profile</h3><div id="' + profileTableId + '"></div></div>' +
                '</div>';

            panelsHtml += '</div>';  // end tab-panel
        }});

        tabBar.innerHTML = tabsHtml;
        panelsDiv.innerHTML = panelsHtml;

        // Render charts & tables for each provider
        providers.forEach(function(prov, i) {{
            var stats = statsList[i] || {{daily:[], by_model:{{}}, by_profile:{{}}}};
            renderDailyChart('chart-daily-' + prov, stats.daily);
            renderBreakdownTable('table-model-' + prov, stats.by_model, 'Model');
            renderBreakdownTable('table-profile-' + prov, stats.by_profile, 'Profile');
        }});
    }});
}}

window.switchTab = function(prov) {{
    document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
    var btn = document.querySelector('.tab-btn[onclick*="' + prov + '"]');
    var panel = document.getElementById('panel-' + prov);
    if (btn) btn.classList.add('active');
    if (panel) panel.classList.add('active');
}};

window.applyDateRange = function(btnEl, provider, range) {{
    // Update pill styling
    var container = btnEl.parentElement;
    container.querySelectorAll('.date-pill').forEach(function(p) {{ p.classList.remove('active'); }});
    btnEl.classList.add('active');

    var start, end, label;
    var today = new Date();
    var d = new Date(today);

    if (range === 'custom') {{
        start = document.getElementById('dsFrom-' + provider).value;
        end = document.getElementById('dsTo-' + provider).value;
        if (!start || !end) return;
        label = start + ' ~ ' + end;
    }} else if (range === 7) {{
        d.setDate(d.getDate() - 7);
        start = d.toISOString().slice(0,10);
        end = today.toISOString().slice(0,10);
        label = '7 days';
    }} else if (range === 30) {{
        d.setDate(d.getDate() - 30);
        start = d.toISOString().slice(0,10);
        end = today.toISOString().slice(0,10);
        label = '30 days';
    }} else if (range === 'month') {{
        start = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().slice(0,10);
        end = today.toISOString().slice(0,10);
        label = 'This month';
    }} else if (range === 'last') {{
        var firstDay = new Date(today.getFullYear(), today.getMonth() - 1, 1);
        var lastDay = new Date(today.getFullYear(), today.getMonth(), 0);
        start = firstDay.toISOString().slice(0,10);
        end = lastDay.toISOString().slice(0,10);
        label = 'Last month';
    }}

    // Update chart title
    var titleEl = document.getElementById('chartTitle-' + provider);
    if (titleEl) titleEl.textContent = 'Daily Spending (' + label + ')';

    // Fetch fresh stats with date range
    fetch('/api/usage/stats?provider=' + encodeURIComponent(provider) + '&start=' + start + '&end=' + end)
        .then(function(r) {{ return r.json(); }})
        .then(function(stats) {{
            // Destroy old chart
            var canvasKey = 'chart-daily-' + provider;
            if (window._charts && window._charts[canvasKey]) {{
                window._charts[canvasKey].destroy();
            }}
            renderDailyChart(canvasKey, stats.daily || []);
            renderBreakdownTable('table-model-' + provider, stats.by_model || {{}}, 'Model');
            renderBreakdownTable('table-profile-' + provider, stats.by_profile || {{}}, 'Profile');
        }})
        .catch(function(e) {{
            console.error('Date filter fetch failed', e);
        }});
}};

window.toggleSidebar = function() {{
    var sb = document.getElementById('sidebar');
    var ov = document.getElementById('sidebarOverlay');
    if (sb) sb.classList.toggle('open');
    if (ov) ov.classList.toggle('open');
}};
window.closeSidebar = function() {{
    var sb = document.getElementById('sidebar');
    var ov = document.getElementById('sidebarOverlay');
    if (sb) sb.classList.remove('open');
    if (ov) ov.classList.remove('open');
}};

// ── Boot ──
buildPage(configuredProviders).catch(function(e) {{
    console.error('Failed to load usage page', e);
}});
</script>
""" + render_sidebar_plugin_js(config, engine) + """
</body>
</html>"""

