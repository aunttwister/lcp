"""Modal HTML snippets for the LCP dashboard.
These are loaded as string constants — no build step required.
"""

# Provider Edit Modal Opened From Sidebar
PROVIDER_EDIT_MODAL_OPENED_FROM_SIDEBAR = """\
<div class="modal-overlay" id="provEditModal">
<div class="modal" style="width:min(560px,95vw)">
  <div class="modal-header">
    <h2 id="pemTitle"></h2>
    <button class="modal-close" onclick="closeProvEditModal()">✕</button>
  </div>
  <div class="modal-body">
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem">
      <span style="font-size:0.75rem;font-weight:600" id="pemStatus"></span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">API Base URL</label>
        <input id="pemUrl" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
      <div>
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">API Key Env Var</label>
        <input id="pemKeyEnv" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
      <div style="grid-column:1/-1">
        <label style="display:block;font-size:0.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:hsl(var(--muted-foreground));margin-bottom:0.25rem">Models (comma-separated)</label>
        <input id="pemModels" style="width:100%;padding:0.375rem 0.5rem;background:hsl(var(--background));border:1px solid hsl(var(--card-border));border-radius:var(--radius);color:hsl(var(--foreground));font-size:0.8125rem;font-family:inherit">
      </div>
    </div>
    <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
      <button class="btn-sm btn-success" id="pemTestBtn" onclick="testPemProvider()">Test Connection</button>
      <button class="btn-sm btn-primary" id="pemSaveBtn" onclick="savePemProvider()">Save Provider</button>
    </div>
    <div id="pemTestResult" class="test-result" style="display:none;margin-top:0.5rem"></div>
  </div>
</div>
</div>
"""

# Api Keys Modal
API_KEYS_MODAL = """\
<div class="modal-overlay" id="keysModal">
<div class="modal" style="width:min(500px,95vw)">
  <div class="modal-header">
    <h2>API Keys</h2>
    <button class="modal-close" onclick="closeKeysModal()">✕</button>
  </div>
  <div class="modal-body">
    <div id="keysList"><div class="empty">Loading...</div></div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm btn-primary" onclick="generateKey()">+ Generate Key</button>
    <button class="btn-sm" onclick="closeKeysModal()">Close</button>
  </div>
</div>
</div>
"""

# Profile Config Modal
PROFILE_CONFIG_MODAL = """\
<div class="modal-overlay" id="profileConfigModal">
<div class="modal" style="width:min(520px,95vw)">
  <div class="modal-header">
    <h2 id="pcmTitle">Profile: L2</h2>
    <button class="modal-close" onclick="closeProfileConfig()">✕</button>
  </div>
  <div class="modal-tabs" id="pcmTabs">
    <button class="modal-tab active" data-tab="pcm-apikeys">API Keys</button>
    <button class="modal-tab" data-tab="pcm-chain">Chain</button>
    <button class="modal-tab" data-tab="pcm-url">URL</button>
  </div>
  <div class="modal-body" id="pcmBody">
    <!-- Tab: API Keys -->
    <div class="tab-panel active" id="panel-pcm-apikeys">
      <div id="pcmKeysList"><div class="empty">Loading...</div></div>
      <div style="margin-top:0.75rem">
        <button class="btn-sm btn-primary" onclick="generateProfileKey()">+ Generate Key</button>
      </div>
    </div>
    <!-- Tab: Provider Chain -->
    <div class="tab-panel" id="panel-pcm-chain">
      <div id="pcmChainList"><div class="empty">Loading...</div></div>
    </div>
    <!-- Tab: Copyable URL -->
    <div class="tab-panel" id="panel-pcm-url">
      <div class="phm-label">Gateway URL</div>
      <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem">
        <code id="pcmUrl" style="flex:1;background:hsl(var(--secondary));padding:0.375rem 0.5rem;border-radius:var(--radius);font-size:0.8125rem;word-break:break-all"></code>
        <button class="btn-sm btn-primary" onclick="copyProfileUrl()">Copy</button>
      </div>
      <div class="phm-label" style="margin-top:1rem">Usage</div>
      <pre id="pcmCurl" style="background:hsl(var(--secondary));padding:0.5rem;border-radius:var(--radius);font-size:0.75rem;overflow-x:auto;margin-top:0.25rem"></pre>
    </div>
  </div>
  <div class="modal-footer">
    <button class="btn-sm" onclick="closeProfileConfig()">Close</button>
  </div>
</div>
</div>
</div>
</div>
"""
