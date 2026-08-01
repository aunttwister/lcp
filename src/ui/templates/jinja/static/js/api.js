// Shared fetch helper — used by all pages.
// Returns a Promise that resolves to parsed JSON.
function api(method, url, body) {
  var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(function (r) { return r.json(); });
}
