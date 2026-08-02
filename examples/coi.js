// Cross-origin isolation on a host that cannot send headers.
//
// onnxruntime-web runs WASM threads only when SharedArrayBuffer exists, which
// browsers expose only on a cross-origin isolated page, which needs COOP+COEP
// response headers. GitHub Pages cannot send headers — so the CPU path there is
// stuck on ONE thread no matter how many cores the machine has. Measured on an
// 18-core Mac: 1.5 tok/s without threads, 4.7 with. Same model, same weights.
//
// A service worker sits between the page and the network and can add the
// headers to responses on the way back, which is enough to satisfy the browser.
// The page then has to reload once for isolation to take effect, because the
// document itself was already fetched without them.
//
// COEP is `credentialless`, not `require-corp`: this page pulls the runtime
// from a CDN and weights from Hugging Face, and require-corp would block every
// cross-origin response that does not carry CORP — which is most of them.
// credentialless keeps them working by sending them without credentials.
//
// Same idea as gzuidhof/coi-serviceworker. Written out here rather than
// vendored so the reload behaviour is inspectable — a service worker that
// reloads the page is exactly the kind of thing you want to be able to read.

if (typeof window === 'undefined') {
  // ── Service worker side ───────────────────────────────────────────────────
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

  self.addEventListener('fetch', (event) => {
    const req = event.request;
    // Range requests must be passed through untouched; rewriting a 206 breaks
    // media and byte-range model loads.
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

    event.respondWith(
      fetch(req.mode === 'no-cors' ? new Request(req, { credentials: 'omit' }) : req)
        .then((res) => {
          if (res.status === 0) return res;   // opaque; headers cannot be set
          const headers = new Headers(res.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
        })
        .catch((e) => new Response(String(e.message), { status: 502 })),
    );
  });
} else {
  // ── Page side ─────────────────────────────────────────────────────────────
  (() => {
    // Already isolated (a host that sends real headers) — nothing to do, and
    // registering would be pure overhead.
    if (window.crossOriginIsolated) return;
    if (!('serviceWorker' in navigator)) return;

    // One reload only. Without this guard a browser that ignores the headers
    // would reload forever, which is far worse than a slow CPU path.
    const RELOADED = 'coi-reloaded';
    navigator.serviceWorker
      .register(window.document.currentScript?.src ?? 'coi.js', { scope: './' })
      .then((reg) => {
        if (reg.active && !navigator.serviceWorker.controller) {
          if (sessionStorage.getItem(RELOADED)) return;
          sessionStorage.setItem(RELOADED, '1');
          window.location.reload();
        }
        reg.addEventListener('updatefound', () => {
          if (sessionStorage.getItem(RELOADED)) return;
          sessionStorage.setItem(RELOADED, '1');
          window.location.reload();
        });
      })
      .catch(() => { /* isolation is an optimisation, never a requirement */ });
  })();
}
