// Run examples/index.html in jsdom with a mock NexusChat.
//
// The page is the thing users actually touch, and none of its wiring — the
// tools editor, the source picker, the snippet, the cache button — is covered
// by the library tests. This boots it for real: real DOM, real event handlers,
// real module body, with only the model swapped out.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { JSDOM } from 'jsdom';

const PAGE = fileURLToPath(new URL('../../examples/index.html', import.meta.url));

/** A NexusChat stand-in: records tools, answers from a script. */
export function mockChatClass({ script = ['answer'], device = 'wasm', dtype = 'q4' } = {}) {
  const state = { loads: [], instances: [] };

  class MockChat {
    constructor() {
      this.tools = new Map();
      this.device = device;
      this.dtype = dtype;
      this.modelId = 'stub/model';
      this.metrics = { summary: () => ({ load_ms_avg: 12, tokens_per_second: 34 }) };
      this.listeners = new Map();
      this.asked = [];
      state.instances.push(this);
    }
    static async load(source, opts) {
      state.loads.push({ source, opts });
      if (state.failNext) { state.failNext = false; throw new Error('boom'); }
      return new MockChat();
    }
    /** Mirrors the real contract: verify, report the attempt, throw on a dud. */
    static async loadForTools(source, opts = {}) {
      const chat = await MockChat.load(source, opts);
      const check = await chat.selfCheck();
      opts.onAttempt?.(check);
      if (!check.ok) {
        throw new Error(
          `no dtype of ${chat.modelId} could call a tool on ${chat.device}. Tried:\n  ${check.dtype}: ${check.detail}`,
        );
      }
      return chat;
    }
    async evalTools(code) {
      this.tools.clear();
      const register = (name, d, p, h) => this.tools.set(name, { d, p, h });
      // Same contract as the real evalTools: user code calling tool(...).
      await new Function('tool', `'use strict';\nreturn (async () => {\n${code}\n})();`)(register);
      return [...this.tools.keys()];
    }
    on(ev, fn) {
      if (!this.listeners.has(ev)) this.listeners.set(ev, new Set());
      this.listeners.get(ev).add(fn);
      return () => this.off(ev, fn);
    }
    off(ev, fn) { this.listeners.get(ev)?.delete(fn); }
    emit(ev, ...args) { for (const f of this.listeners.get(ev) ?? []) f(...args); }
    async selfCheck() {
      this.selfChecked = true;
      return state.selfCheck ?? {
        ok: true, called: true, grounded: true, needed_forcing: false,
        model: 'stub/model', device, dtype, answer: 'QX-7731',
        detail: 'stub/model (wasm/q4) calls tools correctly.',
      };
    }
    async chat(text) {
      this.asked.push(text);
      return script[Math.min(this.asked.length - 1, script.length - 1)];
    }
  }
  return { MockChat, state };
}

/** Embedder + knowledge stand-ins: record what was indexed, answer from script. */
export function mockKnowledge({ answer = 'The refund window is RX-4182 days.', retrieved } = {}) {
  const state = { created: 0, docs: [], asked: [], exported: 0, failNext: null };

  class MockEmbedder {
    static async load(source, opts) {
      if (state.failNext === 'embedder') { state.failNext = null; throw new Error('embedder boom'); }
      const e = new MockEmbedder();
      e.device = 'wasm';
      e.modelId = 'stub/embedder';
      return e;
    }
    async embed() { return new Float32Array([1, 0, 0]); }
    async embedBatch(texts) { return texts.map(() => new Float32Array([1, 0, 0])); }
  }

  class MockKnowledge {
    static async create() {
      state.created++;
      return new MockKnowledge();
    }
    async addDocument(doc) { state.docs.push(doc); return 2; }
    async retrieve(q) {
      return retrieved ?? [{ id: 'doc-0#0', text: 'The refund window for standard orders is RX-4182 days.' }];
    }
    async ask(q) {
      state.asked.push(q);
      if (state.failNext === 'ask') { state.failNext = null; throw new Error('ask boom'); }
      return answer;
    }
    async exportZip() { state.exported++; return new Uint8Array(1024); }
  }

  return { MockEmbedder, MockKnowledge, state };
}

/** Boot the page. Returns the window plus handles for driving it. */
export async function loadPage({ chatClass, knowledge, caches: cacheSeed = {} } = {}) {
  const html = readFileSync(PAGE, 'utf8');
  const { MockChat, state } = chatClass ?? mockChatClass();

  const dom = new JSDOM(html, { runScripts: 'outside-only', url: 'http://localhost:8765/examples/' });
  const { window } = dom;

  // Cache Storage — jsdom has none. Model it as a map of name -> [urls].
  const buckets = new Map(Object.entries(cacheSeed));
  window.caches = {
    keys: async () => [...buckets.keys()],
    open: async (name) => ({ keys: async () => (buckets.get(name) ?? []).map((u) => ({ url: u })) }),
    delete: async (name) => buckets.delete(name),
  };
  window.navigator.storage = { estimate: async () => ({ usage: 512 * 1024 * 1024 }) };
  window.confirm = () => window.__confirmAnswer ?? true;
  window.__buckets = buckets;

  // Take the module body and satisfy its two imports by injection.
  const body = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^import .*$/gm, '');

  const ctx = vm.createContext(window);
  ctx.NexusChat = MockChat;
  const kn = knowledge ?? mockKnowledge();
  ctx.NexusEmbedder = kn.MockEmbedder;
  ctx.NexusKnowledge = kn.MockKnowledge;
  // The export button dynamically imports fflate; jsdom has no module loader
  // for a bare specifier, so hand it a stub that is never actually used to zip.
  ctx.__mockFflate = {};
  ctx.URL = window.URL;
  ctx.describeSource = (s) => ('hub' in s ? `hub:${s.hub}` : 'base' in s ? `${s.base}${s.id}` : 'archive:<file>');
  ctx.transformers = {};
  ctx.console = console;

  await vm.runInContext(`(async () => {\n${body}\n})()`, ctx);
  await new Promise((r) => setTimeout(r, 0));   // let the top-level refreshCache settle

  return {
    window,
    doc: window.document,
    state,
    knowledge: kn.state,
    $: (id) => window.document.getElementById(id),
    click: (id) => window.document.getElementById(id).dispatchEvent(new window.Event('click')),
    fire: (id, type) => window.document.getElementById(id).dispatchEvent(new window.Event(type, { bubbles: true })),
    /** Wait for the page's async handlers to finish. */
    settle: () => new Promise((r) => setTimeout(r, 5)),
  };
}
