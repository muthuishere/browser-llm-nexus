// Knowledge + device tests using fake models — no network, no weights.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NexusKnowledge } from '../dist/knowledge.js';
import { detectDevice, preferredDtypeOrder } from '../dist/runtime.js';
import { MemoryIndex } from '../dist/rag.js';

/** Deterministic fake embedder: bag-of-chars vector, normalized. */
const fakeEmbedder = {
  device: 'wasm',
  async embed(text) {
    return (await this.embedBatch([text]))[0];
  },
  async embedBatch(texts) {
    return texts.map((t) => {
      const v = new Float32Array(26);
      for (const ch of t.toLowerCase()) {
        const i = ch.charCodeAt(0) - 97;
        if (i >= 0 && i < 26) v[i] += 1;
      }
      const norm = Math.hypot(...v) || 1;
      return v.map((x) => x / norm);
    });
  },
  async dispose() {},
};

/** Fake chat that echoes the prompt it received, so we can assert on context. */
function makeFakeChat() {
  const seen = [];
  return {
    seen,
    metrics: { summary: () => ({}) },
    on() {},
    async chat(prompt) {
      seen.push(prompt);
      return 'ANSWER';
    },
    async dispose() {},
  };
}

async function makeKB() {
  const chat = makeFakeChat();
  const kb = await NexusKnowledge.create({ chat, embedder: fakeEmbedder, chunkSize: 120, chunkOverlap: 10 });
  return { kb, chat };
}

test('detectDevice falls back to wasm without WebGPU', async () => {
  assert.equal(await detectDevice('auto'), 'wasm');
});

test('detectDevice honors an explicit choice', async () => {
  assert.equal(await detectDevice('webgpu'), 'webgpu');
  assert.equal(await detectDevice('wasm'), 'wasm');
});

test('detectDevice picks webgpu when an adapter exists', async () => {
  const original = globalThis.navigator;
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: { requestAdapter: async () => ({}) } },
    configurable: true,
  });
  try {
    assert.equal(await detectDevice('auto'), 'webgpu');
  } finally {
    Object.defineProperty(globalThis, 'navigator', { value: original, configurable: true });
  }
});

test('preferredDtypeOrder prefers fp16 on gpu, q4 on cpu', () => {
  assert.equal(preferredDtypeOrder('webgpu')[0], 'fp16');
  assert.equal(preferredDtypeOrder('wasm')[0], 'q4');
});

test('addDocument chunks, embeds and indexes', async () => {
  const { kb } = await makeKB();
  const n = await kb.addDocument({ id: 'doc1', text: 'alpha beta gamma. '.repeat(30) });
  assert.ok(n > 1, 'expected multiple chunks');
  assert.equal(kb.index.size, n);
  assert.equal(kb.docs.size, 1);
});

test('retrieve returns the relevant chunk', async () => {
  const { kb } = await makeKB();
  await kb.addDocument({ id: 'a', text: 'refunds are processed within thirty days of request' });
  await kb.addDocument({ id: 'b', text: 'zzz qqq xxx unrelated jumble' });
  const hits = await kb.retrieve('refund policy', 1);
  assert.equal(hits[0].meta.docId, 'a');
});

test('ask stuffs retrieved context into the user turn', async () => {
  const { kb, chat } = await makeKB();
  await kb.addDocument({ id: 'a', text: 'refunds are processed within thirty days' });
  const answer = await kb.ask('refund policy?');
  assert.equal(answer, 'ANSWER');
  const prompt = chat.seen[0];
  assert.match(prompt, /Context:/);
  assert.match(prompt, /refunds are processed/);
  assert.match(prompt, /Question: refund policy\?/);
});

test('ask without documents skips the context block', async () => {
  const { kb, chat } = await makeKB();
  await kb.ask('hello');
  assert.doesNotMatch(chat.seen[0], /Context:/);
});

test('export/import round-trips the index without re-embedding', async () => {
  const { kb } = await makeKB();
  await kb.addDocument({ id: 'a', title: 'Handbook', text: 'refunds within thirty days' });
  const bundle = await kb.export();

  assert.equal(bundle.version, 1);
  assert.equal(bundle.docs[0].title, 'Handbook');
  assert.ok(bundle.index.chunks.length > 0);

  const restored = await NexusKnowledge.import(bundle, { chat: makeFakeChat(), embedder: fakeEmbedder });
  assert.equal(restored.index.size, kb.index.size);
  assert.equal(restored.docs.get('a').title, 'Handbook');
  const hits = await restored.retrieve('refunds', 1);
  assert.match(hits[0].text, /refunds/);
});

test('export can omit document text (index-only bundle)', async () => {
  const { kb } = await makeKB();
  await kb.addDocument({ id: 'a', text: 'secret internal text' });
  const bundle = await kb.export({ includeText: false });
  assert.equal(bundle.docs[0].text, undefined);
  assert.ok(bundle.index.chunks.length > 0, 'vectors still travel');
});

test('hooks fire for indexing and answering', async () => {
  const { kb } = await makeKB();
  const events = [];
  kb.on('indexed', (id, n) => events.push(['indexed', id, n]));
  kb.on('retrieved', (chunks) => events.push(['retrieved', chunks.length]));
  kb.on('answer', (a) => events.push(['answer', a]));
  await kb.addDocument({ id: 'a', text: 'hello world' });
  await kb.ask('hi');
  assert.deepEqual(events.map((e) => e[0]), ['indexed', 'retrieved', 'answer']);
});

test('metrics track indexing and questions', async () => {
  const { kb } = await makeKB();
  await kb.addDocument({ id: 'a', text: 'some text here' });
  await kb.ask('q');
  const s = kb.metrics.summary();
  assert.ok(s.chunks_indexed >= 1);
  assert.equal(s.questions, 1);
});

test('MemoryIndex.all exposes chunks in insertion order', () => {
  const idx = new MemoryIndex();
  idx.add({ id: 'x', text: 'a', vector: Float32Array.from([1]) });
  idx.add({ id: 'y', text: 'b', vector: Float32Array.from([1]) });
  assert.deepEqual(idx.all().map((c) => c.id), ['x', 'y']);
});
