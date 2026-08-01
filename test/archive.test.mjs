// Artifact export/import: RAG store, model archives, and the knowledge
// composition of both. No network, no real weights.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fflate from 'fflate';
import {
  MemoryIndex,
  indexToFiles,
  indexFromFiles,
  exportIndex,
  importIndex,
} from '../dist/rag.js';
import { exportModel, importModel, inspectModel } from '../dist/model.js';
import { filesFromZip, prefixFiles, stripPrefix, readSource } from '../dist/archive.js';

const Z = { zip: fflate };

function makeIndex() {
  const idx = new MemoryIndex();
  idx.add({ id: 'a#0', text: 'refunds within thirty days', vector: Float32Array.from([0.5, 0.5, 0.5, 0.5]), meta: { docId: 'a' } });
  idx.add({ id: 'a#1', text: 'shipping takes two days', vector: Float32Array.from([0.1, 0.2, 0.3, 0.4]), meta: { docId: 'a' } });
  return idx;
}

// ---------- RAG store ----------

test('rag: file layout is manifest + chunks + binary vectors', () => {
  const files = indexToFiles(makeIndex());
  assert.deepEqual([...files.keys()].sort(), ['chunks.json', 'manifest.json', 'vectors.bin']);

  const manifest = JSON.parse(new TextDecoder().decode(files.get('manifest.json')));
  assert.equal(manifest.kind, 'rag');
  assert.equal(manifest.count, 2);
  assert.equal(manifest.dims, 4);

  assert.equal(files.get('vectors.bin').byteLength, 2 * 4 * 4, '2 chunks x 4 dims x 4 bytes');
  const chunks = JSON.parse(new TextDecoder().decode(files.get('chunks.json')));
  assert.equal(chunks[0].vector, undefined, 'vectors must not be duplicated as JSON');
});

test('rag: file map round-trips', () => {
  const restored = indexFromFiles(indexToFiles(makeIndex()));
  assert.equal(restored.size, 2);
  assert.deepEqual([...restored.all()[0].vector], [0.5, 0.5, 0.5, 0.5]);
  assert.equal(restored.all()[1].meta.docId, 'a');
});

test('rag: zip round-trips and stays searchable', async () => {
  const zipped = await exportIndex(makeIndex(), Z);
  assert.equal(zipped[0], 0x50, 'PK zip header');
  const restored = await importIndex(zipped, Z);
  const hit = restored.search(Float32Array.from([0.5, 0.5, 0.5, 0.5]), 1)[0];
  assert.equal(hit.chunk.id, 'a#0');
});

test('rag: a truncated archive fails loudly', async () => {
  const files = indexToFiles(makeIndex());
  files.delete('vectors.bin');
  assert.throws(() => indexFromFiles(files), /missing vectors\.bin/);
});

// ---------- model archives ----------

/** Serve a fake converted model over a stubbed fetch. */
function stubModelServer(present) {
  const original = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const path = String(url).split('/models/fake%2Fmodel/')[1] ?? String(url).split('/models/fake/model/')[1];
    if (path && present[path] !== undefined) {
      return new Response(new TextEncoder().encode(present[path]), { status: 200 });
    }
    return new Response(null, { status: 404 });
  };
  return () => { globalThis.fetch = original; };
}

/** Minimal Cache API stub so importModel can be tested off-browser. */
function stubCaches() {
  const store = new Map();
  const original = globalThis.caches;
  globalThis.caches = {
    async open() {
      return {
        async put(url, res) { store.set(url, new Uint8Array(await res.arrayBuffer())); },
        async match(url) { return store.has(url) ? new Response(store.get(url)) : undefined; },
        async keys() { return [...store.keys()].map((u) => ({ url: u })); },
      };
    },
  };
  return { store, restore: () => { globalThis.caches = original; } };
}

test('model: exports only the files that exist', async () => {
  const restore = stubModelServer({
    'config.json': '{"model_type":"qwen3"}',
    'tokenizer.json': '{"v":1}',
    'onnx/model_q4.onnx': 'Q4WEIGHTS',
  });
  try {
    const zip = await exportModel('fake/model', { modelsUrl: 'https://host/models/', ...Z });
    const manifest = await inspectModel(zip, Z);
    assert.equal(manifest.kind, 'model');
    assert.equal(manifest.modelId, 'fake/model');
    assert.deepEqual(manifest.dtypes, ['q4']);
    assert.deepEqual(manifest.files.map((f) => f.path), ['config.json', 'tokenizer.json', 'onnx/model_q4.onnx']);
  } finally {
    restore();
  }
});

test('model: dtype filter keeps the archive small', async () => {
  const restore = stubModelServer({
    'config.json': '{}',
    'onnx/model_q4.onnx': 'Q4',
    'onnx/model_fp16.onnx': 'FP16',
  });
  try {
    const zip = await exportModel('fake/model', { modelsUrl: 'https://host/models/', dtypes: ['q4'], ...Z });
    const manifest = await inspectModel(zip, Z);
    assert.deepEqual(manifest.dtypes, ['q4']);
    assert.ok(!manifest.files.some((f) => f.path.includes('fp16')));
  } finally {
    restore();
  }
});

test('model: export fails clearly when nothing is served', async () => {
  const restore = stubModelServer({});
  try {
    await assert.rejects(
      () => exportModel('fake/model', { modelsUrl: 'https://host/models/', ...Z }),
      /no model files found/,
    );
  } finally {
    restore();
  }
});

test('model: import restores every file into the cache under its URL', async () => {
  const restoreFetch = stubModelServer({ 'config.json': '{"a":1}', 'onnx/model_q4.onnx': 'W' });
  const { store, restore: restoreCaches } = stubCaches();
  try {
    const zip = await exportModel('fake/model', { modelsUrl: 'https://host/models/', ...Z });
    const manifest = await importModel(zip, Z);
    assert.equal(manifest.modelId, 'fake/model');
    assert.ok(store.has('https://host/models/fake/model/config.json'));
    assert.equal(new TextDecoder().decode(store.get('https://host/models/fake/model/config.json')), '{"a":1}');
  } finally {
    restoreFetch();
    restoreCaches();
  }
});

test('model: import can re-point URLs to a new host', async () => {
  const restoreFetch = stubModelServer({ 'config.json': '{}' });
  const { store, restore: restoreCaches } = stubCaches();
  try {
    const zip = await exportModel('fake/model', { modelsUrl: 'https://host/models/', ...Z });
    await importModel(zip, { ...Z, rewriteUrl: (_u, path, id) => `https://elsewhere/${id}/${path}` });
    assert.ok(store.has('https://elsewhere/fake/model/config.json'));
  } finally {
    restoreFetch();
    restoreCaches();
  }
});

test('model: rejects a non-model archive', async () => {
  const ragZip = await exportIndex(makeIndex(), Z);
  await assert.rejects(() => importModel(ragZip, Z), /not a model archive/);
});

// ---------- composition helpers ----------

test('prefix/stripPrefix compose and decompose artifacts', () => {
  const files = indexToFiles(makeIndex());
  const nested = prefixFiles(files, 'rag/');
  assert.ok([...nested.keys()].every((k) => k.startsWith('rag/')));
  const back = stripPrefix(nested, 'rag/');
  assert.deepEqual([...back.keys()].sort(), [...files.keys()].sort());
});

test('readSource accepts bytes, ArrayBuffer and Blob', async () => {
  const bytes = new TextEncoder().encode('hello');
  assert.equal(new TextDecoder().decode(await readSource(bytes)), 'hello');
  assert.equal(new TextDecoder().decode(await readSource(bytes.buffer)), 'hello');
  assert.equal(new TextDecoder().decode(await readSource(new Blob([bytes]))), 'hello');
});

test('readSource fetches a URL', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(new TextEncoder().encode('remote'));
  try {
    assert.equal(new TextDecoder().decode(await readSource('https://host/x.zip')), 'remote');
  } finally {
    globalThis.fetch = original;
  }
});

test('zip entries survive a real unzip', async () => {
  const zipped = await exportIndex(makeIndex(), Z);
  const files = await filesFromZip(zipped, Z);
  assert.ok(files.has('manifest.json') && files.has('vectors.bin'));
});
