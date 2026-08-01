// The source contract: where a model comes from is always explicit.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fflate from 'fflate';
import { resolveSource, describeSource, hubRoot } from '../dist/source.js';
import { exportModel } from '../dist/model.js';

const fakeEnv = () => ({ env: {} });

test('hub source enables remote and returns the repo id', async () => {
  const tjs = fakeEnv();
  const id = await resolveSource(tjs, { hub: 'onnx-community/Qwen3-0.6B-ONNX' });
  assert.equal(id, 'onnx-community/Qwen3-0.6B-ONNX');
  assert.equal(tjs.env.allowRemoteModels, true);
  assert.equal(tjs.env.allowLocalModels, false);
});

test('base+id source points at your server, remote disabled', async () => {
  const tjs = fakeEnv();
  const id = await resolveSource(tjs, { base: 'https://host/models/', id: 'Qwen/Qwen3-0.6B' });
  assert.equal(id, 'Qwen/Qwen3-0.6B');
  assert.equal(tjs.env.localModelPath, 'https://host/models/');
  assert.equal(tjs.env.allowRemoteModels, false);
  assert.equal(tjs.env.allowLocalModels, true);
});

test('base without a trailing slash still resolves correctly', async () => {
  const tjs = fakeEnv();
  await resolveSource(tjs, { base: 'https://host/models', id: 'x/y' });
  assert.equal(tjs.env.localModelPath, 'https://host/models/');
});

test('archive source derives the base from the URLs it recorded', async () => {
  // stub a served model, pack it, then resolve from the archive
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.endsWith('/config.json')) return new Response(new TextEncoder().encode('{}'), { status: 200 });
    if (u.endsWith('/onnx/model_q4.onnx')) return new Response(new TextEncoder().encode('W'), { status: 200 });
    return new Response(null, { status: 404 });
  };
  const store = new Map();
  const originalCaches = globalThis.caches;
  globalThis.caches = {
    async open() {
      return { async put(url, res) { store.set(url, new Uint8Array(await res.arrayBuffer())); } };
    },
  };

  try {
    const zip = await exportModel('acme/tiny', { modelsUrl: 'https://host/models/', zip: fflate });
    const tjs = fakeEnv();
    const id = await resolveSource(tjs, { archive: zip, zip: fflate });

    assert.equal(id, 'acme/tiny');
    assert.equal(tjs.env.localModelPath, 'https://host/models/', 'base is derived, not guessed');
    assert.equal(tjs.env.allowRemoteModels, false, 'an archive must not fall back to the network');
    assert.ok(store.has('https://host/models/acme/tiny/config.json'), 'files land in the cache');
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.caches = originalCaches;
  }
});

test('describeSource is readable for logs', () => {
  assert.equal(describeSource({ hub: 'a/b' }), 'hub:a/b');
  assert.equal(describeSource({ base: 'https://h/m', id: 'a/b' }), 'https://h/m/a/b');
  assert.equal(describeSource({ archive: 'https://h/m.zip' }), 'archive:https://h/m.zip');
  assert.equal(describeSource({ archive: new Uint8Array() }), 'archive:<file>');
});

test('hubRoot builds Hugging Face file URLs', () => {
  assert.equal(hubRoot('onnx-community/X'), 'https://huggingface.co/onnx-community/X/resolve/main/');
  assert.equal(hubRoot('a/b', 'dev'), 'https://huggingface.co/a/b/resolve/dev/');
});
