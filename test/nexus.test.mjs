// Tests run against the compiled dist/ (npm test builds first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseToolCalls, stripThinking } from '../dist/toolcalls.js';
import { MemoryIndex, chunkText } from '../dist/rag.js';
import { similarity } from '../dist/embed.js';
import { Metrics } from '../dist/metrics.js';
import { Hooks } from '../dist/hooks.js';
import { toManifest, fromManifest } from '../dist/bundle.js';

const CALL = { name: 'get_weather', arguments: { city: 'Chennai' } };

test('parseToolCalls: qwen/hermes format', () => {
  assert.deepEqual(parseToolCalls('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>'), [CALL]);
});

test('parseToolCalls: unclosed tag (generation cut off)', () => {
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "get_weather", "arguments": {"city": "Chennai"}}'), [CALL]);
});

test('parseToolCalls: mistral format', () => {
  assert.deepEqual(parseToolCalls('[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Chennai"}}]'), [CALL]);
});

test('parseToolCalls: llama bare JSON with parameters', () => {
  assert.deepEqual(parseToolCalls('{"name": "get_weather", "parameters": {"city": "Chennai"}}'), [CALL]);
});

test('parseToolCalls: fenced JSON with string args', () => {
  assert.deepEqual(parseToolCalls('```json\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Chennai\\"}"}\n```'), [CALL]);
});

test('parseToolCalls: openai nested function object', () => {
  assert.deepEqual(parseToolCalls('{"function": {"name": "get_weather", "arguments": {"city": "Chennai"}}}'), [CALL]);
});

test('parseToolCalls: prose yields nothing', () => {
  assert.deepEqual(parseToolCalls('The capital of France is Paris.'), []);
});

test('parseToolCalls: multiple calls', () => {
  const out = parseToolCalls('<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>');
  assert.equal(out.length, 2);
});

test('stripThinking removes think traces', () => {
  assert.equal(stripThinking('<think>hmm reasoning</think>Paris.'), 'Paris.');
  assert.equal(stripThinking('<think>unclosed trace'), '');
});

test('MemoryIndex: search ranks by cosine', () => {
  const idx = new MemoryIndex();
  idx.add({ id: 'a', text: 'about cats', vector: Float32Array.from([1, 0]) });
  idx.add({ id: 'b', text: 'about dogs', vector: Float32Array.from([0, 1]) });
  const hits = idx.search(Float32Array.from([0.9, 0.1]), 2);
  assert.equal(hits[0].chunk.id, 'a');
  assert.ok(hits[0].score > hits[1].score);
});

test('MemoryIndex: serialize round-trip', () => {
  const idx = new MemoryIndex();
  idx.add({ id: 'a', text: 't', vector: Float32Array.from([0.5, 0.5]), meta: { src: 'x' } });
  const restored = MemoryIndex.restore(idx.serialize());
  assert.equal(restored.size, 1);
  assert.equal(restored.search(Float32Array.from([0.5, 0.5]), 1)[0].chunk.meta.src, 'x');
});

test('chunkText splits with overlap and no empties', () => {
  const chunks = chunkText('One sentence. '.repeat(100), 200, 20);
  assert.ok(chunks.length > 3);
  assert.ok(chunks.every((c) => c.length > 0 && c.length <= 220));
});

test('similarity is dot product', () => {
  assert.equal(similarity(Float32Array.from([1, 0]), Float32Array.from([1, 0])), 1);
  assert.equal(similarity(Float32Array.from([1, 0]), Float32Array.from([0, 1])), 0);
});

test('Metrics: counters, timings, tokens/sec', () => {
  const m = new Metrics();
  m.count('tokens_out', 100);
  m.time('generate', 2000);
  const s = m.summary();
  assert.equal(s.tokens_out, 100);
  assert.equal(s.generate_ms_total, 2000);
  assert.equal(s.tokens_per_second, 50);
});

test('Hooks: on/emit/off and unsubscribe', () => {
  const h = new Hooks();
  let n = 0;
  const un = h.on('x', (v) => (n += v));
  h.emit('x', 2);
  un();
  h.emit('x', 5);
  assert.equal(n, 2);
});

test('Hooks: a throwing listener does not break others', () => {
  const h = new Hooks();
  let ok = false;
  h.on('x', () => { throw new Error('boom'); });
  h.on('x', () => (ok = true));
  h.emit('x');
  assert.ok(ok);
});

test('bundle: manifest round-trip', () => {
  const buf = new TextEncoder().encode('weights').buffer;
  const { index, files } = toManifest([{ url: 'https://huggingface.co/x/resolve/main/config.json', data: buf }]);
  assert.equal(index[0].file, '0.bin');
  const back = fromManifest(index, files);
  assert.equal(back[0].url, 'https://huggingface.co/x/resolve/main/config.json');
  assert.equal(back[0].data.byteLength, buf.byteLength);
});
