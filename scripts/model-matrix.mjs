#!/usr/bin/env node
// Which models and quantizations actually call tools?
//
// The unit tests prove the loop is correct GIVEN a model that emits a call.
// They cannot tell you whether a real 0.5B on a given dtype decides to emit
// one — and that is exactly what bit us: schemas present in the prompt, one
// round, zero calls, a degenerate repetition returned as the answer.
//
// This runs the REAL library against REAL weights and reports, per
// model x dtype x question, whether a tool call happened and whether the final
// answer contains the value only the tool could know.
//
// Not part of `npm test` — it downloads hundreds of MB and takes minutes.
//   npm run test:models                     # default matrix
//   npm run test:models -- --dtypes q4,fp16 # narrow it
//   npm run test:models -- --models onnx-community/Qwen2.5-0.5B-Instruct
import * as transformers from '@huggingface/transformers';
import { NexusChat } from '../dist/index.js';

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1].split(',') : dflt;
};

const MODELS = arg('models', [
  'onnx-community/Qwen2.5-0.5B-Instruct',
  'onnx-community/Qwen3-0.6B-ONNX',
  'HuggingFaceTB/SmolLM2-360M-Instruct',
]);
const DTYPES = arg('dtypes', ['q4', 'q8', 'fp16']);

// Deliberately unguessable answers: a correct final answer PROVES a real call.
const TOOLS = [
  ['get_weather', 'Get the current weather for a city. Use this for any weather question.', { city: 'string' },
    async ({ city }) => ({ city, conditions: '31C, humid, light haze', stationId: 'WX-4417' })],
  ['get_time', 'Get the current local time in a city. Use this for any question about the time.', { city: 'string' },
    async ({ city }) => ({ city, time: '19:47', clockId: 'TZ-9082' })],
  ['multiply', 'Multiply two numbers exactly. Use this for arithmetic — never calculate it yourself.',
    { a: 'number', b: 'number' }, async ({ a, b }) => ({ product: Number(a) * Number(b) })],
];

const CASES = [
  // Proof tokens are unguessable by construction: only a real call reveals them.
  { q: "What's the weather in Chennai? Include the station id.", tool: 'get_weather', proof: /WX-?4417/i },
  { q: 'What is 4831 multiplied by 227?', tool: 'multiply', proof: /1[,.]?096[,.]?637/ },
  { q: 'What time is it in Tokyo right now? Include the clock id.', tool: 'get_time', proof: /TZ-?9082/i },
];

const rows = [];
for (const model of MODELS) {
  for (const dtype of DTYPES) {
    let chat;
    const t0 = Date.now();
    try {
      chat = await NexusChat.load({ hub: model }, { transformers, device: 'cpu', dtype });
    } catch (e) {
      rows.push({ model, dtype, load: 'FAILED', note: e.message.slice(0, 70) });
      console.error(`✗ ${model} ${dtype}: ${e.message.slice(0, 90)}`);
      continue;
    }
    const loadS = ((Date.now() - t0) / 1000).toFixed(0);

    for (const c of CASES) {
      chat.reset();
      for (const t of TOOLS) chat.tool(...t);
      let called = null;
      let rawFirst = '';
      chat.on('toolCall', (call) => { called ??= call.name; });
      chat.on('raw', (text, calls, round) => { if (round === 0) rawFirst = text; });

      let answer = '';
      try {
        answer = await chat.chat(c.q);
      } catch (e) {
        answer = `ERROR: ${e.message}`;
      }
      const grounded = c.proof.test(answer);
      const forced = chat.metrics.counters.get('tool_calls_forced') ?? 0;
      const forceFailed = chat.metrics.counters.get('tool_calls_force_failed') ?? 0;
      // A model stuck repeating itself is a distinct, recognisable failure.
      const degenerate = /(.{25,}?)\1{2,}/.test(rawFirst);
      rows.push({
        model, dtype, load: `${loadS}s`, q: c.q.slice(0, 24),
        want: c.tool, called: called ?? '—',
        right: called === c.tool, grounded, degenerate, forced, forceFailed,
        answer: answer.replace(/\s+/g, ' ').slice(0, 70),
      });
      console.error(
        `${called === c.tool && grounded ? '✓' : '✗'} ${model} ${dtype} ` +
        `| ${c.tool} -> ${called ?? 'none'} | grounded:${grounded}` +
        `${forced ? ` | forced:${forced}` : ''}${forceFailed ? ` FORCE-FAILED:${forceFailed}` : ''}` +
        `${degenerate ? ' | DEGENERATE' : ''}`,
      );
    }
    await chat.dispose().catch(() => {});
  }
}

// ── Report ─────────────────────────────────────────────────────────────────
const pad = (s, n) => String(s).padEnd(n).slice(0, n);
console.log('\n' + pad('model', 38) + pad('dtype', 6) + pad('calls', 7) + pad('grounded', 10) + 'verdict');
console.log('-'.repeat(78));
const byCombo = new Map();
for (const r of rows) {
  if (r.load === 'FAILED') continue;
  const k = `${r.model}|${r.dtype}`;
  const v = byCombo.get(k) ?? { ok: 0, n: 0, deg: 0 };
  v.n++; if (r.right && r.grounded) v.ok++; if (r.degenerate) v.deg++;
  byCombo.set(k, v);
}
for (const [k, v] of byCombo) {
  const [model, dtype] = k.split('|');
  const verdict = v.ok === v.n ? 'USABLE' : v.ok === 0 ? 'UNUSABLE' : 'FLAKY';
  console.log(pad(model, 38) + pad(dtype, 6) + pad(`${v.ok}/${v.n}`, 7) + pad(`${v.ok}/${v.n}`, 10) +
    verdict + (v.deg ? `  (${v.deg} degenerate)` : ''));
}

// ── The verdict: one answer per model ──────────────────────────────────────
// The point of running this is to be told what to use, not to read a grid.
// The best quantization is model-specific — Qwen2.5 is reliable at q4 and
// useless at q8; Qwen3 is the exact reverse — so there is nothing to infer
// from another model's result.
console.log('\nVERDICT');
console.log('-'.repeat(78));
let anyUsable = false;
for (const model of MODELS) {
  const combos = [...byCombo.entries()]
    .filter(([k]) => k.startsWith(model + '|'))
    .map(([k, v]) => ({ dtype: k.split('|')[1], ...v }))
    .sort((a, b) => b.ok - a.ok || a.deg - b.deg);
  const best = combos[0];
  if (!best) { console.log(`${model}\n  no result — every dtype failed to load`); continue; }
  if (best.ok === best.n) {
    anyUsable = true;
    console.log(`${model}\n  USE dtype '${best.dtype}' — ${best.ok}/${best.n} tool calls correct`);
    console.log(`  await NexusChat.load({ hub: '${model}' }, { dtype: '${best.dtype}' });`);
  } else if (best.ok > 0) {
    console.log(`${model}\n  BEST is '${best.dtype}' at ${best.ok}/${best.n} — flaky, not safe to ship`);
  } else {
    console.log(`${model}\n  NOT USABLE for tool calling at any dtype tried (${combos.map((c) => c.dtype).join(', ')})`);
    console.log('  Models below ~0.5B generally cannot pick a tool name from a list.');
  }
}

console.log('\nFull rows:');
console.log(JSON.stringify(rows, null, 1));

// Exit non-zero when nothing is shippable, so this can gate CI.
if (!anyUsable) process.exitCode = 1;
