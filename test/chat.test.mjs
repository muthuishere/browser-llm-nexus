// The tool loop, end to end, against a mock LLM — no weights, no network.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NexusChat } from '../dist/chat.js';
import { mockLLM, dialect } from './helpers/mock-llm.mjs';

/** Load a NexusChat backed by the mock runtime. */
async function chatWith(opts = {}) {
  const llm = mockLLM(opts);
  const chat = await NexusChat.load(
    { base: 'https://host/models/', id: 'stub/model' },
    { transformers: llm.transformers, device: 'wasm', dtype: 'q4' },
  );
  return { chat, llm };
}

const weather = (chat, spy = {}) =>
  chat.tool('get_weather', 'Weather for a city', { city: 'string' }, async ({ city }) => {
    spy.city = city;
    return { city, conditions: '31C, humid' };
  });

// ── The loop ────────────────────────────────────────────────────────────────

test('calls the handler, feeds the result back, answers from it', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C and humid in Chennai.'],
  });
  const spy = {};
  weather(chat, spy);

  const answer = await chat.chat("What's the weather in Chennai?");

  assert.equal(spy.city, 'Chennai');
  assert.equal(answer, 'It is 31C and humid in Chennai.');
  assert.equal(llm.rendered.length, 2, 'one round to call, one to answer');
  assert.equal(llm.rendered[0].tools.length, 1, 'schemas reach the template');
  assert.match(llm.toolMsgsAt(1)[0].content, /31C, humid/);
});

test('the rendered prompt actually contains the tool schema and the result', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'done'],
  });
  weather(chat);
  await chat.chat('weather?');

  assert.match(llm.generated[0].prompt, /<tools>[\s\S]*get_weather[\s\S]*<\/tools>/);
  assert.match(llm.generated[1].prompt, /<tool_response>[\s\S]*31C, humid[\s\S]*<\/tool_response>/);
});

test('two tools in one turn both run, in order', async () => {
  const { chat, llm } = await chatWith({
    script: [
      dialect.qwen('get_weather', { city: 'Chennai' }) + '\n' + dialect.qwen('get_time', { city: 'Tokyo' }),
      'Both done.',
    ],
  });
  const order = [];
  weather(chat);
  chat.tool('get_time', 'Time in a city', { city: 'string' }, async ({ city }) => {
    order.push('time');
    return { city, time: '19:00' };
  });
  chat.on('toolCall', (c) => order.push(c.name === 'get_weather' ? 'weather' : 'time-seen'));

  await chat.chat('weather and time?');

  assert.equal(chat.metrics.counters.get('tool_calls'), 2);
  assert.equal(llm.toolMsgsAt(1).length, 2, 'both results fed back before answering');
  assert.deepEqual(llm.toolMsgsAt(1).map((m) => m.name), ['get_weather', 'get_time']);
});

test('chained rounds: a second call after seeing the first result', async () => {
  const { chat, llm } = await chatWith({
    script: [
      dialect.qwen('get_weather', { city: 'Chennai' }),
      dialect.qwen('get_time', { city: 'Chennai' }),
      'It is 31C at 19:00.',
    ],
  });
  weather(chat);
  chat.tool('get_time', 'Time', { city: 'string' }, async () => ({ time: '19:00' }));

  const answer = await chat.chat('weather and time?');

  assert.equal(answer, 'It is 31C at 19:00.');
  assert.equal(llm.rounds, 3);
  assert.equal(llm.toolMsgsAt(2).length, 2, 'round 3 sees both results');
});

test('maxRounds is a real bound, and it is reported', async () => {
  const { chat, llm } = await chatWith({
    script: [() => dialect.qwen('get_weather', { city: 'Chennai' })], // never stops calling
  });
  weather(chat);
  chat.maxRounds = 3;

  const answer = await chat.chat('weather?');

  assert.match(answer, /exceeded 3 rounds/);
  assert.equal(llm.rounds, 3);
});

// ── Every dialect a real model emits ────────────────────────────────────────

for (const family of ['qwen', 'hermes', 'mistral', 'llama', 'fenced', 'openai', 'thinking']) {
  test(`${family}-style tool calls drive the loop`, async () => {
    const { chat } = await chatWith({
      family,
      script: [dialect[family]('get_weather', { city: 'Chennai' }), 'It is 31C.'],
    });
    const spy = {};
    weather(chat, spy);

    const answer = await chat.chat('weather?');

    assert.equal(spy.city, 'Chennai', `${family} args did not reach the handler`);
    assert.equal(answer, 'It is 31C.');
  });
}

test('reasoning traces are stripped from the final answer', async () => {
  const { chat } = await chatWith({ script: ['<think>hmm, no tool needed</think>\nJust 42.'] });
  assert.equal(await chat.chat('what is 42?'), 'Just 42.');
});

test('enable_thinking:false is passed to the template', async () => {
  const { chat, llm } = await chatWith({ script: ['hi'] });
  await chat.chat('hello');
  assert.equal(llm.rendered[0].opts.enable_thinking, false);
  assert.equal(llm.rendered[0].opts.add_generation_prompt, true);
});

// ── The two-phase system prompt (regression) ────────────────────────────────

// The instruction that makes a small model CALL a tool ("you MUST call the
// tool instead of guessing") reads, once results are in context, as "the tool
// has not been called yet" — so the model apologises for a failure that never
// happened instead of reading the result above it. Measured on Qwen2.5-0.5B:
// "It looks like there was an error while fetching the weather information."
test('the system prompt switches from call-phase to answer-phase', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);

  await chat.chat('weather?');

  assert.equal(llm.systemAt(0), chat.systemPrompt, 'round 1 steers toward calling');
  assert.equal(llm.systemAt(1), chat.answerPrompt, 'round 2 steers toward answering');
  assert.notEqual(chat.systemPrompt, chat.answerPrompt);
});

test('a caller-written system message is never rewritten', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'done'],
  });
  weather(chat);
  chat.messages.push({ role: 'system', content: 'You are a pirate.' });

  await chat.chat('weather?');

  for (let i = 0; i < llm.rendered.length; i++) assert.equal(llm.systemAt(i), 'You are a pirate.');
});

test('a customized systemPrompt is still swapped for the answer phase', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'done'],
  });
  weather(chat);
  chat.systemPrompt = 'Always use a tool.';
  chat.answerPrompt = 'Now report the numbers.';

  await chat.chat('weather?');

  assert.equal(llm.systemAt(0), 'Always use a tool.');
  assert.equal(llm.systemAt(1), 'Now report the numbers.');
});

test('no tools: no system prompt injected, one round, no schemas', async () => {
  const { chat, llm } = await chatWith({ script: ['Just an answer.'] });
  const answer = await chat.chat('hello');
  assert.equal(answer, 'Just an answer.');
  assert.equal(llm.rounds, 1);
  assert.equal(llm.systemAt(0), undefined);
  assert.equal(llm.rendered[0].tools, undefined);
});

// ── Failure paths ───────────────────────────────────────────────────────────

test('a throwing handler is reported to the model, not thrown at the caller', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'X' }), 'That lookup failed.'],
  });
  chat.tool('get_weather', 'Weather', { city: 'string' }, async () => {
    throw new Error('upstream down');
  });

  const answer = await chat.chat('weather?');

  assert.equal(answer, 'That lookup failed.');
  assert.match(llm.toolMsgsAt(1)[0].content, /upstream down/);
  assert.equal(chat.metrics.counters.get('tool_calls_failed'), 1);
  assert.equal(chat.metrics.counters.get('tool_calls_ok'), undefined);
});

test('unknown tool names are not dispatched', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('nope', {})] });
  weather(chat);
  const answer = await chat.chat('weather?');
  assert.match(answer, /nope/, 'unmatched call text becomes the answer');
  assert.equal(chat.metrics.counters.get('tool_calls'), undefined);
});

test('a call with no arguments still dispatches', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('ping', {}), 'pong received'] });
  let ran = false;
  chat.tool('ping', 'Ping', {}, async () => { ran = true; return { ok: true }; });
  await chat.chat('ping?');
  assert.equal(ran, true);
});

// ── Schemas ─────────────────────────────────────────────────────────────────

test('shorthand expands, and everything is required by default', async () => {
  const { chat } = await chatWith({ script: ['x'] });
  chat.tool('t', 'desc', { city: 'string', n: 'number' }, async () => 1);
  const p = chat.toolSchemas[0].function.parameters;
  assert.deepEqual(p.properties, { city: { type: 'string' }, n: { type: 'number' } });
  assert.deepEqual(p.required, ['city', 'n']);
});

test('explicit required and full property objects are preserved', async () => {
  const { chat } = await chatWith({ script: ['x'] });
  chat.tool('t', 'desc', { q: 'string', limit: { type: 'number', description: 'max' } },
    async () => 1, { required: ['q'] });
  const p = chat.toolSchemas[0].function.parameters;
  assert.deepEqual(p.required, ['q']);
  assert.equal(p.properties.limit.description, 'max');
});

test('tool() chains and re-registering a name replaces it', async () => {
  const { chat } = await chatWith({ script: ['x'] });
  const same = chat.tool('a', 'A', {}, async () => 1).tool('b', 'B', {}, async () => 2);
  assert.equal(same, chat);
  chat.tool('a', 'A2', {}, async () => 3);
  assert.equal(chat.toolSchemas.length, 2);
  assert.equal(chat.toolSchemas.find((s) => s.function.name === 'a').function.description, 'A2');
});

// ── evalTools: the single editable tools file ───────────────────────────────

test('evalTools registers a whole file of tools and returns their names', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('add', { a: 2, b: 3 }), 'It is 5.'] });
  const names = await chat.evalTools(`
    tool('add', 'Add two numbers', { a: 'number', b: 'number' }, async ({ a, b }) => ({ sum: a + b }));
    tool('now', 'Current time', {}, async () => ({ t: 'noon' }));
  `);
  assert.deepEqual(names, ['add', 'now']);
  assert.equal(await chat.chat('2+3?'), 'It is 5.');
});

test('evalTools replaces the previous set', async () => {
  const { chat } = await chatWith({ script: ['x'] });
  weather(chat);
  const names = await chat.evalTools("tool('only', 'One', {}, async () => 1);");
  assert.deepEqual(names, ['only']);
  assert.equal(chat.toolSchemas.length, 1);
});

test('evalTools surfaces a syntax error instead of registering junk', async () => {
  const { chat } = await chatWith({ script: ['x'] });
  await assert.rejects(() => chat.evalTools('tool(((('), (e) => e instanceof SyntaxError || /Unexpected/.test(e.message));
  assert.equal(chat.toolSchemas.length, 0);
});

test('evalTools handlers can close over async work', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('slow', {}), 'ok'] });
  await chat.evalTools(`
    tool('slow', 'Slow', {}, async () => { await new Promise(r => setTimeout(r, 5)); return { done: true }; });
  `);
  let result;
  chat.on('toolCall', (_c, r) => { result = r; });
  await chat.chat('go');
  assert.deepEqual(result, { done: true });
});

// ── Hooks, memory, metrics, lifecycle ───────────────────────────────────────

test('token and round hooks fire; unsubscribing stops them', async () => {
  const { chat } = await chatWith({ script: ['hello there'] });
  const tokens = [];
  const rounds = [];
  const off = chat.on('token', (t) => tokens.push(t));
  chat.on('round', (r) => rounds.push(r));

  await chat.chat('hi');
  assert.equal(tokens.join(''), 'hello there');
  assert.deepEqual(rounds, [0]);

  off();
  tokens.length = 0;
  await chat.chat('again');
  assert.equal(tokens.length, 0);
});

test('the answer hook carries the final text', async () => {
  const { chat } = await chatWith({ script: ['final'] });
  let seen;
  chat.on('answer', (a) => { seen = a; });
  await chat.chat('q');
  assert.equal(seen, 'final');
});

test('conversation memory accumulates across turns, reset clears it', async () => {
  const { chat, llm } = await chatWith({ script: ['one', 'two'] });
  await chat.chat('first');
  await chat.chat('second');

  const second = llm.rendered[1].messages;
  assert.deepEqual(second.map((m) => m.role), ['user', 'assistant', 'user']);
  assert.equal(second[0].content, 'first');

  chat.reset();
  assert.deepEqual(chat.messages, []);
});

test('metrics record load, generation and tokens', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('get_weather', { city: 'X' }), 'ok'] });
  weather(chat);
  await chat.chat('weather?');
  const s = chat.metrics.summary();
  assert.ok(s.load_ms_avg >= 0, 'load timed');
  assert.equal(chat.metrics.counters.get('chats'), 1);
  assert.equal(chat.metrics.counters.get('tool_calls_ok'), 1);
  assert.ok(chat.metrics.counters.get('tokens_out') > 0);
});

test('load passes device and dtype through to the pipeline', async () => {
  const llm = mockLLM({ script: ['x'] });
  const chat = await NexusChat.load(
    { base: 'https://host/models/', id: 'stub/model' },
    { transformers: llm.transformers, device: 'webgpu', dtype: 'fp16' },
  );
  assert.equal(llm.transformers.lastPipeline.task, 'text-generation');
  assert.equal(llm.transformers.lastPipeline.opts.device, 'webgpu');
  assert.equal(llm.transformers.lastPipeline.opts.dtype, 'fp16');
  assert.equal(chat.device, 'webgpu');
  assert.equal(chat.dtype, 'fp16');
  assert.equal(chat.modelId, 'stub/model');
});

test('maxNewTokens is forwarded, with a default', async () => {
  const { chat, llm } = await chatWith({ script: ['a', 'b'] });
  await chat.chat('q');
  assert.equal(llm.generated[0].opts.max_new_tokens, 256);
  await chat.chat('q', { maxNewTokens: 32 });
  assert.equal(llm.generated[1].opts.max_new_tokens, 32);
});

test('works without a TextStreamer (lite transformers builds)', async () => {
  const { chat } = await chatWith({ script: ['no streamer here'], streamer: false });
  const tokens = [];
  chat.on('token', (t) => tokens.push(t));
  assert.equal(await chat.chat('hi'), 'no streamer here');
  assert.equal(tokens.length, 0, 'no tokens streamed, but the answer still lands');
});

test('dispose releases the pipeline', async () => {
  const { chat, llm } = await chatWith({ script: ['x'] });
  await chat.dispose();
  assert.equal(llm.generator.disposed, true);
});

test('a mistral-templated model round-trips through the loop', async () => {
  const { chat, llm } = await chatWith({
    family: 'mistral',
    template: 'mistral',
    script: [dialect.mistral('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);
  assert.equal(await chat.chat('weather?'), 'It is 31C.');
  assert.match(llm.generated[0].prompt, /\[AVAILABLE_TOOLS\][\s\S]*get_weather/);
  assert.match(llm.generated[1].prompt, /\[TOOL_RESULTS\][\s\S]*31C/);
});

// ── Observability: why did the loop do that? ────────────────────────────────

test('prompt and raw hooks expose every round', async () => {
  const { chat } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);
  const prompts = [];
  const raws = [];
  chat.on('prompt', (p, round) => prompts.push({ round, hasTools: /get_weather/.test(p) }));
  chat.on('raw', (text, calls, round) => raws.push({ round, calls: calls.length, text }));

  await chat.chat('weather?');

  assert.deepEqual(prompts.map((p) => p.round), [0, 1]);
  assert.equal(prompts[0].hasTools, true, 'the schema really is in the prompt');
  assert.deepEqual(raws.map((r) => r.round), [0, 1]);
  assert.equal(raws[0].calls, 1, 'round 0 parsed a call');
  assert.equal(raws[1].calls, 0, 'round 1 was the answer');
  assert.match(raws[0].text, /<tool_call>/, 'raw is pre-parse text');
});

// This is the failure seen on the live demo: tools registered, one generation,
// zero tool calls, and a degenerate repetition returned as the answer. Without
// the raw hook it is indistinguishable from a parser bug.
test('a model that answers instead of calling is visible, not silent', async () => {
  const degenerate = 'The weather in Chennai is described as follows: '.repeat(6);
  const { chat, llm } = await chatWith({ script: [degenerate] });
  weather(chat);
  const seen = [];
  chat.on('raw', (text, calls, round) => seen.push({ round, calls: calls.length, text }));

  const answer = await chat.chat("What's the weather in Chennai?");

  assert.equal(llm.rounds, 1, 'one generation, exactly what the metrics showed');
  assert.equal(seen.length, 1);
  assert.equal(seen[0].calls, 0, 'nothing parsed — the model never called');
  assert.equal(chat.metrics.counters.get('tool_calls'), undefined);
  assert.equal(answer, degenerate.trim(), 'the raw text became the answer');
});

test('a call to an unregistered tool is distinguishable from no call', async () => {
  const { chat } = await chatWith({ script: [dialect.qwen('get_stock', { ticker: 'X' })] });
  weather(chat);
  let parsedCount = null;
  chat.on('raw', (_t, calls) => { parsedCount = calls.length; });

  await chat.chat('stock?');

  assert.equal(parsedCount, 1, 'the parser DID find a call — it just was not registered');
  assert.equal(chat.metrics.counters.get('tool_calls'), undefined, 'and it was not dispatched');
});

test('the prompt hook shows tool results arriving in round 2', async () => {
  const { chat } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);
  const prompts = [];
  chat.on('prompt', (p) => prompts.push(p));

  await chat.chat('weather?');

  assert.doesNotMatch(prompts[0], /tool_response/, 'round 1 has no results yet');
  assert.match(prompts[1], /<tool_response>[\s\S]*31C, humid/, 'round 2 carries the result');
});
