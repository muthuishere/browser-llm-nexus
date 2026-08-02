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
// The live-demo failure: the model narrates instead of calling. The library
// no longer accepts that — it regenerates with the call syntax already open,
// which leaves no continuation that isn't a call.
test('a model that answers instead of calling is forced to call', async () => {
  const degenerate = 'The weather in Chennai is described as follows: '.repeat(6);
  const { chat, llm } = await chatWith({
    // 1st: prose. 2nd (forced): completes the primed call. 3rd: the answer.
    script: [degenerate, 'get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>', 'It is 31C.'],
  });
  const spy = {};
  weather(chat, spy);

  const answer = await chat.chat("What's the weather in Chennai?");

  assert.equal(spy.city, 'Chennai', 'the tool ran despite the model declining');
  assert.equal(answer, 'It is 31C.');
  assert.equal(chat.metrics.counters.get('tool_calls_forced'), 1);
  assert.equal(chat.metrics.counters.get('tool_calls_ok'), 1);
  assert.equal(llm.rounds, 3, 'free turn, forced turn, answer turn');
});

test('the forced turn primes the prompt with the call syntax', async () => {
  const { chat, llm } = await chatWith({
    script: ['just prose', 'get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>', 'done'],
  });
  weather(chat);
  await chat.chat('weather?');

  // The system prompt contains a <tool_call> EXAMPLE, so presence proves
  // nothing — priming is about what the prompt ENDS with, i.e. what the model
  // has to continue from.
  assert.doesNotMatch(llm.generated[0].prompt, /\{"name": "$/, 'the free turn is not primed');
  assert.match(llm.generated[1].prompt, /<tool_call>\n\{"name": "$/, 'the forced turn is');
});

// A forced call is only worth keeping if it names a real tool. A hallucinated
// name is worse than the answer the model gave us unprompted.
test('a forced call naming an unknown tool is discarded, not dispatched', async () => {
  const prose = 'I cannot help with that.';
  const { chat } = await chatWith({
    script: [prose, 'not_a_real_tool", "arguments": {}}\n</tool_call>'],
  });
  weather(chat);

  const answer = await chat.chat('hello');

  assert.equal(answer, prose, 'the original answer survives');
  assert.equal(chat.metrics.counters.get('tool_calls_force_failed'), 1);
  assert.equal(chat.metrics.counters.get('tool_calls'), undefined, 'nothing dispatched');
});

test('forcing never happens once tool results exist', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);

  await chat.chat('weather?');

  assert.equal(chat.metrics.counters.get('tool_calls_forced'), undefined);
  assert.equal(llm.rounds, 2, 'no extra generation — the answer turn is allowed to be prose');
});

test('toolChoice:none leaves a declining model alone', async () => {
  const prose = 'I think it is warm.';
  const { chat, llm } = await chatWith({ script: [prose] });
  weather(chat);

  const answer = await chat.chat('weather?', { toolChoice: 'none' });

  assert.equal(answer, prose);
  assert.equal(llm.rounds, 1, 'no forced retry');
  assert.equal(chat.metrics.counters.get('tool_calls_forced'), undefined);
});

test('toolChoice:required skips the free turn entirely', async () => {
  const { chat, llm } = await chatWith({
    script: ['get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>', 'It is 31C.'],
  });
  const spy = {};
  weather(chat, spy);

  const answer = await chat.chat('weather?', { toolChoice: 'required' });

  assert.equal(spy.city, 'Chennai');
  assert.equal(answer, 'It is 31C.');
  assert.match(llm.generated[0].prompt, /<tool_call>\n\{"name": "$/, 'primed from round 0');
  assert.equal(llm.rounds, 2, 'no wasted free turn');
});

test('no tools registered: nothing is ever forced', async () => {
  const { chat, llm } = await chatWith({ script: ['just chatting'] });
  const answer = await chat.chat('hello');
  assert.equal(answer, 'just chatting');
  assert.equal(llm.rounds, 1);
  assert.equal(chat.metrics.counters.get('tool_calls_forced'), undefined);
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

test('the forced attempt is observable even when discarded', async () => {
  const { chat } = await chatWith({
    script: ['I think it is warm.', 'rain", "arguments": {"station_id": "CHM01"}}'],
  });
  weather(chat);
  const attempts = [];
  chat.on('forced', (text, salvaged, round) => attempts.push({ text, salvaged, round }));

  await chat.chat('weather?');

  assert.equal(attempts.length, 1, 'the forced turn is reported');
  assert.equal(attempts[0].salvaged, null, 'nothing trustworthy came back');
  assert.match(attempts[0].text, /rain/, 'the hallucinated name is visible');
});

// Safety: a model that names a tool it was not given must never be quietly
// mapped onto whatever tool happens to be registered. Calling the wrong tool
// is worse than not calling one.
test('a hallucinated name is never mapped onto the only registered tool', async () => {
  const { chat } = await chatWith({
    script: ['prose', 'launch_missiles", "arguments": {}}'],
  });
  let ran = false;
  chat.tool('delete_account', 'Delete the account', {}, async () => { ran = true; return {}; });

  await chat.chat('hello');

  assert.equal(ran, false, 'the single registered tool was NOT dispatched');
  assert.equal(chat.metrics.counters.get('tool_calls_force_failed'), 1);
});

// ── The answer phase must return prose, never call syntax ──────────────────
// Observed live on the deployed demo, WebGPU/q4: round 1 called get_weather
// correctly and the tool returned "31C, humid, light haze" — then round 2
// emitted a bare "<tool_call>" which parsed to nothing and was handed to the
// user AS the answer. The tool worked; the answer was garbage.

test('a dangling tool-call fragment is never returned as the answer', async () => {
  const { chat, llm } = await chatWith({
    script: [
      dialect.qwen('get_weather', { city: 'Chennai' }),
      '<tool_call>',                        // the live failure, exactly
      'It is 31C and humid in Chennai.',    // the retry, with tools removed
    ],
  });
  weather(chat);

  const answer = await chat.chat('weather?');

  assert.equal(answer, 'It is 31C and humid in Chennai.');
  assert.doesNotMatch(answer, /<tool_call>/);
  assert.equal(chat.metrics.counters.get('answer_retried_without_tools'), 1);
  // The retry must not show the model the schemas again, or it can loop.
  assert.equal(llm.rendered[2].tools, undefined, 'retry drops the tool schemas');
});

test('prose with a trailing call fragment keeps the prose', async () => {
  const { chat } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.\n<tool_call>\n{"name":'],
  });
  weather(chat);

  const answer = await chat.chat('weather?');

  assert.equal(answer, 'It is 31C.');
  assert.equal(chat.metrics.counters.get('answer_retried_without_tools'), undefined, 'no retry needed');
});

test('fragment stripping only applies after tool results exist', async () => {
  // With no tools registered there is no answer phase, so text is untouched.
  const { chat } = await chatWith({ script: ['Here is some ```json {"a":1}``` inline.'] });
  const answer = await chat.chat('hi');
  assert.match(answer, /```json/, 'a plain chat answer is not rewritten');
});

// ── selfCheck: can THIS model actually do it? ───────────────────────────────
// The published matrix cannot cover a model someone just uploaded from a zip
// or served from their own host, so the model gets asked directly.

const SENSOR = (args = { id: 'A9' }) => dialect.qwen('lookup_sensor', args);

test('selfCheck passes when the model calls and reports the token', async () => {
  const { chat } = await chatWith({ script: [SENSOR(), 'The reading is QX-7731.'] });
  const r = await chat.selfCheck();

  assert.equal(r.ok, true);
  assert.equal(r.called, true);
  assert.equal(r.grounded, true);
  assert.equal(r.needed_forcing, false);
  assert.match(r.detail, /calls tools correctly/);
});

test('selfCheck fails loudly when the model never calls', async () => {
  const { chat } = await chatWith({ script: ['The sensor probably reads about 40.', 'still no call'] });
  const r = await chat.selfCheck();

  assert.equal(r.ok, false);
  assert.equal(r.called, false);
  assert.match(r.detail, /does not call tools/);
  assert.match(r.detail, /below ~0\.5B/, 'says what to do about it');
});

// The dangerous case: it calls, then misreports the value. Answers look right.
test('selfCheck fails when the tool ran but the answer is not grounded', async () => {
  const { chat } = await chatWith({ script: [SENSOR(), 'The reading is 42.'] });
  const r = await chat.selfCheck();

  assert.equal(r.called, true);
  assert.equal(r.grounded, false);
  assert.equal(r.ok, false);
  assert.match(r.detail, /may look right and be wrong/);
});

test('selfCheck reports when the call only happened under forcing', async () => {
  const { chat } = await chatWith({
    script: ['prose, no call', 'lookup_sensor", "arguments": {"id": "A9"}}', 'The reading is QX-7731.'],
  });
  const r = await chat.selfCheck();

  assert.equal(r.ok, true);
  assert.equal(r.needed_forcing, true);
  assert.match(r.detail, /only when the call syntax is forced/);
});

test('selfCheck restores the caller tools and conversation', async () => {
  const { chat } = await chatWith({ script: ['hello there', SENSOR(), 'The reading is QX-7731.'] });
  weather(chat);
  // toolChoice:'none' so this setup turn cannot consume the probe script.
  await chat.chat('hello', { toolChoice: 'none' });
  const toolsBefore = chat.toolSchemas.map((s) => s.function.name);
  const messagesBefore = chat.messages.length;

  await chat.selfCheck();

  assert.deepEqual(chat.toolSchemas.map((s) => s.function.name), toolsBefore, 'tools restored');
  assert.equal(chat.messages.length, messagesBefore, 'history restored');
  assert.equal(chat.messages.some((m) => /sensor/i.test(m.content)), false, 'no probe left behind');
});

test('selfCheck carries the model, device and dtype it tested', async () => {
  const { chat } = await chatWith({ script: [SENSOR(), 'QX-7731'] });
  const r = await chat.selfCheck();
  assert.equal(r.model, 'stub/model');
  assert.equal(r.device, 'wasm');
  assert.equal(r.dtype, 'q4');
});

// ── Decoding ────────────────────────────────────────────────────────────────
// Greedy decoding has no escape from a loop: once a phrase is the argmax it
// stays the argmax. No system prompt fixes that, so the penalty is not optional
// and must reach EVERY generation — including the forced and answer-phase ones,
// which are exactly the turns a looping model ruins.

test('every generation carries a repetition penalty', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C in Chennai.'],
  });
  weather(chat);
  await chat.chat('Weather in Chennai?');

  assert.ok(llm.generated.length >= 2, 'more than one generation happened');
  for (const [i, g] of llm.generated.entries()) {
    assert.equal(g.opts.repetition_penalty, 1.1, `generation ${i} has the penalty`);
  }
});

test('the forced generation carries it too', async () => {
  const { chat, llm } = await chatWith({
    script: ['I cannot help with that.', dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);
  await chat.chat('Weather in Chennai?');

  const forced = llm.generated.find((g) => g.prompt.includes('<tool_call>\n{"name": "'));
  assert.ok(forced, 'a forced turn happened');
  assert.equal(forced.opts.repetition_penalty, 1.1);
});

test('repetitionPenalty is overridable, including off', async () => {
  const { chat, llm } = await chatWith({ script: ['hi'] });
  await chat.chat('hi', { repetitionPenalty: 1 });
  assert.equal(llm.generated.at(-1).opts.repetition_penalty, 1);
});

// ── loadForTools ────────────────────────────────────────────────────────────
// The dtype a host serves and the dtype that can call a tool are different
// facts. load() answers the first; this answers the second, which is the only
// one a caller actually cares about.

/** Serve `have` dtypes over HEAD; behaviour per dtype comes from `byDtype`. */
function dtypeWorld({ have = ['q4', 'q8'], byDtype = {} } = {}) {
  const originalFetch = globalThis.fetch;
  const FILES = { q4: 'model_q4.onnx', q8: 'model_quantized.onnx', fp16: 'model_fp16.onnx', fp32: 'model.onnx' };
  globalThis.fetch = async (url) => {
    const served = have.some((d) => String(url).endsWith(FILES[d]));
    return new Response(null, { status: served ? 200 : 404 });
  };
  const seen = [];
  const disposed = [];
  const transformers = {
    env: { allowRemoteModels: false, allowLocalModels: true, localModelPath: 'https://host/models/', backends: { onnx: { wasm: {} } } },
    async pipeline(_task, _id, opts) {
      seen.push(opts.dtype);
      const plan = byDtype[opts.dtype] ?? { throws: true };
      if (plan.throws) throw new Error(`cannot run ${opts.dtype}`);
      let n = 0;
      const gen = async () => ({ 0: { generated_text: plan.script[Math.min(n++, plan.script.length - 1)] } , length: 1 });
      const g = async (...a) => { const r = await gen(...a); return [r[0]]; };
      g.tokenizer = {
        apply_chat_template: (messages) =>
          messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>`).join('\n') + '\n<|im_start|>assistant\n',
      };
      g.dispose = async () => disposed.push(opts.dtype);
      return g;
    },
  };
  return { transformers, seen, disposed, restore: () => { globalThis.fetch = originalFetch; } };
}

const CALLS_OK = { script: [SENSOR(), 'The reading is QX-7731.'] };
const NEVER_CALLS = { script: ['I cannot do that.', 'I cannot do that.', 'I cannot do that.', 'I cannot do that.'] };

test('loadForTools skips a dtype that cannot call and keeps one that can', async () => {
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q4: NEVER_CALLS, q8: CALLS_OK } });
  try {
    const chat = await NexusChat.loadForTools(
      { base: 'https://host/models/', id: 'stub/model' },
      { transformers: w.transformers, device: 'wasm' },
    );
    assert.equal(chat.dtype, 'q8', 'settled on the dtype that works');
    assert.deepEqual(w.seen, ['q4', 'q8'], 'tried in preference order');
    assert.deepEqual(w.disposed, ['q4'], 'the rejected model was freed, the kept one was not');
  } finally { w.restore(); }
});

test('loadForTools throws naming every dtype it tried, rather than returning a dud', async () => {
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q4: NEVER_CALLS, q8: NEVER_CALLS } });
  try {
    await assert.rejects(
      NexusChat.loadForTools({ base: 'https://host/models/', id: 'stub/model' }, { transformers: w.transformers, device: 'wasm' }),
      (e) => /could call a tool/.test(e.message) && /q4:/.test(e.message) && /q8:/.test(e.message),
    );
    assert.deepEqual(w.disposed, ['q4', 'q8'], 'nothing left resident');
  } finally { w.restore(); }
});

test('a dtype that is served but cannot even load is a failed candidate, not a crash', async () => {
  // fp16 throws inside the ONNX session on some runtimes despite being served.
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q8: CALLS_OK } });   // q4 throws
  try {
    const chat = await NexusChat.loadForTools(
      { base: 'https://host/models/', id: 'stub/model' },
      { transformers: w.transformers, device: 'wasm' },
    );
    assert.equal(chat.dtype, 'q8');
  } finally { w.restore(); }
});

test('loadForTools reports every attempt so a UI can narrate the retry', async () => {
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q4: NEVER_CALLS, q8: CALLS_OK } });
  const attempts = [];
  try {
    await NexusChat.loadForTools(
      { base: 'https://host/models/', id: 'stub/model' },
      { transformers: w.transformers, device: 'wasm', onAttempt: (a) => attempts.push(a) },
    );
    assert.deepEqual(attempts.map((a) => [a.dtype, a.ok]), [['q4', false], ['q8', true]]);
  } finally { w.restore(); }
});

test('an explicit dtype is the only candidate, but is still verified', async () => {
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q4: NEVER_CALLS, q8: CALLS_OK } });
  try {
    await assert.rejects(
      NexusChat.loadForTools(
        { base: 'https://host/models/', id: 'stub/model' },
        { transformers: w.transformers, device: 'wasm', dtype: 'q4' },
      ),
      /could call a tool/,
    );
    assert.deepEqual(w.seen, ['q4'], 'did not go looking past the dtype it was given');
  } finally { w.restore(); }
});

test('allowForcing:false rejects a model that only calls when primed', async () => {
  const forcedOnly = { script: ['I cannot do that.', SENSOR(), 'The reading is QX-7731.'] };
  const w = dtypeWorld({ have: ['q4', 'q8'], byDtype: { q4: forcedOnly, q8: CALLS_OK } });
  try {
    const chat = await NexusChat.loadForTools(
      { base: 'https://host/models/', id: 'stub/model' },
      { transformers: w.transformers, device: 'wasm', allowForcing: false },
    );
    assert.equal(chat.dtype, 'q8', 'passed over the one that needed priming');
  } finally { w.restore(); }
});

test('the failure does not recommend the model that just failed', async () => {
  const w = dtypeWorld({ have: ['q4'], byDtype: { q4: NEVER_CALLS } });
  try {
    await assert.rejects(
      NexusChat.loadForTools({ hub: 'onnx-community/Qwen3-0.6B-ONNX' }, { transformers: w.transformers, device: 'wasm' }),
      (e) => !/try onnx-community\/Qwen3-0\.6B-ONNX/.test(e.message),
    );
  } finally { w.restore(); }
});

// ── Stepwise strategy ───────────────────────────────────────────────────────
// The point is structural, not statistical: the model SELECTS a tool from the
// registered list and never WRITES a name, and arguments arrive as bare values
// so there is no JSON to malform.

test('stepwise builds the call from closed answers, never asking for JSON', async () => {
  const { chat, llm } = await chatWith({ script: ['get_weather', 'Chennai"', 'It is 31C in Chennai.'] });
  const spy = {};
  weather(chat, spy);
  const answer = await chat.chat('Weather in Chennai?', { strategy: 'stepwise' });

  assert.equal(spy.city, 'Chennai', 'the primed value was extracted, not the sentence');
  assert.match(answer, /31C/);
  // No generation ever asked the template for tool schemas.
  assert.equal(llm.rendered.slice(0, 2).every((r) => !r.tools), true, 'no tool schemas in the step prompts');
});

test('stepwise cannot dispatch a tool the model invented', async () => {
  // 'rain' is exactly what Qwen2.5-0.5B at q8 emits when primed inline.
  const { chat } = await chatWith({ script: ['rain', 'I cannot help with that.'] });
  const spy = {};
  weather(chat, spy);
  await chat.chat('Weather in Chennai?', { strategy: 'stepwise' });
  assert.equal(spy.city, undefined, 'no handler ran');
  assert.equal(chat.metrics.counters.get('tool_calls'), undefined);
  assert.equal(chat.metrics.counters.get('tool_calls_stepwise_declined'), 1);
});

test('stepwise strips a sentence the model wrapped the value in', async () => {
  const { chat } = await chatWith({ script: ['get_weather', 'The city is Chennai', 'ok'] });
  const spy = {};
  weather(chat, spy);
  await chat.chat('Weather in Chennai?', { strategy: 'stepwise' });
  assert.equal(spy.city, 'Chennai', 'leading "The <key> is" removed');
});

test('stepwise coerces a numeric argument out of prose', async () => {
  const { chat } = await chatWith({ script: ['add', '4,831', '227', 'The sum is 5058.'] });
  const got = {};
  chat.tool('add', 'Add two numbers', { a: 'number', b: 'number' }, async ({ a, b }) => {
    Object.assign(got, { a, b });
    return { sum: a + b };
  });
  await chat.chat('Add 4831 and 227', { strategy: 'stepwise' });
  assert.deepEqual(got, { a: 4831, b: 227 }, 'numbers, not strings, commas removed');
});

test('stepwise selecting NONE falls through to a normal answer', async () => {
  const { chat } = await chatWith({ script: ['NONE', 'Paris is the capital of France.'] });
  const spy = {};
  weather(chat, spy);
  const answer = await chat.chat('What is the capital of France?', { strategy: 'stepwise' });
  assert.equal(spy.city, undefined);
  assert.match(answer, /Paris/);
});

test('stepwise and inline produce the same downstream transcript', async () => {
  const inline = await chatWith({ script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'] });
  weather(inline.chat);
  await inline.chat.chat('Weather in Chennai?');

  const step = await chatWith({ script: ['get_weather', 'Chennai"', 'It is 31C.'] });
  weather(step.chat);
  await step.chat.chat('Weather in Chennai?', { strategy: 'stepwise' });

  const shape = (c) => c.messages.map((m) => m.role);
  assert.deepEqual(shape(step.chat), shape(inline.chat), 'same message roles in the same order');
  const toolMsg = (c) => c.messages.find((m) => m.role === 'tool');
  assert.deepEqual(toolMsg(step.chat).content, toolMsg(inline.chat).content, 'identical tool result message');
});

test('stepwise primes the value so the model has nowhere to put a sentence', async () => {
  const { chat, llm } = await chatWith({ script: ['get_weather', 'Chennai"', 'ok'] });
  weather(chat);
  await chat.chat('Weather in Chennai?', { strategy: 'stepwise' });

  const argPrompt = llm.generated[1].prompt;
  assert.ok(argPrompt.endsWith('city = "'), `arg prompt must open the value; ends with ${JSON.stringify(argPrompt.slice(-24))}`);
});

test('stepwise primes a numeric argument without a quote', async () => {
  const { chat, llm } = await chatWith({ script: ['add', '4831', '227', 'ok'] });
  chat.tool('add', 'Add two numbers', { a: 'number', b: 'number' }, async () => ({ sum: 0 }));
  await chat.chat('Add them', { strategy: 'stepwise' });
  assert.ok(llm.generated[1].prompt.endsWith('a = '), 'numbers are not quoted');
});

// ── Automatic escalation ────────────────────────────────────────────────────
// The caller should not have to know which models need stepwise. Inline first,
// then primed, then stepwise — and only when the earlier ones actually failed.

test('a model that cannot produce the call format is rescued without being asked', async () => {
  // 1: refuses. 2: primed, names a tool that does not exist. 3-4: stepwise. 5: answer.
  const { chat } = await chatWith({
    script: ['I cannot help with that.', '{"name": "rain", "arguments": {}}', 'get_weather', 'Chennai"', 'It is 31C.'],
  });
  const spy = {};
  weather(chat, spy);
  const answer = await chat.chat('Weather in Chennai?');

  assert.equal(spy.city, 'Chennai', 'the handler ran after escalation');
  assert.match(answer, /31C/);
  assert.equal(chat.metrics.counters.get('tool_calls_force_failed'), 1, 'inline and priming both failed first');
  assert.equal(chat.metrics.counters.get('tool_calls_stepwise_rescued'), 1);
});

test('escalation does not fire when inline already worked', async () => {
  const { chat, llm } = await chatWith({
    script: [dialect.qwen('get_weather', { city: 'Chennai' }), 'It is 31C.'],
  });
  weather(chat);
  await chat.chat('Weather in Chennai?');
  assert.equal(chat.metrics.counters.get('tool_calls_stepwise_rescued'), undefined);
  assert.equal(llm.generated.length, 2, 'no extra round trips were spent');
});

test("strategy:'inline' opts out of the rescue entirely", async () => {
  const { chat } = await chatWith({
    script: ['I cannot help.', '{"name": "rain", "arguments": {}}', 'get_weather', 'Chennai"', 'nope'],
  });
  const spy = {};
  weather(chat, spy);
  await chat.chat('Weather in Chennai?', { strategy: 'inline' });
  assert.equal(spy.city, undefined, 'no handler ran');
  assert.equal(chat.metrics.counters.get('tool_calls_stepwise_rescued'), undefined);
});
