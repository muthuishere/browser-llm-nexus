// The tool loop, with a stubbed generator — no weights, no network.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NexusChat } from '../dist/chat.js';

/** A generator that records every rendered prompt and replays scripted turns. */
function stubChat(turns) {
  const rendered = [];
  const tokenizer = {
    // Stand-in for a real chat template: enough structure to assert against.
    apply_chat_template: (messages, opts) => {
      rendered.push({ messages: structuredClone(messages), tools: opts.tools });
      return messages.map((m) => `<${m.role}>${m.content}`).join('\n');
    },
  };
  let i = 0;
  const generator = async () => [{ generated_text: turns[Math.min(i++, turns.length - 1)] }];
  generator.tokenizer = tokenizer;
  const chat = new NexusChat(generator, 'q4', 'wasm', {}, 'stub/model');
  return { chat, rendered };
}

const CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Chennai"}}\n</tool_call>';

test('tool loop: calls the handler and feeds the result back', async () => {
  const { chat, rendered } = stubChat([CALL, 'It is 31C and humid in Chennai.']);
  let seen = null;
  chat.tool('get_weather', 'Weather for a city', { city: 'string' }, async ({ city }) => {
    seen = city;
    return { city, conditions: '31C, humid' };
  });

  const answer = await chat.chat("What's the weather in Chennai?");

  assert.equal(seen, 'Chennai');
  assert.equal(answer, 'It is 31C and humid in Chennai.');
  assert.equal(rendered.length, 2, 'one render to call the tool, one to answer');
  assert.equal(rendered[0].tools.length, 1, 'tool schemas reach the template');
  // The result was fed back as a tool message before the second generation.
  const fedBack = rendered[1].messages.find((m) => m.role === 'tool');
  assert.equal(fedBack.name, 'get_weather');
  assert.match(fedBack.content, /31C, humid/);
});

// Regression: the instruction that makes a small model CALL a tool ("you MUST
// call the tool instead of guessing") reads, once results are in context, as
// "the tool has not been called yet" — so the model apologises for a failure
// that never happened instead of reading the result above it. Measured on
// Qwen2.5-0.5B: "there was an error while fetching the weather information".
test('the system prompt switches from call-phase to answer-phase', async () => {
  const { chat, rendered } = stubChat([CALL, 'It is 31C in Chennai.']);
  chat.tool('get_weather', 'Weather for a city', { city: 'string' }, async () => ({ ok: true }));

  await chat.chat("What's the weather in Chennai?");

  const sysOf = (r) => r.messages.find((m) => m.role === 'system').content;
  assert.equal(sysOf(rendered[0]), chat.systemPrompt, 'round 1 steers toward calling');
  assert.equal(sysOf(rendered[1]), chat.answerPrompt, 'round 2 steers toward answering');
  assert.notEqual(chat.systemPrompt, chat.answerPrompt);
});

test('a caller-written system message is never rewritten', async () => {
  const { chat, rendered } = stubChat([CALL, 'done']);
  chat.tool('get_weather', 'Weather', { city: 'string' }, async () => ({ ok: true }));
  chat.messages.push({ role: 'system', content: 'You are a pirate.' });

  await chat.chat('weather?');

  for (const r of rendered) {
    assert.equal(r.messages.find((m) => m.role === 'system').content, 'You are a pirate.');
  }
});

test('no tools registered: no system prompt is injected, one round only', async () => {
  const { chat, rendered } = stubChat(['Just an answer.']);
  const answer = await chat.chat('hello');
  assert.equal(answer, 'Just an answer.');
  assert.equal(rendered.length, 1);
  assert.equal(rendered[0].messages.some((m) => m.role === 'system'), false);
  assert.equal(rendered[0].tools, undefined);
});

test('a throwing handler is reported to the model, not thrown at the caller', async () => {
  const { chat, rendered } = stubChat([CALL, 'Sorry, that lookup failed.']);
  chat.tool('get_weather', 'Weather', { city: 'string' }, async () => {
    throw new Error('upstream down');
  });

  const answer = await chat.chat('weather?');

  assert.equal(answer, 'Sorry, that lookup failed.');
  assert.match(rendered[1].messages.find((m) => m.role === 'tool').content, /upstream down/);
  assert.equal(chat.metrics.counters.get('tool_calls_failed'), 1);
});

test('unknown tool names are ignored rather than dispatched', async () => {
  const { chat } = stubChat(['<tool_call>\n{"name": "nope", "arguments": {}}\n</tool_call>']);
  chat.tool('get_weather', 'Weather', { city: 'string' }, async () => ({ ok: true }));
  const answer = await chat.chat('weather?');
  // No registered tool matched, so the text is treated as the final answer.
  assert.match(answer, /nope/);
});
