// A mock LLM: real chat templates, real call dialects, no weights and no network.
//
// The point is to exercise the actual loop in chat.ts — template rendering,
// parsing, dispatch, feed-back, phase switching — against the formats real
// models emit, without downloading a model to find out something broke.

/** Format a tool call the way each model family actually emits one. */
export const dialect = {
  qwen: (name, args) => `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`,
  hermes: (name, args) => `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`,
  mistral: (name, args) => `[TOOL_CALLS] ${JSON.stringify([{ name, arguments: args }])}`,
  llama: (name, args) => JSON.stringify({ name, parameters: args }),
  fenced: (name, args) => '```json\n' + JSON.stringify({ name, arguments: args }) + '\n```',
  // Arguments as a JSON *string* — extremely common from quantized models.
  openai: (name, args) =>
    JSON.stringify({ function: { name, arguments: JSON.stringify(args) } }),
  thinking: (name, args) =>
    `<think>The user wants ${name}. I should call it.</think>\n<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`,
};

/** Chat templates, close enough to the real ones to assert against. */
const templates = {
  qwen(messages, tools) {
    let out = '';
    for (const m of messages) {
      if (m.role === 'system') {
        out += `<|im_start|>system\n${m.content}`;
        if (tools?.length) {
          out += `\n\n# Tools\n<tools>\n${tools.map((t) => JSON.stringify(t)).join('\n')}\n</tools>`;
        }
        out += '<|im_end|>\n';
      } else if (m.role === 'tool') {
        out += `<|im_start|>user\n<tool_response>\n${m.content}\n</tool_response><|im_end|>\n`;
      } else {
        out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      }
    }
    return out + '<|im_start|>assistant\n';
  },
  mistral(messages, tools) {
    let out = tools?.length ? `[AVAILABLE_TOOLS] ${JSON.stringify(tools)}[/AVAILABLE_TOOLS]` : '';
    for (const m of messages) {
      if (m.role === 'user') out += `[INST] ${m.content}[/INST]`;
      else if (m.role === 'tool') out += `[TOOL_RESULTS] ${m.content}[/TOOL_RESULTS]`;
      else out += m.content ?? '';
    }
    return out;
  },
};

/**
 * Build a mock transformers runtime.
 *
 * `script` drives generation: each entry is the raw text the "model" returns
 * for that round — a string, or a function given { messages, tools, round }.
 * The last entry repeats if the loop runs longer.
 */
export function mockLLM({ family = 'qwen', template = 'qwen', script = [], streamer = true } = {}) {
  const rendered = [];   // every apply_chat_template call
  const generated = [];  // every generation's options
  let round = 0;

  const tokenizer = {
    apply_chat_template(messages, opts = {}) {
      rendered.push({ messages: structuredClone(messages), tools: opts.tools, opts });
      return (templates[template] ?? templates.qwen)(messages, opts.tools);
    },
  };

  const generator = async (prompt, opts = {}) => {
    const ctx = { ...rendered[rendered.length - 1], round, prompt, dialect: dialect[family] };
    const entry = script[Math.min(round, script.length - 1)];
    round++;
    let text = typeof entry === 'function' ? await entry(ctx) : (entry ?? '');
    generated.push({ prompt, opts });
    // Drive the streamer the way a real pipeline does, so 'token' hooks fire.
    if (opts.streamer?.callback_function) {
      for (const chunk of String(text).match(/.{1,8}/gs) ?? []) opts.streamer.callback_function(chunk);
    }
    return [{ generated_text: text }];
  };
  generator.tokenizer = tokenizer;
  generator.dispose = async () => { generator.disposed = true; };

  const transformers = {
    env: {},
    pipeline: async (task, modelId, opts) => {
      transformers.lastPipeline = { task, modelId, opts };
      return generator;
    },
    ...(streamer
      ? {
          TextStreamer: class {
            constructor(tok, o) { this.callback_function = o.callback_function; }
          },
        }
      : {}),
  };

  return {
    transformers,
    generator,
    rendered,
    generated,
    get rounds() { return round; },
    /** The system message content as the model saw it in round `i`. */
    systemAt: (i) => rendered[i]?.messages.find((m) => m.role === 'system')?.content,
    /** Tool-result messages the model saw in round `i`. */
    toolMsgsAt: (i) => rendered[i]?.messages.filter((m) => m.role === 'tool') ?? [],
  };
}
