import { Hooks } from './hooks.ts';
import { Metrics } from './metrics.ts';
import { parseToolCalls, stripThinking, type ToolCall } from './toolcalls.ts';
import { resolveTransformers, detectDtype, type RuntimeOptions, type TransformersLike } from './runtime.ts';

export type ToolHandler = (args: Record<string, unknown>) => unknown | Promise<unknown>;

export interface ToolSchema {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: { type: 'object'; properties: Record<string, unknown>; required: string[] };
  };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
}

export interface LoadOptions extends RuntimeOptions {
  dtype?: string | 'auto';
  device?: string;
  onProgress?: (p: unknown) => void;
}

export interface ChatOptions {
  maxNewTokens?: number;
}

type ChatEvents = {
  token: [string];
  toolCall: [ToolCall, unknown];
  round: [number];
  answer: [string];
  metric: [string, number];
};

/** Tool-calling chat over a converted browser model.
 *
 *   const chat = await NexusChat.load('Qwen/Qwen3-0.6B');
 *   chat.tool('get_weather', 'Current weather', { city: 'string' }, getWeather);
 *   chat.on('token', t => render(t));
 *   const answer = await chat.chat('Weather in Chennai?');
 */
export class NexusChat extends Hooks<ChatEvents> {
  readonly metrics = new Metrics();
  maxRounds = 4;
  systemPrompt =
    'You have access to tools. When a question relates to a tool, you MUST call the tool instead of guessing or inventing data. Answer from tool results.';
  messages: ChatMessage[] = [];

  private tools = new Map<string, { schema: ToolSchema; handler: ToolHandler }>();

  private constructor(
    private generator: any,
    readonly dtype: string,
    private tjs: TransformersLike,
  ) {
    super();
  }

  static async load(modelId: string, opts: LoadOptions = {}): Promise<NexusChat> {
    const tjs = await resolveTransformers(opts);
    const dtype = !opts.dtype || opts.dtype === 'auto' ? await detectDtype(tjs, modelId) : opts.dtype;
    const t0 = Date.now();
    const generator = await tjs.pipeline('text-generation', modelId, {
      dtype,
      device: opts.device ?? 'wasm',
      progress_callback: opts.onProgress,
    });
    const chat = new NexusChat(generator, dtype, tjs);
    chat.metrics.time('load', Date.now() - t0);
    return chat;
  }

  /** Register a tool. Properties accept shorthand: { city: 'string' }. */
  tool(
    name: string,
    description: string,
    properties: Record<string, string | Record<string, unknown>>,
    handler: ToolHandler,
    opts: { required?: string[] } = {},
  ): this {
    const props = Object.fromEntries(
      Object.entries(properties).map(([k, v]) => [k, typeof v === 'string' ? { type: v } : v]),
    );
    this.tools.set(name, {
      schema: {
        type: 'function',
        function: {
          name,
          description,
          parameters: { type: 'object', properties: props, required: opts.required ?? Object.keys(props) },
        },
      },
      handler,
    });
    return this;
  }

  get toolSchemas(): ToolSchema[] {
    return [...this.tools.values()].map((t) => t.schema);
  }

  /** Evaluate user-written JS that defines tools via `tool(...)` — the
   *  decorator pattern as a function. Replaces existing tools. */
  async evalTools(code: string): Promise<string[]> {
    this.tools.clear();
    const register = (
      name: string,
      description: string,
      properties: Record<string, string | Record<string, unknown>>,
      handler: ToolHandler,
    ) => this.tool(name, description, properties, handler);
    const fn = new Function('tool', `'use strict';\nreturn (async () => {\n${code}\n})();`);
    await fn(register);
    return [...this.tools.keys()];
  }

  private async generate(opts: ChatOptions): Promise<string> {
    const tok = this.generator.tokenizer;
    const prompt: string = tok.apply_chat_template(this.messages, {
      tools: this.tools.size ? this.toolSchemas : undefined,
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    let tokens = 0;
    const streamer = this.tjs.TextStreamer
      ? new this.tjs.TextStreamer(tok, {
          skip_prompt: true,
          callback_function: (t: string) => {
            tokens++;
            this.emit('token', t);
          },
        })
      : undefined;
    const out: any = await this.metrics.measure('generate', () =>
      this.generator(prompt, {
        max_new_tokens: opts.maxNewTokens ?? 256,
        do_sample: false,
        return_full_text: false,
        streamer,
      }),
    );
    this.metrics.count('tokens_out', tokens);
    return out[0].generated_text as string;
  }

  /** Chat with the automatic tool loop; returns the final grounded answer. */
  async chat(userText: string, opts: ChatOptions = {}): Promise<string> {
    if (this.tools.size && !this.messages.some((m) => m.role === 'system')) {
      this.messages.unshift({ role: 'system', content: this.systemPrompt });
    }
    this.messages.push({ role: 'user', content: userText });
    this.metrics.count('chats');

    for (let round = 0; round < this.maxRounds; round++) {
      this.emit('round', round);
      const raw = await this.generate(opts);
      const calls = parseToolCalls(raw).filter((c) => this.tools.has(c.name));
      if (!calls.length) {
        const answer = stripThinking(raw);
        this.messages.push({ role: 'assistant', content: answer });
        this.emit('answer', answer);
        return answer;
      }
      this.messages.push({ role: 'assistant', content: raw });
      for (const call of calls) {
        this.metrics.count('tool_calls');
        let result: unknown;
        try {
          result = await this.tools.get(call.name)!.handler((call.arguments as Record<string, unknown>) ?? {});
          this.metrics.count('tool_calls_ok');
        } catch (e) {
          result = { error: String((e as Error).message ?? e) };
          this.metrics.count('tool_calls_failed');
        }
        this.emit('toolCall', call, result);
        this.messages.push({ role: 'tool', name: call.name, content: JSON.stringify(result) });
      }
    }
    const answer = `tool loop exceeded ${this.maxRounds} rounds`;
    this.messages.push({ role: 'assistant', content: answer });
    this.emit('answer', answer);
    return answer;
  }

  reset(): void {
    this.messages = [];
  }

  dispose(): Promise<void> {
    return this.generator.dispose();
  }
}
