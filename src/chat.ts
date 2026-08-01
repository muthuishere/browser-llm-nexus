import { Hooks } from './hooks.ts';
import { Metrics } from './metrics.ts';
import { parseToolCalls, stripThinking, type ToolCall } from './toolcalls.ts';
import { resolveTransformers, detectDtype, detectDevice, type Device, type RuntimeOptions, type TransformersLike } from './runtime.ts';
import { dtypeProbe, resolveSource, type ModelSource } from './source.ts';

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
  /** Quantization variant. 'auto' probes which variants the source has. */
  dtype?: string | 'auto';
  /** 'auto' (default) uses WebGPU when available, else WASM/CPU. */
  device?: Device;
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
  /** The prompt as the model actually received it, per round. The single most
   *  useful thing to see when a model won't call a tool — it shows whether the
   *  schemas and tool results really made it into the template. */
  prompt: [string, number];
  /** Raw generation before parsing, per round, with what was parsed out of it.
   *  "The model answered instead of calling" and "the model called but we
   *  failed to parse it" look identical from the outside without this. */
  raw: [string, ToolCall[], number];
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
  /** Steers the model *toward* calling a tool, before any results exist. */
  systemPrompt =
    'You have access to tools. When a question relates to a tool, you MUST call the tool instead of guessing or inventing data. Answer from tool results.';
  /** Replaces `systemPrompt` once tool results are in the conversation.
   *
   *  These have to be two different instructions. "You MUST call the tool
   *  instead of guessing" is what makes a small model emit a call in round one
   *  — and the same sentence, still in context in round two, reads as *the
   *  tool has not been called yet*, so the model apologises for a failure that
   *  never happened instead of reading the result sitting right above it.
   *  Measured on Qwen2.5-0.5B: the call-phase prompt answers "there was an
   *  error while fetching the weather information" with a perfectly good
   *  tool_response in context; this one reports the value. */
  answerPrompt =
    'You have received tool results. Report them to the user in a sentence. Never invent data.';
  messages: ChatMessage[] = [];

  private tools = new Map<string, { schema: ToolSchema; handler: ToolHandler }>();

  private constructor(
    private generator: any,
    readonly dtype: string,
    readonly device: string,
    private tjs: TransformersLike,
    readonly modelId: string,
  ) {
    super();
  }

  /**
   * Load a chat model. The source is always explicit — this library never
   * guesses a host or a path convention:
   *
   *   NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' })   // Hugging Face
   *   NexusChat.load({ base: '/models/', id: 'Qwen/Qwen3-0.6B' }) // your server
   *   NexusChat.load({ archive: fileFromInput })                  // a portable zip
   *   NexusChat.load({ archive: 'https://host/model.zip' })
   */
  static async load(source: ModelSource, opts: LoadOptions = {}): Promise<NexusChat> {
    const tjs = await resolveTransformers(opts);
    const modelId = await resolveSource(tjs, source);
    const device = await detectDevice(opts.device ?? 'auto');
    const dtype =
      !opts.dtype || opts.dtype === 'auto'
        ? await detectDtype(tjs, modelId, device, dtypeProbe(source, tjs))
        : opts.dtype;
    const t0 = Date.now();
    const generator = await tjs.pipeline('text-generation', modelId, {
      dtype,
      device,
      progress_callback: opts.onProgress,
    });
    const chat = new NexusChat(generator, dtype, device, tjs, modelId);
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

  private async generate(opts: ChatOptions, round = 0): Promise<string> {
    const tok = this.generator.tokenizer;
    // Once results are in, swap the call-phase instruction for the answer-phase
    // one. Only our own injected system message is touched — a system message
    // the caller wrote themselves is left exactly as they wrote it.
    const answering = this.messages.some((m) => m.role === 'tool');
    const messages = answering
      ? this.messages.map((m) =>
          m.role === 'system' && m.content === this.systemPrompt ? { ...m, content: this.answerPrompt } : m,
        )
      : this.messages;
    const prompt: string = tok.apply_chat_template(messages, {
      tools: this.tools.size ? this.toolSchemas : undefined,
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    });
    this.emit('prompt', prompt, round);
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
      const raw = await this.generate(opts, round);
      const parsed = parseToolCalls(raw);
      const calls = parsed.filter((c) => this.tools.has(c.name));
      // Emit what was parsed, not just what survived the name filter — a call
      // to a tool that isn't registered is a different problem from no call at
      // all, and both end the loop the same silent way.
      this.emit('raw', raw, parsed, round);
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
