import { Hooks } from './hooks.ts';
import { Metrics } from './metrics.ts';
import { parseToolCalls, salvageToolCall, stripCallFragments, stripThinking, type ToolCall } from './toolcalls.ts';
import { resolveTransformers, detectDtype, availableDtypes, detectDevice, type Device, type RuntimeOptions, type TransformersLike } from './runtime.ts';
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
  /** Whether a tool call is optional or mandatory for this turn.
   *
   *  'auto' (default) — generate normally; if the model produced no tool call
   *  and tools are registered, generate ONCE more with the call syntax already
   *  started, which leaves it no way to answer except by completing a call.
   *  'required'  — start the call syntax immediately, skipping the free turn.
   *  'none'      — never force; the model answers or it doesn't. */
  toolChoice?: 'auto' | 'required' | 'none';
  /** Divides the logit of any token already generated, so the next one is less
   *  likely to be the same. Decoding is greedy, and greedy decoding on a small
   *  model has no escape from a loop: once a phrase becomes the argmax it stays
   *  the argmax forever. This is the only thing that breaks that cycle — no
   *  system prompt can, because the loop is not a comprehension failure.
   *
   *  1.1 by default: enough to break loops, mild enough that the structural
   *  tokens JSON legitimately repeats (`"`, `,`, `:`) still win their positions.
   *  Set 1 to disable. Above ~1.2 tool-call JSON starts to malform. */
  repetitionPenalty?: number;
}

/** Verdict from {@link NexusChat.selfCheck}. */
export interface ToolCallCheck {
  /** Called a tool AND answered from its result. The only value worth gating on. */
  ok: boolean;
  called: boolean;
  grounded: boolean;
  /** True when the model only called because the library primed the syntax —
   *  it works, but it is closer to the edge than a model that volunteers. */
  needed_forcing: boolean;
  model: string;
  device: string;
  dtype: string;
  answer: string;
  /** One sentence you can show a user verbatim. */
  detail: string;
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
  /** A forced attempt: the primed generation, what was salvaged from it (null
   *  if nothing trustworthy), and the round. Without this a discarded forced
   *  turn is invisible — you see "no tool call" and cannot tell whether the
   *  model refused, named something unregistered, or emitted unparseable text. */
  forced: [string, ToolCall | null, number];
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
  /** Steers the model *toward* calling a tool, before any results exist.
   *
   *  Small models follow a worked example far better than an instruction, so
   *  this shows the exact bytes expected rather than describing them. Measured
   *  on Qwen2.5-0.5B: the earlier prose-only prompt left q8 calling 0/3. */
  systemPrompt =
    'You have access to tools. You MUST call a tool instead of guessing, ' +
    'calculating, or inventing data — even if you think you know the answer.\n' +
    'To call a tool, reply with ONLY this, and nothing else:\n' +
    '<tool_call>\n{"name": "the_tool_name", "arguments": {"arg": "value"}}\n</tool_call>\n' +
    'Do not explain what you are about to do. Do not describe the tool. ' +
    'Emit the tool call itself.';

  /** How a forced tool call is started. The parser accepts this tag from any
   *  model family, so priming it works even where the model was trained on a
   *  different call syntax. */
  toolCallPrefix = '<tool_call>\n{"name": "';
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

  /** Load a model that provably calls tools — or fail loudly saying nothing did.
   *
   *  `load()` picks the first dtype the host *serves*, which is a statement
   *  about the host and not about whether anything works. This asks the only
   *  question that matters: it loads a candidate, runs {@link selfCheck} against
   *  a throwaway tool returning an unguessable token, and keeps the first one
   *  that both calls the tool and answers from its result. A candidate that
   *  fails is disposed before the next is tried, so only one model is resident.
   *
   *      const chat = await NexusChat.loadForTools({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });
   *
   *  Pass an explicit `dtype` and that is the only candidate — this still tells
   *  you whether it works, it just will not go looking for another. Every
   *  attempt is reported through `onAttempt` so a UI can narrate the retry
   *  rather than appear to hang on a second download.
   *
   *  Cost is the honest tradeoff: a rejected candidate was still downloaded.
   *  Weights are cached, so it is paid once per dtype per browser. */
  static async loadForTools(
    source: ModelSource,
    opts: LoadOptions & {
      /** Called after each candidate is judged, pass or fail. */
      onAttempt?: (check: ToolCallCheck) => void;
      /** Accept a model that only calls when the syntax is primed. Default true —
       *  forcing is a supported path, not a defect. Set false to demand a model
       *  that volunteers the call unaided. */
      allowForcing?: boolean;
    } = {},
  ): Promise<NexusChat> {
    const tjs = await resolveTransformers(opts);
    const modelId = await resolveSource(tjs, source);
    const device = await detectDevice(opts.device ?? 'auto');
    const candidates =
      opts.dtype && opts.dtype !== 'auto'
        ? [opts.dtype]
        : await availableDtypes(tjs, modelId, device, dtypeProbe(source, tjs));

    const tried: ToolCallCheck[] = [];
    for (const dtype of candidates) {
      let chat: NexusChat;
      try {
        chat = await NexusChat.load(source, { ...opts, dtype, transformers: tjs });
      } catch (err) {
        // A dtype the host serves can still fail to run — fp16 on some
        // runtimes throws inside the session rather than 404ing. That is a
        // failed candidate, not a failed load.
        tried.push({
          ok: false, called: false, grounded: false, needed_forcing: false,
          model: modelId, device, dtype, answer: '',
          detail: `${modelId} (${device}/${dtype}) failed to load: ${(err as Error).message}`,
        });
        opts.onAttempt?.(tried[tried.length - 1]!);
        continue;
      }
      const check = await chat.selfCheck();
      tried.push(check);
      opts.onAttempt?.(check);
      if (check.ok && (opts.allowForcing !== false || !check.needed_forcing)) {
        chat.metrics.count('dtypes_rejected', tried.length - 1);
        return chat;
      }
      await chat.dispose();
    }

    // Suggesting the model that just failed reads as a bug, so only name it
    // when it is not the one in hand.
    const KNOWN_GOOD = 'onnx-community/Qwen3-0.6B-ONNX';
    throw new Error(
      `no dtype of ${modelId} could call a tool on ${device}. Tried:\n` +
        tried.map((t) => `  ${t.dtype}: ${t.detail}`).join('\n') +
        (modelId.includes(KNOWN_GOOD)
          ? ''
          : `\nModels below ~0.5B generally cannot pick a tool name from a list; try ${KNOWN_GOOD}.`),
    );
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

  private async generate(opts: ChatOptions, round = 0, prefix = '', withTools = true): Promise<string> {
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
    // A prefix is appended AFTER the generation prompt, so the model resumes
    // mid-token-stream with the call syntax already open. There is no valid
    // continuation that is prose — that is what makes the call happen rather
    // than merely being requested.
    const prompt: string =
      tok.apply_chat_template(messages, {
        tools: withTools && this.tools.size ? this.toolSchemas : undefined,
        tokenize: false,
        add_generation_prompt: true,
        enable_thinking: false,
      }) + prefix;
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
        repetition_penalty: opts.repetitionPenalty ?? 1.1,
        return_full_text: false,
        streamer,
      }),
    );
    this.metrics.count('tokens_out', tokens);
    // The prefix was our text, not the model's, but the parser has to see the
    // whole call — so stitch it back on.
    return prefix + (out[0].generated_text as string);
  }

  /** Chat with the automatic tool loop; returns the final grounded answer. */
  async chat(userText: string, opts: ChatOptions = {}): Promise<string> {
    if (this.tools.size && !this.messages.some((m) => m.role === 'system')) {
      this.messages.unshift({ role: 'system', content: this.systemPrompt });
    }
    this.messages.push({ role: 'user', content: userText });
    this.metrics.count('chats');

    const choice = opts.toolChoice ?? 'auto';
    const answered = () => this.messages.some((m) => m.role === 'tool');

    for (let round = 0; round < this.maxRounds; round++) {
      this.emit('round', round);
      // 'required' skips the free turn on the first round; after results exist
      // the model must be free to answer, or the loop could never terminate.
      const forceNow = choice === 'required' && !answered();
      let raw = await this.generate(opts, round, forceNow ? this.toolCallPrefix : '');
      let parsed = parseToolCalls(raw);
      let calls = parsed.filter((c) => this.tools.has(c.name));

      // The model declined to call anything. Asking again politely does not
      // work on small models — so generate once more with the call syntax
      // already open, leaving no continuation that isn't a call. Only before
      // any results exist: afterwards, "no call" is the correct final answer.
      if (!calls.length && choice === 'auto' && this.tools.size && !answered() && !forceNow) {
        this.metrics.count('tool_calls_forced');
        const forced = await this.generate(opts, round, this.toolCallPrefix);
        // Lenient on purpose: a primed small model routinely emits nearly-valid
        // JSON, and discarding a correctly-named call over a missing brace
        // wastes the whole forced turn.
        const salvaged = salvageToolCall(forced, [...this.tools.keys()]);
        this.emit('forced', forced, salvaged, round);
        const forcedParsed = salvaged ? [salvaged] : parseToolCalls(forced);
        const forcedCalls = forcedParsed.filter((c) => this.tools.has(c.name));
        // Keep the forced turn only if it named a real tool; a hallucinated
        // name is worse than the answer the model gave us unprompted.
        if (forcedCalls.length) {
          raw = forced;
          parsed = forcedParsed;
          calls = forcedCalls;
        } else {
          // Loud on purpose. The model was handed open call syntax and still
          // could not name a registered tool — that is a capability limit, not
          // a transient miss, and silently returning its prose hides it.
          this.metrics.count('tool_calls_force_failed');
          const named = forced.match(/"name"\s*:\s*"([^"]{1,40})"/)?.[1];
          if (named) this.emit('metric', `forced_call_named_unknown_tool:${named}`, 1);
        }
      }

      // Emit what was parsed, not just what survived the name filter — a call
      // to a tool that isn't registered is a different problem from no call at
      // all, and both end the loop the same silent way.
      this.emit('raw', raw, parsed, round);
      if (!calls.length) {
        let answer = stripThinking(raw);
        if (answered()) {
          // Answer phase. The model still sees the tool schemas and sometimes
          // opens a call again and stops; that fragment is not an answer.
          const cleaned = stripCallFragments(answer);
          if (!cleaned) {
            // Nothing but call syntax came back. Ask once more with the tools
            // removed from the template, so prose is the only thing it can
            // produce. Observed live: the tool ran, and the user was shown the
            // literal text "<tool_call>".
            this.metrics.count('answer_retried_without_tools');
            answer = stripCallFragments(stripThinking(await this.generate(opts, round, '', false)));
          } else {
            answer = cleaned;
          }
        }
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

  /** Can THIS model, as loaded, actually call a tool and answer from it?
   *
   *  The published matrix cannot cover a model someone just uploaded from a
   *  zip or served from their own host, so ask the model itself: register a
   *  throwaway tool whose result is a token nothing could guess, ask for it,
   *  and see whether the token comes back in the answer. Roughly one
   *  generation pair — cheap next to loading the weights.
   *
   *  Registered tools and conversation history are saved and restored, so this
   *  is safe to run immediately after load. Note that `token`/`toolCall`/`raw`
   *  hooks DO fire during the check; ignore them by their round if your UI
   *  cares.
   *
   *    const check = await chat.selfCheck();
   *    if (!check.ok) warn(check.detail);
   */
  async selfCheck(opts: ChatOptions = {}): Promise<ToolCallCheck> {
    const savedTools = new Map(this.tools);
    const savedMessages = this.messages;
    this.tools = new Map();
    this.messages = [];

    // Unguessable by construction: the only way into the answer is a real call.
    const TOKEN = 'QX-7731';
    let called = false;
    this.tool(
      'lookup_sensor',
      'Read the current value of a sensor by its id. Use this for any sensor question.',
      { id: 'string' },
      async ({ id }) => {
        called = true;
        return { id, reading: TOKEN };
      },
    );

    let answer = '';
    try {
      answer = await this.chat('What is the reading of sensor A9? Include the reading exactly.', opts);
    } catch (e) {
      answer = `error: ${String((e as Error).message ?? e)}`;
    }

    const forced = (this.metrics.counters.get('tool_calls_forced') ?? 0) > 0;
    const grounded = answer.includes(TOKEN);
    this.tools = savedTools;
    this.messages = savedMessages;

    const ok = called && grounded;
    const detail = ok
      ? `${this.modelId} (${this.device}/${this.dtype}) calls tools correctly` +
        (forced ? ', but only when the call syntax is forced — expect the occasional miss.' : '.')
      : called
        ? `${this.modelId} (${this.device}/${this.dtype}) calls tools but does not report the result accurately — answers may look right and be wrong.`
        : `${this.modelId} (${this.device}/${this.dtype}) does not call tools. Try another quantization, or a larger model — below ~0.5B this usually cannot be fixed.`;

    return { ok, called, grounded, needed_forcing: forced, model: this.modelId, device: this.device, dtype: this.dtype, answer, detail };
  }

  reset(): void {
    this.messages = [];
  }

  dispose(): Promise<void> {
    return this.generator.dispose();
  }
}
