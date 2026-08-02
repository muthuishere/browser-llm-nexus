# browser-llm-nexus

[![npm](https://img.shields.io/npm/v/browser-llm-nexus?color=cb3837&logo=npm)](https://www.npmjs.com/package/browser-llm-nexus)
[![install size](https://img.shields.io/badge/deps-0-brightgreen)](https://www.npmjs.com/package/browser-llm-nexus)
[![types](https://img.shields.io/badge/types-included-blue?logo=typescript&logoColor=white)](https://www.npmjs.com/package/browser-llm-nexus)
[![provenance](https://img.shields.io/badge/npm-signed%20provenance-6f42c1?logo=github)](https://www.npmjs.com/package/browser-llm-nexus#provenance)
[![license](https://img.shields.io/npm/l/browser-llm-nexus)](./LICENSE)

**One interface for running models in the browser.** Any ONNX model, from anywhere —
the Hub, your own host, a zip URL, a file on disk — on WebGPU or CPU, with tool calling,
embeddings, RAG, offline bundles and metrics. One small hooks-style TypeScript library
over [Transformers.js](https://github.com/huggingface/transformers.js).

Running a model in a tab means absorbing four kinds of variation at once: where the model
comes from, which backend the browser has, which quantization actually works, and which
call syntax the model family speaks. This library absorbs all four so your code doesn't
change when any of them does.

```bash
npm install browser-llm-nexus
```

📦 **[browser-llm-nexus on npm](https://www.npmjs.com/package/browser-llm-nexus)** — zero runtime
dependencies, TypeScript types included, published from CI with signed provenance.

Standalone by design: no server of ours, no bundled weights, no assumed layout.
Transformers.js is an *injectable* peer dependency, and **where a model comes from is
always something you state** — never guessed.

```ts
import { NexusChat } from 'browser-llm-nexus';

const chat = await NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });  // Hugging Face
const chat = await NexusChat.load({ base: '/models/', id: 'Qwen/Qwen3-0.6B' }); // your server
const chat = await NexusChat.load({ archive: fileTheUserPicked });              // a portable zip
const chat = await NexusChat.load({ archive: 'https://host/model.zip' });
```

Every loader picks **WebGPU when the browser has it, WASM/CPU otherwise** — your code
doesn't change either way. GPU is an accelerator here, never a requirement, so the same
page works on a workstation and a locked-down laptop. dtype selection follows the
backend, among the variants your source actually has — `q4` first on both, because a
faster answer that silently corrupts a tool result is worse than a slower correct one
(measured: fp16 read `1096637` back as `109,663,700`).

```ts
await NexusChat.load(source);                     // auto: webgpu → wasm
await NexusChat.load(source, { device: 'wasm' }); // force CPU
await NexusChat.load(source, { dtype: 'q4' });    // skip probing
```

## Chat with tool calling

```ts
// loadForTools, not load: it verifies the model can actually call a tool at the
// quantization it picked, and moves to the next one if it can't.
const chat = await NexusChat.loadForTools({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });

chat.tool('get_weather', 'Get current weather for a city',
  { city: 'string' },                                        // shorthand JSON schema
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());

chat.on('token', t => render(t));                            // hooks, not callbacks
chat.on('toolCall', (call, result) => console.log(call, result));

const answer = await chat.chat('What is the weather in Chennai?');
console.log(chat.metrics.summary());   // { load_ms_avg, tokens_per_second, tool_calls_ok, … }
```

The tool loop is automatic: parse (Qwen/Hermes `<tool_call>`, Mistral `[TOOL_CALLS]`,
Llama bare JSON, fenced JSON) → run your handler → feed the result back → grounded final
answer. Multi-round, multi-tool, with an anti-hallucination system prompt and
reasoning-model handling (`enable_thinking: false`, `<think>` stripping).

**A tool call always happens.** If a turn produces none while tools are registered, the
next generation starts with the call syntax already open, leaving no continuation that
isn't a call — and the result is kept only if it names a tool you registered. Decoding
carries a repetition penalty by default, because greedy decoding cannot escape a loop
on its own and no system prompt can fix that.

Which quantization a model can call tools at is **model-specific and inverted** between
models (Qwen2.5-0.5B works at `q4` and not `q8`; Qwen3-0.6B is the reverse), so
`loadForTools()` measures it instead of guessing — or throws naming every dtype it tried:

```ts
const chat = await NexusChat.loadForTools(source);   // proven, or it throws
const check = await chat.selfCheck();                // ask any loaded model directly
```

Dynamic tools from user-written JS (the decorator pattern as a function):

```ts
await chat.evalTools(`
  tool('get_watch_count', 'How many watches the user owns', {},
    async () => ({ watches: 7 }));
`);
```

## What else is out there, and where this differs

Running an LLM in a browser is not a new idea, and this is **a set of abstractions built on
Transformers.js**, not a new inference engine. Checked August 2026 — this area moves, so
re-check before trusting any of it.

**Chrome's built-in AI (Gemini Nano, the Prompt API)** — Chrome 138, stable. The model ships
with the browser, so there is **no download at all**, and it is free and fast. If you target
Chrome and don't need a specific model, use it. Its docs describe structured output via
`responseConstraint` (a JSON Schema), but **no tool calling** — constraining a reply's shape
is not the same as picking a function, filling its arguments, running it, and answering from
the result. It is also Chrome-only, the model is whichever one Google ships, and you cannot
substitute your own fine-tune or carry it to an air-gapped machine.

**WebLLM (MLC)** — faster than anything here; it compiles models to WebGPU kernels. It
**does** have function calling, and it is worth being precise about the state of it: the
project describes it as beta, limited to **one round** and to the **Hermes-2-Pro** models.
It also **requires WebGPU**, so there is no fallback on the GPU-less corporate laptop —
which is exactly the machine where "this data must not leave the page" is a hard requirement
rather than a preference.

**Transformers.js** — this library is built on it. If you want to run a model in a tab and
nothing more, use it directly: it already does device selection, dtype and the pipeline API.
You do not need this.

**TensorFlow.js** — still good for vision, audio and classical models. Not where transformer
work happens now.

### So what is actually here

Abstractions over the messy parts, not a faster runtime.

**A tool loop that survives a small model.** Multi-round and multi-tool, across model
families rather than one: Qwen/Hermes `<tool_call>`, Mistral `[TOOL_CALLS]`, Llama bare
JSON, fenced JSON and nested OpenAI-style objects all parse to the same call. It forces a
call when the model would rather narrate, salvages near-miss JSON, refuses to dispatch a
name you never registered, and — when a model simply cannot emit the format — stops asking
for JSON and builds the call from closed questions instead. That escalation is automatic.

**It tells you when it won't work.** `loadForTools()` gives a candidate a throwaway tool
returning an unguessable token and keeps the first quantization that both calls it and
answers from the result, or throws naming every one it tried. Which quantization works is
model-specific and doesn't transfer between models, so it is measured rather than assumed.
[What we measured](https://muthuishere.github.io/browser-llm-nexus/verified-models/),
failures included.

**Embeddings and RAG in the same place.** One engine stack for chat *and* embeddings —
chunking, a vector index, similarity search and retrieval-grounded answers, with no second
runtime and no server. Neither Chrome's Prompt API nor WebLLM gives you the retrieval half;
you would assemble it yourself.

**Export and import as first-class operations.** A chat model, an embedding model, a vector
index, or an entire knowledge base packs into **one zip** and restores on a machine with no
network — nothing re-embedded, nothing re-downloaded. This is the part that is genuinely
hard to assemble yourself, and it is why the offline/air-gapped case is the one where there
is no better alternative to reach for.

### Where it is the wrong choice

If you can call an API, call an API — a frontier model is far more capable, and your time
costs more than the tokens. This makes sense when the data cannot leave the machine, when
there is no network at all, or when per-user inference cost has to be zero. Outside those
three, it is the harder path.

## Three portable artifacts

Everything that can travel is an independent artifact with its own export/import, and
every importer accepts a **URL, a `File` from an `<input>`, a `Blob`, or raw bytes** — so
"load from a server" and "load from a file the user picked" are the same call.

```ts
import { exportModel, importModel, exportIndex, importIndex, hubRoot } from 'browser-llm-nexus';

// 1. a chat model — an embedding model packs identically
const zip = await exportModel('Qwen/Qwen3-0.6B', { modelsUrl: '/models/', dtypes: ['q4'] });
const zip2 = await exportModel('onnx-community/Qwen3-0.6B-ONNX', { root: hubRoot('onnx-community/Qwen3-0.6B-ONNX') });
const chat = await NexusChat.load({ archive: zip });

// 2. a RAG store — vectors as raw Float32, not JSON numbers
const ragZip = await exportIndex(index);
const restored = await importIndex(ragZip);
```

`inspectModel(source)` reads an archive's manifest without restoring anything.

A model archive is just:

```
manifest.json   { kind: 'model', modelId, dtypes, files: [{ file, url, path }] }
files/0.bin     each file's bytes
```

Import writes every file into the Cache API under the URL the runtime will request, so
loading afterwards makes zero network calls. The format is plain zip + JSON — any server
or build step can produce it.

## Offline knowledge system in one object

Documents in, grounded answers out — chunking, embedding, indexing, retrieval and
context assembly handled for you.

```ts
import { NexusKnowledge } from 'browser-llm-nexus';

const kb = await NexusKnowledge.create({
  chat: { hub: 'onnx-community/Qwen3-0.6B-ONNX' },
  embedder: { hub: 'Xenova/bge-small-en-v1.5' },   // this is also the default
});
await kb.addDocument({ id: 'handbook', title: 'Handbook', text: handbookText });

kb.on('token', t => render(t));
const answer = await kb.ask('What is the refund policy?');
```

A knowledge archive is simply the three artifacts composed into one zip:

```
manifest.json          { kind: 'knowledge', models, docs, contains }
rag/                   the vector store
models/chat/model.zip  optional — a full model archive, nested
models/embedder/model.zip
```

Ship it somewhere with no internet:

```ts
const zip = await kb.exportZip({ includeModels: true });   // rag + both models
download(zip, 'handbook-kb.zip');

// on the air-gapped machine — restores weights and vectors, re-embeds nothing
const kb2 = await NexusKnowledge.importZip(fileFromInput);
const kb3 = await NexusKnowledge.importZip('/bundles/handbook-kb.zip');
```

`exportZip({ includeText: false })` ships vectors without the source text when the
documents themselves shouldn't travel, and `NexusKnowledge.inspect(source)` reads the
manifest without loading any models.

## Embeddings + RAG on their own

```ts
import { NexusEmbedder, MemoryIndex, chunkText } from 'browser-llm-nexus';

const embedder = await NexusEmbedder.load({ hub: 'Xenova/bge-small-en-v1.5' });
const index = new MemoryIndex();

const chunks = chunkText(documentText);
const vectors = await embedder.embedBatch(chunks);
chunks.forEach((text, i) => index.add({ id: String(i), text, vector: vectors[i] }));

const context = index.contextFor(await embedder.embed(question), 5);
const answer = await chat.chat(`Context:\n${context}\n\nQuestion: ${question}`);
```

## Injectable runtime

```ts
import * as transformers from '@huggingface/transformers';   // or a lite/custom build
const chat = await NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' }, { transformers });
```

`fflate` is an optional peer dependency, used only for zip work; pass your own with
`{ zip }` or let it load from a CDN.

## Where models come from

Anything serving the standard Transformers.js layout works — the Hugging Face Hub, your
own static host, or a build step of your own. If you need to convert a PyTorch model to
that layout yourself, [hf2browser](https://github.com/muthuishere/hf2browser) is one tool
that does it (and serves model archives in the format above), but nothing here depends on
it.

## Used by

- **[hf2browser](https://github.com/muthuishere/hf2browser)** — converts any Hugging Face LLM
  to the browser layout, then chats with it through this library (its CPU verifier runs the
  tool loop here, so what it certifies is what a page does). It also generates a **single
  self-contained `chat.html`** per model — this library from a CDN, the weights from a
  `model.zip` — so a converted model ships as two static files you can host anywhere.
- **[offline-llm-knowledge-system](https://github.com/muthuishere/offline-llm-knowledge-system)** —
  packages documents + a model into a portable zip you open offline; this library is its
  `transformersjs` engine, the one that needs no GPU.

## Develop

```bash
npm install && npm test      # tsc build + node --test (53 tests, no network)
```
