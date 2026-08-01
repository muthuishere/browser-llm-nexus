# browser-llm-nexus

[![npm](https://img.shields.io/npm/v/browser-llm-nexus?color=cb3837&logo=npm)](https://www.npmjs.com/package/browser-llm-nexus)
[![install size](https://img.shields.io/badge/deps-0-brightgreen)](https://www.npmjs.com/package/browser-llm-nexus)
[![types](https://img.shields.io/badge/types-included-blue?logo=typescript&logoColor=white)](https://www.npmjs.com/package/browser-llm-nexus)
[![provenance](https://img.shields.io/badge/npm-signed%20provenance-6f42c1?logo=github)](https://www.npmjs.com/package/browser-llm-nexus#provenance)
[![license](https://img.shields.io/npm/l/browser-llm-nexus)](./LICENSE)

**Run LLMs in the browser — GPU or CPU, same API.** Tool calling, embeddings, RAG,
offline bundles, and metrics in one small hooks-style TypeScript library over
[Transformers.js](https://github.com/huggingface/transformers.js).

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
backend (fp16 first on GPU, q4 first on CPU) among the variants your source actually has.

```ts
await NexusChat.load(source);                     // auto: webgpu → wasm
await NexusChat.load(source, { device: 'wasm' }); // force CPU
await NexusChat.load(source, { dtype: 'q4' });    // skip probing
```

## Chat with tool calling

```ts
const chat = await NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });

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

Dynamic tools from user-written JS (the decorator pattern as a function):

```ts
await chat.evalTools(`
  tool('get_watch_count', 'How many watches the user owns', {},
    async () => ({ watches: 7 }));
`);
```

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
