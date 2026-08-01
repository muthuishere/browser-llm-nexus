# browser-llm-nexus

**Run LLMs in the browser — GPU or CPU, same API.** Tool calling, embeddings, RAG,
offline knowledge bundles, and metrics in one small hooks-style TypeScript library over
[Transformers.js](https://github.com/huggingface/transformers.js) (injectable — bring
the full build, a lite build, or anything shape-compatible).

```bash
npm install browser-llm-nexus
```

By default every loader picks **WebGPU when the browser has it, WASM/CPU otherwise** —
your code doesn't change, and neither does the API. GPU is an accelerator here, never a
requirement, so the same page works on a workstation and a locked-down laptop.

```ts
await NexusChat.load(model);                    // auto: webgpu → wasm
await NexusChat.load(model, { device: 'wasm' }); // force CPU
```

dtype selection follows the backend too (fp16 first on GPU, q4 first on CPU) among the
variants that actually exist for your model.

Models come from [hf2browser](https://github.com/muthuishere/hf2browser), which
converts any Hugging Face LLM to the layout this library loads — or point
`modelsUrl` at any host serving the same layout.

## Chat with tool calling

```ts
import { NexusChat } from 'browser-llm-nexus';

const chat = await NexusChat.load('Qwen/Qwen3-0.6B');       // auto-picks best dtype

chat.tool('get_weather', 'Get current weather for a city',
  { city: 'string' },                                        // shorthand JSON schema
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json());

chat.on('token', t => render(t));                            // hooks, not callbacks
chat.on('toolCall', (call, result) => console.log(call, result));

const answer = await chat.chat('What is the weather in Chennai?');
console.log(chat.metrics.summary());   // { load_ms_avg, tokens_per_second, tool_calls_ok, ... }
```

The tool loop is automatic: parse (Qwen/Hermes `<tool_call>`, Mistral `[TOOL_CALLS]`,
Llama bare JSON, fenced JSON) → run your handler → feed the result back → grounded
final answer. Multi-round, multi-tool, with an anti-hallucination system prompt and
reasoning-model handling (`enable_thinking: false`, `<think>` stripping).

Dynamic tools from user-written JS (the decorator pattern as a function):

```ts
await chat.evalTools(`
  tool('get_watch_count', 'How many watches the user owns', {},
    async () => ({ watches: 7 }));
`);
```

## Offline knowledge system in one object

Documents in, grounded answers out — chunking, embedding, indexing, retrieval and
context assembly handled for you. This is the whole
[offline-llm-knowledge-system](https://github.com/muthuishere/offline-llm-knowledge-system)
pattern as an API.

```ts
import { NexusKnowledge } from 'browser-llm-nexus';

const kb = await NexusKnowledge.create({ chat: 'Qwen/Qwen3-0.6B' });  // embedder defaults to bge-small
await kb.addDocument({ id: 'handbook', title: 'Handbook', text: handbookText });

kb.on('token', t => render(t));
const answer = await kb.ask('What is the refund policy?');
```

Ship it somewhere with no internet:

```ts
const bundle = await kb.export({ includeModels: true });  // index + weights from the browser cache
await fs.writeFile('kb.json', JSON.stringify(bundle));

// on the air-gapped machine — restores weights and vectors, re-embeds nothing
const kb2 = await NexusKnowledge.import(JSON.parse(await fs.readFile('kb.json')));
```

`export({ includeText: false })` ships vectors without the source text when the
documents themselves shouldn't travel.

## Embeddings + RAG

```ts
import { NexusEmbedder, MemoryIndex, chunkText } from 'browser-llm-nexus';

const embedder = await NexusEmbedder.load('BAAI/bge-small-en-v1.5');
const index = new MemoryIndex();

const chunks = chunkText(documentText);
const vectors = await embedder.embedBatch(chunks);
chunks.forEach((text, i) => index.add({ id: String(i), text, vector: vectors[i] }));

const context = index.contextFor(await embedder.embed(question), 5);
const answer = await chat.chat(`Context:\n${context}\n\nQuestion: ${question}`);
```

`MemoryIndex.serialize()/restore()` round-trips through JSON for persistence.

## Offline bundles

```ts
import { exportCache, importCache, toManifest } from 'browser-llm-nexus';

// machine A (online): after loading models once
const entries = await exportCache();          // drain transformers-cache
const { index, files } = toManifest(entries); // zip-ready {file,url} + blobs

// machine B (air-gapped): restore, then loads make zero network calls
await importCache(entries);
```

Same Cache-API contract as
[offline-llm-knowledge-system](https://github.com/muthuishere/offline-llm-knowledge-system)'s
embed-cache.

## Injectable runtime

```ts
import * as transformers from '@huggingface/transformers';   // or a lite build
const chat = await NexusChat.load(model, { transformers, modelsUrl: '/models/' });
```

## Develop

```bash
npm install && npm test      # tsc build + node --test (17 tests, no network)
```

Currently lives inside the hf2browser repo (`libs/browser-llm-nexus`); designed to
extract to its own repo/npm package unchanged.
