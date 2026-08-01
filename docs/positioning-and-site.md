# Positioning & docs site — browser-llm-nexus

*Decided in the huddle of 2026-08-01. This is the spec the `site/` scaffold implements.
Everything below is grounded in shipped code — no aspirational claims.*

---

## 1. The one-line position

> **Private LLM in any browser. GPU optional.**

And the sister line, which is the reason this library has the name it has:

> **toolnexus gives any LLM tools. browser-llm-nexus gives you the LLM.**
> Same tool schema — one runs on your server in six languages, one runs in the tab with
> no server at all.

## 2. The enemy

Not Transformers.js — that is the engine, and it is a peer dependency by design.

The enemy is the assumption that **running a model yourself means either a server or a
gaming GPU.** Every other "LLM in the browser" project is, in practice, a WebGPU showcase:
no adapter, no product. That excludes the managed corporate laptop with GPU acceleration
disabled by policy — which is *exactly* the machine where "this data must not leave the
page" is a hard requirement rather than a preference.

`src/runtime.ts:28` is the answer to that, in code:

```ts
/** Pick the fastest available backend: WebGPU when the browser exposes a usable
 *  adapter, otherwise WASM (CPU). Everything in this library works on both —
 *  GPU is an accelerator, never a requirement. */
```

That comment is the positioning. It just wasn't on a page anyone could read.

## 3. The three things a reader must retain

1. **Runs anywhere** — WebGPU → WASM/CPU, no code change, dtype ladder follows the backend
   (`fp16` first on GPU, `q4` first on CPU) among the variants the source actually serves.
2. **Calls your tools** — one parser covering Qwen/Hermes `<tool_call>`, Mistral
   `[TOOL_CALLS]`, Llama bare JSON, fenced JSON, nested OpenAI shapes, and string-encoded
   argument objects. Tool calling is what makes a 0.6B model useful.
3. **Ships offline in a zip** — export a model, an index, or a whole knowledge base;
   import it air-gapped; **nothing re-embeds, nothing downloads**.

Point 3 is the genuinely uncontested one and was buried near the bottom of the README.

## 4. Search intent — the distribution bet

Nobody types "browser-llm-nexus". The queries that exist and have no good answer today:

| Query | Landing page |
|---|---|
| transformers.js tool calling | `/tool-calling/` |
| offline RAG in the browser | `/knowledge/` |
| run LLM in browser without WebGPU | `/gpu-or-cpu/` |
| local LLM no server | `/` |

Consequence for the IA: **a page per search intent, not a page per class.** Each of those
three pages must stand alone with its own hook — a visitor arriving from search has never
seen the homepage and never will.

## 5. Site decisions

- **Free path only.** GitHub Pages at `https://muthuishere.github.io/browser-llm-nexus`.
  No custom domain, no Cloudflare, no shared `nexus.*` home. Revisit only if there's budget.
- **Astro + Starlight**, same as toolnexus: `site/` subfolder, `starlight-llms-txt`,
  the shared `deemwar.css` theme, `pages.yml` gated on `paths: site/**`.
- **Dropped from the toolnexus setup:** `starlight-sidebar-topics` and `LanguagePicker` —
  both exist because toolnexus is six languages. A topic picker on a one-language site is
  noise.
- **Kept:** `starlight-llms-txt`. Free, and it's how agents find the library.
- **No hand-written API reference.** It rots by 0.5.0. Generate from the `.d.ts` files
  `tsc` already emits, or leave it out.

## 6. Information architecture

Eight pages, all sourced from the existing README prose:

| Page | Slug | Role |
|---|---|---|
| Home | `/` | splash — headline, three points, sister line |
| Quickstart | `quickstart` | install → load → stream → one tool → metrics |
| GPU or CPU — same API | `gpu-or-cpu` | **differentiator + search intent** |
| Tool calling | `tool-calling` | **differentiator + search intent** |
| Knowledge & offline bundles | `knowledge` | **differentiator + search intent** |
| Embeddings & RAG | `embeddings-rag` | the pieces underneath, used directly |
| Where models come from | `model-sources` | the three explicit `ModelSource` shapes |
| Together with toolnexus | `toolnexus` | the sister page — honest version |

## 7. The sister claim, checked

Dileep's challenge in the huddle was: is the shared tool contract real, or marketing?
Checked against source — **it is real at the layer that matters, and not at the authoring
layer.**

- toolnexus: `defineTool({ name, description, inputSchema, run })` — `js/src/native.ts:19`
- browser-llm-nexus: `chat.tool(name, description, propsShorthand, handler, { required })`
  — `src/chat.ts:96`

Different call shapes, but `src/chat.ts:100-112` normalises the shorthand into exactly the
same `{ type: 'function', function: { name, description, parameters } }` envelope toolnexus
emits. So the *wire contract* is already shared; only the ergonomics differ.

**Decision: do not spend a release making the authoring APIs match.** Say it on the sister
page with both snippets side by side, and ship a `defineTool`-shaped adapter (~15 lines) if
someone actually asks. Cost: zero. Honesty: intact.

## 8. Bug found while writing this — fixed

Writing the "GPU or CPU" page surfaced a real one, and the README's own headline example
was the thing that broke.

**Symptom.** `NexusChat.load({ hub: … })` with auto dtype threw
`no dtype variant found for onnx-community/Qwen3-0.6B-ONNX under /models/`.

**Cause.** `resolveSource` deliberately does *not* set `env.localModelPath` on the `hub`
branch — hub loads are remote. But `detectDtype` probed
`${env.localModelPath}${modelId}/onnx/…` unconditionally, so a hub load probed whatever
stale local base happened to be set. Every candidate 404'd. No test covered a hub load
end-to-end, so it never showed up.

**Fix.** Probing location is now the *source's* business, not the runtime's:

- `src/runtime.ts` — `detectDtype` takes an optional `DtypeProbe` (`(file) => string`).
  Without one it keeps the old local-base behavior, which is correct for `base` and
  `archive`.
- `src/source.ts` — `dtypeProbe(source, tjs)` returns a Hub probe for `hub` sources
  (`<host><repo>/resolve/<revision>/onnx/<file>`, honoring a configured `remoteHost`
  mirror) and `undefined` otherwise.
- `src/chat.ts` — passes it through.

**Verified.** Four new tests in `test/source.test.mjs` (57 pass, was 53), plus a live check
against Hugging Face: `webgpu → fp16`, `wasm → q4` — exactly the documented ladder.
