# Your LLM Doesn't Always Need a Server. Sometimes It Doesn't Even Need a GPU.

*Why I built browser-llm-nexus — and, more usefully, when you should and shouldn't reach for it.*

---

There's a demo you've seen a dozen times by now. Someone loads a language model into a
browser tab, types a prompt, and tokens stream out with no server anywhere. The comments
fill up with "wow, WebGPU is here."

Then you try it on your work laptop and get a blank page.

That gap is the entire reason this library exists. Not because the demos are fake — they
work fine on a MacBook Pro with an unlocked GPU. But the machine where "this data must
never leave the page" is an actual *requirement*, rather than a nice-to-have, is almost
never that machine. It's a managed corporate laptop in a hospital, a bank, or a law firm,
with GPU acceleration disabled by policy and a browser three versions behind.

The demo excludes precisely the user who needed it most.

## The assumption worth attacking

The received wisdom is that running a model yourself means one of two things: you rent a
server and send your data to it, or you require a machine with a decent GPU.

**browser-llm-nexus takes a third position: GPU is an accelerator, never a requirement.**

Every loader asks the browser for a GPU adapter. If there's one, it uses WebGPU. If there
isn't — or if requesting it throws, which happens more than you'd think behind enterprise
policy — it falls back to WASM on the CPU and keeps going.

```ts
const chat = await NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });

console.log(chat.device);  // 'webgpu' or 'wasm' — whichever this browser can do
```

Your code doesn't change. There's no capability check to write, no "unsupported browser"
branch, no second code path to maintain. The same page works on a workstation and on a
locked-down laptop, and the only difference is how long it takes.

The quantization follows the backend automatically, because what's fast on a GPU isn't
what's fast on a CPU. On WebGPU it tries `fp16` first; on WASM it tries `q4` first. It
probes which variants your model host actually serves and takes the first one that exists,
rather than assuming.

Slow but correct beats fast but blank.

## Tool calling is what makes a small model useful

Here's the thing people get wrong about small models: they judge them on what the model
*knows*.

A 0.6B model in a browser tab knows very little, and it will confidently make things up.
Judged as an oracle, it's useless.

But it doesn't need to be an oracle. It needs to work out *which of your functions to
call*, and that's a much easier job — one that works reliably at sizes that fit in a tab.

```ts
chat.tool(
  'get_weather',
  'Get current weather for a city',
  { city: 'string' },
  async ({ city }) => (await fetch(`/api/weather?c=${city}`)).json(),
);

const answer = await chat.chat('What is the weather in Chennai?');
```

The loop runs itself: the model emits a call, the library parses it, runs your handler,
feeds the result back, and asks again until you get an answer grounded in real data.

The unglamorous work here is parsing. Every open model family invented its own way of
saying "call this function" — Qwen and Hermes emit `<tool_call>` tags, Mistral emits
`[TOOL_CALLS]`, Llama emits bare JSON, some emit JSON in a markdown fence, and quantized
models routinely emit the arguments object as a *string* of JSON inside the arguments
field. One parser handles all of it, in order, so switching models doesn't mean rewriting
your integration.

There's also a default system prompt that pushes hard against guessing, and reasoning
models get their `<think>` blocks stripped before parsing. Small details, but they're the
difference between a demo and something you'd ship.

## The part I actually think is uncontested

Everything above, someone else could build. This next bit is the reason I kept going.

Embedding a document corpus is the slow, expensive part of retrieval. So do it once, at
build time, and ship the *result*:

```ts
const zip = await kb.exportZip({ includeModels: true });   // vectors + both models
```

Hand that file to a machine with no internet:

```ts
const kb = await NexusKnowledge.importZip(fileFromInput);
const answer = await kb.ask('What is the refund policy?');
```

The weights are restored into the browser cache and the vectors are restored as vectors.
**Nothing downloads. Nothing re-embeds.** A field technician with a laptop and a USB stick
gets a working, grounded assistant over your documentation, in a place with no signal.

If the source documents themselves shouldn't travel — customer records, licensed content —
export with `includeText: false`. Retrieval still works; the prose stays home.

I don't know of another way to get that today.

## So: when should you use this?

Let me be more useful than a feature list.

**Reach for it when the constraint is that data must not leave the machine.** Not "we'd
prefer it didn't" — when it genuinely can't. Regulated content, client-confidential
material, internal documents that aren't cleared for a third-party API. The strongest
version of this argument is that it doesn't leave for *your* server either, which means
there's no breach surface to defend and no data-processing agreement to negotiate.

**Reach for it when there's no network.** Air-gapped environments, field work, ships,
factory floors, conference wifi. An offline bundle is a file, and files work everywhere.

**Reach for it when per-token cost dominates.** If you're doing high-volume, low-complexity
work — classification, extraction, routing, autocomplete over a fixed corpus — inference on
the user's own hardware costs you nothing per call, forever. That math changes what
features are viable.

**Reach for it when latency matters more than depth.** No round trip. No cold start on
someone else's queue.

## And when should you not?

I'd rather you find this out here than three weeks in.

**Don't use it when you need a frontier model.** You're running something in the range of
0.5B to 3B parameters. It will not reason through your hardest problems, write your
best code, or handle a 200k-token context. If quality is the binding constraint, call a
real API — that's what [toolnexus](https://github.com/muthuishere/toolnexus), the sister
library, is for.

**Don't use it when first-load weight is unacceptable.** Even a small quantized model is
hundreds of megabytes on first visit. It caches afterwards, but if your users bounce in
eight seconds, this is the wrong tool.

**Don't use it for a public consumer site with no repeat visits.** The economics invert:
you're paying bandwidth to ship weights to people who use them once.

**Don't use it when CPU-only would actually be too slow for the job.** The fallback keeps
the page *working*, not fast. For an interactive chat on an old laptop, "working" may not
be good enough — measure it with your real users before committing.

That's a real list, and I'd rather it cost me a few users than have someone adopt this for
a job it can't do.

## A note on how it's built

Three things I decided early and haven't regretted.

**No bundled weights, no assumed host.** Where a model comes from is always something you
state — a Hugging Face repo, a folder you serve, or a portable archive. There's no default
location and no `/models/` convention baked in. If you don't say, the library doesn't
guess.

**Transformers.js is injectable.** It's the engine, and it's a peer dependency you can
swap — pin it, use a lite build, patch it. The library doesn't own your runtime.

**Nothing is aspirational.** Every capability here has tests behind it, and when a bug
turns up it gets a regression test. A recent one is instructive: loading from the Hugging
Face Hub with automatic quantization selection was probing the wrong location entirely and
throwing — and the reason nobody caught it is that no test covered that path end to end.
It's fixed, the fix has four tests, and the design changed so that *where to probe* is now
the model source's responsibility instead of a shared default someone can forget to set.

That's the kind of bug you only find by writing the documentation honestly enough to check
your own claims.

## Try it

```bash
npm install browser-llm-nexus
```

Docs: **https://muthuishere.github.io/browser-llm-nexus/**
Source: **https://github.com/muthuishere/browser-llm-nexus**

If you're doing the server-side version of this problem — giving any LLM tools, MCP
servers, and agent skills across six languages — that's
[toolnexus](https://muthuishere.github.io/toolnexus/). Same idea, other side of the wire.

If you build something with either, I'd genuinely like to hear about it.
