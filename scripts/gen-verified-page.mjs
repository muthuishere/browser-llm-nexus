// Render data/verified-models.json into a docs page.
//
// The table is generated, never hand-written: a measured claim that someone
// edited by hand is no longer a measured claim. Re-run after test:models.
import { readFileSync, writeFileSync } from 'node:fs';

const d = JSON.parse(readFileSync(new URL('../data/verified-models.json', import.meta.url), 'utf8'));
const r = d.retrieval;
// Data becomes MDX, and MDX reads { } as a JSX expression and < as a tag. A
// measured note like {"name": "rain"} is ordinary prose here and a syntax
// error there, so escape at the boundary rather than sanitising the data.
const mdx = (t) => String(t).replace(/[{}<>]/g, (c) => `\\${c}`);
const MARK = { usable: '✅ usable', flaky: '⚠️ flaky', poor: '⚠️ poor', broken: '❌ broken', unusable: '❌ 0/3' };
const dt = ['q4', 'q8', 'fp16'];

const embedderList = r.embedders
  .map((e) => `- **${e.id}** @ ${e.dtype} — recall@1 **${e.recallAt1 * 100}%**, margin ${e.margin}` +
              (e.note ? ` — ${mdx(e.note)}` : ''))
  .join('\n');

const rows = d.models.map((m) => {
  const cells = dt.map((k) => {
    const v = m.dtypes[k];
    // Plain markdown, not inline HTML: MDX parses attributes as JSX, where
    // style= must be an object, and the build fails on a raw style string.
    return v ? `${MARK[v.verdict]} · ${v.called}/${d.questions}` : '–';
  });
  return `| [${m.id}](https://huggingface.co/${m.id}) | ${m.params} | ${cells.join(' | ')} |`;
}).join('\n');

const notes = d.models.flatMap((m) =>
  dt.filter((k) => m.dtypes[k]?.note).map((k) => `- **${m.id} @ ${k}** — ${mdx(m.dtypes[k].note)}`),
).join('\n');

writeFileSync(new URL('../site/src/content/docs/verified-models.mdx', import.meta.url), `---
title: Which models actually work
description: Measured results for tool calling in the browser, per model and per quantization — produced by a real-weights harness, not curated by hand.
---

{/* GENERATED from data/verified-models.json by scripts/gen-verified-page.mjs — do not edit. */}

Every row here was produced by \`npm run test:models\` against **real weights**, asking
${d.questions} questions whose answers are unguessable tokens — so a correct reply proves a tool
call really happened rather than the model guessing well.

Measured **${d.measuredOn}** on **${d.device}**, library **${d.library}**.

| Model | Params | q4 | q8 | fp16 |
|---|---|---|---|---|
${rows}

${notes ? `### Why the marked ones fail\n\n${notes}\n` : ''}
## Retrieval — which embedding models find the right chunk

\`npm run test:rag\` measures the other half: given a question, does the **right document**
come back? The corpus is built to be hard — every question's key phrase appears in **two**
documents and only a qualifier (standard/express, production/staging, severity one/two)
picks the right one, so lexical overlap alone cannot answer it. Answers are unguessable
tokens, and the harness is checked against itself: swapping in random vectors drops
recall@1 from 10/10 to 1/10, so a good score is not an artefact of an easy corpus.

${embedderList}

**Margin is the number to read, not recall.** All three retrieve correctly on every
question; what separates them is how far ahead the right document scores over the
near-miss. ${r.embedders[0].id} leads by ${r.embedders[0].margin}, ${r.embedders.at(-1).id}
by only ${r.embedders.at(-1).margin} — correct today, and much closer to a coin flip on a
corpus slightly harder than this one.

### End to end

With **${r.endToEnd.embedder}** retrieving and **${r.endToEnd.chat}** answering:
**${r.endToEnd.groundedAnswers} grounded answers** — the reply contained the unguessable
token from the correct document, so retrieval and generation both did their job. The whole
knowledge base exported to a **${r.endToEnd.knowledgeZipKB} KB zip** and, after reimport,
returned identical retrieval${r.endToEnd.reimportRetrievalIdentical ? '' : ' — MISMATCH'}.

Note the contrast with tool calling above: retrieval is **far** more reliable at these model
sizes. Summarising a retrieved paragraph is a much easier task than choosing a function and
emitting valid JSON, which is why an offline knowledge system is a better fit for a small
in-browser model than an agent is.

## Reading this table

**Only models we measured appear here.** Anything else is untested rather than bad —
${d.verdicts.untested.replace(/^Not measured by us\. /, '')}

**${mdx(d.floor)}**

The important thing this table shows is that **the best quantization is model-specific and
does not transfer**: Qwen2.5-0.5B is reliable at q4 and poor at q8, while Qwen3-0.6B is fine
at both and broken at fp16. There is no ordering that wins everywhere, which is exactly why
[\`loadForTools()\`](/browser-llm-nexus/tool-calling/#just-give-me-a-model-that-calls-tools)
measures instead of assuming.

## Check your own model

This table cannot cover a model you converted yourself or loaded from your own host. Ask it
directly — one throwaway tool, one unguessable token, a few seconds:

\`\`\`ts
const chat = await NexusChat.loadForTools(source);  // picks a working dtype, or throws
const check = await chat.selfCheck();
// { ok, called, grounded, needed_forcing, dtype, detail }
\`\`\`

To reproduce this whole table locally:

\`\`\`bash
npm run test:models
\`\`\`
`);
console.log('wrote site/src/content/docs/verified-models.mdx');
