// Render data/verified-models.json into a docs page.
//
// The table is generated, never hand-written: a measured claim that someone
// edited by hand is no longer a measured claim. Re-run after test:models.
import { readFileSync, writeFileSync } from 'node:fs';

const d = JSON.parse(readFileSync(new URL('../data/verified-models.json', import.meta.url), 'utf8'));
// Data becomes MDX, and MDX reads { } as a JSX expression and < as a tag. A
// measured note like {"name": "rain"} is ordinary prose here and a syntax
// error there, so escape at the boundary rather than sanitising the data.
const mdx = (t) => String(t).replace(/[{}<>]/g, (c) => `\\${c}`);
const MARK = { usable: '✅ usable', flaky: '⚠️ flaky', poor: '⚠️ poor', broken: '❌ broken', unusable: '❌ 0/3' };
const dt = ['q4', 'q8', 'fp16'];

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
