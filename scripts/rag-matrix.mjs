// Does retrieval actually work? Real embedding models, real documents.
//
//   npm run test:rag
//
// The knowledge tests use a fake bag-of-chars embedder, which proves the
// plumbing and nothing about whether the right chunk comes back. This asks the
// question that matters, against weights, on a corpus built with near-miss
// distractors so a good score cannot be an accident.
//
// Measured per embedding model and dtype:
//   recall@1 / recall@5   did the GOLD document rank first / in the top five
//   MRR                   1/rank of the gold document, averaged
//   margin                gold score minus best distractor score; near zero
//                         means it is guessing between look-alikes
//   round-trip            exported and reimported index returns identical hits
//
// Exits non-zero if no model reaches a usable bar, so CI can gate on it.
import * as transformers from '@huggingface/transformers';
import { NexusEmbedder, NexusChat, NexusKnowledge, MemoryIndex, chunkText, exportIndex, importIndex } from '../dist/index.js';
import { DOCS, QUESTIONS } from './rag-corpus.mjs';
import * as fflate from 'fflate';

const MODELS = process.env.RAG_MODELS?.split(',') ?? [
  'Xenova/all-MiniLM-L6-v2',
  'Xenova/bge-small-en-v1.5',
  'Xenova/gte-small',
];
const DTYPES = process.env.RAG_DTYPES?.split(',') ?? ['q8', 'fp32'];
const K = 5;

/** Build an index over the corpus, one chunk per document region. */
async function buildIndex(embedder) {
  const index = new MemoryIndex();
  const texts = [];
  const owners = [];
  for (const doc of DOCS) {
    for (const [i, text] of chunkText(doc.text, 500, 50).entries()) {
      texts.push(text);
      owners.push({ id: `${doc.id}#${i}`, docId: doc.id });
    }
  }
  const vectors = await embedder.embedBatch(texts);
  texts.forEach((text, i) =>
    index.add({ id: owners[i].id, text, vector: vectors[i], meta: { docId: owners[i].docId } }),
  );
  return index;
}

function scoreOne(hits, gold) {
  const rank = hits.findIndex((h) => h.chunk.meta.docId === gold);       // 0-based, -1 = missing
  const goldScore = rank >= 0 ? hits[rank].score : 0;
  const bestOther = hits.find((h) => h.chunk.meta.docId !== gold)?.score ?? 0;
  return { rank, margin: goldScore - bestOther };
}

const rows = [];
for (const modelId of MODELS) {
  for (const dtype of DTYPES) {
    let embedder;
    const t0 = Date.now();
    try {
      embedder = await NexusEmbedder.load({ hub: modelId }, { device: 'cpu', dtype, transformers });
    } catch (e) {
      rows.push({ modelId, dtype, error: String(e.message).slice(0, 80) });
      console.log(`✗ ${modelId} ${dtype} — failed to load: ${String(e.message).slice(0, 70)}`);
      continue;
    }

    const index = await buildIndex(embedder);
    let at1 = 0, at5 = 0, mrrSum = 0, marginSum = 0;
    const misses = [];

    for (const { q, gold } of QUESTIONS) {
      const qv = await embedder.embed(q);
      const hits = index.search(qv, K);
      const { rank, margin } = scoreOne(hits, gold);
      if (rank === 0) at1++;
      if (rank >= 0) { at5++; mrrSum += 1 / (rank + 1); }
      marginSum += margin;
      if (rank !== 0) misses.push({ q, gold, got: hits[0]?.chunk.meta.docId ?? '—', rank });
    }

    // Export/import fidelity — a headline claim, so measure it rather than
    // trust it: the restored index must return the same ranking, not merely
    // load without throwing.
    let roundTrip = 'ok';
    try {
      const zip = await exportIndex(index, { zip: fflate });
      const restored = await importIndex(zip, { zip: fflate });
      for (const { q } of QUESTIONS) {
        const qv = await embedder.embed(q);
        const a = index.search(qv, K).map((h) => h.chunk.id).join(',');
        const b = restored.search(qv, K).map((h) => h.chunk.id).join(',');
        if (a !== b) { roundTrip = 'DIFFERS'; break; }
      }
    } catch (e) {
      roundTrip = 'FAILED: ' + String(e.message).slice(0, 40);
    }

    const n = QUESTIONS.length;
    const row = {
      modelId, dtype,
      at1: at1 / n, at5: at5 / n,
      mrr: mrrSum / n,
      margin: marginSum / n,
      roundTrip,
      secs: (Date.now() - t0) / 1000,
      misses,
    };
    rows.push(row);
    console.log(
      `${at1 === n ? '✓' : at5 === n ? '~' : '✗'} ${modelId} ${dtype} ` +
      `recall@1:${at1}/${n} recall@5:${at5}/${n} mrr:${row.mrr.toFixed(2)} ` +
      `margin:${row.margin.toFixed(3)} round-trip:${roundTrip} ${row.secs.toFixed(0)}s`,
    );
    for (const m of misses) console.log(`    miss: "${m.q}" → ${m.got} (gold ${m.gold}, rank ${m.rank})`);
    await embedder.dispose?.();
  }
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const ok = rows.filter((r) => !r.error && r.at1 === 1 && r.roundTrip === 'ok');
console.log('\n' + '─'.repeat(78));
console.log('model                          dtype  recall@1  recall@5  mrr    margin  verdict');
console.log('─'.repeat(78));
for (const r of rows) {
  if (r.error) { console.log(`${r.modelId.padEnd(30)} ${r.dtype.padEnd(6)} load failed`); continue; }
  const verdict = r.roundTrip !== 'ok' ? 'ROUND-TRIP BROKEN'
    : r.at1 === 1 ? 'USABLE'
    : r.at5 === 1 ? 'RERANK NEEDED'
    : 'UNUSABLE';
  console.log(
    `${r.modelId.padEnd(30)} ${r.dtype.padEnd(6)} ${String(r.at1 * 100).padStart(7)}% ` +
    `${String(r.at5 * 100).padStart(8)}% ${r.mrr.toFixed(2).padStart(6)} ${r.margin.toFixed(3).padStart(7)}  ${verdict}`,
  );
}

console.log('\nVERDICT');
console.log('─'.repeat(78));
if (!ok.length) {
  console.log('No embedding model retrieved the right document first on every question.');
  console.log('Retrieval is the foundation — a wrong chunk cannot be recovered downstream.');
  process.exit(1);
}
const best = ok.sort((a, b) => b.margin - a.margin)[0];
console.log(`USE ${best.modelId} at dtype '${best.dtype}'`);
console.log(`  ${best.at1 * 100}% recall@1 on ${QUESTIONS.length} questions against near-miss distractors,`);
console.log(`  mean margin ${best.margin.toFixed(3)} over the best wrong answer.`);
console.log(`  await NexusEmbedder.load({ hub: '${best.modelId}' }, { dtype: '${best.dtype}' });`);
console.log('\nFull rows:');
console.log(JSON.stringify(rows.map(({ misses, ...r }) => r), null, 1));


// ── Phase 2: the whole thing, end to end ────────────────────────────────────
// Retrieval scoring above says the right chunk comes back. That is necessary
// and not sufficient — the claim on the box is a GROUNDED ANSWER, and the zip
// that carries the whole knowledge base to a machine with no network. Both are
// asserted in the README, so both get measured.
if (process.env.RAG_SKIP_E2E) {
  console.log('\nskipping end-to-end (RAG_SKIP_E2E set)');
  process.exit(0);
}

const CHAT = process.env.RAG_CHAT ?? 'onnx-community/Qwen3-0.6B-ONNX';
console.log(`\nEND TO END — ${best.modelId} + ${CHAT}`);
console.log('─'.repeat(78));

const embedder = await NexusEmbedder.load({ hub: best.modelId }, { device: 'cpu', dtype: best.dtype, transformers });
const chat = await NexusChat.load({ hub: CHAT }, { device: 'cpu', dtype: 'q4', transformers });
const kb = await NexusKnowledge.create({ chat, embedder, transformers });
for (const doc of DOCS) await kb.addDocument(doc);

let grounded = 0, wrongFact = 0;
for (const { q, answer } of QUESTIONS) {
  const said = String(await kb.ask(q, { maxNewTokens: 96 }));
  const hit = said.includes(answer);
  if (hit) grounded++;
  // Did it answer with a DIFFERENT document's token? That is the failure mode
  // that matters: confidently right-shaped and wrong, which a human would not
  // catch without knowing the corpus.
  else if (QUESTIONS.some((o) => o.answer !== answer && said.includes(o.answer))) wrongFact++;
  console.log(`  ${hit ? '✓' : '✗'} ${q}`);
  if (!hit) console.log(`      wanted ${answer}, said: ${JSON.stringify(said.slice(0, 110))}`);
}

// The one-zip claim: export everything, reimport, ask again with no re-embedding.
let zipVerdict = 'ok';
try {
  const zip = await kb.exportZip({ zip: fflate });
  const kb2 = await NexusKnowledge.importZip(zip, { chat, embedder, transformers, zip: fflate });
  const probe = QUESTIONS[0];
  const before = (await kb.retrieve(probe.q, 5)).map((c) => c.id).join(',');
  const after = (await kb2.retrieve(probe.q, 5)).map((c) => c.id).join(',');
  if (before !== after) zipVerdict = `DIFFERS (${before} vs ${after})`;
  console.log(`\n  knowledge zip: ${(zip.length / 1024).toFixed(0)} KB, retrieval after reimport: ${zipVerdict}`);
} catch (e) {
  zipVerdict = 'FAILED: ' + e.message;
  console.log(`\n  knowledge zip: ${zipVerdict}`);
}

const n = QUESTIONS.length;
console.log(`\n  grounded answers: ${grounded}/${n}` + (wrongFact ? `   answered with the WRONG document's fact: ${wrongFact}` : ''));
await chat.dispose();

if (zipVerdict !== 'ok') {
  console.log('\nExport/import is a headline claim and it did not hold. Failing.');
  process.exit(1);
}
