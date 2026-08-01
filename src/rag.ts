import { similarity } from './embed.ts';

export interface Chunk {
  id: string;
  text: string;
  vector: Float32Array;
  meta?: Record<string, unknown>;
}

export interface SearchHit {
  chunk: Chunk;
  score: number;
}

/** In-memory vector index: add chunks, cosine top-k search, JSON round-trip.
 *  Deliberately dependency-free — for thousands of chunks this is plenty;
 *  swap in a real store when you outgrow it. */
export class MemoryIndex {
  private chunks: Chunk[] = [];

  add(chunk: Chunk): void {
    this.chunks.push(chunk);
  }

  addAll(chunks: Chunk[]): void {
    this.chunks.push(...chunks);
  }

  get size(): number {
    return this.chunks.length;
  }

  /** Top-k most similar chunks to the query vector. */
  search(queryVector: Float32Array, k = 5): SearchHit[] {
    return this.chunks
      .map((chunk) => ({ chunk, score: similarity(queryVector, chunk.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /** Assemble a context block from top hits (for stuffing into a user turn). */
  contextFor(queryVector: Float32Array, k = 5, maxChars = 4000): string {
    let out = '';
    for (const hit of this.search(queryVector, k)) {
      if (out.length + hit.chunk.text.length > maxChars) break;
      out += hit.chunk.text + '\n\n';
    }
    return out.trim();
  }

  /** Serialize to a plain JSON-able object (vectors as number arrays). */
  serialize(): { chunks: Array<{ id: string; text: string; vector: number[]; meta?: Record<string, unknown> }> } {
    return {
      chunks: this.chunks.map((c) => ({ id: c.id, text: c.text, vector: Array.from(c.vector), meta: c.meta })),
    };
  }

  static restore(data: ReturnType<MemoryIndex['serialize']>): MemoryIndex {
    const idx = new MemoryIndex();
    for (const c of data.chunks) {
      idx.add({ id: c.id, text: c.text, vector: Float32Array.from(c.vector), meta: c.meta });
    }
    return idx;
  }
}

/** Split text into ~size-char chunks on sentence-ish boundaries. */
export function chunkText(text: string, size = 500, overlap = 50): string[] {
  const out: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + size, text.length);
    if (end < text.length) {
      const cut = text.lastIndexOf('. ', end);
      if (cut > start + size / 2) end = cut + 1;
    }
    out.push(text.slice(start, end).trim());
    start = end - overlap > start ? end - overlap : end;
  }
  return out.filter(Boolean);
}
