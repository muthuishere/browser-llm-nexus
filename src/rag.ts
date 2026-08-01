import { similarity } from './embed.ts';
import {
  encodeJSON,
  decodeJSON,
  filesToZip,
  filesFromZip,
  requireFile,
  type ArchiveSource,
  type ZipOptions,
} from './archive.ts';

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

  /** All indexed chunks (insertion order). */
  all(): readonly Chunk[] {
    return this.chunks;
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

/** Serialize an index to a flat file map — vectors stay raw Float32.
 *
 *    manifest.json  { kind: 'rag', count, dims, createdAt }
 *    chunks.json    [{ id, text, meta }]
 *    vectors.bin    Float32 matrix, row-major
 */
export function indexToFiles(index: MemoryIndex): Map<string, Uint8Array> {
  const chunks = index.all();
  const dims = chunks[0]?.vector.length ?? 0;
  const files = new Map<string, Uint8Array>();

  files.set(
    'manifest.json',
    encodeJSON({ kind: 'rag', count: chunks.length, dims, createdAt: new Date().toISOString() }),
  );
  files.set('chunks.json', encodeJSON(chunks.map((c) => ({ id: c.id, text: c.text, meta: c.meta }))));

  const matrix = new Float32Array(chunks.length * dims);
  chunks.forEach((c, i) => matrix.set(c.vector, i * dims));
  files.set('vectors.bin', new Uint8Array(matrix.buffer));
  return files;
}

export function indexFromFiles(files: Map<string, Uint8Array>): MemoryIndex {
  const manifest = decodeJSON<{ dims: number }>(requireFile(files, 'manifest.json'));
  const meta = decodeJSON<Array<{ id: string; text: string; meta?: Record<string, unknown> }>>(
    requireFile(files, 'chunks.json'),
  );
  const raw = requireFile(files, 'vectors.bin');
  // Copy first: the zip reader's buffer may not be 4-byte aligned.
  const matrix = new Float32Array(raw.slice().buffer);
  const dims = manifest.dims;

  const index = new MemoryIndex();
  meta.forEach((c, i) => {
    index.add({
      id: c.id,
      text: c.text,
      meta: c.meta,
      vector: matrix.slice(i * dims, (i + 1) * dims),
    });
  });
  return index;
}

/** Pack a vector store into a portable zip. */
export function exportIndex(index: MemoryIndex, opts: ZipOptions = {}): Promise<Uint8Array> {
  return filesToZip(indexToFiles(index), { level: 6, ...opts });
}

/** Read a vector store back from a zip, URL, File or bytes. */
export async function importIndex(source: ArchiveSource, opts: ZipOptions = {}): Promise<MemoryIndex> {
  return indexFromFiles(await filesFromZip(source, opts));
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
