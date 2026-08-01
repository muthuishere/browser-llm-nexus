/** Export/import of the Transformers.js browser cache — the offline-bundle
 *  pattern: drain `caches.open('transformers-cache')` into portable entries,
 *  restore them on another machine so model loads make zero network calls.
 *  (Same contract offline-llm-knowledge-system uses for its embed-cache.) */

export interface CacheEntry {
  url: string;
  data: ArrayBuffer;
}

const CACHE_NAME = 'transformers-cache';

/** Drain the Transformers.js cache into portable entries.
 *  Pass a filter to keep only some URLs (default: everything). */
export async function exportCache(filter: (url: string) => boolean = () => true): Promise<CacheEntry[]> {
  const cache = await caches.open(CACHE_NAME);
  const entries: CacheEntry[] = [];
  for (const req of await cache.keys()) {
    if (!filter(req.url)) continue;
    const res = await cache.match(req);
    if (!res) continue;
    const data = await res.arrayBuffer();
    if (data.byteLength > 0) entries.push({ url: req.url, data });
  }
  return entries;
}

/** Restore entries into the Transformers.js cache under their original URLs. */
export async function importCache(entries: CacheEntry[]): Promise<number> {
  const cache = await caches.open(CACHE_NAME);
  for (const e of entries) {
    await cache.put(e.url, new Response(e.data));
  }
  return entries.length;
}

/** Manifest form: {file, url} pairs + binary blobs, ready for zipping
 *  (matches offline-llm-knowledge-system's embed-cache/index.json layout). */
export function toManifest(entries: CacheEntry[]): {
  index: Array<{ file: string; url: string }>;
  files: Map<string, ArrayBuffer>;
} {
  const index: Array<{ file: string; url: string }> = [];
  const files = new Map<string, ArrayBuffer>();
  entries.forEach((e, i) => {
    const file = `${i}.bin`;
    index.push({ file, url: e.url });
    files.set(file, e.data);
  });
  return { index, files };
}

export function fromManifest(
  index: Array<{ file: string; url: string }>,
  files: Map<string, ArrayBuffer>,
): CacheEntry[] {
  return index
    .filter((e) => files.has(e.file))
    .map((e) => ({ url: e.url, data: files.get(e.file)! }));
}
