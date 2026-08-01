/** Zip plumbing shared by every portable artifact in this library.
 *
 *  Everything serializes to a flat `Map<path, bytes>` first — that map can be
 *  zipped, written to OPFS, uploaded, or nested inside a bigger archive under
 *  a prefix. Binary stays binary the whole way: a Float32 vector costs 4 bytes
 *  per dimension here, versus ~10 as a JSON number.
 */

/** A zip implementation. fflate's `zip`/`unzip` match this shape. */
export interface ZipLike {
  zip: (
    files: Record<string, Uint8Array>,
    opts: Record<string, unknown>,
    cb: (err: Error | null, data: Uint8Array) => void,
  ) => void;
  unzip: (
    data: Uint8Array,
    cb: (err: Error | null, files: Record<string, Uint8Array>) => void,
  ) => void;
}

export interface ZipOptions {
  /** Inject fflate (or anything with the same zip/unzip shape). Defaults to CDN. */
  zip?: ZipLike;
  /** Deflate level 0-9. Default 0 — model weights and vectors are already
   *  dense, so compressing them burns CPU for almost nothing. Raise it for
   *  text-heavy archives. */
  level?: number;
}

/** Anything you can hand to an importer: bytes, a fetched buffer, a File from
 *  an <input type="file">, or a URL to fetch. */
export type ArchiveSource = Uint8Array | ArrayBuffer | Blob | string;

const FFLATE_CDN = 'https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js';

async function resolveZip(zip?: ZipLike): Promise<ZipLike> {
  return zip ?? ((await import(/* @vite-ignore */ FFLATE_CDN)) as unknown as ZipLike);
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export const encodeJSON = (v: unknown): Uint8Array => encoder.encode(JSON.stringify(v));
export const decodeJSON = <T>(b: Uint8Array): T => JSON.parse(decoder.decode(b)) as T;

/** Read `source` into bytes — fetching it first when given a URL. */
export async function readSource(source: ArchiveSource): Promise<Uint8Array> {
  if (typeof source === 'string') {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`fetch ${source} failed: ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  return new Uint8Array(await source.arrayBuffer());
}

export async function filesToZip(
  files: Map<string, Uint8Array>,
  opts: ZipOptions = {},
): Promise<Uint8Array> {
  const impl = await resolveZip(opts.zip);
  const record: Record<string, Uint8Array> = {};
  for (const [name, data] of files) record[name] = data;
  const level = opts.level ?? 0;
  return new Promise((resolve, reject) => {
    impl.zip(record, { level }, (err, data) => (err ? reject(err) : resolve(data)));
  });
}

export async function filesFromZip(
  source: ArchiveSource,
  opts: ZipOptions = {},
): Promise<Map<string, Uint8Array>> {
  const impl = await resolveZip(opts.zip);
  const bytes = await readSource(source);
  const record = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    impl.unzip(bytes, (err, f) => (err ? reject(err) : resolve(f)));
  });
  return new Map(Object.entries(record));
}

/** Nest a file map under a prefix (used to compose artifacts into one archive). */
export function prefixFiles(files: Map<string, Uint8Array>, prefix: string): Map<string, Uint8Array> {
  const out = new Map<string, Uint8Array>();
  for (const [name, data] of files) out.set(`${prefix}${name}`, data);
  return out;
}

/** Pull one nested artifact back out of a composed archive. */
export function stripPrefix(files: Map<string, Uint8Array>, prefix: string): Map<string, Uint8Array> {
  const out = new Map<string, Uint8Array>();
  for (const [name, data] of files) {
    if (name.startsWith(prefix)) out.set(name.slice(prefix.length), data);
  }
  return out;
}

export function requireFile(files: Map<string, Uint8Array>, name: string): Uint8Array {
  const f = files.get(name);
  if (!f) throw new Error(`archive is missing ${name}`);
  return f;
}
