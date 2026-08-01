/** Portable model artifacts — a chat model or an embedding model, packed as a
 *  zip you can move to a machine with no internet.
 *
 *    manifest.json   { kind: 'model', modelId, createdAt, files: [{file, url}] }
 *    files/0.bin     each file's bytes, in manifest order
 *
 *  Import restores every file into the Cache API under the URL the runtime
 *  will request, so the next `load()` makes zero network calls. That is the
 *  same mechanism offline-llm-knowledge-system uses for its embed model.
 */
import {
  encodeJSON,
  decodeJSON,
  filesToZip,
  filesFromZip,
  requireFile,
  type ArchiveSource,
  type ZipOptions,
} from './archive.ts';
import { CACHE_NAME } from './bundle.ts';

/** Files a converted Transformers.js model folder can contain. Missing ones
 *  are skipped, so this list can stay generous. */
export const MODEL_FILES = [
  'config.json',
  'generation_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'preprocessor_config.json',
  'vocab.json',
  'merges.txt',
  'added_tokens.json',
  'chat_template.jinja',
] as const;

export const ONNX_FILES = [
  'onnx/model_q4.onnx',
  'onnx/model_quantized.onnx',
  'onnx/model_fp16.onnx',
  'onnx/model.onnx',
  'onnx/model.onnx_data',
] as const;

export interface ModelManifest {
  kind: 'model';
  modelId: string;
  createdAt: string;
  /** Which onnx variants are inside. */
  dtypes: string[];
  files: Array<{ file: string; url: string; path: string }>;
}

export interface ExportModelOptions extends ZipOptions {
  /** Where the model is served from. Default '/models/'. */
  modelsUrl?: string;
  /** Full base URL to fetch files from, overriding `modelsUrl + modelId`.
   *  Use `hfRoot(repo)` to pack straight from the Hugging Face Hub. */
  root?: string;
  /** Only include these dtypes (e.g. ['q4']). Default: every variant present. */
  dtypes?: string[];
  /** Extra relative paths to include. */
  extraFiles?: string[];
  onProgress?: (file: string, index: number, total: number) => void;
}

const DTYPE_OF: Record<string, string> = {
  'onnx/model_q4.onnx': 'q4',
  'onnx/model_quantized.onnx': 'q8',
  'onnx/model_fp16.onnx': 'fp16',
  'onnx/model.onnx': 'fp32',
  'onnx/model.onnx_data': 'fp32',
};

function baseUrl(modelsUrl?: string): string {
  const base = modelsUrl ?? '/models/';
  return new URL(base, typeof location !== 'undefined' ? location.href : 'file:///').href;
}

/** Pack a served model into a portable zip. */
export async function exportModel(modelId: string, opts: ExportModelOptions = {}): Promise<Uint8Array> {
  const root = `${baseUrl(opts.modelsUrl)}${modelId}/`;
  const wanted = [
    ...MODEL_FILES,
    ...ONNX_FILES.filter((f) => !opts.dtypes || opts.dtypes.includes(DTYPE_OF[f]!)),
    ...(opts.extraFiles ?? []),
  ];

  const files = new Map<string, Uint8Array>();
  const entries: ModelManifest['files'] = [];
  const dtypes = new Set<string>();

  for (const [i, path] of wanted.entries()) {
    const url = root + path;
    opts.onProgress?.(path, i, wanted.length);
    let res: Response;
    try {
      res = await fetch(url);
    } catch {
      continue;
    }
    if (!res.ok) continue;
    const data = new Uint8Array(await res.arrayBuffer());
    if (!data.byteLength) continue;
    const name = `files/${entries.length}.bin`;
    files.set(name, data);
    entries.push({ file: name, url, path });
    if (DTYPE_OF[path]) dtypes.add(DTYPE_OF[path]!);
  }

  if (!entries.length) throw new Error(`no model files found under ${root}`);

  const manifest: ModelManifest = {
    kind: 'model',
    modelId,
    createdAt: new Date().toISOString(),
    dtypes: [...dtypes],
    files: entries,
  };
  files.set('manifest.json', encodeJSON(manifest));
  return filesToZip(files, opts);
}

export interface ImportModelOptions extends ZipOptions {
  /** Re-point the recorded URLs (e.g. when the app is served elsewhere now). */
  rewriteUrl?: (url: string, path: string, modelId: string) => string;
  /** Also register under this base, so `load()` from a local path hits cache. */
  modelsUrl?: string;
}

/** Restore a model zip into the browser cache. After this, loading the model
 *  needs no network. Returns the manifest so callers know what they got. */
export async function importModel(
  source: ArchiveSource,
  opts: ImportModelOptions = {},
): Promise<ModelManifest> {
  const files = await filesFromZip(source, opts);
  const manifest = decodeJSON<ModelManifest>(requireFile(files, 'manifest.json'));
  if (manifest.kind !== 'model') throw new Error(`not a model archive (kind=${manifest.kind})`);

  const cache = await caches.open(CACHE_NAME);
  const localRoot = opts.modelsUrl ? `${baseUrl(opts.modelsUrl)}${manifest.modelId}/` : null;

  for (const entry of manifest.files) {
    const data = files.get(entry.file);
    if (!data) continue;
    const urls = new Set<string>([
      opts.rewriteUrl ? opts.rewriteUrl(entry.url, entry.path, manifest.modelId) : entry.url,
    ]);
    if (localRoot) urls.add(localRoot + entry.path);
    for (const url of urls) {
      // slice() gives a standalone ArrayBuffer — a Uint8Array view isn't a valid BodyInit.
      await cache.put(url, new Response(data.slice().buffer, { status: 200 }));
    }
  }
  return manifest;
}

/** Read a model archive's manifest without restoring anything. */
export async function inspectModel(source: ArchiveSource, opts: ZipOptions = {}): Promise<ModelManifest> {
  const files = await filesFromZip(source, opts);
  return decodeJSON<ModelManifest>(requireFile(files, 'manifest.json'));
}
