/** Shared runtime plumbing: the injectable Transformers.js implementation and
 *  backend selection. Nothing here assumes a particular model host — where a
 *  model comes from is always stated explicitly (see source.ts). */

export interface TransformersLike {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<any>;
  env: any;
  TextStreamer?: new (tokenizer: any, opts: Record<string, unknown>) => unknown;
}

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

export interface RuntimeOptions {
  /** Bring your own transformers implementation (full build, lite build, or a
   *  custom one). Defaults to loading it from a CDN. */
  transformers?: TransformersLike;
}

export async function resolveTransformers(opts: RuntimeOptions = {}): Promise<TransformersLike> {
  return opts.transformers ?? ((await import(/* @vite-ignore */ CDN)) as TransformersLike);
}

export type Device = 'auto' | 'webgpu' | 'wasm' | 'cpu' | (string & {});

/** Pick the fastest available backend: WebGPU when the browser exposes a usable
 *  adapter, otherwise WASM (CPU). Everything in this library works on both —
 *  GPU is an accelerator, never a requirement. */
export async function detectDevice(preferred: Device = 'auto'): Promise<string> {
  if (preferred !== 'auto') return preferred;
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)?.gpu;
  if (gpu) {
    try {
      if (await gpu.requestAdapter()) return 'webgpu';
    } catch { /* fall through to wasm */ }
  }
  return 'wasm';
}

export const DTYPE_FILES: Record<string, string> = {
  q4: 'model_q4.onnx',
  q8: 'model_quantized.onnx',
  fp16: 'model_fp16.onnx',
  fp32: 'model.onnx',
};
export const DTYPE_ORDER = ['q4', 'q8', 'fp16', 'fp32'] as const;

/** dtype to prefer on a given backend.
 *
 *  q4 first on BOTH backends. WebGPU used to prefer fp16 — the conventional
 *  choice, since a GPU has the memory bandwidth for it — and that made small
 *  models unreliable at the one job they are good for. Measured on
 *  Qwen2.5-0.5B-Instruct over three tool-calling questions
 *  (`npm run test:models`):
 *
 *    q4    3/3 called and answered from the result
 *    fp16  3/3 called, 2/3 correct — it read 1096637 back as "109,663,700"
 *    q8    0/3 — narrated ("I would need to use a specific tool") and never called
 *
 *  A faster answer that silently corrupts a tool result is worse than a slower
 *  correct one, so correctness picks the default and fp16 stays one explicit
 *  option away. Larger models may well prefer fp16; state it when you know. */
export function preferredDtypeOrder(device: string): readonly string[] {
  return device === 'webgpu' ? (['q4', 'fp16', 'q8', 'fp32'] as const) : DTYPE_ORDER;
}

/** Builds the location of one of a model's dtype files. Sources differ in
 *  layout — a served folder puts them at `<base><id>/onnx/…`, the Hub at
 *  `<host><repo>/resolve/<revision>/onnx/…` — so the source decides, not this
 *  module. See `dtypeProbe` in source.ts. */
export type DtypeProbe = (file: string) => string;

/** Probe which dtype variants exist for a model and return the best one for
 *  this backend. Works for any host serving the standard onnx/ layout.
 *
 *  Without a `probe`, falls back to the local-model base — correct for `base`
 *  and `archive` sources, which point `env.localModelPath` at their files. A
 *  `hub` source has no local base and must pass one. */
export async function detectDtype(
  tjs: TransformersLike,
  modelId: string,
  device = 'wasm',
  probe?: DtypeProbe,
): Promise<string> {
  return (await availableDtypes(tjs, modelId, device, probe))[0]!;
}

/** Every dtype variant the source actually serves, best-first.
 *
 *  This answers "what is on the host", which is NOT the same question as "what
 *  works". Availability is knowable from a HEAD request; whether a model can
 *  call a tool at a given quantization is only knowable by running it — the two
 *  come apart badly, and in both directions (Qwen2.5-0.5B calls tools at q4 and
 *  not at q8; Qwen3-0.6B is the reverse). `detectDtype` takes the first of
 *  these and hopes; {@link NexusChat.loadForTools} walks the whole list and
 *  verifies. */
export async function availableDtypes(
  tjs: TransformersLike,
  modelId: string,
  device = 'wasm',
  probe?: DtypeProbe,
): Promise<string[]> {
  const base: string | undefined = tjs.env.localModelPath;
  if (!probe && !base) throw new Error('cannot probe dtypes without a base URL — pass an explicit dtype');
  const at: DtypeProbe =
    probe ?? ((file) => `${base}${base!.endsWith('/') ? '' : '/'}${modelId}/onnx/${file}`);
  const found: string[] = [];
  for (const d of preferredDtypeOrder(device)) {
    const path = at(DTYPE_FILES[d]!);
    const httpBase = /^https?:\/\//.test(path);
    try {
      if (httpBase) {
        const res = await fetch(path, { method: 'HEAD' });
        if (res.ok) found.push(d);
      } else {
        // Filesystem base (Node): probing over fetch would not work.
        // Specifier via a variable so bundlers and browser builds ignore it.
        const nodeFs = 'node:fs/promises';
        const { stat } = (await import(/* @vite-ignore */ nodeFs)) as { stat: (p: string) => Promise<unknown> };
        await stat(path);
        found.push(d);
      }
    } catch { /* keep probing */ }
  }
  if (!found.length) throw new Error(`no dtype variant found for ${modelId} under ${at('onnx/')}`);
  return found;
}
