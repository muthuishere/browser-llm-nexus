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

/** dtype that actually works well on a given backend. WebGPU prefers fp16;
 *  WASM/CPU prefers the quantized variants. */
export function preferredDtypeOrder(device: string): readonly string[] {
  return device === 'webgpu' ? (['fp16', 'q4', 'q8', 'fp32'] as const) : DTYPE_ORDER;
}

/** Probe which dtype variants exist for a model and return the best one for
 *  this backend. Works for any host serving the standard onnx/ layout. */
export async function detectDtype(tjs: TransformersLike, modelId: string, device = 'wasm'): Promise<string> {
  const base: string | undefined = tjs.env.localModelPath;
  if (!base) throw new Error('cannot probe dtypes without a base URL — pass an explicit dtype');
  const httpBase = /^https?:\/\//.test(base);
  for (const d of preferredDtypeOrder(device)) {
    const path = `${base}${base.endsWith('/') ? '' : '/'}${modelId}/onnx/${DTYPE_FILES[d]}`;
    try {
      if (httpBase) {
        const res = await fetch(path, { method: 'HEAD' });
        if (res.ok) return d;
      } else {
        // Filesystem base (Node): probing over fetch would not work.
        // Specifier via a variable so bundlers and browser builds ignore it.
        const nodeFs = 'node:fs/promises';
        const { stat } = (await import(/* @vite-ignore */ nodeFs)) as { stat: (p: string) => Promise<unknown> };
        await stat(path);
        return d;
      }
    } catch { /* keep probing */ }
  }
  throw new Error(`no dtype variant found for ${modelId} under ${base}`);
}
