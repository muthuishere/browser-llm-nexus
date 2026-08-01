/** Shared runtime plumbing: the injectable Transformers.js implementation and
 *  local-model environment setup. Users can pass the full @huggingface/transformers,
 *  a lite build, or anything shape-compatible. */

export interface TransformersLike {
  pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<any>;
  env: any;
  TextStreamer?: new (tokenizer: any, opts: Record<string, unknown>) => unknown;
}

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

export interface RuntimeOptions {
  /** Bring your own transformers implementation (full, lite, custom). Defaults to CDN import. */
  transformers?: TransformersLike;
  /** Base URL/path the converted model folders are served from. Default '/models/'. */
  modelsUrl?: string;
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

/** dtype that actually works well on a given backend. WebGPU prefers fp16;
 *  WASM/CPU prefers the quantized variants. */
export function preferredDtypeOrder(device: string): readonly string[] {
  return device === 'webgpu' ? (['fp16', 'q4', 'q8', 'fp32'] as const) : DTYPE_ORDER;
}

export async function resolveTransformers(opts: RuntimeOptions): Promise<TransformersLike> {
  const tjs = opts.transformers ?? ((await import(/* @vite-ignore */ CDN)) as TransformersLike);
  const base = opts.modelsUrl ?? '/models/';
  tjs.env.allowLocalModels = true;
  tjs.env.allowRemoteModels = false;
  tjs.env.localModelPath = new URL(base, typeof location !== 'undefined' ? location.href : 'file:///').href;
  return tjs;
}

export const DTYPE_FILES: Record<string, string> = {
  q4: 'model_q4.onnx',
  q8: 'model_quantized.onnx',
  fp16: 'model_fp16.onnx',
  fp32: 'model.onnx',
};
export const DTYPE_ORDER = ['q4', 'q8', 'fp16', 'fp32'] as const;

/** Probe which dtype variants exist for a converted model and return the best
 *  one for the given backend. */
export async function detectDtype(tjs: TransformersLike, modelId: string, device = 'wasm'): Promise<string> {
  for (const d of preferredDtypeOrder(device)) {
    try {
      const res = await fetch(`${tjs.env.localModelPath}${modelId}/onnx/${DTYPE_FILES[d]}`, { method: 'HEAD' });
      if (res.ok) return d;
    } catch { /* keep probing */ }
  }
  throw new Error(`no converted dtype found for ${modelId} under ${tjs.env.localModelPath}`);
}
