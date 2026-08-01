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

/** Probe which dtype variants exist for a converted model and return the best. */
export async function detectDtype(tjs: TransformersLike, modelId: string): Promise<string> {
  for (const d of DTYPE_ORDER) {
    try {
      const res = await fetch(`${tjs.env.localModelPath}${modelId}/onnx/${DTYPE_FILES[d]}`, { method: 'HEAD' });
      if (res.ok) return d;
    } catch { /* keep probing */ }
  }
  throw new Error(`no converted dtype found for ${modelId} under ${tjs.env.localModelPath}`);
}
