import { Metrics } from './metrics.ts';
import { resolveTransformers, detectDevice, type Device, type RuntimeOptions } from './runtime.ts';

export interface EmbedOptions extends RuntimeOptions {
  dtype?: string;
  /** 'auto' (default) uses WebGPU when available, else WASM/CPU. */
  device?: Device;
  onProgress?: (p: unknown) => void;
  /** Load from the HF Hub instead of local converted models. */
  remote?: boolean;
}

/** Embedding model wrapper (feature-extraction) with batching + similarity. */
export class NexusEmbedder {
  readonly metrics = new Metrics();

  private constructor(private extractor: any, readonly device: string) {}

  static async load(modelId: string, opts: EmbedOptions = {}): Promise<NexusEmbedder> {
    const tjs = await resolveTransformers(opts);
    if (opts.remote) {
      tjs.env.allowRemoteModels = true;
      tjs.env.allowLocalModels = false;
    }
    const device = await detectDevice(opts.device ?? 'auto');
    const extractor = await tjs.pipeline('feature-extraction', modelId, {
      dtype: opts.dtype ?? 'q8',
      device,
      progress_callback: opts.onProgress,
    });
    return new NexusEmbedder(extractor, device);
  }

  /** Embed one text into a normalized vector. */
  async embed(text: string): Promise<Float32Array> {
    return (await this.embedBatch([text]))[0]!;
  }

  /** Embed many texts; returns one normalized vector per text. */
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const out: any = await this.metrics.measure('embed', () =>
      this.extractor(texts, { pooling: 'mean', normalize: true }),
    );
    this.metrics.count('texts_embedded', texts.length);
    const [n, dim] = [out.dims[0] as number, out.dims[1] as number];
    const data = out.data as Float32Array;
    return Array.from({ length: n }, (_, i) => data.slice(i * dim, (i + 1) * dim));
  }

  dispose(): Promise<void> {
    return this.extractor.dispose();
  }
}

/** Cosine similarity of two normalized vectors (= dot product). */
export function similarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}
