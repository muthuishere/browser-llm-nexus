import { Metrics } from './metrics.ts';
import { resolveTransformers, detectDevice, type Device, type RuntimeOptions } from './runtime.ts';
import { resolveSource, type ModelSource } from './source.ts';

export interface EmbedOptions extends RuntimeOptions {
  /** Quantization variant. Default q8 — a good size/quality point for embedders. */
  dtype?: string;
  /** 'auto' (default) uses WebGPU when available, else WASM/CPU. */
  device?: Device;
  onProgress?: (p: unknown) => void;
}

/** Embedding model wrapper (feature-extraction) with batching + similarity. */
export class NexusEmbedder {
  readonly metrics = new Metrics();

  private constructor(private extractor: any, readonly device: string, readonly modelId: string) {}

  /**
   * Load an embedding model from an explicit source:
   *
   *   NexusEmbedder.load({ hub: 'Xenova/bge-small-en-v1.5' })
   *   NexusEmbedder.load({ base: '/models/', id: 'BAAI/bge-small-en-v1.5' })
   *   NexusEmbedder.load({ archive: fileFromInput })
   */
  static async load(source: ModelSource, opts: EmbedOptions = {}): Promise<NexusEmbedder> {
    const tjs = await resolveTransformers(opts);
    const modelId = await resolveSource(tjs, source);
    const device = await detectDevice(opts.device ?? 'auto');
    const extractor = await tjs.pipeline('feature-extraction', modelId, {
      dtype: opts.dtype ?? 'q8',
      device,
      progress_callback: opts.onProgress,
    });
    return new NexusEmbedder(extractor, device, modelId);
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
