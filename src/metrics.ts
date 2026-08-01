/** Lightweight metrics collector: counters, timings, and derived rates.
 *  Chat/embed classes feed it automatically; read `summary()` any time or
 *  subscribe to the owning class's 'metric' hook. */
export interface MetricEvent {
  name: string;
  value: number;
  unit: 'ms' | 'count' | 'tokens' | 'tokens/s';
}

export class Metrics {
  counters = new Map<string, number>();
  timings = new Map<string, number[]>();

  count(name: string, by = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + by);
  }

  time(name: string, ms: number): void {
    if (!this.timings.has(name)) this.timings.set(name, []);
    this.timings.get(name)!.push(ms);
  }

  /** Time an async operation and record it under name. */
  async measure<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const t0 = Date.now();
    try {
      return await fn();
    } finally {
      this.time(name, Date.now() - t0);
    }
  }

  summary(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, v] of this.counters) out[k] = v;
    for (const [k, arr] of this.timings) {
      const total = arr.reduce((a, b) => a + b, 0);
      out[`${k}_ms_total`] = total;
      out[`${k}_ms_avg`] = Math.round(total / arr.length);
      out[`${k}_runs`] = arr.length;
    }
    const genMs = this.timings.get('generate')?.reduce((a, b) => a + b, 0) ?? 0;
    const tokens = this.counters.get('tokens_out') ?? 0;
    if (genMs > 0 && tokens > 0) out.tokens_per_second = Math.round((tokens / genMs) * 1000 * 10) / 10;
    return out;
  }

  reset(): void {
    this.counters.clear();
    this.timings.clear();
  }
}
