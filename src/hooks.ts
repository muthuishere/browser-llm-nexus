/** Tiny typed hooks mixin: on/off/emit. Every nexus class extends this so
 *  consumers can subscribe (`nexus.on('token', fn)`) instead of threading
 *  callbacks through every call. */
type Listener = (...args: unknown[]) => void;

export class Hooks<Events extends Record<string, unknown[]>> {
  #listeners = new Map<keyof Events, Set<Listener>>();

  /** Subscribe. Returns an unsubscribe function. */
  on<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): () => void {
    if (!this.#listeners.has(event)) this.#listeners.set(event, new Set());
    this.#listeners.get(event)!.add(fn as Listener);
    return () => this.off(event, fn);
  }

  off<K extends keyof Events>(event: K, fn: (...args: Events[K]) => void): void {
    this.#listeners.get(event)?.delete(fn as Listener);
  }

  emit<K extends keyof Events>(event: K, ...args: Events[K]): void {
    for (const fn of this.#listeners.get(event) ?? []) {
      try {
        fn(...args);
      } catch (e) {
        console.error(`hook ${String(event)} threw`, e);
      }
    }
  }
}
