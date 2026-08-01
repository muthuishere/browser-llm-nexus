/** Where a model comes from — always stated, never guessed.
 *
 * There is no default location and no "/models/" convention: a caller must say
 * whether the weights come from a portable archive, a folder served somewhere,
 * or the Hugging Face Hub. Nothing in this library assumes a particular server.
 */
import { importModel } from './model.ts';
import type { ArchiveSource, ZipLike } from './archive.ts';
import type { DtypeProbe, TransformersLike } from './runtime.ts';

export type ModelSource =
  /** A portable model archive: a URL, a File from an <input>, a Blob, or bytes.
   *  Restored into the browser cache, so loading needs no further network. */
  | { archive: ArchiveSource; zip?: ZipLike }
  /** A model folder served over HTTP: files live at `${base}${id}/config.json` etc. */
  | { base: string; id: string }
  /** The Hugging Face Hub, by repo id. */
  | { hub: string };

const withSlash = (u: string): string => (u.endsWith('/') ? u : `${u}/`);

const inBrowser = (): boolean => typeof location !== 'undefined';

/** In a browser, a base is a URL. Off-browser (Node, tests) a non-URL base is a
 *  filesystem path and must stay one — Transformers.js reads it with fs. */
const absolute = (u: string): string => {
  if (!inBrowser() && !/^https?:\/\//.test(u)) return u;
  return new URL(u, inBrowser() ? location.href : 'file:///').href;
};

/** Point the runtime at `source` and return the model id to load. */
export async function resolveSource(tjs: TransformersLike, source: ModelSource): Promise<string> {
  if ('hub' in source) {
    tjs.env.allowRemoteModels = true;
    tjs.env.allowLocalModels = false;
    return source.hub;
  }

  if ('base' in source) {
    tjs.env.allowLocalModels = true;
    tjs.env.allowRemoteModels = false;
    tjs.env.localModelPath = withSlash(absolute(source.base));
    return source.id;
  }

  // Archive: restore it, then serve the model from the URLs it recorded so the
  // runtime's requests hit the cache entries we just wrote.
  const manifest = await importModel(source.archive, { zip: source.zip });
  const first = manifest.files[0];
  if (!first) throw new Error(`model archive for ${manifest.modelId} is empty`);

  const root = first.url.slice(0, first.url.length - first.path.length); // .../<id>/
  const base = root.slice(0, root.length - (manifest.modelId.length + 1)); // .../
  tjs.env.allowLocalModels = true;
  tjs.env.allowRemoteModels = false;
  tjs.env.localModelPath = withSlash(base);
  return manifest.modelId;
}

/** Where to probe for a model's dtype variants.
 *
 *  `base` and `archive` sources point `env.localModelPath` at their files, so
 *  the default probe already finds them. A `hub` source deliberately leaves
 *  that path alone — its files live at `<host><repo>/resolve/<revision>/`, which
 *  no local base ever points at — so it must say where to look. */
export function dtypeProbe(source: ModelSource, tjs: TransformersLike): DtypeProbe | undefined {
  if (!('hub' in source)) return undefined;
  const host: string = tjs.env.remoteHost ?? 'https://huggingface.co/';
  const template: string = tjs.env.remotePathTemplate ?? '{model}/resolve/{revision}/';
  const repoPath = template.replace('{model}', source.hub).replace('{revision}', 'main');
  return (file) => `${withSlash(host)}${repoPath}onnx/${file}`;
}

/** Human-readable description of a source, for logs and metrics. */
export function describeSource(source: ModelSource): string {
  if ('hub' in source) return `hub:${source.hub}`;
  if ('base' in source) return `${withSlash(source.base)}${source.id}`;
  return typeof source.archive === 'string' ? `archive:${source.archive}` : 'archive:<file>';
}

/** Base URL for a Hugging Face repo's files — handy with `exportModel`. */
export const hubRoot = (repo: string, revision = 'main'): string =>
  `https://huggingface.co/${repo}/resolve/${revision}/`;
