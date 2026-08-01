// browser-llm-nexus — run LLMs in the browser, GPU or CPU, same API.
//
// Three independently portable artifacts, each with its own export/import:
//   • chat model     — exportModel / importModel / NexusChat.fromArchive
//   • embedding model— exportModel / importModel / NexusEmbedder.fromArchive
//   • RAG store      — exportIndex / importIndex
// NexusKnowledge composes all three into a single offline bundle.

export {
  NexusChat,
  type ToolHandler,
  type ToolSchema,
  type ChatMessage,
  type LoadOptions,
  type ChatOptions,
  type ToolCallCheck,
} from './chat.ts';

export { NexusEmbedder, similarity, type EmbedOptions } from './embed.ts';
export { resolveSource, describeSource, dtypeProbe, hubRoot, type ModelSource } from './source.ts';

export {
  MemoryIndex,
  chunkText,
  indexToFiles,
  indexFromFiles,
  exportIndex,
  importIndex,
  type Chunk,
  type SearchHit,
} from './rag.ts';

export {
  exportModel,
  importModel,
  inspectModel,
  MODEL_FILES,
  ONNX_FILES,
  type ModelManifest,
  type ExportModelOptions,
  type ImportModelOptions,
} from './model.ts';

export {
  NexusKnowledge,
  type KnowledgeOptions,
  type KnowledgeDoc,
  type KnowledgeManifest,
  type ExportKnowledgeOptions,
} from './knowledge.ts';

export {
  exportCache,
  importCache,
  toManifest,
  fromManifest,
  CACHE_NAME,
  type CacheEntry,
} from './bundle.ts';

export {
  filesToZip,
  filesFromZip,
  prefixFiles,
  stripPrefix,
  readSource,
  encodeJSON,
  decodeJSON,
  requireFile,
  type ZipLike,
  type ZipOptions,
  type ArchiveSource,
} from './archive.ts';

export { Metrics, type MetricEvent } from './metrics.ts';
export { parseToolCalls, stripThinking, type ToolCall } from './toolcalls.ts';
export { Hooks } from './hooks.ts';
export {
  resolveTransformers,
  detectDtype,
  detectDevice,
  preferredDtypeOrder,
  DTYPE_FILES,
  DTYPE_ORDER,
  type TransformersLike,
  type RuntimeOptions,
  type Device,
} from './runtime.ts';
