// browser-llm-nexus — run LLMs in the browser on plain CPU.
// Tool calling, embeddings, RAG, offline bundles, metrics. Hooks everywhere.
export { NexusChat, type ToolHandler, type ToolSchema, type ChatMessage, type LoadOptions, type ChatOptions } from './chat.ts';
export { NexusEmbedder, similarity, type EmbedOptions } from './embed.ts';
export { MemoryIndex, chunkText, type Chunk, type SearchHit } from './rag.ts';
export { NexusKnowledge, type KnowledgeOptions, type KnowledgeDoc, type KnowledgeBundle } from './knowledge.ts';
export { exportCache, importCache, toManifest, fromManifest, type CacheEntry } from './bundle.ts';
export { Metrics, type MetricEvent } from './metrics.ts';
export { parseToolCalls, stripThinking, type ToolCall } from './toolcalls.ts';
export { Hooks } from './hooks.ts';
export { resolveTransformers, detectDtype, detectDevice, preferredDtypeOrder, DTYPE_FILES, DTYPE_ORDER, type TransformersLike, type RuntimeOptions, type Device } from './runtime.ts';
