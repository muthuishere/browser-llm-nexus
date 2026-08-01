import { Hooks } from './hooks.ts';
import { Metrics } from './metrics.ts';
import { NexusChat, type LoadOptions } from './chat.ts';
import { NexusEmbedder, type EmbedOptions } from './embed.ts';
import { MemoryIndex, chunkText, type Chunk } from './rag.ts';
import { exportCache, importCache, type CacheEntry } from './bundle.ts';

export interface KnowledgeOptions {
  chat: string | NexusChat;
  embedder?: string | NexusEmbedder;
  chatOptions?: LoadOptions;
  embedOptions?: EmbedOptions;
  /** Chunking: characters per chunk and overlap. */
  chunkSize?: number;
  chunkOverlap?: number;
  /** How many chunks to retrieve per question. */
  topK?: number;
}

export interface KnowledgeDoc {
  id: string;
  title?: string;
  text: string;
  meta?: Record<string, unknown>;
}

/** A portable knowledge bundle: the index plus (optionally) the model weights
 *  captured from the browser cache, so the whole thing works air-gapped. */
export interface KnowledgeBundle {
  version: 1;
  createdAt: string;
  models: { chat: string; embedder: string };
  index: ReturnType<MemoryIndex['serialize']>;
  docs: Array<Omit<KnowledgeDoc, 'text'> & { text?: string }>;
  cache?: Array<{ url: string; data: number[] }>;
}

type KnowledgeEvents = {
  indexing: [string, number];
  indexed: [string, number];
  retrieved: [Chunk[], string];
  token: [string];
  answer: [string];
};

const DEFAULT_EMBEDDER = 'Xenova/bge-small-en-v1.5';

/**
 * Offline knowledge system in one object: documents in, grounded answers out.
 * Runs on WebGPU when available, CPU/WASM otherwise — same API either way.
 *
 *   const kb = await NexusKnowledge.create({ chat: 'Qwen/Qwen3-0.6B' });
 *   await kb.addDocument({ id: 'handbook', text: handbookText });
 *   const answer = await kb.ask('What is the refund policy?');
 *
 *   const bundle = await kb.export({ includeModels: true });  // ship it offline
 *   const kb2 = await NexusKnowledge.import(bundle);
 */
export class NexusKnowledge extends Hooks<KnowledgeEvents> {
  readonly metrics = new Metrics();
  readonly index = new MemoryIndex();
  readonly docs = new Map<string, KnowledgeDoc>();

  chunkSize: number;
  chunkOverlap: number;
  topK: number;

  private constructor(
    readonly chat: NexusChat,
    readonly embedder: NexusEmbedder,
    readonly modelIds: { chat: string; embedder: string },
    opts: KnowledgeOptions,
  ) {
    super();
    this.chunkSize = opts.chunkSize ?? 500;
    this.chunkOverlap = opts.chunkOverlap ?? 50;
    this.topK = opts.topK ?? 5;
  }

  static async create(opts: KnowledgeOptions): Promise<NexusKnowledge> {
    const chatId = typeof opts.chat === 'string' ? opts.chat : '(provided)';
    const chat =
      typeof opts.chat === 'string' ? await NexusChat.load(opts.chat, opts.chatOptions) : opts.chat;

    const embedderSpec = opts.embedder ?? DEFAULT_EMBEDDER;
    const embedderId = typeof embedderSpec === 'string' ? embedderSpec : '(provided)';
    const embedder =
      typeof embedderSpec === 'string'
        ? await NexusEmbedder.load(embedderSpec, opts.embedOptions)
        : embedderSpec;

    const kb = new NexusKnowledge(chat, embedder, { chat: chatId, embedder: embedderId }, opts);
    kb.chat.on('token', (t) => kb.emit('token', t));
    return kb;
  }

  /** Chunk, embed and index a document. */
  async addDocument(doc: KnowledgeDoc): Promise<number> {
    this.docs.set(doc.id, doc);
    const texts = chunkText(doc.text, this.chunkSize, this.chunkOverlap);
    this.emit('indexing', doc.id, texts.length);
    const vectors = await this.metrics.measure('index', () => this.embedder.embedBatch(texts));
    texts.forEach((text, i) =>
      this.index.add({
        id: `${doc.id}#${i}`,
        text,
        vector: vectors[i]!,
        meta: { docId: doc.id, title: doc.title, ...doc.meta },
      }),
    );
    this.metrics.count('chunks_indexed', texts.length);
    this.emit('indexed', doc.id, texts.length);
    return texts.length;
  }

  async addDocuments(docs: KnowledgeDoc[]): Promise<number> {
    let total = 0;
    for (const d of docs) total += await this.addDocument(d);
    return total;
  }

  /** Retrieve the chunks most relevant to a question. */
  async retrieve(question: string, k = this.topK): Promise<Chunk[]> {
    const qv = await this.embedder.embed(question);
    const chunks = this.index.search(qv, k).map((h) => h.chunk);
    this.emit('retrieved', chunks, question);
    return chunks;
  }

  /**
   * Retrieval-augmented answer. Context goes in the user turn — small models
   * attend to it far more reliably than to a system prompt.
   */
  async ask(question: string, opts: { k?: number; maxNewTokens?: number } = {}): Promise<string> {
    const chunks = await this.retrieve(question, opts.k ?? this.topK);
    this.metrics.count('questions');
    const context = chunks.map((c) => c.text).join('\n\n');
    const prompt = context
      ? `Use the context below to answer. If the context does not contain the answer, say so.\n\nContext:\n${context}\n\nQuestion: ${question}`
      : question;
    const answer = await this.chat.chat(prompt, { maxNewTokens: opts.maxNewTokens });
    this.emit('answer', answer);
    return answer;
  }

  /** Serialize to a portable bundle. With includeModels, also captures the
   *  browser's model cache so the bundle runs with no network at all. */
  async export(opts: { includeModels?: boolean; includeText?: boolean } = {}): Promise<KnowledgeBundle> {
    const bundle: KnowledgeBundle = {
      version: 1,
      createdAt: new Date().toISOString(),
      models: this.modelIds,
      index: this.index.serialize(),
      docs: [...this.docs.values()].map((d) => ({
        id: d.id,
        title: d.title,
        meta: d.meta,
        ...(opts.includeText === false ? {} : { text: d.text }),
      })),
    };
    if (opts.includeModels) {
      const entries = await exportCache();
      bundle.cache = entries.map((e) => ({ url: e.url, data: Array.from(new Uint8Array(e.data)) }));
    }
    return bundle;
  }

  /** Restore a bundle: rehydrates the index (and model cache, if bundled)
   *  without re-embedding anything. */
  static async import(bundle: KnowledgeBundle, opts: Partial<KnowledgeOptions> = {}): Promise<NexusKnowledge> {
    if (bundle.cache?.length) {
      const entries: CacheEntry[] = bundle.cache.map((c) => ({
        url: c.url,
        data: Uint8Array.from(c.data).buffer,
      }));
      await importCache(entries);
    }
    const kb = await NexusKnowledge.create({
      chat: opts.chat ?? bundle.models.chat,
      embedder: opts.embedder ?? bundle.models.embedder,
      ...opts,
    });
    kb.index.addAll([...MemoryIndex.restore(bundle.index).all()]);
    for (const d of bundle.docs) kb.docs.set(d.id, { ...d, text: d.text ?? '' });
    return kb;
  }

  async dispose(): Promise<void> {
    await Promise.all([this.chat.dispose(), this.embedder.dispose()]);
  }
}
