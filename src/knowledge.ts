import { Hooks } from './hooks.ts';
import { Metrics } from './metrics.ts';
import { NexusChat, type LoadOptions } from './chat.ts';
import { NexusEmbedder, type EmbedOptions } from './embed.ts';
import { MemoryIndex, chunkText, indexToFiles, indexFromFiles, type Chunk } from './rag.ts';
import { exportModel, importModel, type ExportModelOptions } from './model.ts';
import {
  encodeJSON,
  decodeJSON,
  filesToZip,
  filesFromZip,
  prefixFiles,
  stripPrefix,
  requireFile,
  type ArchiveSource,
  type ZipOptions,
} from './archive.ts';

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

export interface KnowledgeManifest {
  kind: 'knowledge';
  version: 1;
  createdAt: string;
  models: { chat: string; embedder: string };
  docs: Array<Omit<KnowledgeDoc, 'text'> & { text?: string }>;
  /** Which parts are actually inside this archive. */
  contains: { rag: boolean; chatModel: boolean; embedModel: boolean };
}

export interface ExportKnowledgeOptions extends ZipOptions {
  /** Bundle the chat model weights (big — makes the archive self-contained). */
  includeChatModel?: boolean;
  /** Bundle the embedding model weights. */
  includeEmbedModel?: boolean;
  /** Shorthand for both of the above. */
  includeModels?: boolean;
  /** Keep the original document text in the archive. Default true. */
  includeText?: boolean;
  /** Passed through when packing models (dtype filter, modelsUrl, progress). */
  modelOptions?: ExportModelOptions;
  onProgress?: (stage: string) => void;
}

type KnowledgeEvents = {
  indexing: [string, number];
  indexed: [string, number];
  retrieved: [Chunk[], string];
  token: [string];
  answer: [string];
};

const DEFAULT_EMBEDDER = 'Xenova/bge-small-en-v1.5';
const RAG_PREFIX = 'rag/';
const CHAT_MODEL_PREFIX = 'models/chat/';
const EMBED_MODEL_PREFIX = 'models/embedder/';

/**
 * Offline knowledge system: documents in, grounded answers out — and the whole
 * thing packs into one zip.
 *
 * This is a *composition* of three independently portable artifacts: the chat
 * model, the embedding model, and the RAG store. Each of those has its own
 * export/import (`exportModel`/`importModel`, `exportIndex`/`importIndex`) if
 * you want to ship them separately; this class just zips all three together.
 *
 *   const kb = await NexusKnowledge.create({ chat: 'Qwen/Qwen3-0.6B' });
 *   await kb.addDocument({ id: 'handbook', text: handbookText });
 *   const answer = await kb.ask('What is the refund policy?');
 *
 *   const zip = await kb.exportZip({ includeModels: true });
 *   const kb2 = await NexusKnowledge.importZip(zip);   // works offline
 */
export class NexusKnowledge extends Hooks<KnowledgeEvents> {
  readonly metrics = new Metrics();
  index = new MemoryIndex();
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

  /** Serialize to a flat file map: manifest + rag/ + optionally the models. */
  async toFiles(opts: ExportKnowledgeOptions = {}): Promise<Map<string, Uint8Array>> {
    const wantChat = opts.includeChatModel ?? opts.includeModels ?? false;
    const wantEmbed = opts.includeEmbedModel ?? opts.includeModels ?? false;

    const files = new Map<string, Uint8Array>();
    opts.onProgress?.('rag');
    for (const [name, data] of prefixFiles(indexToFiles(this.index), RAG_PREFIX)) {
      files.set(name, data);
    }

    if (wantChat) {
      opts.onProgress?.('chat model');
      const zip = await exportModel(this.modelIds.chat, { ...opts.modelOptions, zip: opts.zip });
      files.set(`${CHAT_MODEL_PREFIX}model.zip`, zip);
    }
    if (wantEmbed) {
      opts.onProgress?.('embedding model');
      const zip = await exportModel(this.modelIds.embedder, { ...opts.modelOptions, zip: opts.zip });
      files.set(`${EMBED_MODEL_PREFIX}model.zip`, zip);
    }

    const manifest: KnowledgeManifest = {
      kind: 'knowledge',
      version: 1,
      createdAt: new Date().toISOString(),
      models: this.modelIds,
      docs: [...this.docs.values()].map((d) => ({
        id: d.id,
        title: d.title,
        meta: d.meta,
        ...(opts.includeText === false ? {} : { text: d.text }),
      })),
      contains: { rag: true, chatModel: wantChat, embedModel: wantEmbed },
    };
    files.set('manifest.json', encodeJSON(manifest));
    return files;
  }

  /** Pack everything into one zip — the format to actually ship. */
  async exportZip(opts: ExportKnowledgeOptions = {}): Promise<Uint8Array> {
    return filesToZip(await this.toFiles(opts), { level: 6, ...opts });
  }

  /**
   * Restore from an archive: rehydrates the vector store (no re-embedding) and
   * any bundled model weights, then loads the models — from cache, so an
   * archive exported with `includeModels` needs no network at all.
   * Accepts a URL, a File from an <input>, a Blob, or raw bytes.
   */
  static async importZip(
    source: ArchiveSource,
    opts: Partial<KnowledgeOptions> & ZipOptions = {},
  ): Promise<NexusKnowledge> {
    const files = await filesFromZip(source, opts);
    const manifest = decodeJSON<KnowledgeManifest>(requireFile(files, 'manifest.json'));
    if (manifest.kind !== 'knowledge') throw new Error(`not a knowledge archive (kind=${manifest.kind})`);

    const chatZip = files.get(`${CHAT_MODEL_PREFIX}model.zip`);
    if (chatZip) await importModel(chatZip, { zip: opts.zip });
    const embedZip = files.get(`${EMBED_MODEL_PREFIX}model.zip`);
    if (embedZip) await importModel(embedZip, { zip: opts.zip });

    const kb = await NexusKnowledge.create({
      chat: opts.chat ?? manifest.models.chat,
      embedder: opts.embedder ?? manifest.models.embedder,
      ...opts,
    });
    kb.index = indexFromFiles(stripPrefix(files, RAG_PREFIX));
    for (const d of manifest.docs) kb.docs.set(d.id, { ...d, text: d.text ?? '' });
    return kb;
  }

  /** Peek at an archive's manifest without loading any models. */
  static async inspect(source: ArchiveSource, opts: ZipOptions = {}): Promise<KnowledgeManifest> {
    const files = await filesFromZip(source, opts);
    return decodeJSON<KnowledgeManifest>(requireFile(files, 'manifest.json'));
  }

  async dispose(): Promise<void> {
    await Promise.all([this.chat.dispose(), this.embedder.dispose()]);
  }
}
