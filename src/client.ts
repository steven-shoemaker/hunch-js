import { TypeSafeClient } from "@typesafe-ai/sdk";
import { type CacheStore, MemoryCache, fileCache } from "./cache.js";
import { HunchError } from "./errors.js";
import { type LanguageModel, asLLM } from "./llm.js";
import { ShapePolicy } from "./shapes.js";

/** Anything with the SDK's systemOne shape. Inject one in tests. */
export interface JevLike {
  systemOne(request: { state: unknown; questions: Record<string, unknown> }): PromiseLike<unknown>;
}

export interface ClientOptions {
  /** Defaults to TYPESAFE_API_KEY. */
  apiKey?: string;
  /** Jev model; defaults to TYPESAFE_DEFAULT_MODEL or the SDK default. */
  model?: string;
  /** An existing TypeSafeClient, or any object with systemOne(). */
  client?: JevLike;
  /** LLM for generate / discover / refine / escalation: an adapter or a function (system, user) => text. */
  llm?: LanguageModel | ((system: string, user: string) => string | Promise<string>);
  /** A CacheStore, or a directory path for a file cache (Node). Defaults to memory. */
  cache?: CacheStore | string;
  policy?: ShapePolicy;
  /** Requests in flight at once. Default 16. */
  maxConcurrency?: number;
  /** Cap on requests per second. */
  maxRps?: number;
  /** "skip": a request that fails after the SDK's retries gives null for its rows, with a warning. */
  errors?: "raise" | "skip";
  onProgress?: (done: number, total: number, label: string) => void;
  onWarning?: (message: string) => void;
}

export interface Usage {
  calls: number;
  hits: number;
  inputTokens: number;
  outputTokens: number;
  model?: string;
}

export class Client {
  readonly jev: JevLike;
  readonly llm?: LanguageModel;
  readonly policy: ShapePolicy;
  readonly maxConcurrency: number;
  readonly maxRps?: number;
  readonly errors: "raise" | "skip";
  readonly onProgress?: ClientOptions["onProgress"];
  readonly onWarning: (message: string) => void;
  readonly usage: Usage = { calls: 0, hits: 0, inputTokens: 0, outputTokens: 0 };
  #cache: Promise<CacheStore>;
  #nextSlot = 0;

  constructor(options: ClientOptions = {}) {
    if (options.errors && options.errors !== "raise" && options.errors !== "skip") {
      throw new HunchError('errors must be "raise" or "skip".');
    }
    if (options.maxRps !== undefined && options.maxRps <= 0) throw new HunchError("maxRps must be positive.");
    this.jev =
      options.client ??
      (new TypeSafeClient({
        ...(options.apiKey ? { apiKey: options.apiKey } : {}),
        ...(options.model ? { defaultModel: options.model } : {}),
      }) as unknown as JevLike);
    this.llm = asLLM(options.llm);
    this.policy = options.policy ?? new ShapePolicy();
    this.maxConcurrency = Math.max(1, options.maxConcurrency ?? 16);
    this.maxRps = options.maxRps;
    this.errors = options.errors ?? "raise";
    this.onProgress = options.onProgress;
    this.onWarning = options.onWarning ?? ((m) => console.warn(m));
    const cache = options.cache;
    this.#cache = Promise.resolve(typeof cache === "string" ? fileCache(cache) : (cache ?? new MemoryCache()));
  }

  cache(): Promise<CacheStore> {
    return this.#cache;
  }

  async clearCache(): Promise<void> {
    await (await this.#cache).clear?.();
  }

  /** Milliseconds to wait before the next request under maxRps. */
  reserveSlot(): number {
    if (!this.maxRps) return 0;
    const now = Date.now();
    const start = Math.max(now, this.#nextSlot);
    this.#nextSlot = start + 1000 / this.maxRps;
    return start - now;
  }
}

let current: Client | undefined;

/** Set the default client used when a call doesn't pass { client }. */
export function configure(options: ClientOptions = {}): Client {
  current = new Client(options);
  return current;
}

export function defaultClient(): Client {
  current ??= new Client();
  return current;
}

export function resolve(client?: Client): Client {
  return client ?? defaultClient();
}
