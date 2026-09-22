import { HunchError } from "./errors.js";

/** An LLM that proposes text. hunch only ever sends one system and one user message. */
export interface LanguageModel {
  readonly name: string;
  complete(args: { system: string; user: string }): Promise<string>;
}

export type LLMInput = LanguageModel | ((system: string, user: string) => string | Promise<string>);

/** Accept an adapter or any function (system, user) => text. */
export function asLLM(llm?: LLMInput | null): LanguageModel | undefined {
  if (llm === undefined || llm === null) return undefined;
  if (typeof llm === "object" && typeof llm.complete === "function") return llm;
  if (typeof llm === "function") {
    const fn = llm;
    return {
      name: fn.name || "function",
      async complete({ system, user }) {
        const text = await fn(system, user);
        if (typeof text !== "string" || !text.trim()) throw new HunchError("The llm function returned no text.");
        return text.trim();
      },
    };
  }
  throw new HunchError("llm must have complete({ system, user }) or be a function (system, user) => text.");
}

export function isLLM(value: unknown): boolean {
  return (
    value !== null &&
    typeof value !== "string" &&
    ((typeof value === "object" && typeof (value as LanguageModel).complete === "function") || typeof value === "function")
  );
}

export interface OpenAICompatOptions {
  apiKey: string;
  model: string;
  baseURL?: string;
  headers?: Record<string, string>;
  body?: Record<string, unknown>;
  timeoutMs?: number;
}

/** Chat Completions for OpenAI-compatible hosts. */
export function openAICompat({ apiKey, model, baseURL = "https://api.openai.com/v1", headers = {}, body = {}, timeoutMs = 60_000 }: OpenAICompatOptions): LanguageModel {
  const url = `${baseURL.replace(/\/$/, "")}/chat/completions`;
  return {
    name: model,
    async complete({ system, user }) {
      const response = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", "User-Agent": "hunch-js", ...headers },
        body: JSON.stringify({ model, messages: [{ role: "system", content: system }, { role: "user", content: user }], ...body }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) throw new HunchError(`Language model request failed (${response.status}). ${(await response.text()).slice(0, 400)}`);
      const data = (await response.json()) as { choices?: { message?: { content?: string } }[] };
      const text = data.choices?.[0]?.message?.content;
      if (!text?.trim()) throw new HunchError("Language model returned no text.");
      return text.trim();
    },
  };
}

const env = (name: string) => (typeof process !== "undefined" ? process.env[name] : undefined);

function need(value: string | undefined, message: string): string {
  if (!value) throw new HunchError(message);
  return value;
}

export function openai({ apiKey, model = "gpt-5-mini", ...rest }: Partial<OpenAICompatOptions> = {}): LanguageModel {
  return openAICompat({ apiKey: need(apiKey ?? env("OPENAI_API_KEY"), "Set OPENAI_API_KEY or pass apiKey to openai()."), model, ...rest });
}

export function openrouter({ apiKey, model = "z-ai/glm-5.3-flash", ...rest }: Partial<OpenAICompatOptions> = {}): LanguageModel {
  return openAICompat({
    apiKey: need(apiKey ?? env("OPENROUTER_API_KEY"), "Set OPENROUTER_API_KEY or pass apiKey to openrouter()."),
    model,
    baseURL: "https://openrouter.ai/api/v1",
    headers: { "HTTP-Referer": "https://github.com/steven-shoemaker/hunch", "X-Title": "hunch" },
    ...rest,
  });
}

export function cerebras({ apiKey, model = "gpt-oss-120b", ...rest }: Partial<OpenAICompatOptions> = {}): LanguageModel {
  return openAICompat({ apiKey: need(apiKey ?? env("CEREBRAS_API_KEY"), "Set CEREBRAS_API_KEY or pass apiKey to cerebras()."), model, baseURL: "https://api.cerebras.ai/v1", ...rest });
}

/** Azure OpenAI through its OpenAI-compatible v1 endpoint. */
export function azure({ endpoint, deployment, apiKey, timeoutMs }: { endpoint?: string; deployment: string; apiKey?: string; timeoutMs?: number }): LanguageModel {
  const base = need(endpoint ?? env("AZURE_OPENAI_ENDPOINT"), "Set AZURE_OPENAI_ENDPOINT or pass endpoint to azure().");
  const key = need(apiKey ?? env("AZURE_OPENAI_API_KEY"), "Set AZURE_OPENAI_API_KEY or pass apiKey to azure().");
  return openAICompat({ apiKey: key, model: deployment, baseURL: `${base.replace(/\/$/, "")}/openai/v1`, timeoutMs });
}

/** A local model served by Ollama. */
export function ollama(model = "llama3.2", { host = "http://localhost:11434", timeoutMs = 120_000 } = {}): LanguageModel {
  return openAICompat({ apiKey: "ollama", model, baseURL: `${host.replace(/\/$/, "")}/v1`, timeoutMs });
}

/** Claude through the official @anthropic-ai/sdk (install it alongside hunch). Reads ANTHROPIC_API_KEY. */
export function anthropic({ apiKey, model = "claude-opus-5", maxTokens = 16_000, client }: { apiKey?: string; model?: string; maxTokens?: number; client?: unknown } = {}): LanguageModel {
  let sdk: Promise<{ messages: { create(args: unknown): Promise<unknown> } }> | undefined;
  const get = () =>
    (sdk ??= client
      ? Promise.resolve(client as { messages: { create(args: unknown): Promise<unknown> } })
      : import("@anthropic-ai/sdk")
          .then((m) => new m.default(apiKey ? { apiKey } : {}) as unknown as { messages: { create(args: unknown): Promise<unknown> } })
          .catch(() => {
            throw new HunchError("hunch anthropic() needs @anthropic-ai/sdk: npm install @anthropic-ai/sdk");
          }));
  return {
    name: model,
    async complete({ system, user }) {
      const response = (await (await get()).messages.create({
        model,
        max_tokens: maxTokens,
        system,
        messages: [{ role: "user", content: user }],
      })) as { stop_reason?: string; content: { type: string; text?: string }[] };
      if (response.stop_reason === "refusal") throw new HunchError(`${model} declined the request.`);
      const text = response.content.filter((b) => b.type === "text").map((b) => b.text ?? "").join("").trim();
      if (!text) throw new HunchError("Language model returned no text.");
      return text;
    },
  };
}

export function stripFence(raw: string): string {
  const text = raw.trim();
  if (!text.startsWith("```")) return text;
  const lines = text.split("\n").slice(1);
  if (lines.length && lines[lines.length - 1].trim() === "```") lines.pop();
  return lines.join("\n").trim();
}
