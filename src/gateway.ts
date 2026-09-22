/**
 * Reach Jev through a gateway instead of TypeSafe directly: OpenRouter or Vercel AI Gateway.
 * Each adapter has the SDK's systemOne({ state, questions }) shape. Retries 408, 429, 5xx, and 529.
 */

import type { JevLike } from "./client.js";
import { HunchError } from "./errors.js";

const RETRY = new Set([408, 429, 500, 502, 503, 504, 529]);
const env = (name: string) => (typeof process !== "undefined" ? process.env[name] : undefined);

async function post(url: string, headers: Record<string, string>, body: unknown, timeoutMs: number, retries = 3): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    let status = 0;
    let detail = "";
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "User-Agent": "hunch-js", ...headers },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.ok) return response.json();
      status = response.status;
      detail = (await response.text()).slice(0, 400);
      if (!RETRY.has(status) || attempt === retries) throw new HunchError(`Jev gateway request failed (${status}). ${detail}`.trim());
    } catch (error) {
      if (error instanceof HunchError) throw error;
      if (attempt === retries) throw new HunchError(`Jev gateway request failed: ${String(error)}`);
    }
    await new Promise((r) => setTimeout(r, Math.min(8000, 500 * 2 ** attempt) * (1 - 0.25 * Math.random())));
  }
}

/** Jev's own confidence formula: the top probability rescaled from uniform (0) to certain (1). */
export function jevConfidence(probabilities: Record<string, number>): number {
  const values = Object.values(probabilities).map(Number);
  const n = values.length;
  return n < 2 ? 0 : Math.max(0, (n * Math.max(...values) - 1) / (n - 1));
}

/** Jev through OpenRouter's decisions endpoint (alpha). Same body as TypeSafe's API. */
export function openRouterJev({ apiKey, model = "~typesafe/jev-latest", timeoutMs = 30_000 }: { apiKey?: string; model?: string; timeoutMs?: number } = {}): JevLike {
  const key = apiKey ?? env("OPENROUTER_API_KEY");
  if (!key) throw new HunchError('gateway "openrouter" needs OPENROUTER_API_KEY or apiKey.');
  return {
    async systemOne({ state, questions }) {
      const body = await post(
        "https://openrouter.ai/api/alpha/decisions",
        { Authorization: `Bearer ${key}`, "HTTP-Referer": "https://github.com/steven-shoemaker/hunch-js", "X-Title": "hunch" },
        { model, state, questions },
        timeoutMs,
      );
      return { model: body.model ?? model, answers: body.answers ?? {}, usage: body.usage };
    },
  };
}

/**
 * Jev through Vercel AI Gateway's evaluation-model endpoint. The gateway calls a noul question
 * "boolean", drops score legends, and reports confidence separately; this translates both ways.
 */
export function vercelJev({ apiKey, model = "typesafe-ai/jev", timeoutMs = 30_000 }: { apiKey?: string; model?: string; timeoutMs?: number } = {}): JevLike {
  const key = apiKey ?? env("AI_GATEWAY_API_KEY");
  if (!key) throw new HunchError('gateway "vercel" needs AI_GATEWAY_API_KEY or apiKey.');
  return {
    async systemOne({ state, questions }) {
      const qs = questions as Record<string, { type: string; criteria?: unknown }>;
      const wire = Object.fromEntries(Object.entries(qs).map(([id, q]) => [id, q.type === "noul" ? { ...q, type: "boolean" } : q]));
      const body = await post(
        "https://ai-gateway.vercel.sh/v4/ai/evaluation-model",
        {
          Authorization: `Bearer ${key}`,
          "Ai-Gateway-Protocol-Version": "0.0.1",
          "Ai-Gateway-Auth-Method": "api-key",
          "Ai-Evaluation-Model-Specification-Version": "4",
          "Ai-Model-Id": model,
        },
        { state, questions: wire },
        timeoutMs,
      );
      const reported: Record<string, number> = body.providerMetadata?.typesafe?.confidence ?? {};
      const answers: Record<string, unknown> = {};
      for (const [id, raw] of Object.entries<any>(body.answers ?? {})) {
        if (raw.type === "boolean") {
          answers[id] = { type: "noul", noul: Number(raw.probability) };
          continue;
        }
        const probabilities = raw.probabilities ?? {};
        const confidence = reported[id] ?? jevConfidence(probabilities);
        if (raw.type === "score") {
          const legend = Object.fromEntries(((qs[id]?.criteria as unknown[]) ?? []).map((c, i) => [String(i), String(c)]));
          answers[id] = { type: "score", score: raw.score, probabilities, legend, confidence };
        } else {
          answers[id] = { type: "choice", choice: raw.choice, probabilities, confidence };
        }
      }
      const u = body.usage ?? {};
      return { model, answers, usage: { input_tokens: u.inputTokens ?? u.input_tokens, output_tokens: u.outputTokens ?? u.output_tokens } };
    },
  };
}
