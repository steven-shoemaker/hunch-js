import { AsyncLocalStorage } from "node:async_hooks";
import type { Client } from "./client.js";
import { HunchError } from "./errors.js";
import { stableStringify, toJson } from "./json.js";

/** One Jev answer as plain JSON. */
export type Raw =
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "noul"; noul: number }
  | { type: "score"; score: number; confidence: number; legend: Record<string, string>; probabilities: Record<string, number> };

export type Question = { type: "choice" | "noul" | "score"; instructions?: unknown; criteria?: unknown };

export interface Plan {
  /** Distinct uncached requests that would be sent. Rematches aren't counted. */
  requests: number;
  questions: number;
  items: number;
  calls: { label: string; requests: number }[];
}

const planStore = new AsyncLocalStorage<Plan>();

/**
 * Count what the calls inside fn would send to Jev, without sending anything.
 * Verbs return placeholder answers inside, so the rest of fn keeps running. Nothing is cached.
 */
export async function dryRun(fn: () => unknown | Promise<unknown>): Promise<Plan> {
  const plan: Plan = { requests: 0, questions: 0, items: 0, calls: [] };
  await planStore.run(plan, async () => {
    await fn();
  });
  return plan;
}

export function buildState(item: unknown, context: unknown): Record<string, unknown> {
  const state: Record<string, unknown> = { input: toJson(item) };
  if (context === undefined || context === null) return state;
  if (typeof context === "object" && !Array.isArray(context)) {
    for (const [key, value] of Object.entries(context as Record<string, unknown>)) {
      if (key === "input") throw new HunchError("context cannot use the reserved key 'input'.");
      state[key] = toJson(value);
    }
  } else {
    state.context = toJson(context);
  }
  return state;
}

export function mergeContext(base: unknown, extra: unknown): unknown {
  if (extra === undefined || extra === null) return base;
  if (base === undefined || base === null) return extra;
  const asObject = (c: unknown) => (typeof c === "object" && !Array.isArray(c) ? (c as object) : { context: c });
  return { ...asObject(base), ...asObject(extra) };
}

/** Answer every question for every state. One request per distinct uncached state. */
export async function run(
  client: Client,
  states: unknown[],
  questions: Record<string, Question>,
  label = "hunch",
): Promise<Record<string, Raw | null>[]> {
  const ids = Object.keys(questions);
  if (!ids.length) throw new HunchError("No questions to ask.");
  const encoded = Object.fromEntries(ids.map((id) => [id, stableStringify(questions[id])]));
  const cache = await client.cache();
  const results: Record<string, Raw | null>[] = states.map(() => ({}));
  const todo = new Map<string, number[]>();

  for (const [index, state] of states.entries()) {
    const key = stableStringify(state);
    for (const id of ids) {
      const hit = (await cache.get(`${key}|${encoded[id]}`)) as Raw | undefined;
      if (hit !== undefined) {
        results[index][id] = hit;
        client.usage.hits += 1;
      }
    }
    if (Object.keys(results[index]).length < ids.length) {
      const list = todo.get(key) ?? [];
      list.push(index);
      todo.set(key, list);
    }
  }

  const keys = [...todo.keys()];
  const plan = planStore.getStore();
  if (plan) {
    plan.items += states.length;
    plan.requests += keys.length;
    plan.calls.push({ label, requests: keys.length });
    for (const key of keys) {
      const first = todo.get(key)![0];
      const missing = ids.filter((id) => !(id in results[first]));
      plan.questions += missing.length;
      const fake = Object.fromEntries(missing.map((id) => [id, placeholder(questions[id])]));
      for (const index of todo.get(key)!) Object.assign(results[index], fake);
    }
    return results;
  }

  const failures: unknown[] = [];
  let done = 0;
  const one = async (key: string): Promise<Record<string, Raw | null>> => {
    const first = todo.get(key)![0];
    const missing = Object.fromEntries(ids.filter((id) => !(id in results[first])).map((id) => [id, questions[id]]));
    const wait = client.reserveSlot();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    try {
      const response = (await client.jev.systemOne({ state: states[first], questions: missing })) as {
        model?: string;
        answers?: Record<string, unknown>;
        usage?: { input_tokens?: number; output_tokens?: number };
      };
      client.usage.calls += 1;
      if (response.model) client.usage.model = response.model;
      client.usage.inputTokens += response.usage?.input_tokens ?? 0;
      client.usage.outputTokens += response.usage?.output_tokens ?? 0;
      const answers: Record<string, Raw> = {};
      for (const id of Object.keys(missing)) {
        answers[id] = normalize(response.answers?.[id], id);
        await cache.set(`${key}|${encoded[id]}`, answers[id]);
      }
      return answers;
    } catch (error) {
      if (client.errors === "raise") throw error;
      failures.push(error);
      return Object.fromEntries(Object.keys(missing).map((id) => [id, null]));
    } finally {
      done += 1;
      client.onProgress?.(done, keys.length, label);
    }
  };

  const fetched = await pool(keys, client.maxConcurrency, one);
  keys.forEach((key, i) => {
    for (const index of todo.get(key)!) Object.assign(results[index], fetched[i]);
  });
  if (failures.length) {
    client.onWarning(
      `hunch ${label}: ${failures.length} of ${keys.length} requests failed and were skipped (their rows are null). First error: ${String(failures[0])}`,
    );
  }
  return results;
}

/** Run fn over items with at most `limit` in flight; results keep input order. */
export async function pool<T, R>(items: T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

export function isDryRun(): boolean {
  return planStore.getStore() !== undefined;
}

function normalize(payload: unknown, id: string): Raw {
  const p = payload as Record<string, unknown> | undefined;
  if (!p) throw new HunchError(`Jev response is missing an answer for ${JSON.stringify(id)}.`);
  if (typeof p.noul === "number") return { type: "noul", noul: p.noul };
  if (p.legend) {
    return {
      type: "score",
      score: Number(p.score),
      confidence: Number(p.confidence ?? 0),
      legend: Object.fromEntries(Object.entries(p.legend as object).map(([k, v]) => [String(k), String(v)])),
      probabilities: Object.fromEntries(Object.entries(p.probabilities as object).map(([k, v]) => [String(k), Number(v)])),
    };
  }
  if (p.choice === undefined || !p.probabilities) throw new HunchError(`Jev answer ${JSON.stringify(id)} has an unrecognized shape.`);
  return {
    type: "choice",
    choice: String(p.choice),
    confidence: Number(p.confidence ?? 0),
    probabilities: Object.fromEntries(Object.entries(p.probabilities as object).map(([k, v]) => [String(k), Number(v)])),
  };
}

function placeholder(question: Question): Raw {
  if (question.type === "noul") return { type: "noul", noul: 0 };
  if (question.type === "score") {
    const legend = Object.fromEntries((question.criteria as unknown[]).map((c, i) => [String(i), String(c)]));
    return { type: "score", score: 0, confidence: 1, legend, probabilities: Object.fromEntries(Object.keys(legend).map((k) => [k, k === "0" ? 1 : 0])) };
  }
  const keys = Object.keys(question.criteria as object);
  return { type: "choice", choice: keys[0], confidence: 1, probabilities: Object.fromEntries(keys.map((k) => [k, k === keys[0] ? 1 : 0])) };
}
