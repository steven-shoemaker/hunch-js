/** An LLM and Jev together: escalate, generate, discover, refine, verify. The LLM proposes; Jev decides or checks. */

import type { Answer } from "./answer.js";
import { type Client, resolve } from "./client.js";
import { buildState, pool, run } from "./engine.js";
import { HunchError } from "./errors.js";
import { stableStringify, toJson } from "./json.js";
import { type LLMInput, type LanguageModel, asLLM, stripFence } from "./llm.js";
import { type Description, type Out, check, classify, isAnswer, pairs } from "./verbs.js";

// ----------------------------------------------------------------------------- escalate

/** Send the rows at idxs to an LLM, which must pick one of the same labels. Used by classify({ unsure: llm }). */
export async function escalate(
  jev: Client,
  llm: LLMInput,
  list: unknown[],
  idxs: number[],
  answers: (Answer | null | string)[],
  keys: string[],
  describe: Record<string, Description>,
  instructions: unknown,
  context: unknown,
): Promise<(Answer | null | string)[]> {
  const model = asLLM(llm)!;
  // every row gets a `by`, so it's there whether or not anything escalated
  const out = answers.map((a) => (isAnswer(a) && !a.by ? { ...a, by: "jev" } : a));
  if (!idxs.length) return out;
  const system =
    "You are the careful second opinion for a fast classifier that was unsure about this input. " +
    'Choose exactly one label from `labels`. Reply with ONLY JSON: {"label": "<one of the labels>"}.';
  const byItem = new Map<string, number[]>();
  for (const i of idxs) {
    const key = stableStringify(list[i]);
    byItem.set(key, [...(byItem.get(key) ?? []), i]);
  }
  const cache = await jev.cache();
  const order = [...byItem.keys()];
  const decided = await pool(order, jev.maxConcurrency, async (key) => {
    const i = byItem.get(key)![0];
    const first = out[i] as Answer;
    const payload = {
      task: instructions ?? "Which label best describes the input?",
      labels: Object.fromEntries(keys.map((k) => [k, describe[k] ?? null])),
      input: toJson(list[i]),
      context: toJson(context),
      first_pass_probabilities: Object.fromEntries(Object.entries(first.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3)),
    };
    const cacheKey = `escalate|${model.name}|${stableStringify(payload)}`;
    const hit = (await cache.get(cacheKey)) as { label: string } | undefined;
    if (hit) return hit.label;
    let user = JSON.stringify(payload);
    for (let attempt = 0; attempt < 2; attempt++) {
      const label = pickLabel(await model.complete({ system, user }), keys);
      if (label) {
        await cache.set(cacheKey, { label });
        return label;
      }
      user += `\n\nYour reply was not one of the labels. Choose exactly one of: ${JSON.stringify(keys)}`;
    }
    return null;
  });
  let misses = 0;
  order.forEach((key, n) => {
    for (const i of byItem.get(key)!) {
      const label = decided[n];
      if (!label) {
        misses += 1;
        continue;
      }
      const first = out[i] as Answer;
      out[i] = { ...first, label, p: first.probabilities[label] ?? 0, by: "llm" };
    }
  });
  if (misses) jev.onWarning(`hunch escalate: the LLM gave no valid label for ${misses} rows; kept Jev's answer.`);
  return out;
}

function pickLabel(text: string, keys: string[]): string | null {
  const raw = stripFence(text);
  let candidate = raw.replace(/^"|"$/g, "").trim();
  try {
    const data = JSON.parse(raw);
    candidate = String(typeof data === "object" && data !== null ? (data as { label?: unknown }).label : data);
  } catch {
    // plain text answer
  }
  if (keys.includes(candidate)) return candidate;
  return keys.find((k) => k.toLowerCase() === candidate.toLowerCase()) ?? null;
}

// ----------------------------------------------------------------------------- generate

export interface GenerateOptions<T> {
  n?: number;
  instructions?: string;
  context?: unknown;
  /** JSON Schema for one item. Defaults to a string. With zod: z.toJSONSchema(MySchema). */
  schema?: Record<string, unknown>;
  /** Validate and convert one parsed item, e.g. MySchema.parse. Throwing rejects the item. */
  parse?: (value: unknown) => T;
  llm?: LLMInput;
  /** Draw new items instead of returning the cached ones. */
  fresh?: boolean;
  client?: Client;
}

const BATCH = 25;

/**
 * An LLM makes n example items. Returns one item when n is 1, else exactly n distinct items.
 * Large n is drawn in batches that avoid repeats, and results are cached until fresh: true.
 */
export async function generate<T = string>(options: GenerateOptions<T> = {}): Promise<T | T[]> {
  const n = options.n ?? 1;
  if (n < 1) throw new HunchError("generate() needs n >= 1.");
  const jev = resolve(options.client);
  const model = asLLM(options.llm) ?? jev.llm;
  if (!model) throw new HunchError("generate() needs a language model: pass llm here or to configure().");
  const schema = options.schema ?? { type: "string" };
  const parse = options.parse ?? ((v: unknown) => v as T);
  const cache = await jev.cache();
  const key = `generate|${stableStringify({ model: model.name, schema, instructions: options.instructions, context: options.context, n })}`;
  if (!options.fresh) {
    const hit = (await cache.get(key)) as unknown[] | undefined;
    if (hit) {
      jev.usage.hits += 1;
      const values = hit.map(parse);
      return n === 1 ? values[0] : values;
    }
  }
  const items: T[] = [];
  const raws: unknown[] = [];
  const seen = new Set<string>();
  for (let round = 1; items.length < n; round++) {
    if (round > 2 * Math.ceil(n / BATCH) + 2) throw new HunchError(`generate() got only ${items.length} distinct items of ${n}.`);
    const want = Math.min(BATCH, n - items.length);
    const batch = await draw(model, schema, want, options.instructions, options.context, raws.slice(-50));
    for (const value of batch) {
      const fingerprint = stableStringify(value);
      if (seen.has(fingerprint)) continue;
      let parsed: T;
      try {
        parsed = parse(value);
      } catch {
        continue;
      }
      seen.add(fingerprint);
      raws.push(value);
      items.push(parsed);
      if (items.length === n) break;
    }
  }
  await cache.set(key, raws);
  return n === 1 ? items[0] : items;
}

async function draw(model: LanguageModel, schema: Record<string, unknown>, n: number, instructions: unknown, context: unknown, avoid: unknown[]): Promise<unknown[]> {
  const system =
    "You generate example data. Reply with ONLY a JSON array of exactly " +
    `${n} distinct items, each matching this JSON Schema. No prose, no markdown fences.\n${JSON.stringify(schema)}`;
  const payload = { instructions, context: toJson(context), n, already_have_do_not_repeat: avoid.length ? avoid : undefined };
  let user = JSON.stringify(payload);
  let lastError = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = stripFence(await model.complete({ system, user }));
    try {
      const value = JSON.parse(text);
      const list = Array.isArray(value) ? value : [value];
      if (list.length >= Math.min(n, 1)) return list;
      lastError = `expected ${n} items, got ${list.length}`;
    } catch (error) {
      if (n === 1 && schema.type === "string") return [text];
      lastError = String(error).slice(0, 300);
    }
    user = `${JSON.stringify(payload)}\n\nYour previous reply was invalid: ${lastError}\nReply again with only a valid JSON array.`;
  }
  throw new HunchError(`generate() could not get valid JSON from the model: ${lastError}`);
}

// ----------------------------------------------------------------------------- discover

export interface DiscoverOptions {
  instructions?: string;
  columns?: readonly string[];
  sample?: number;
  /** Add an "other" category so rows that fit nothing aren't forced into one. Default true. */
  other?: boolean;
  llm?: LLMInput;
  fresh?: boolean;
  client?: Client;
}

/** An LLM reads a sample and proposes n categories as { name: description }, ready for classify(). */
export async function discover(input: readonly unknown[], n = 8, options: DiscoverOptions = {}): Promise<Record<string, string>> {
  if (n < 2) throw new HunchError("discover() needs n >= 2.");
  const rows = options.columns
    ? input.map((r) => Object.fromEntries(options.columns!.map((c) => [c, (r as Record<string, unknown>)[c]])))
    : [...input];
  const distinct = [...new Map(rows.filter((r) => r !== null && r !== undefined).map((r) => [stableStringify(r), r])).values()];
  if (!distinct.length) throw new HunchError("discover() needs some data.");
  const sample = options.sample ?? 100;
  const step = Math.max(1, Math.floor(distinct.length / sample));
  const examples = distinct.filter((_, i) => i % step === 0).slice(0, sample).map(toJson);
  const proposed = (await generate<{ name: string; description: string }>({
    n,
    instructions:
      `Propose ${n} categories that together cover these examples. Short names in plain words. Mutually exclusive. ` +
      "Each description says in one sentence what belongs and what does not. No catch-all category." +
      (options.instructions ? ` Group them ${options.instructions}.` : ""),
    context: { examples },
    schema: { type: "object", properties: { name: { type: "string" }, description: { type: "string" } }, required: ["name", "description"] },
    parse: (v) => {
      const o = v as { name?: unknown; description?: unknown };
      if (typeof o?.name !== "string" || typeof o?.description !== "string") throw new Error("bad category");
      return { name: o.name, description: o.description };
    },
    llm: options.llm,
    fresh: options.fresh,
    client: options.client,
  })) as { name: string; description: string }[];
  const categories: Record<string, string> = {};
  for (const { name, description } of Array.isArray(proposed) ? proposed : [proposed]) {
    const clean = name.trim();
    if (clean && clean.toLowerCase() !== "other" && !Object.keys(categories).some((k) => k.toLowerCase() === clean.toLowerCase())) {
      categories[clean] = description.trim();
    }
  }
  if (options.other ?? true) categories.other = "fits none of the other categories";
  return categories;
}

// ----------------------------------------------------------------------------- refine

export interface Refined {
  text: string;
  /** true when every check held; null when a check request was skipped. */
  passed: boolean | null;
  /** LLM rewrites it took. 0 means the input already passed. */
  rounds: number;
  /** Checks still failing at the end. */
  failed: string[];
  original: string;
}

export interface RefineOptions {
  instructions?: string;
  rounds?: number;
  threshold?: number;
  context?: unknown;
  llm?: LLMInput;
  detail?: boolean;
  client?: Client;
}

/**
 * Rewrite text with an LLM until Jev agrees every check holds, or rounds run out.
 * checks are statements a good version makes true. Only failing drafts go back to the LLM.
 */
export async function refine<I extends string | readonly string[]>(input: I, checks: readonly string[] | Record<string, string>, options: RefineOptions & { detail: true }): Promise<Out<I, Refined>>;
export async function refine<I extends string | readonly string[]>(input: I, checks: readonly string[] | Record<string, string>, options?: RefineOptions): Promise<Out<I, string>>;
export async function refine(input: string | readonly string[], checks: readonly string[] | Record<string, string>, options: RefineOptions = {}): Promise<unknown> {
  const jev = resolve(options.client);
  const model = asLLM(options.llm) ?? jev.llm;
  if (!model) throw new HunchError("refine() needs a language model: pass llm here or to configure().");
  const named = Array.isArray(checks) ? Object.fromEntries(checks.map((c, i) => [`c${i}`, c])) : (checks as Record<string, string>);
  if (!Object.keys(named).length) throw new HunchError("refine() needs at least one check.");
  const rounds = options.rounds ?? 3;
  if (rounds < 0) throw new HunchError("refine() rounds must be >= 0.");
  const many = Array.isArray(input);
  const originals = (many ? [...input] : [input as string]).map((t) => t ?? "");
  const drafts = [...originals];
  const used = originals.map(() => 0);
  const failing: (string[] | null)[] = originals.map(() => null);
  let pending = originals.map((_, i) => i);
  const system =
    "Rewrite the text so every requirement holds. Keep everything that already works, including meaning and length, " +
    "unless a requirement says otherwise. Reply with ONLY the rewritten text.";

  for (let round = 0; round <= rounds && pending.length; round++) {
    const verdicts = (await check(pending.map((i) => drafts[i]), named, { context: options.context, threshold: options.threshold, detail: true, client: jev })) as Record<string, { value: boolean | null } | null>[];
    const still: number[] = [];
    pending.forEach((i, k) => {
      const v = verdicts[k];
      if (Object.values(v).some((f) => f === null)) {
        failing[i] = null;
        return;
      }
      const bad = Object.keys(named).filter((name) => !v[name]!.value);
      failing[i] = bad;
      if (bad.length) still.push(i);
    });
    if (round === rounds || !still.length) break;
    const texts = await pool(still, jev.maxConcurrency, async (i) =>
      stripFence(
        await model.complete({
          system,
          user: JSON.stringify({
            text: drafts[i],
            failed_requirements: (failing[i] ?? []).map((n) => named[n]),
            all_requirements: Object.values(named),
            instructions: options.instructions,
            context: toJson(options.context),
          }),
        }),
      ),
    );
    still.forEach((i, k) => {
      drafts[i] = texts[k];
      used[i] += 1;
    });
    pending = still;
  }
  const results: Refined[] = drafts.map((text, i) => ({
    text,
    passed: failing[i] === null ? null : failing[i]!.length === 0,
    rounds: used[i],
    failed: (failing[i] ?? []).map((n) => (Array.isArray(checks) ? named[n] : n)),
    original: originals[i],
  }));
  const out = options.detail ? results : results.map((r) => r.text);
  return many ? out : out[0];
}

// ----------------------------------------------------------------------------- verify

export const VERDICTS = {
  supported: "the source states or directly implies every part of the claim",
  contradicted: "the source says something that conflicts with the claim",
  "not mentioned": "the source doesn't address the claim, or supports only part of it",
} as const;

export type Verdict = keyof typeof VERDICTS | "misquoted";

/**
 * Check each claim against its source: "supported", "contradicted", "not mentioned", or "misquoted".
 * source is one document for all claims, or an array aligned with claims. Quoted text in a claim
 * must appear in the source word for word, or the claim is "misquoted".
 */
export async function verify<I extends string | readonly string[]>(claims: I, source: unknown, options: { context?: unknown; detail: true; client?: Client }): Promise<Out<I, (Answer & { label: Verdict }) | null>>;
export async function verify<I extends string | readonly string[]>(claims: I, source: unknown, options?: { context?: unknown; detail?: boolean; client?: Client }): Promise<Out<I, Verdict | null>>;
export async function verify(claims: string | readonly string[], source: unknown, options: { context?: unknown; detail?: boolean; client?: Client } = {}): Promise<unknown> {
  const many = Array.isArray(claims);
  const list = many ? [...claims] : [claims as string];
  const sources = Array.isArray(source) ? [...source] : list.map(() => source);
  if (sources.length !== list.length) throw new HunchError(`verify() got ${list.length} claims and ${sources.length} sources.`);
  const answers = (await classify(pairs(list, sources, ["claim", "source"]), VERDICTS, {
    instructions: { question: "How does the source relate to the claim?", rules: "Judge only what the source states. Don't use outside knowledge." },
    context: options.context,
    detail: true,
    client: options.client,
  })) as (Answer | null)[];
  // ponytail: asks Jev even for misquoted rows, then overrides; skip them first if volume matters
  const out = list.map((claim, i) =>
    misquoted(claim, sources[i]) ? ({ label: "misquoted", probabilities: { misquoted: 1 }, p: 1, confidence: 1, shape: "sure", by: "quote check" } as Answer) : answers[i],
  );
  const values = options.detail ? out : out.map((a) => (a ? a.label : null));
  return many ? values : values[0];
}

function misquoted(claim: unknown, source: unknown): boolean {
  const quotes = [...String(claim).matchAll(/["“]([^"”]{8,})["”]/g)].map((m) => m[1]);
  if (!quotes.length) return false;
  const norm = (t: unknown) => String(t).toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  const haystack = norm(source);
  return quotes.some((q) => !haystack.includes(norm(q)));
}

export { buildState, run };
