import type { Answer, Feeling, MultiAnswer, Pick, Ranked, Rating } from "./answer.js";
import { feeling } from "./answer.js";
import { type Client, resolve } from "./client.js";
import { type Question, type Raw, buildState, mergeContext, run } from "./engine.js";
import { HunchError } from "./errors.js";
import { toJson } from "./json.js";
import { type LLMInput, isLLM } from "./llm.js";

export const MAX_CHOICE_OPTIONS = 255;

/** One value in, one answer out; an array in, an array out. */
export type Out<I, R> = I extends readonly unknown[] ? R[] : R;

/** A label's description: text, an object ({ what, not_for, examples }), or null. */
export type Description = string | Record<string, unknown> | unknown[] | null;
export type Labels = readonly string[] | Readonly<Record<string, Description>> | Tree;

export interface BaseOptions {
  /** Extra state that rides along with every item: a policy, an ICP, a diff. */
  context?: unknown;
  /** For object items, the keys Jev should read. */
  columns?: readonly string[];
  client?: Client;
}

// ----------------------------------------------------------------------------- items

function items(input: unknown, columns?: readonly string[]): { list: unknown[]; many: boolean } {
  const many = Array.isArray(input);
  let list = many ? [...(input as unknown[])] : [input];
  if (columns) {
    list = list.map((item) => {
      if (item === null || typeof item !== "object" || Array.isArray(item)) throw new HunchError("columns only applies to object items.");
      const missing = columns.filter((c) => !(c in (item as object)));
      if (missing.length) throw new HunchError(`columns names keys the items don't have: ${missing.join(", ")}`);
      return Object.fromEntries(columns.map((c) => [c, (item as Record<string, unknown>)[c]]));
    });
  }
  return { list, many };
}

function pack<R>(values: R[], many: boolean): R | R[] {
  return many ? values : values[0];
}

// ----------------------------------------------------------------------------- pairs and trees

/**
 * A label tree for classify(): { Parent: { Child: description | null, Sub: { ... } } }.
 * Marked explicitly because a plain object's values may be object descriptions, not subtrees.
 */
export class Tree {
  constructor(readonly nodes: Readonly<Record<string, unknown>>) {}
}

export const tree = (nodes: Record<string, unknown>) => new Tree(nodes);

/**
 * Line up two aligned arrays so any verb can compare them row by row.
 * Each item becomes { a, b } (or your names): hunch.score(pairs(crm, vendors), [...], { instructions: "Are a and b the same company?" })
 */
export function pairs<A, B>(a: readonly A[], b: readonly B[], names: [string, string] = ["a", "b"]): Record<string, A | B>[] {
  if (a.length !== b.length) throw new HunchError(`pairs() got ${a.length} and ${b.length} items; they must line up.`);
  return a.map((x, i) => ({ [names[0]]: x, [names[1]]: b[i] }));
}

// ----------------------------------------------------------------------------- ask specs

const KEEP = Symbol("keep");
type Policy = typeof KEEP | "rematch" | LLMInput | string;

export interface ClassifyOptions extends BaseOptions {
  instructions?: string | Record<string, unknown>;
  multiLabel?: boolean;
  /** multiLabel cutoff. */
  threshold?: number;
  /** For rows where two labels were close: "rematch", an LLM, or a literal to return. */
  split?: Policy;
  /** For rows where the evidence was flat: an LLM, or a literal to return (e.g. "review"). */
  unsure?: Policy;
  /** { child: parent }: answer with the parent when the children are too close to call. */
  backoff?: Record<string, string>;
  /** Paths kept per level when labels is a Tree. */
  beam?: number;
  detail?: boolean;
}

export interface ClassifySpec {
  kind: "classify";
  labels: Labels;
  instructions?: string | Record<string, unknown>;
  split?: Policy;
  unsure?: Policy;
  context?: unknown;
}
export interface RateSpec {
  kind: "rate";
  levels: readonly string[];
  instructions?: string | Record<string, unknown>;
  context?: unknown;
}
export interface CheckSpec {
  kind: "check";
  statement: string;
  criteria?: { true?: string | null; false?: string | null };
  threshold?: number;
  uncertain?: [number, number];
  context?: unknown;
}
export type Spec = ClassifySpec | RateSpec | CheckSpec;

export const Classify = (labels: Labels, options: Omit<ClassifySpec, "kind" | "labels"> = {}): ClassifySpec => ({ kind: "classify", labels, ...options });
export const Rate = (levels: readonly string[], options: Omit<RateSpec, "kind" | "levels"> | string = {}): RateSpec => ({
  kind: "rate",
  levels,
  ...(typeof options === "string" ? { instructions: options } : options),
});
export const Check = (statement: string, options: Omit<CheckSpec, "kind" | "statement"> = {}): CheckSpec => ({ kind: "check", statement, ...options });

// ----------------------------------------------------------------------------- ask

/**
 * Ask several questions about the same items, one Jev request per item.
 * Returns { name: answer } per item. A spec's own context is merged over the call's context;
 * questions whose merged context differs go in separate requests.
 */
export async function ask<I>(
  input: I,
  specs: Record<string, Spec>,
  options: BaseOptions & { detail?: boolean } = {},
): Promise<Out<I, Record<string, unknown>>> {
  const names = Object.keys(specs);
  if (!names.length) throw new HunchError("ask() needs at least one question.");
  const jev = resolve(options.client);
  const { list, many } = items(input, options.columns);
  const built: Record<string, Question> = {};
  const readers: Record<string, (raw: Raw) => unknown> = {};
  const contexts: Record<string, unknown> = {};
  for (const name of names) {
    const spec = specs[name];
    if (spec.kind === "classify") {
      const { keys, describe } = labelsOf(spec.labels);
      built[name] = choiceQuestion(spec.instructions, keys, describe);
      readers[name] = (raw) => answerOf(jev, raw);
    } else if (spec.kind === "rate") {
      built[name] = { type: "score", instructions: spec.instructions ?? "Where on this scale does the input fall?", criteria: levelsOf(spec.levels) };
      readers[name] = (raw) => ratingOf(jev, raw);
    } else if (spec.kind === "check") {
      const [low, high] = band(spec.uncertain, spec.threshold ?? 0.5);
      built[name] = noulQuestion(spec.statement, spec.criteria);
      readers[name] = (raw) => feeling((raw as { noul: number }).noul, high, low);
    } else {
      throw new HunchError(`ask() question ${JSON.stringify(name)} must be Classify, Rate, or Check.`);
    }
    contexts[name] = mergeContext(options.context, spec.context);
  }

  const groups = new Map<string, string[]>();
  for (const name of names) {
    const key = JSON.stringify(toJson(contexts[name]));
    groups.set(key, [...(groups.get(key) ?? []), name]);
  }
  const rows: Record<string, unknown>[] = list.map(() => ({}));
  for (const group of groups.values()) {
    const states = list.map((item) => buildState(item, contexts[group[0]]));
    const raws = await run(jev, states, Object.fromEntries(group.map((n) => [n, built[n]])), "ask");
    raws.forEach((raw, i) => {
      for (const n of group) rows[i][n] = raw[n] === null ? null : readers[n](raw[n]!);
    });
  }
  for (const name of names) {
    const spec = specs[name];
    if (spec.kind === "classify" && (spec.split !== undefined || spec.unsure !== undefined)) {
      const { keys, describe } = labelsOf(spec.labels);
      const answers = await applyPolicies(jev, list, rows.map((r) => r[name] as Answer | null), keys, describe, spec.instructions, contexts[name], spec.split ?? KEEP, spec.unsure ?? KEEP, !!options.detail);
      answers.forEach((a, i) => (rows[i][name] = a));
    }
  }
  const out = rows.map((row) => Object.fromEntries(names.map((n) => [n, options.detail ? row[n] : bare(row[n])])));
  return pack(out, many) as Out<I, Record<string, unknown>>;
}

// ----------------------------------------------------------------------------- classify

/**
 * Assign each item one of the labels, or several with multiLabel.
 * labels: an array, { label: description }, or a Tree (returns "Parent > Child > Leaf").
 */
export async function classify<I>(input: I, labels: Labels, options: ClassifyOptions & { detail: true }): Promise<Out<I, Answer | null>>;
export async function classify<I>(input: I, labels: Labels, options?: ClassifyOptions): Promise<Out<I, string | string[] | null>>;
export async function classify(input: unknown, labels: Labels, options: ClassifyOptions = {}): Promise<unknown> {
  const jev = resolve(options.client);
  const { list, many } = items(input, options.columns);
  const detail = !!options.detail;

  if (labels instanceof Tree) {
    if (options.multiLabel) throw new HunchError("multiLabel doesn't work with a label tree.");
    const answers = await classifyTree(jev, list, labels, options.beam ?? 3, options.instructions, options.context);
    return pack(answers.map((a) => (detail ? a : bare(a))), many);
  }
  const { keys, describe } = labelsOf(labels);
  const states = list.map((item) => buildState(item, options.context));

  if (options.multiLabel) {
    const threshold = options.threshold ?? 0.5;
    const questions = Object.fromEntries(
      keys.map((key, i) => [`l${i}`, noulQuestion({ question: "Does this label apply to the input?", label: key, ...(describe[key] != null ? { definition: describe[key] } : {}), ...(options.instructions ? { guidance: options.instructions } : {}) })]),
    );
    const raws = await run(jev, states, questions, "classify");
    const out = raws.map((raw) => {
      if (keys.some((_, i) => raw[`l${i}`] === null)) return null;
      const probabilities = Object.fromEntries(keys.map((key, i) => [key, (raw[`l${i}`] as { noul: number }).noul]));
      const chosen = keys.filter((k) => probabilities[k] >= threshold).sort((a, b) => probabilities[b] - probabilities[a]);
      const multi: MultiAnswer = { labels: chosen, probabilities };
      return detail ? multi : chosen;
    });
    return pack(out, many);
  }

  const raws = await run(jev, states, { q: choiceQuestion(options.instructions, keys, describe) }, "classify");
  let answers: (Answer | null | string)[] = raws.map((raw) => (raw.q === null ? null : answerOf(jev, raw.q!)));
  if (options.backoff) answers = answers.map((a) => (isAnswer(a) ? backOff(jev, a, options.backoff!) : a));
  answers = await applyPolicies(jev, list, answers, keys, describe, options.instructions, options.context, options.split ?? KEEP, options.unsure ?? KEEP, detail);
  return pack(answers.map((a) => (detail ? a : bare(a))), many);
}

async function applyPolicies(
  jev: Client,
  list: unknown[],
  answers: (Answer | null | string)[],
  keys: string[],
  describe: Record<string, Description>,
  instructions: ClassifyOptions["instructions"],
  context: unknown,
  split: Policy,
  unsure: Policy,
  detail: boolean,
): Promise<(Answer | null | string)[]> {
  if (split === KEEP && unsure === KEEP) return answers;
  const literal = (p: Policy) => p !== KEEP && p !== "rematch" && !isLLM(p);
  if (detail && (literal(split) || literal(unsure))) {
    throw new HunchError("Literal split / unsure values need detail: false; branch on answer.shape for custom logic.");
  }
  let out = [...answers];
  for (const [policy, shape] of [[split, "split"], [unsure, "unsure"]] as const) {
    if (isLLM(policy)) {
      const { escalate } = await import("./combine.js");
      const idxs = out.flatMap((a, i) => (isAnswer(a) && a.shape === shape ? [i] : []));
      out = await escalate(jev, policy as LLMInput, list, idxs, out, keys, describe, instructions, context);
    }
  }
  if (split === "rematch") {
    const groups = new Map<string, number[]>();
    out.forEach((a, i) => {
      if (isAnswer(a) && a.shape === "split") {
        const pair = top2(a).join("\u0000");
        groups.set(pair, [...(groups.get(pair) ?? []), i]);
      }
    });
    for (const [pair, idxs] of groups) {
      const two = pair.split("\u0000");
      const question = choiceQuestion({ question: instructions ?? "Which of these two labels fits the input better?", note: "Only these two options apply." }, two, describe);
      const raws = await run(jev, idxs.map((i) => buildState(list[i], context)), { q: question }, "rematch");
      raws.forEach((raw, k) => {
        if (raw.q) out[idxs[k]] = answerOf(jev, raw.q);
      });
    }
  } else if (literal(split)) {
    out = out.map((a) => (isAnswer(a) && a.shape === "split" ? (split as string) : a));
  }
  if (literal(unsure)) out = out.map((a) => (isAnswer(a) && a.shape === "unsure" ? (unsure as string) : a));
  return out;
}

function backOff(jev: Client, answer: Answer | null, parents: Record<string, string>): Answer | null {
  if (!answer) return answer;
  const tagged: Answer = { ...answer, by: answer.by ?? "jev" };
  if (tagged.shape === "sure") return tagged;
  const mass: Record<string, number> = {};
  for (const [label, p] of Object.entries(tagged.probabilities)) {
    const parent = parents[label] ?? label;
    mass[parent] = (mass[parent] ?? 0) + p;
  }
  const [parent, p] = Object.entries(mass).sort((a, b) => b[1] - a[1])[0];
  if (parent === tagged.label || p < jev.policy.surePeak) return tagged;
  return { label: parent, probabilities: mass, p, confidence: tagged.confidence, shape: jev.policy.classify(mass), by: "parent" };
}

async function classifyTree(jev: Client, list: unknown[], labels: Tree, beam: number, instructions: ClassifyOptions["instructions"], context: unknown): Promise<(Answer | null)[]> {
  if (beam < 1) throw new HunchError("beam must be at least 1.");
  const isNode = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);
  const nodeAt = (path: string[]): unknown => path.reduce<unknown>((here, step) => (here as Record<string, unknown>)[step], labels.nodes);
  let beams: { path: string[]; logp: number }[][] = list.map(() => [{ path: [], logp: 0 }]);

  for (;;) {
    const todo = new Map<string, number[]>();
    beams.forEach((paths, i) => {
      for (const { path } of paths) {
        if (isNode(nodeAt(path))) {
          const key = JSON.stringify(path);
          if (!(todo.get(key) ?? []).includes(i)) todo.set(key, [...(todo.get(key) ?? []), i]);
        }
      }
    });
    if (!todo.size) break;
    const probs = new Map<string, Record<string, number> | null>();
    for (const [key, idxs] of todo) {
      const path = JSON.parse(key) as string[];
      const children = nodeAt(path) as Record<string, unknown>;
      const criteria = Object.fromEntries(Object.entries(children).map(([k, v]) => [k, isNode(v) ? { includes: Object.keys(v) } : ((v as Description) ?? null)]));
      const question: Question = {
        type: "choice",
        instructions: { question: instructions ?? "Which category best describes the input?", ...(path.length ? { within: path.join(" > ") } : {}) },
        criteria,
      };
      const raws = await run(jev, idxs.map((i) => buildState(list[i], context)), { q: question }, `classify ${path.at(-1) ?? "tree"}`);
      raws.forEach((raw, k) => probs.set(`${key}|${idxs[k]}`, raw.q && raw.q.type === "choice" ? raw.q.probabilities : null));
    }
    beams = beams.map((paths, i) => {
      const grown: { path: string[]; logp: number }[] = [];
      for (const { path, logp } of paths) {
        if (!isNode(nodeAt(path))) {
          grown.push({ path, logp });
          continue;
        }
        const p = probs.get(`${JSON.stringify(path)}|${i}`);
        if (!p) continue;
        for (const [child, q] of Object.entries(p)) grown.push({ path: [...path, child], logp: logp + Math.log(Math.max(q, 1e-12)) });
      }
      return grown.sort((a, b) => b.logp / b.path.length - a.logp / a.path.length).slice(0, beam);
    });
  }

  return beams.map((paths) => {
    if (!paths.length) return null;
    const scores = Object.fromEntries(paths.map(({ path, logp }) => [path.join(" > "), Math.exp(logp / path.length)]));
    const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
    const probabilities = Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, v / total]));
    const [best, bestScore] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    return { label: best, probabilities, p: probabilities[best], confidence: bestScore, shape: jev.policy.classify(probabilities) };
  });
}

// ----------------------------------------------------------------------------- score

export interface ScoreOptions extends BaseOptions {
  /** The question, or { dimension: question } to score several in one request. */
  instructions?: string | Record<string, string>;
  detail?: boolean;
}

/** Place each item on an ordered scale (2 to 10 levels, worst first). Returns 0 .. levels.length - 1. */
export async function score<I>(input: I, levels: readonly string[], options: ScoreOptions & { detail: true }): Promise<Out<I, Rating | Record<string, Rating> | null>>;
export async function score<I>(input: I, levels: readonly string[], options?: ScoreOptions): Promise<Out<I, number | Record<string, number> | null>>;
export async function score(input: unknown, levels: readonly string[], options: ScoreOptions = {}): Promise<unknown> {
  const jev = resolve(options.client);
  const { list, many } = items(input, options.columns);
  const criteria = levelsOf(levels);
  const named = typeof options.instructions === "object";
  const dims: Record<string, string> = named ? (options.instructions as Record<string, string>) : { score: (options.instructions as string) ?? "Where on this scale does the input fall?" };
  if (!Object.keys(dims).length) throw new HunchError("Need at least one dimension.");
  const questions = Object.fromEntries(Object.entries(dims).map(([name, text]) => [name, { type: "score" as const, instructions: text, criteria }]));
  const raws = await run(jev, list.map((item) => buildState(item, options.context)), questions, "score");
  const out = raws.map((raw) => {
    const ratings = Object.fromEntries(Object.keys(dims).map((d) => [d, raw[d] === null ? null : ratingOf(jev, raw[d]!)]));
    const shaped = options.detail ? ratings : Object.fromEntries(Object.entries(ratings).map(([k, r]) => [k, r ? r.score : null]));
    return named ? shaped : shaped.score;
  });
  return pack(out, many);
}

// ----------------------------------------------------------------------------- check

export interface CheckOptions extends BaseOptions {
  criteria?: { true?: string | null; false?: string | null };
  threshold?: number;
  /** [low, high]: P(yes) >= high is true, <= low is false, in between is null ("maybe"). */
  uncertain?: [number, number];
  detail?: boolean;
}

/** Does the statement hold for each item? Pass { name: statement } to check several in one request. */
export async function check<I>(input: I, statement: string | Record<string, string>, options: CheckOptions & { detail: true }): Promise<Out<I, Feeling | Record<string, Feeling> | null>>;
export async function check<I>(input: I, statement: string | Record<string, string>, options?: CheckOptions): Promise<Out<I, boolean | null | Record<string, boolean | null>>>;
export async function check(input: unknown, statement: string | Record<string, string>, options: CheckOptions = {}): Promise<unknown> {
  const [low, high] = band(options.uncertain, options.threshold ?? 0.5);
  const jev = resolve(options.client);
  const { list, many } = items(input, options.columns);
  const named = typeof statement === "object";
  const dims: Record<string, string> = named ? statement : { check: statement };
  if (!Object.keys(dims).length) throw new HunchError("A statement is required.");
  const questions = Object.fromEntries(Object.entries(dims).map(([name, text]) => [name, noulQuestion(text, options.criteria)]));
  const raws = await run(jev, list.map((item) => buildState(item, options.context)), questions, "check");
  const out = raws.map((raw) => {
    const feelings = Object.fromEntries(Object.keys(dims).map((d) => [d, raw[d] === null ? null : feeling((raw[d] as { noul: number }).noul, high, low)]));
    const shaped = options.detail ? feelings : Object.fromEntries(Object.entries(feelings).map(([k, f]) => [k, f ? f.value : null]));
    return named ? shaped : shaped.check;
  });
  return pack(out, many);
}

// ----------------------------------------------------------------------------- where

/** Semantic filter: the items the statement holds for, strongest match first. */
export async function where<T>(input: readonly T[], statement: string, options: Omit<CheckOptions, "uncertain"> & { detail: true }): Promise<{ item: T; index: number; match: boolean; p: number | null }[]>;
export async function where<T>(input: readonly T[], statement: string, options?: Omit<CheckOptions, "uncertain">): Promise<T[]>;
export async function where<T>(input: readonly T[], statement: string, options: Omit<CheckOptions, "uncertain"> = {}): Promise<unknown> {
  if (!Array.isArray(input)) throw new HunchError("where() needs an array.");
  const feelings = (await check(input, statement, { ...options, detail: true })) as (Feeling | null)[];
  const rows = input.map((item, index) => ({ item, index, match: !!feelings[index]?.value, p: feelings[index]?.p ?? null }));
  if (options.detail) return rows;
  return rows.filter((r) => r.match).sort((a, b) => (b.p ?? 0) - (a.p ?? 0)).map((r) => r.item);
}

// ----------------------------------------------------------------------------- pick

export interface PickOptions extends BaseOptions {
  /** Also ask whether any candidate fits; return null when P(fits) < noneThreshold. */
  none?: boolean;
  noneThreshold?: number;
  detail?: boolean;
}

/** Choose the single best candidate; Jev compares them head to head. More than 255 run as heats and a final. */
export async function pick<T>(candidates: readonly T[], instructions: string, options: PickOptions & { detail: true }): Promise<Pick<T>>;
export async function pick<T>(candidates: readonly T[], instructions: string, options?: PickOptions): Promise<T | null>;
export async function pick<T>(candidates: readonly T[], instructions: string, options: PickOptions = {}): Promise<unknown> {
  const jev = resolve(options.client);
  const { list } = items(candidates, options.columns);
  if (!list.length) throw new HunchError("pick() needs at least one candidate.");
  if (!instructions?.trim()) throw new HunchError("pick() needs instructions saying what 'best' means.");

  const heat = async (group: number[], final: boolean) => {
    const askFits = !!options.none && final;
    if (group.length === 1 && !askFits) return { winner: group[0], ranked: [{ index: group[0], p: 1 }], confidence: 1, shape: "sure" as const, fits: undefined as number | undefined };
    const criteria = Object.fromEntries(group.map((i) => [`c${i}`, toJson(list[i])]));
    const questions: Record<string, Question> = {};
    if (group.length > 1) questions.q = { type: "choice", instructions: { task: instructions, note: "Each option is one candidate." }, criteria };
    const task: Record<string, unknown> = { task: instructions };
    if (askFits) {
      task.candidates = criteria;
      questions.fits = noulQuestion("Does at least one of the candidates actually satisfy the task well?", {
        true: "at least one candidate is a genuinely good fit for the task",
        false: "none of the candidates is a good fit; picking any would be settling",
      });
    }
    const [raw] = await run(jev, [buildState(task, options.context)], questions, "pick");
    if (Object.values(raw).some((v) => v === null)) throw new HunchError("pick() request failed; see the warning.");
    const fits = askFits ? (raw.fits as { noul: number }).noul : undefined;
    if (!raw.q) return { winner: group[0], ranked: [{ index: group[0], p: 1 }], confidence: 1, shape: "sure" as const, fits };
    const q = raw.q as Extract<Raw, { type: "choice" }>;
    const ranked = Object.entries(q.probabilities).map(([cid, p]) => ({ index: Number(cid.slice(1)), p })).sort((a, b) => b.p - a.p);
    return { winner: Number(q.choice.slice(1)), ranked, confidence: q.confidence, shape: jev.policy.classify(q.probabilities), fits };
  };

  let field = list.map((_, i) => i);
  while (field.length > MAX_CHOICE_OPTIONS) {
    const next: number[] = [];
    for (let i = 0; i < field.length; i += MAX_CHOICE_OPTIONS) next.push((await heat(field.slice(i, i + MAX_CHOICE_OPTIONS), false)).winner);
    field = next;
  }
  const result = await heat(field, true);
  const empty = !!options.none && result.fits !== undefined && result.fits < (options.noneThreshold ?? 0.5);
  const detail: Pick<T> = {
    winner: empty ? null : candidates[result.winner],
    index: empty ? null : result.winner,
    ranked: result.ranked.map((r) => ({ candidate: candidates[r.index], index: r.index, p: r.p })),
    confidence: result.confidence,
    shape: result.shape,
    ...(result.fits !== undefined ? { fits: result.fits } : {}),
  };
  return options.detail ? detail : detail.winner;
}

// ----------------------------------------------------------------------------- rank

export interface RankOptions extends BaseOptions {
  weights?: Record<string, number>;
  /** What the candidates are ranked for; every candidate is scored against it. */
  query?: unknown;
}

/** Score every candidate on each dimension, weight, and sort best first. */
export async function rank<T>(candidates: readonly T[], dimensions: string | Record<string, string>, levels: readonly string[], options: RankOptions = {}): Promise<Ranked<T>[]> {
  if (!Array.isArray(candidates)) throw new HunchError("rank() needs an array of candidates.");
  const dims = typeof dimensions === "string" ? { score: dimensions } : dimensions;
  const names = Object.keys(dims);
  const w = Object.fromEntries(names.map((n) => [n, options.weights?.[n] ?? 1]));
  const total = Object.values(w).reduce((a, b) => a + b, 0);
  if (total <= 0) throw new HunchError("rank() weights must sum to more than zero.");
  const context = options.query === undefined ? options.context : mergeContext(options.context, { query: options.query });
  const ratings = (await score([...candidates], levels, { instructions: dims, context, columns: options.columns, client: options.client, detail: true })) as Record<string, Rating | null>[];
  return ratings
    .map((r, index) => ({
      item: candidates[index],
      index,
      composite: names.some((n) => !r[n]) ? Number.NaN : names.reduce((sum, n) => sum + w[n] * r[n]!.normalized, 0) / total,
      ratings: r,
    }))
    .sort((a, b) => (Number.isNaN(a.composite) ? 1 : Number.isNaN(b.composite) ? -1 : b.composite - a.composite));
}

// ----------------------------------------------------------------------------- helpers

export function labelsOf(labels: Labels): { keys: string[]; describe: Record<string, Description> } {
  if (labels instanceof Tree) throw new HunchError("A Tree only works with classify().");
  if (typeof labels === "string") throw new HunchError("labels must be an array or an object, not a string.");
  const keys = Array.isArray(labels) ? labels.map(String) : Object.keys(labels);
  const describe = Array.isArray(labels) ? {} : (labels as Record<string, Description>);
  if (!keys.length) throw new HunchError("classify() needs at least one label.");
  if (new Set(keys).size !== keys.length) throw new HunchError("Labels must be unique.");
  if (keys.length > MAX_CHOICE_OPTIONS) throw new HunchError(`classify() takes at most ${MAX_CHOICE_OPTIONS} labels.`);
  return { keys, describe };
}

function levelsOf(levels: readonly string[]): string[] {
  if (typeof levels === "string") throw new HunchError("levels must be an array of level descriptions.");
  if (levels.length < 2 || levels.length > 10) throw new HunchError("levels needs 2–10 entries.");
  return levels.map(String);
}

export function choiceQuestion(instructions: unknown, keys: readonly string[], describe: Record<string, Description>): Question {
  return {
    type: "choice",
    instructions: instructions ?? "Which label best describes the input?",
    criteria: Object.fromEntries(keys.map((k) => [k, describe[k] ?? null])),
  };
}

export function noulQuestion(instructions: unknown, criteria?: { true?: string | null; false?: string | null }): Question {
  return { type: "noul", instructions, ...(criteria ? { criteria: { true: criteria.true ?? null, false: criteria.false ?? null } } : {}) };
}

function band(uncertain: [number, number] | undefined, threshold: number): [number | undefined, number] {
  if (!uncertain) return [undefined, threshold];
  const [low, high] = uncertain;
  if (!(0 <= low && low < high && high <= 1)) throw new HunchError("uncertain needs [low, high] with 0 <= low < high <= 1.");
  return [low, high];
}

export function answerOf(jev: Client, raw: Raw): Answer {
  if (raw.type !== "choice") throw new HunchError("Expected a choice answer.");
  return { label: raw.choice, probabilities: raw.probabilities, p: raw.probabilities[raw.choice] ?? 0, confidence: raw.confidence, shape: jev.policy.classify(raw.probabilities) };
}

function ratingOf(jev: Client, raw: Raw): Rating {
  if (raw.type !== "score") throw new HunchError("Expected a score answer.");
  const legend = Object.fromEntries(Object.entries(raw.legend).map(([k, v]) => [Number(k), v]));
  const probabilities = Object.fromEntries(Object.entries(raw.probabilities).map(([k, v]) => [Number(k), v]));
  const levels = Object.keys(legend).map(Number);
  const nearest = levels.reduce((best, k) => (Math.abs(k - raw.score) < Math.abs(best - raw.score) ? k : best), levels[0]);
  const top = Math.max(...levels);
  const byLevel = Object.fromEntries(Object.entries(probabilities).map(([k, p]) => [legend[Number(k)], p]));
  return { score: raw.score, level: legend[nearest], normalized: top ? raw.score / top : 0, probabilities, confidence: raw.confidence, shape: jev.policy.classify(byLevel) };
}

export function isAnswer(value: unknown): value is Answer {
  return value !== null && typeof value === "object" && "label" in (value as object) && "shape" in (value as object);
}

function top2(answer: Answer): string[] {
  return Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k]) => k);
}

function bare(value: unknown): unknown {
  if (value === null || value === undefined) return null;
  if (isAnswer(value)) return value.label;
  if (typeof value === "object" && "score" in (value as object) && "level" in (value as object)) return (value as Rating).score;
  if (typeof value === "object" && "verdict" in (value as object)) return (value as Feeling).value;
  if (typeof value === "object" && "labels" in (value as object)) return (value as MultiAnswer).labels;
  return value;
}
