/** route(): turn answers into outcomes with ordered rules. No requests; it reads answers you already have. */

import type { Answer, Feeling, MultiAnswer, Rating } from "./answer.js";
import { HunchError } from "./errors.js";

/**
 * A condition on one answer:
 * - number: P(yes), P of the chosen label, or the score is at least this
 * - string: the label is this (or, for a name ending in .shape, the shape is this)
 * - [label, minP]: the label is this with probability at least minP
 * - string[]: the label is one of these
 * - boolean: a check came back true / false (never matches null, i.e. "maybe" or skipped)
 * - function: your own test on the answer
 */
export type Condition = number | string | boolean | [string, number] | readonly string[] | ((answer: any) => boolean);
export type Rules = Record<string, Record<string, Condition>>;

const MISSING = Symbol("missing");

/**
 * Pick an outcome for each item: the first rule whose conditions all hold, else fallback.
 *
 *   hunch.route(answers, {
 *     page:    { urgent: 0.8 },
 *     billing: { topic: ["billing", 0.7] },
 *     review:  { "topic.shape": "unsure" },
 *   }, { default: "triage" })
 *
 * answers: what ask() / classify() / check() / score() returned, ideally with detail: true.
 */
export function route<A>(answers: A, rules: Rules, options: { default?: string | null } = {}): A extends readonly unknown[] ? (string | null)[] : string | null {
  const entries = Object.entries(rules);
  if (!entries.length) throw new HunchError("route() needs at least one rule.");
  const compiled = entries.map(([outcome, conds]) => [outcome, Object.entries(conds).map(([name, cond]) => [name, test(name, cond)] as const)] as const);
  const fallback = options.default ?? null;
  const one = (row: unknown) => {
    const record = row !== null && typeof row === "object" && !Array.isArray(row) && !isAnswerLike(row) ? (row as Record<string, unknown>) : { _: row };
    for (const [outcome, tests] of compiled) if (tests.every(([name, t]) => t(read(record, name)))) return outcome;
    return fallback;
  };
  return (Array.isArray(answers) ? answers.map(one) : one(answers)) as never;
}

function isAnswerLike(v: unknown): boolean {
  return typeof v === "object" && v !== null && ("shape" in v || "verdict" in v || ("labels" in v && "probabilities" in v));
}

function read(row: Record<string, unknown>, name: string): unknown {
  const dot = name.lastIndexOf(".");
  const part = dot > 0 ? name.slice(dot + 1) : "";
  const known = ["p", "shape", "level", "score", "label"].includes(part);
  const base = known ? name.slice(0, dot) : name;
  let value: unknown = base in row ? row[base] : Object.keys(row).length === 1 && "_" in row ? row._ : MISSING;
  if (value === MISSING || value === null || value === undefined) return value ?? null;
  if (!known) return value;
  return typeof value === "object" && part in (value as object) ? (value as Record<string, unknown>)[part] : MISSING;
}

function labelOf(v: any): unknown {
  if (v && typeof v === "object") {
    if ("label" in v) return v.label;
    if ("verdict" in v) return v.value;
  }
  return v;
}

function probOf(v: any): number | null {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v && typeof v === "object") {
    if ("level" in v && "score" in v) return (v as Rating).score;
    if ("p" in v) return (v as Answer | Feeling).p;
  }
  return null;
}

function labelsOf(v: any): unknown[] {
  return v && typeof v === "object" && Array.isArray((v as MultiAnswer).labels) ? (v as MultiAnswer).labels : Array.isArray(v) ? v : [labelOf(v)];
}

function test(name: string, cond: Condition): (value: unknown) => boolean {
  const ok = (v: unknown) => v !== MISSING && v !== null && v !== undefined;
  if (typeof cond === "function") return (v) => ok(v) && !!cond(v);
  if (typeof cond === "boolean") return (v) => ok(v) && labelOf(v) === cond;
  if (typeof cond === "number") return (v) => ok(v) && (probOf(v) ?? -Infinity) >= cond;
  if (Array.isArray(cond)) {
    if (cond.length === 2 && typeof cond[1] === "number") {
      const [label, floor] = cond as [string, number];
      return (v) => ok(v) && labelOf(v) === label && (probOf(v) ?? 0) >= floor;
    }
    if (cond.some((c) => typeof c !== "string")) throw new HunchError(`route() condition for ${JSON.stringify(name)}: use [label, minP] or a list of labels.`);
    return (v) => ok(v) && labelsOf(v).some((l) => (cond as readonly string[]).includes(String(l)));
  }
  return (v) => ok(v) && labelsOf(v).some((l) => String(l) === cond);
}
