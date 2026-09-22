/** extract(): code finds candidate values in the text; Jev picks which one is the answer. Values are always copied from the text. */

import type { Answer } from "./answer.js";
import { resolve } from "./client.js";
import { buildState, pool, run } from "./engine.js";
import { HunchError } from "./errors.js";
import type { BaseOptions, Out } from "./verbs.js";

const MONTH = String.raw`(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?`;

/** Built-in candidate finders. Use a name ("email"), your own RegExp, or a function. */
export const PATTERNS: Record<string, string> = {
  email: String.raw`[\w.+-]+@[\w-]+(?:\.[\w-]+)+`,
  url: String.raw`https?://[^\s<>"')]+`,
  money: String.raw`(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b)`,
  number: String.raw`-?\d[\d,]*(?:\.\d+)?%?`,
  percent: String.raw`-?\d+(?:\.\d+)?\s?%`,
  phone: String.raw`\+?\d[\d\s().-]{7,}\d`,
  date: String.raw`\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|${MONTH}\s\d{1,2}(?:st|nd|rd|th)?,?\s\d{4}|\d{1,2}(?:st|nd|rd|th)?\s${MONTH},?\s\d{4})\b`,
};

export type Finder = string | RegExp | ((text: string) => Iterable<string>);
export type FieldSpec = Finder | [Finder, string];

const MAX_CANDIDATES = 254;
const NONE = "none";

/**
 * Pull named values out of text. Returns null for a field the text doesn't state.
 *
 *   await extract(invoice, { total: ["money", "the amount due, not a subtotal"], due: "date", email: "email" })
 *
 * Every candidate goes to Jev with the words around it, plus a "none" option; all fields for an
 * item go in one request, and an item with no candidates for a field costs nothing for it.
 */
export async function extract<I>(input: I, fields: Record<string, FieldSpec>, options: BaseOptions & { detail: true }): Promise<Out<I, Record<string, Answer<string | null> | null>>>;
export async function extract<I>(input: I, fields: Record<string, FieldSpec>, options?: BaseOptions & { detail?: boolean }): Promise<Out<I, Record<string, string | null>>>;
export async function extract(input: unknown, fields: Record<string, FieldSpec>, options: BaseOptions & { detail?: boolean } = {}): Promise<unknown> {
  const names = Object.keys(fields);
  if (!names.length) throw new HunchError("extract() needs at least one field.");
  const jev = resolve(options.client);
  const specs = Object.fromEntries(names.map((name) => [name, spec(name, fields[name])]));
  const many = Array.isArray(input);
  const texts = (many ? (input as unknown[]) : [input]).map((item) => textOf(item, options.columns));

  const one = async (text: string) => {
    const questions: Record<string, { type: "choice"; instructions: unknown; criteria: unknown }> = {};
    const options_: Record<string, Record<string, string>> = {};
    for (const [name, { find, description }] of Object.entries(specs)) {
      const found = candidates(text, find);
      if (!found.length) continue;
      const label = name.replace(/_/g, " ");
      options_[name] = Object.fromEntries(found.map(([value], i) => [`c${i}`, value]));
      questions[name] = {
        type: "choice",
        instructions: {
          question: `Which candidate is the ${label} the text states?`,
          ...(description ? { field: description } : {}),
          rules: `Pick the candidate the text gives as the ${label}. Pick none if the text doesn't state a ${label}, or if no candidate is it.`,
        },
        criteria: {
          ...Object.fromEntries(found.map(([value, inContext], i) => [`c${i}`, { value, in_context: inContext }])),
          [NONE]: `the text doesn't state a ${label}, or none of the candidates is it`,
        },
      };
    }
    const row: Record<string, Answer<string | null> | null> = Object.fromEntries(names.map((n) => [n, null]));
    if (!Object.keys(questions).length) return row;
    const [raw] = await run(jev, [buildState(text, options.context)], questions, "extract");
    for (const name of Object.keys(questions)) {
      const answer = raw[name];
      if (!answer || answer.type !== "choice") continue;
      const ids = options_[name];
      const probabilities: Record<string, number> = {};
      for (const [cid, p] of Object.entries(answer.probabilities)) {
        const key = cid === NONE ? "null" : ids[cid];
        probabilities[key] = (probabilities[key] ?? 0) + p;
      }
      const value = answer.choice === NONE ? null : ids[answer.choice];
      row[name] = { label: value, probabilities, p: probabilities[String(value)] ?? 0, confidence: answer.confidence, shape: jev.policy.classify(probabilities) };
    }
    return row;
  };

  const distinct = [...new Set(texts)];
  const done = await pool(distinct, jev.maxConcurrency, one);
  const byText = new Map(distinct.map((t, i) => [t, done[i]]));
  const rows = texts.map((t) => {
    const row = byText.get(t)!;
    return options.detail ? row : Object.fromEntries(Object.entries(row).map(([k, a]) => [k, a ? a.label : null]));
  });
  return many ? rows : rows[0];
}

function spec(name: string, field: FieldSpec): { find: (text: string) => string[]; description?: string } {
  let finder: Finder = field as Finder;
  let description: string | undefined;
  if (Array.isArray(field)) {
    if (field.length !== 2) throw new HunchError(`extract() field ${JSON.stringify(name)}: use [finder, description].`);
    [finder, description] = field as [Finder, string];
  }
  if (typeof finder === "string") {
    try {
      finder = new RegExp(PATTERNS[finder] ?? finder, "g");
    } catch (error) {
      throw new HunchError(`extract() field ${JSON.stringify(name)}: ${JSON.stringify(finder)} is not a known finder or a valid regex (${String(error)}).`);
    }
  }
  if (finder instanceof RegExp) {
    const re = finder.flags.includes("g") ? finder : new RegExp(finder.source, `${finder.flags}g`);
    return { find: (text) => [...text.matchAll(re)].map((m) => m[0]), description };
  }
  if (typeof finder === "function") {
    const fn = finder;
    return { find: (text) => [...fn(text)].map(String), description };
  }
  throw new HunchError(`extract() field ${JSON.stringify(name)} must be a finder name, RegExp, or function.`);
}

function candidates(text: string, find: (text: string) => string[]): [string, string][] {
  const out = new Map<string, string>();
  for (const raw of find(text)) {
    const value = raw.trim();
    if (!value || out.has(value)) continue;
    const at = text.indexOf(value);
    const start = Math.max(0, at - 60);
    const end = at + value.length + 60;
    out.set(value, (start ? "…" : "") + text.slice(start, end).replace(/\n/g, " ") + (end < text.length ? "…" : ""));
    if (out.size >= MAX_CANDIDATES) break;
  }
  return [...out.entries()];
}

function textOf(item: unknown, columns?: readonly string[]): string {
  if (item === null || item === undefined) return "";
  if (typeof item === "object") {
    const entries = Object.entries(item as Record<string, unknown>).filter(([k, v]) => v != null && (!columns || columns.includes(k)));
    return entries.map(([k, v]) => `${k}: ${v}`).join("\n");
  }
  return String(item);
}

export { pool };
