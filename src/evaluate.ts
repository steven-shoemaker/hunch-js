/** Measure hunch against labeled data: how often it's right, where it's wrong, which cutoff to use. */

import type { Answer, Feeling } from "./answer.js";
import { HunchError } from "./errors.js";

export interface Evaluation {
  n: number;
  accuracy: number;
  /** "truth -> predicted" -> count. */
  confusion: Record<string, number>;
  /** shape -> { n, accuracy }. Empty unless predictions carry shapes (detail: true). */
  byShape: Record<string, { n: number; accuracy: number }>;
  errors: { index: number; truth: string; predicted: string }[];
  /** Rows with no prediction (skipped requests). */
  missing: number;
}

/** Compare classify() output (labels or Answers from detail: true) with the true labels. */
export function evaluate(predicted: readonly (string | Answer | null)[], truth: readonly unknown[]): Evaluation {
  if (predicted.length !== truth.length) throw new HunchError(`predictions and truth differ in length (${predicted.length} vs ${truth.length}).`);
  const confusion: Record<string, number> = {};
  const shapes: Record<string, boolean[]> = {};
  const errors: Evaluation["errors"] = [];
  let missing = 0;
  let correct = 0;
  predicted.forEach((p, index) => {
    if (p === null || p === undefined) {
      missing += 1;
      return;
    }
    const got = String(typeof p === "object" ? p.label : p);
    const want = String(truth[index]);
    confusion[`${want} -> ${got}`] = (confusion[`${want} -> ${got}`] ?? 0) + 1;
    if (typeof p === "object") (shapes[p.shape] ??= []).push(got === want);
    if (got === want) correct += 1;
    else errors.push({ index, truth: want, predicted: got });
  });
  const n = predicted.length - missing;
  if (!n) throw new HunchError("evaluate() found no rows with both a prediction and a truth.");
  const order = ["sure", "split", "unsure"];
  const byShape = Object.fromEntries(
    Object.entries(shapes)
      .sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]))
      .map(([s, v]) => [s, { n: v.length, accuracy: v.filter(Boolean).length / v.length }]),
  );
  return { n, accuracy: correct / n, confusion, byShape, errors, missing };
}

export interface Threshold {
  threshold: number;
  precision: number;
  recall: number;
  n: number;
  positives: number;
}

/**
 * Find the cutoff for check() / where() from labeled rows. probabilities: P(yes) per row, as numbers
 * or Feelings from check({ detail: true }). precision: the lowest cutoff keeping that share of matches
 * correct. recall: the highest cutoff still catching that share. Neither: maximize F1.
 */
export function tuneThreshold(
  probabilities: readonly (number | Feeling | null)[],
  truth: readonly boolean[],
  { precision, recall }: { precision?: number; recall?: number } = {},
): Threshold {
  if (precision !== undefined && recall !== undefined) throw new HunchError("Pass precision or recall, not both.");
  if (probabilities.length !== truth.length) throw new HunchError("probabilities and truth differ in length.");
  const rows = probabilities.flatMap((p, i) => (p === null ? [] : [{ p: typeof p === "number" ? p : p.p, t: !!truth[i] }]));
  if (!rows.length) throw new HunchError("tuneThreshold() found no rows with both a probability and a truth.");
  const positives = rows.filter((r) => r.t).length;
  if (!positives) throw new HunchError("tuneThreshold() needs at least one true row.");
  const at = (cut: number): Threshold => {
    const kept = rows.filter((r) => r.p >= cut);
    const hits = kept.filter((r) => r.t).length;
    return { threshold: cut, precision: kept.length ? hits / kept.length : 1, recall: hits / positives, n: rows.length, positives };
  };
  const candidates = [...new Set(rows.map((r) => r.p))].sort((a, b) => a - b).map(at);
  if (precision !== undefined) {
    const ok = candidates.filter((c) => c.precision >= precision);
    if (!ok.length) {
      const best = candidates.reduce((a, b) => (b.precision > a.precision ? b : a));
      throw new HunchError(`No cutoff reaches precision ${precision}; best is ${best.precision.toFixed(2)} at ${best.threshold.toFixed(2)}.`);
    }
    return ok[0];
  }
  if (recall !== undefined) return candidates.filter((c) => c.recall >= recall).at(-1)!;
  const f1 = (c: Threshold) => (c.precision + c.recall ? (2 * c.precision * c.recall) / (c.precision + c.recall) : 0);
  return candidates.reduce((a, b) => (f1(b) > f1(a) ? b : a));
}
