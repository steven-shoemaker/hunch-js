import type { Shape } from "./shapes.js";

/** A resolved Choice: one label from a closed set, plus the whole distribution. */
export interface Answer<L = string> {
  label: L;
  /** P(option) for every option, keyed by the option's string form. */
  probabilities: Record<string, number>;
  /** P of the chosen label. */
  p: number;
  /** How peaked the distribution is. Not whether the label is true. */
  confidence: number;
  shape: Shape;
  /** Who chose the label when an LLM policy or backoff was set: "jev", "llm", "parent", "quote check". */
  by?: string;
}

/** A resolved Score: a position along ordered levels. */
export interface Rating {
  /** Probability-weighted position, 0 .. levels.length - 1. Can land between levels. */
  score: number;
  /** The nearest level. */
  level: string;
  /** score rescaled to 0..1 so ratings with different level counts compare. */
  normalized: number;
  probabilities: Record<number, number>;
  confidence: number;
  shape: Shape;
}

/** A resolved Noul: P(yes), and the verdict under the call's threshold. */
export interface Feeling {
  p: number;
  /** true / false, or null for "maybe" when check() was given uncertain=[low, high]. */
  value: boolean | null;
  verdict: "yes" | "no" | "maybe";
}

export interface MultiAnswer<L = string> {
  labels: L[];
  probabilities: Record<string, number>;
}

export interface Pick<T> {
  /** The chosen candidate, or null when pick({ none: true }) found nothing that fits. */
  winner: T | null;
  /** Index of the winner in the input, or null. */
  index: number | null;
  ranked: { candidate: T; index: number; p: number }[];
  confidence: number;
  shape: Shape;
  /** With none: true, P(at least one candidate satisfies the task). */
  fits?: number;
}

export interface Ranked<T> {
  item: T;
  index: number;
  /** Weighted mean of normalized dimension scores, 0..1. NaN when a request was skipped. */
  composite: number;
  ratings: Record<string, Rating | null>;
}

export function feeling(p: number, threshold: number, low?: number): Feeling {
  const verdict = p >= threshold ? "yes" : low === undefined || p <= low ? "no" : "maybe";
  return { p, verdict, value: verdict === "yes" ? true : verdict === "no" ? false : null };
}
