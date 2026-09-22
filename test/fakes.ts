import { Client, type ClientOptions } from "../src/index.js";

export type Q = { type: "choice" | "noul" | "score"; instructions?: unknown; criteria?: unknown };
export type Handler = (state: any, questions: Record<string, Q>) => Record<string, unknown>;

/** Stands in for TypeSafeClient. handler(state, questions) returns { qid: answer }. */
export class FakeJev {
  calls: { state: any; questions: Record<string, Q> }[] = [];
  constructor(private handler: Handler) {}
  async systemOne({ state, questions }: { state: unknown; questions: Record<string, unknown> }) {
    this.calls.push({ state, questions: questions as Record<string, Q> });
    return { model: "jev-test", usage: { input_tokens: 10, output_tokens: 2 }, answers: this.handler(state, questions as Record<string, Q>) };
  }
}

export const choice = (pick: string, probabilities: Record<string, number>, confidence = 0.85) => ({ type: "choice", choice: pick, probabilities, confidence });
export const noul = (p: number) => ({ type: "noul", noul: p });
export const scored = (value: number, probabilities: Record<string, number>, legend: Record<string, string>, confidence = 0.9) => ({
  type: "score",
  score: value,
  probabilities,
  legend,
  confidence,
});

/** Pick the first option confidently for every choice; yes for nouls; top level for scores. */
export const firstOption: Handler = (_state, questions) =>
  Object.fromEntries(
    Object.entries(questions).map(([id, q]) => {
      if (q.type === "choice") {
        const keys = Object.keys(q.criteria as object);
        return [id, choice(keys[0], Object.fromEntries(keys.map((k, i) => [k, i === 0 ? 0.9 : 0.1 / Math.max(1, keys.length - 1)])))];
      }
      if (q.type === "noul") return [id, noul(0.9)];
      const levels = q.criteria as string[];
      const top = levels.length - 1;
      return [id, scored(top, { [top]: 1 }, Object.fromEntries(levels.map((l, i) => [i, l])))];
    }),
  );

export class FakeLLM {
  readonly name = "fake";
  calls: { system: string; user: string }[] = [];
  private queue: string[];
  constructor(replies: string | string[]) {
    this.queue = Array.isArray(replies) ? [...replies] : [replies];
  }
  async complete({ system, user }: { system: string; user: string }) {
    this.calls.push({ system, user });
    const next = this.queue.shift();
    if (next === undefined) throw new Error("FakeLLM has no replies left.");
    return next;
  }
}

export const client = (handler: Handler, options: ClientOptions = {}) => {
  const fake = new FakeJev(handler);
  return { fake, jev: new Client({ client: fake, onWarning: () => {}, ...options }) };
};
