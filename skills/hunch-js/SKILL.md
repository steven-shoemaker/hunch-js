---
name: hunch-js
description: >
  Use the hunch TypeScript library (npm install hunch-jev) to put judgment into code: label,
  score, filter, extract, rank, or verify strings and arrays with Jev (TypeSafe's classifier),
  with an optional LLM that only proposes. Use when a TypeScript/JavaScript task needs semantic
  judgment over data, such as categorizing items, filtering by meaning, triage, pulling values
  out of text, picking the best draft, finding themes, checking an LLM's output against a
  source, or replacing prompt-then-parse LLM code with typed answers.
---

# Using hunch in TypeScript

Every verb is async. One value in gives one answer; an array in gives an array in the same
order. Objects are rows: Jev sees every field unless you pass `columns`. Jev decides; an LLM,
if configured, only proposes. Code owns thresholds, weights, and actions.

```ts
import * as hunch from "hunch-jev";                 // npm install hunch-jev
hunch.configure({ apiKey: process.env.TYPESAFE_API_KEY, llm: hunch.anthropic(), cache: ".hunch-cache" });
```

LLM adapters: `hunch.anthropic()` (needs @anthropic-ai/sdk), `hunch.openai()`, `hunch.azure({ deployment })`,
`hunch.openrouter()`, `hunch.ollama("model")`, or any `(system, user) => text`. Only `generate`,
`discover`, `refine`, and escalation need one. Never hardcode keys.
To reach Jev through OpenRouter or Vercel AI Gateway: `configure({ gateway: "openrouter" | "vercel" })`.

## Pick the verb

| The task | Call |
| --- | --- |
| One of known categories | `classify(items, labels)`; labels as `{ label: description }` when names are ambiguous |
| Several may apply | `classify(items, labels, { multiLabel: true })` |
| Categories form a hierarchy | `classify(items, new hunch.Tree({...}))` or `{ backoff: { child: parent } }` |
| Ordered scale (severity, fit) | `score(items, levels)`; levels are 2 to 10 concrete situations, worst first |
| Yes / no | `check(items, statement)`; `{ uncertain: [0.3, 0.7] }` returns null for maybe |
| Keep matching items | `where(items, statement)` |
| Pull a value out of text | `extract(items, { total: ["money", "the amount due"], due: "date" })` |
| Several questions per item | `ask(items, { a: Classify([...]), b: Rate([...]), c: Check("...") })` |
| Best of N, or none | `pick(candidates, "what best means", { none: true })` |
| Order on weighted criteria | `rank(candidates, { dim: "question" }, levels, { weights, query })` |
| Compare two things item by item | any verb on `hunch.pairs(a, b)` |
| Turn answers into actions (queues, alerts) | `route(answers, { page: { urgent: 0.8 }, billing: { topic: ["billing", 0.7] } }, { default })` |
| Is an LLM/agent claim supported? | `verify(claims, source)` returns supported / contradicted / not mentioned / misquoted |
| Unknown categories | `discover(items, n)` then `classify(items, result)` |
| Draft text that must meet rules | `refine(text, checks, { context })` |
| Fake or seed data | `generate({ n, instructions, schema, parse })` |
| Is it accurate on my data? | `evaluate(pred, truth)`, `tuneThreshold(p, truth, { precision: 0.9 })` |

## Uncertainty

`{ detail: true }` returns probabilities and a shape: `sure`, `split` (two close), `unsure` (flat).

```ts
await hunch.classify(titles, LEVELS, { split: "rematch", unsure: "review" });
await hunch.classify(titles, LEVELS, { unsure: hunch.anthropic() });   // escalate only hard items
```

Confidence means a peaked distribution, not a correct answer. Label 50 to 100 items and run
`evaluate` before trusting a pipeline; pick `check` / `where` cutoffs with `tuneThreshold`.

## Writing good questions

- Include an `other` label when inputs may fit nothing.
- `where` / `check` statements about evidence filter well; predictions ("might cancel") sit near
  0.3 to 0.4 when the item says nothing, so rank them (`detail: true`, sort by `p`).
- In `ask`, put context only on the spec that needs it, or it colors every answer.
- `refine` checks only test what you list: add an accuracy check plus the facts as `context`.

## Scale

- Pass whole arrays; never loop calling a verb per item. Duplicates are asked once and cached.
- `configure({ maxConcurrency, maxRps, errors: "skip", onProgress })`. With `errors: "skip"`,
  failed items are null and a rerun re-sends only those.
- `await hunch.dryRun(async () => { ... })` counts requests without sending anything.

## Don'ts

- Don't prompt an LLM and parse JSON for a closed-set decision; use classify / check / score / extract.
- Don't let an LLM make the final call; route it through pick, verify, or escalation.
