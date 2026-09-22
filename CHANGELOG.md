# Changelog

## 0.2.0

- `route(answers, rules, { default })` turns answers into outcomes with ordered rules: minimum probabilities, labels, `[label, p]` pairs, label lists, shapes, or functions. Sends no requests.
- `configure({ gateway: "openrouter" | "vercel" })` reaches Jev through OpenRouter or Vercel AI Gateway, with retries. `openRouterJev()` and `vercelJev()` are exported for custom clients.

## 0.1.0

First release: a TypeScript port of [hunch for Python](https://github.com/steven-shoemaker/hunch) 0.9.

- Jev verbs: `classify` (labels, descriptions, multi-label, `Tree`, `backoff`, `split` / `unsure` policies), `score`, `check` (`uncertain`), `where`, `extract`, `ask` with `Classify` / `Rate` / `Check`, `pick` (`none`), `rank` (`weights`, `query`), `pairs`, `verify`.
- LLM verbs: `generate` (JSON Schema + `parse`), `discover`, `refine`, and escalation of shaky items.
- `evaluate` and `tuneThreshold`.
- Dedupe, memory or file cache (or any `get`/`set` store), `maxConcurrency`, `maxRps`, `errors: "skip"`, `dryRun`, `onProgress`.
- LLM adapters: `anthropic` (official SDK), `openai`, `azure`, `openrouter`, `cerebras`, `ollama`, or any function.
