# Changelog

## 0.1.0

First release: a TypeScript port of [hunch for Python](https://github.com/steven-shoemaker/hunch) 0.9.

- Jev verbs: `classify` (labels, descriptions, multi-label, `Tree`, `backoff`, `split` / `unsure` policies), `score`, `check` (`uncertain`), `where`, `extract`, `ask` with `Classify` / `Rate` / `Check`, `pick` (`none`), `rank` (`weights`, `query`), `pairs`, `verify`.
- LLM verbs: `generate` (JSON Schema + `parse`), `discover`, `refine`, and escalation of shaky items.
- `evaluate` and `tuneThreshold`.
- Dedupe, memory or file cache (or any `get`/`set` store), `maxConcurrency`, `maxRps`, `errors: "skip"`, `dryRun`, `onProgress`.
- LLM adapters: `anthropic` (official SDK), `openai`, `azure`, `openrouter`, `cerebras`, `ollama`, or any function.
