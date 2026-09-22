/**
 * hunch: Jev judgments as TypeScript functions over arrays.
 * Jev decides; your code owns the workflow; an LLM, if configured, only proposes.
 */

export type { Answer, Feeling, MultiAnswer, Pick, Ranked, Rating } from "./answer.js";
export { type CacheStore, MemoryCache, fileCache } from "./cache.js";
export { Client, type ClientOptions, type JevLike, type Usage, configure, defaultClient } from "./client.js";
export { type DiscoverOptions, type GenerateOptions, type Refined, type RefineOptions, type Verdict, VERDICTS, discover, generate, refine, verify } from "./combine.js";
export { type Plan, dryRun } from "./engine.js";
export { HunchError } from "./errors.js";
export { type Evaluation, type Threshold, evaluate, tuneThreshold } from "./evaluate.js";
export { type FieldSpec, type Finder, PATTERNS, extract } from "./extract.js";
export { type LLMInput, type LanguageModel, anthropic, azure, cerebras, ollama, openAICompat, openai, openrouter } from "./llm.js";
export { type Shape, ShapePolicy } from "./shapes.js";
export {
  Check,
  type CheckOptions,
  type CheckSpec,
  Classify,
  type ClassifyOptions,
  type ClassifySpec,
  type Description,
  type Labels,
  type Out,
  type PickOptions,
  Rate,
  type RankOptions,
  type RateSpec,
  type ScoreOptions,
  type Spec,
  Tree,
  ask,
  check,
  classify,
  pairs,
  pick,
  rank,
  score,
  tree,
  where,
} from "./verbs.js";

export const VERSION = "0.1.0";
