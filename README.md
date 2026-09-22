# hunch for TypeScript

![hunch — lists in, lists out](https://raw.githubusercontent.com/steven-shoemaker/hunch/main/docs/banner.png)

**Judgment as a TypeScript function.** Ask a question about a string or a whole array, and get an answer for every item that your code can branch on.

```ts
import * as hunch from "hunch-jev";

const themes = await hunch.classify(reviews, ["delivery", "food quality", "pricing", "app"]);
const angry = await hunch.where(reviews, "the customer is angry");
const totals = await hunch.extract(invoices, { total: "money", due: "date" });
```

```bash
npm install hunch-jev
```

Node 20 or newer. Also available [for Python](https://github.com/steven-shoemaker/hunch) (`pip install hunch-jev`), with the same verbs.

Using a coding agent? Give it the skill so it uses these verbs instead of writing prompt-and-parse code:

```bash
npx skills add steven-shoemaker/hunch-js --skill hunch-js
```

## How it works

**Jev does the judging.** [Jev](https://docs.typesafe.ai) is TypeSafe's classifier with frontier-level intelligence, and it needs no fine-tuning. You send it some state and a typed question. It answers with a probability distribution instead of prose. It knows three kinds of question, and every hunch verb is built from them:

| Jev primitive | Answers | Used by |
| --- | --- | --- |
| Choice | one option from a closed set | `classify`, `pick`, `extract`, `verify` |
| Score | a position on ordered levels | `score`, `rank` |
| Noul | the probability that a statement is true | `check`, `where`, `classify({ multiLabel: true })` |

**Your data goes in and comes back in the same shape.** One value returns one answer. An array returns an array, in the same order. An object is treated as a row, so Jev sees every field unless you pass `columns`. Anything in `context`, such as a policy, an ICP, or a diff, rides along with every item.

**One request per distinct item.** Duplicate values are asked once. Several questions about the same item go in a single request. Requests run in parallel, 16 at a time by default, and answers are cached.

**Every answer keeps its probabilities.** By default you get the plain answer. `detail: true` gives the whole distribution, plus a **shape**: `sure` when one option dominates, `split` when two are close, and `unsure` when the evidence is flat. Shapes tell your code which items to trust, re-ask, or send to a person.

**An LLM may propose, but only Jev decides.** A few verbs use an LLM to write things: fake data, category names, rewrites, and second opinions on hard items. Jev still makes every call, and your code keeps the thresholds and weights. If you never configure an LLM, you never need one.

## Setup

```ts
import * as hunch from "hunch-jev";

hunch.configure({ apiKey: process.env.TYPESAFE_API_KEY });   // or just set TYPESAFE_API_KEY
hunch.configure({ llm: hunch.anthropic(), cache: ".hunch-cache" });
```

The LLM can be any of these. Only `generate`, `discover`, `refine`, and escalation use it.

```ts
hunch.anthropic()                          // Claude via @anthropic-ai/sdk (npm install @anthropic-ai/sdk)
hunch.openai({ model: "gpt-5-mini" })
hunch.azure({ deployment: "my-gpt" })      // Azure OpenAI
hunch.openrouter({ model: "z-ai/glm-5.3-flash" })
hunch.ollama("qwen3")                      // a local model
(system, user) => myGateway(system, user)  // anything else
```

Every verb also takes `{ client }` if you'd rather not use the default.

## The verbs

### classify: which label?

```ts
await hunch.classify("This product is amazing!", ["positive", "negative", "neutral"]);   // "positive"
await hunch.classify(titles, { Sales: "sells to customers", Engineering: "builds the product" });
```

Labels can be an array, or an object of label to description for when the names alone are ambiguous. Useful options:

- `multiLabel: true` asks one yes/no per label and returns every label that applies.
- `split: "rematch"` re-asks between the top two labels, only for items where those two were close.
- `unsure: "review"` returns that value for items where the evidence was flat. Passing an LLM instead, like `unsure: hunch.anthropic()`, sends only those items to the LLM, which must choose from the same labels.
- `backoff: { Laptops: "Computers", Tablets: "Computers" }` answers with the parent when Jev can't tell the children apart.
- For a taxonomy, pass a `Tree`. Jev walks it level by level, keeping the best few paths:

```ts
const catalog = new hunch.Tree({
  Electronics: { Phones: null, Computers: { Laptops: null, Tablets: null } },
  Home: { Kitchen: null, Furniture: null },
});
await hunch.classify("MacBook Air 13-inch", catalog);   // "Electronics > Computers > Laptops"
```

### score: where on a scale?

```ts
await hunch.score("Checkout is down for everyone", ["cosmetic", "degraded, workaround exists", "blocked", "outage"]);
// a number from 0 to 3, e.g. 2.99
```

Returns a position from 0 to one less than the number of levels, and it can land between two levels. Give 2 to 10 levels, worst first, written as concrete situations rather than degrees. Pass `instructions: { hook: "...", clarity: "..." }` to score several dimensions in one request.

### check: is it true?

```ts
await hunch.check("BUY NOW!!! Limited offer", "is unsolicited advertising");   // true
await hunch.check(reviews, "the customer will cancel", { uncertain: [0.3, 0.7] });   // [true, null, false, ...]
```

Returns `true` when P(yes) reaches `threshold` (0.5 by default). With `uncertain: [low, high]`, items in between come back `null`, so borderline cases go to review instead of being forced to a side. Pass an object of statements to check several in one request.

### where: which items match?

```ts
await hunch.where(profiles, "might yell at a waiter for getting their order wrong");
await hunch.where(leads, "is an economic buyer for a product like ours", { columns: ["title", "company"], threshold: 0.7 });
```

A semantic `WHERE` clause. It returns the matching items, strongest match first. Statements about evidence in the item filter well. Predictions about behavior sit near 0.3 to 0.4 when the item says nothing either way, so rank those with `detail: true` and sort by `p` instead of filtering.

### extract: what's the value?

```ts
await hunch.extract(invoice, {
  total: ["money", "the amount due, not a subtotal or tax line"],
  due: "date",
  billingEmail: ["email", "where to send payment questions"],
  poNumber: /PO-\d+/,
});
// { total: "$1,240.00", due: "October 1, 2026", billingEmail: "ap@northwind.com", poNumber: null }
```

Code finds every candidate, meaning every dollar amount, date, or email. Jev picks the one that answers the field, seeing the words around each. The value is always copied from the text, never written by a model, and a field the text doesn't state comes back `null`. Built-in finders are `email`, `url`, `money`, `number`, `percent`, `phone`, and `date`; anything else is a `RegExp` or a function.

### ask: several questions at once

```ts
import { Classify, Rate, Check } from "hunch-jev";

const triaged = await hunch.ask(tickets, {
  kind: Classify(["bug", "feature request", "question"]),
  severity: Rate(["cosmetic", "degraded", "blocked", "outage"]),
  angry: Check("the customer is frustrated", { uncertain: [0.3, 0.7] }),
});
// [{ kind: "bug", severity: 2.98, angry: true }, ...]
```

Every question about an item goes in one request. Each spec takes the same options as its verb. When only one question should see some context, put `context` on that spec, so it doesn't color the other answers.

### pick: which one is best?

```ts
await hunch.pick(drafts, "most likely to get a reply from a busy CFO");
await hunch.pick(dateIdeas, "a quiet first date for someone who hates noise", { none: true });   // null if nothing fits
```

The candidates go head to head in one question. A head-to-head always crowns someone, so `none: true` also asks whether anything actually fits, in the same request.

### rank: order them

```ts
const ranked = await hunch.rank(
  candidates,
  { experience: "How relevant is their experience?", writing: "How clear is their writing?" },
  ["weak", "okay", "strong", "excellent"],
  { weights: { experience: 2, writing: 1 }, query: "Senior data engineer, remote, healthcare" },
);
// [{ item, index, composite, ratings }, ...] best first
```

Scores every candidate on each dimension, normalizes, and applies your weights. `query` is what they're being ranked for, which turns this into a reranker.

### pairs: compare two things

```ts
const same = await hunch.score(hunch.pairs(crmAccounts, vendorRecords), ["different companies", "related", "the same company"], {
  instructions: "Are a and b the same company?",
});
```

`pairs(a, b)` lines up two arrays item by item, so any verb can compare them. Use it to deduplicate records, match leads to accounts, or grade answers against references.

### verify: is it supported?

```ts
await hunch.verify(["skips None values", "adds a cache", 'says "filter out empty strings"'], diff);
// ["supported", "contradicted", "misquoted"]
```

Checks each claim against one source, or one source per claim. It tells "the source says otherwise" apart from "the source doesn't say", which comes back as `"not mentioned"`. Text a claim puts in quotes must appear in the source word for word, which catches made-up quotes. Use it as the last step of anything an LLM or agent produced.

### generate, discover, refine: with an LLM

```ts
const drafts = await hunch.generate({ n: 20, instructions: "cold emails to CFOs about expense software" });
const themes = await hunch.discover(tickets, 8, { instructions: "by what the customer needs" });   // { name: description }
const labeled = await hunch.classify(tickets, themes);
const final = await hunch.refine(draft, { accurate: "matches the facts in context", short: "is under 120 words" }, { context: { facts } });
```

- `generate` makes example data. Pass a JSON Schema as `schema` and a validator as `parse` for typed objects (with zod: `schema: z.toJSONSchema(Profile), parse: Profile.parse`). Large `n` is drawn in batches without repeats, and results are cached until `fresh: true`.
- `discover` has the LLM read a sample and propose categories with descriptions, including an `other` bucket. The result goes straight into `classify`.
- `refine` has the LLM rewrite until Jev confirms every check. Only failing drafts go back, each told which checks it missed. Checks only test what you list, so include an accuracy check and give the facts as context.

### evaluate, tuneThreshold: should I trust it?

```ts
const pred = await hunch.classify(sample.map((s) => s.title), LEVELS, { detail: true });
hunch.evaluate(pred, sample.map((s) => s.trueLevel));
// e.g. { accuracy: 0.91, byShape: { sure: { n: 71, accuracy: 0.98 }, split: ..., unsure: ... }, errors: [...] }

const p = await hunch.check(sample, "is a buyer", { detail: true });
hunch.tuneThreshold(p, sample.map((s) => s.isBuyer), { precision: 0.9 });
```

Label 50 to 100 items by hand. `evaluate` shows accuracy overall and per shape, and lists every miss. `tuneThreshold` finds the `check` / `where` cutoff that hits the precision or recall you need.

## Probabilities and shapes

With `detail: true`, `classify` returns `Answer` objects `{ label, p, probabilities, confidence, shape, by? }`, `score` returns `Rating` objects `{ score, level, normalized, probabilities, confidence, shape }`, and `check` returns `Feeling` objects `{ p, value, verdict }`. The `by` field says who decided when an LLM or backoff was allowed to: `"jev"`, `"llm"`, or `"parent"`.

Shapes come from `new ShapePolicy({ surePeak: 0.8, unsurePeak: 0.5, splitMargin: 0.15, splitMass: 0.75 })`, which you can pass as `policy` to `configure`. They're computed from the probabilities alone. A `sure` shape, or a high confidence, means the distribution is peaked. It does not mean the answer is correct. That's what `evaluate` is for.

## Running on real data

```ts
hunch.configure({ cache: ".hunch-cache", maxConcurrency: 16, maxRps: 20, errors: "skip", onProgress: (done, total) => bar.update(done / total) });

const plan = await hunch.dryRun(async () => {
  await hunch.ask(rows, { ... });
});
// e.g. { requests: 8214, questions: 16428, items: 50000 }: nothing was sent
```

- **Cache.** `cache` takes a directory path, which uses Node's file system, or any object with `get` and `set`, such as Redis. By default, answers are cached in memory.
- **Failures.** With `errors: "skip"`, a request that still fails after the SDK's retries returns `null` for its items, with a warning. Running again re-sends only those items.
- **Rate limits.** `maxRps` caps requests per second, and `maxConcurrency` caps requests in flight.
- **Progress.** `onProgress(done, total, label)` fires after every request.
- **Usage.** `hunch.defaultClient().usage` reports calls, cache hits, and tokens.

## What this is not

Jev never invents labels. Whatever you pass as labels is the whole set of allowed answers, and that constraint is the point. LLMs only propose. They have no tools and take no actions. If you want open-ended writing or a multi-step agent, this is the wrong library, on purpose.

## License

MIT. Jev and TypeSafe are [typesafe.ai](https://typesafe.ai); this library is not affiliated.
