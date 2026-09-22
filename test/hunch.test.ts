import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  Check,
  Classify,
  Client,
  Rate,
  ShapePolicy,
  Tree,
  ask,
  check,
  classify,
  discover,
  dryRun,
  evaluate,
  extract,
  fileCache,
  generate,
  pairs,
  pick,
  rank,
  refine,
  score,
  tuneThreshold,
  verify,
  where,
} from "../src/index.js";
import { FakeJev, FakeLLM, choice, client, firstOption, noul, scored } from "./fakes.js";

const salesOrEng = (state: any) => {
  const pick = String(state.input).includes("Sales") ? "Sales" : "Engineering";
  return { q: choice(pick, pick === "Sales" ? { Sales: 0.9, Engineering: 0.1 } : { Sales: 0.1, Engineering: 0.9 }) };
};

describe("shapes", () => {
  it("reads only the probabilities", () => {
    const policy = new ShapePolicy();
    expect(policy.classify({ a: 0.92, b: 0.05, c: 0.03 })).toBe("sure");
    expect(policy.classify({ a: 0.48, b: 0.44, c: 0.08 })).toBe("split");
    expect(policy.classify({ a: 0.34, b: 0.33, c: 0.33 })).toBe("unsure");
  });
});

describe("classify", () => {
  it("returns a bare label for one item and sends the input as state", async () => {
    const { fake, jev } = client(salesOrEng);
    expect(await classify("VP Sales", ["Sales", "Engineering"], { client: jev })).toBe("Sales");
    expect(fake.calls[0].state).toEqual({ input: "VP Sales" });
  });

  it("asks each distinct value once and keeps the array shape", async () => {
    const { fake, jev } = client(salesOrEng);
    expect(await classify(["VP Sales", "Staff Engineer", "VP Sales"], ["Sales", "Engineering"], { client: jev })).toEqual(["Sales", "Engineering", "Sales"]);
    expect(fake.calls).toHaveLength(2);
    await classify(["VP Sales"], ["Sales", "Engineering"], { client: jev });
    expect(fake.calls).toHaveLength(2); // cached
    expect(jev.usage.hits).toBe(1);
  });

  it("sends label descriptions, context, and only the chosen columns", async () => {
    const { fake, jev } = client(salesOrEng);
    await classify([{ title: "VP Sales", ssn: "x" }], { Sales: "sells things", Engineering: { what: "builds things" } }, { client: jev, context: { company: "Acme" }, columns: ["title"] });
    expect(fake.calls[0].state).toEqual({ input: { title: "VP Sales" }, company: "Acme" });
    expect(fake.calls[0].questions.q.criteria).toEqual({ Sales: "sells things", Engineering: { what: "builds things" } });
    await expect(classify("x", ["a"], { client: jev, context: { input: 1 } })).rejects.toThrow("reserved");
  });

  it("gives the distribution and shape with detail", async () => {
    const { jev } = client(salesOrEng);
    const answer = await classify("VP Sales", ["Sales", "Engineering"], { client: jev, detail: true });
    expect(answer).toMatchObject({ label: "Sales", p: 0.9, shape: "sure" });
  });

  it("multi-label is one noul per label, thresholded in code", async () => {
    const { fake, jev } = client(() => ({ l0: noul(0.9), l1: noul(0.2), l2: noul(0.6) }));
    expect(await classify("romcom about AI", ["comedy", "action", "sci-fi"], { client: jev, multiLabel: true })).toEqual(["comedy", "sci-fi"]);
    expect(await classify("romcom about AI", ["comedy", "action", "sci-fi"], { client: jev, multiLabel: true, threshold: 0.8 })).toEqual(["comedy"]);
    expect(fake.calls).toHaveLength(1);
  });

  it("rematches split rows between their top two and marks unsure rows", async () => {
    const { fake, jev } = client((state, questions) => {
      const q = questions.q;
      if (Object.keys(q.criteria as object).length === 2) return { q: choice("Director", { Manager: 0.15, Director: 0.85 }) };
      if (state.input === "Head of Sales") return { q: choice("Manager", { IC: 0.04, Manager: 0.5, Director: 0.46 }) };
      if (state.input === "Consultant") return { q: choice("IC", { IC: 0.36, Manager: 0.33, Director: 0.31 }) };
      return { q: choice("IC", { IC: 0.95, Manager: 0.03, Director: 0.02 }) };
    });
    const titles = ["Staff Engineer", "Head of Sales", "Consultant", "Head of Sales"];
    expect(await classify(titles, ["IC", "Manager", "Director"], { client: jev, split: "rematch", unsure: "review" })).toEqual(["IC", "Director", "review", "Director"]);
    expect(Object.keys(fake.calls.at(-1)!.questions.q.criteria as object)).toEqual(["Manager", "Director"]);
  });

  it("escalates only unsure rows to an LLM that must answer from the labels", async () => {
    const llm = new FakeLLM('{"label": "Director"}');
    const { jev } = client((state) =>
      state.input === "Head of Growth"
        ? { q: choice("Manager", { IC: 0.34, Manager: 0.36, Director: 0.3 }) }
        : { q: choice("IC", { IC: 0.95, Manager: 0.03, Director: 0.02 }) },
    );
    expect(await classify(["Engineer", "Head of Growth", "Head of Growth"], ["IC", "Manager", "Director"], { client: jev, unsure: llm })).toEqual(["IC", "Director", "Director"]);
    expect(llm.calls).toHaveLength(1);
    const detail = (await classify(["Engineer", "Head of Growth"], ["IC", "Manager", "Director"], { client: jev, unsure: llm, detail: true })) as any[];
    expect(detail.map((a) => a.by)).toEqual(["jev", "llm"]);
    expect(llm.calls).toHaveLength(1); // decision cached
    expect(await classify("Head of Growth", ["IC", "Manager", "Director"], { client: jev, unsure: () => "director" })).toBe("Director");
  });

  it("backs off to the parent when children are too close", async () => {
    const { jev } = client(() => ({ q: choice("Laptops", { Laptops: 0.46, Tablets: 0.44, Kitchen: 0.1 }, 0.2) }));
    const parents = { Laptops: "Computers", Tablets: "Computers", Kitchen: "Home" };
    expect(await classify("device", ["Laptops", "Tablets", "Kitchen"], { client: jev, backoff: parents })).toBe("Computers");
    expect(await classify("device", ["Laptops", "Tablets", "Kitchen"], { client: jev })).toBe("Laptops");
  });

  it("walks a Tree with beam search", async () => {
    const want: Record<string, number> = { Electronics: 0.7, Home: 0.3, Computers: 0.8, Phones: 0.2, Laptops: 0.9, Tablets: 0.1, Kitchen: 0.6, Garden: 0.4 };
    const { fake, jev } = client((_s, questions) => {
      const probs = Object.fromEntries(Object.keys(questions.q.criteria as object).map((k) => [k, want[k]]));
      const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
      return { q: choice(top, probs) };
    });
    const catalog = new Tree({ Electronics: { Phones: "handsets", Computers: { Laptops: null, Tablets: null } }, Home: { Kitchen: null, Garden: null } });
    expect(await classify(["macbook", "thinkpad"], catalog, { client: jev, beam: 2 })).toEqual(["Electronics > Computers > Laptops", "Electronics > Computers > Laptops"]);
    const parent = fake.calls.find((c) => "Computers" in (c.questions.q.criteria as object))!;
    expect((parent.questions.q.criteria as any).Computers).toEqual({ includes: ["Laptops", "Tablets"] });
  });

  it("validates labels", async () => {
    const { jev } = client(salesOrEng);
    await expect(classify("x", ["a", "a"], { client: jev })).rejects.toThrow("unique");
    await expect(classify("x", [], { client: jev })).rejects.toThrow("at least one");
    await expect(classify("x", Array.from({ length: 256 }, (_, i) => String(i)), { client: jev })).rejects.toThrow("255");
  });
});

describe("score, check, where", () => {
  const LEVELS = ["weak", "okay", "strong", "excellent"];
  const legend = Object.fromEntries(LEVELS.map((l, i) => [i, l]));

  it("scores one dimension or several in one request", async () => {
    const { fake, jev } = client((_s, qs) => Object.fromEntries(Object.keys(qs).map((k) => [k, scored(k === "clarity" ? 0.4 : 2.6, k === "clarity" ? { 0: 0.7, 1: 0.3 } : { 3: 0.85, 2: 0.15 }, legend)])));
    expect(await score("great tweet", LEVELS, { client: jev, instructions: "How strong is the hook?" })).toBeCloseTo(2.6);
    const rating = (await score("great tweet", LEVELS, { client: jev, instructions: "How strong is the hook?", detail: true })) as any;
    expect(rating).toMatchObject({ level: "excellent", shape: "sure" });
    expect(rating.normalized).toBeCloseTo(2.6 / 3);
    expect(await score(["a"], LEVELS, { client: jev, instructions: { hook: "?", clarity: "?" } })).toEqual([{ hook: 2.6, clarity: 0.4 }]);
    expect(Object.keys(fake.calls.at(-1)!.questions)).toEqual(["hook", "clarity"]);
    await expect(score("x", ["one"], { client: jev })).rejects.toThrow("2–10");
  });

  it("check gives maybe as null inside the uncertain band, validated before sending", async () => {
    const ps: Record<string, number> = { yes: 0.9, borderline: 0.5, no: 0.1 };
    const { fake, jev } = client((s, qs) => Object.fromEntries(Object.keys(qs).map((k) => [k, noul(ps[s.input])])));
    expect(await check(["yes", "borderline", "no"], "x", { client: jev, uncertain: [0.3, 0.7] })).toEqual([true, null, false]);
    expect(await check(["yes", "borderline", "no"], "x", { client: jev })).toEqual([true, true, false]);
    const calls = fake.calls.length;
    await expect(check("x", "y", { client: jev, uncertain: [0.7, 0.3] })).rejects.toThrow("uncertain");
    expect(fake.calls.length).toBe(calls);
  });

  it("where keeps matching items, strongest first", async () => {
    const { jev } = client((s) => ({ check: noul(s.input.bio.includes("cat") ? 0.95 : s.input.bio.includes("purr") ? 0.6 : 0.1) }));
    const people = [{ name: "Amanda", bio: "two cats" }, { name: "Dave", bio: "tv" }, { name: "Elena", bio: "I purr" }];
    expect((await where(people, "likes cats", { client: jev })).map((p) => p.name)).toEqual(["Amanda", "Elena"]);
    const detail = await where(people, "likes cats", { client: jev, detail: true });
    expect(detail.map((r) => r.match)).toEqual([true, false, true]);
  });
});

describe("ask", () => {
  it("sends every question in one request, and keeps per-question context apart", async () => {
    const { fake, jev } = client(firstOption);
    const out = await ask(["bio"], {
      vibe: Classify(["outdoorsy", "homebody"]),
      flag: Check("has a red flag", { uncertain: [0.3, 0.7] }),
      fit: Rate(["lo", "hi"], { instructions: "fit?", context: { looking_for: "runner" } }),
    }, { client: jev, context: { city: "Denver" } });
    expect(out).toEqual([{ vibe: "outdoorsy", flag: true, fit: 1 }]);
    expect(fake.calls).toHaveLength(2);
    const byQuestions = Object.fromEntries(fake.calls.map((c) => [Object.keys(c.questions).sort().join(","), c.state]));
    expect(byQuestions["flag,vibe"]).toEqual({ input: "bio", city: "Denver" });
    expect(byQuestions.fit).toEqual({ input: "bio", city: "Denver", looking_for: "runner" });
  });
});

describe("pick, rank, pairs, verify", () => {
  it("pick compares candidates head to head and can say none", async () => {
    const handler = (fits: number) => (state: any, qs: any) => {
      const out: Record<string, unknown> = {};
      if (qs.q) {
        const ids = Object.keys(qs.q.criteria);
        const best = ids.reduce((a, b) => (String(qs.q.criteria[b]).length > String(qs.q.criteria[a]).length ? b : a));
        out.q = choice(best, Object.fromEntries(ids.map((c) => [c, c === best ? 0.8 : 0.2 / (ids.length - 1)])));
      }
      if (qs.fits) {
        expect(state.input.candidates).toBeDefined();
        out.fits = noul(fits);
      }
      return out;
    };
    const tweets = ["short", "a much longer tweet here", "mid length"];
    expect(await pick(tweets, "best", { client: client(handler(0.9)).jev })).toBe("a much longer tweet here");
    const bad = client(handler(0.2));
    expect(await pick(tweets, "best", { client: bad.jev, none: true })).toBeNull();
    expect(Object.keys(bad.fake.calls[0].questions).sort()).toEqual(["fits", "q"]);
    const field = Array.from({ length: 300 }, (_, i) => `c${String(i).padStart(4, "0")}`);
    field[299] = "x".repeat(40);
    const big = client(handler(0.9));
    expect(await pick(field, "best", { client: big.jev })).toBe("x".repeat(40));
    expect(big.fake.calls).toHaveLength(3);
  });

  it("rank weights normalized scores and sends the query", async () => {
    const legend = { 0: "weak", 1: "okay", 2: "strong" };
    const values: Record<string, Record<string, number>> = { a: { hook: 2, clarity: 0 }, b: { hook: 1, clarity: 2 } };
    const { fake, jev } = client((s, qs) => Object.fromEntries(Object.keys(qs).map((k) => [k, scored(values[s.input][k], { [values[s.input][k]]: 1 }, legend)])));
    const dims = { hook: "hook?", clarity: "clear?" };
    expect((await rank(["a", "b"], dims, Object.values(legend), { client: jev })).map((r) => r.item)).toEqual(["b", "a"]);
    const weighted = await rank(["a", "b"], dims, Object.values(legend), { client: jev, weights: { hook: 3, clarity: 1 }, query: "q" });
    expect(weighted.map((r) => r.item)).toEqual(["a", "b"]);
    expect(weighted[0].composite).toBeCloseTo(0.75);
    expect(fake.calls.at(-1)!.state.query).toBe("q");
  });

  it("pairs lines two arrays up", async () => {
    expect(pairs([1], [2], ["claim", "source"])).toEqual([{ claim: 1, source: 2 }]);
    expect(() => pairs([1, 2], [1])).toThrow("line up");
  });

  it("verify tells contradicted from not mentioned and catches fabricated quotes", async () => {
    const { fake, jev } = client((state, qs) => {
      const { claim, source } = state.input;
      const pick = source.includes(`not ${claim}`) ? "contradicted" : source.includes(claim) ? "supported" : "not mentioned";
      return { q: choice(pick, Object.fromEntries(Object.keys(qs.q.criteria as object).map((k) => [k, k === pick ? 0.9 : 0.05]))) };
    });
    const source = "the change skips None values. it is not cached.";
    expect(await verify(["skips None values", "cached", "renames the function", 'says "filter empty strings please"'], source, { client: jev })).toEqual([
      "supported",
      "contradicted",
      "not mentioned",
      "misquoted",
    ]);
    expect(fake.calls[0].state.input).toEqual({ claim: "skips None values", source });
    await expect(verify(["a", "b"], ["one"], { client: jev })).rejects.toThrow("2 claims and 1 sources");
  });
});

describe("extract", () => {
  const INVOICE = "Subtotal $1,100.00. Tax $140.00. Total due $1,240.00 by 2026-10-01. Questions: ap@vendor.com";
  const handler = (_s: any, qs: any) =>
    Object.fromEntries(
      Object.entries(qs).map(([name, q]: [string, any]) => {
        const ids = Object.fromEntries(Object.entries(q.criteria).filter(([c]) => c !== "none").map(([c, v]: [string, any]) => [c, v.value]));
        const want = name === "total" ? Object.keys(ids).find((c) => ids[c].includes("1,240"))! : name === "email" ? "none" : Object.keys(ids)[0];
        return [name, choice(want, Object.fromEntries(Object.keys(q.criteria).map((c) => [c, c === want ? 0.9 : 0.02])))];
      }),
    );

  it("copies values from the text, one request per text", async () => {
    const { fake, jev } = client(handler);
    const out = await extract(INVOICE, { total: ["money", "the amount due"], due_date: "date", email: "email", po_number: /PO-\d+/ }, { client: jev });
    expect(out).toEqual({ total: "$1,240.00", due_date: "2026-10-01", email: null, po_number: null });
    expect(fake.calls).toHaveLength(1);
    expect(Object.keys(fake.calls[0].questions)).not.toContain("po_number");
    expect((fake.calls[0].questions.total.criteria as any).c2.in_context).toContain("Total due");
    const many = await extract([INVOICE, "nothing", INVOICE], { total: "money" }, { client: jev });
    expect(many.map((r) => r.total)).toEqual(["$1,240.00", null, "$1,240.00"]);
    expect(await extract("red green blue", { color: (t) => t.split(" ") }, { client: jev })).toEqual({ color: "red" });
  });
});

describe("LLM verbs", () => {
  it("generate batches large n, avoids repeats, and caches until fresh", async () => {
    const first = JSON.stringify(Array.from({ length: 25 }, (_, i) => `t${i}`));
    const second = JSON.stringify(["t0", "t1", ...Array.from({ length: 21 }, (_, i) => `t${i + 25}`)]);
    const llm = new FakeLLM([first, second, JSON.stringify(["t46", "t47", "t48"])]);
    const { jev } = client(firstOption, { llm });
    const out = await generate({ n: 48 , client: jev });
    expect(out).toEqual(Array.from({ length: 48 }, (_, i) => `t${i}`));
    expect(llm.calls[1].user).toContain("already_have_do_not_repeat");
    const cached = new FakeLLM(['["a", "b"]', '["c", "d"]']);
    const c2 = client(firstOption, { llm: cached }).jev;
    expect(await generate({ n: 2, instructions: "x", client: c2 })).toEqual(["a", "b"]);
    expect(await generate({ n: 2, instructions: "x", client: c2 })).toEqual(["a", "b"]);
    expect(await generate({ n: 2, instructions: "x", client: c2, fresh: true })).toEqual(["c", "d"]);
  });

  it("discover returns labels ready for classify", async () => {
    const llm = new FakeLLM(JSON.stringify([
      { name: "shipping", description: "late or damaged" },
      { name: "billing", description: "charges" },
      { name: "Other", description: "dropped" },
    ]));
    const { jev } = client(firstOption, { llm });
    expect(await discover(["box crushed", "charged twice", "box crushed"], 3, { client: jev, instructions: "by complaint" })).toEqual({
      shipping: "late or damaged",
      billing: "charges",
      other: "fits none of the other categories",
    });
    expect(JSON.parse(llm.calls[0].user).context.examples).toEqual(["box crushed", "charged twice"]);
  });

  it("refine rewrites only failing drafts until the checks pass", async () => {
    const llm = new FakeLLM(["hunch: npm i hunch-jev"]);
    const { jev } = client((s, qs) => Object.fromEntries(Object.entries(qs).map(([k, q]) => [k, noul(q.instructions !== "shows the install command" || s.input.includes("npm i") ? 0.9 : 0.1)])), { llm });
    expect(await refine(["try hunch", "npm i hunch-jev now"], ["shows the install command"], { client: jev })).toEqual(["hunch: npm i hunch-jev", "npm i hunch-jev now"]);
    expect(llm.calls).toHaveLength(1);
    const stuck = client((_s, qs) => Object.fromEntries(Object.keys(qs).map((k) => [k, noul(0.1)])), { llm: new FakeLLM(["a", "b"]) });
    const result = (await refine("x", { install: "shows it" }, { client: stuck.jev, rounds: 2, detail: true })) as any;
    expect(result).toMatchObject({ passed: false, rounds: 2, failed: ["install"], original: "x" });
  });
});

describe("measurement", () => {
  it("evaluate reports accuracy by shape and the misses", () => {
    const ev = evaluate(
      [
        { label: "a", probabilities: {}, p: 0.9, confidence: 0.9, shape: "sure" },
        { label: "b", probabilities: {}, p: 0.5, confidence: 0.2, shape: "split" },
        null,
      ],
      ["a", "a", "a"],
    );
    expect(ev).toMatchObject({ n: 2, accuracy: 0.5, missing: 1 });
    expect(ev.byShape).toEqual({ sure: { n: 1, accuracy: 1 }, split: { n: 1, accuracy: 0 } });
    expect(ev.errors).toEqual([{ index: 1, truth: "a", predicted: "b" }]);
  });

  it("tuneThreshold picks cutoffs for precision, recall, and F1", () => {
    const p = [0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3];
    const truth = [true, true, false, true, false, true, false];
    expect(tuneThreshold(p, truth, { precision: 1 })).toMatchObject({ threshold: 0.9, recall: 0.5 });
    expect(tuneThreshold(p, truth, { recall: 0.75 })).toMatchObject({ threshold: 0.7, precision: 0.75 });
    expect(() => tuneThreshold([0.9, 0.8], [false, true], { precision: 1 })).toThrow("No cutoff");
  });
});

describe("running on real data", () => {
  it("errors: skip keeps good rows, warns, and retries only failures", async () => {
    const warnings: string[] = [];
    const fake = new FakeJev((s, qs) => {
      if (s.input === "boom") throw new Error("503 from Jev");
      return firstOption(s, qs);
    });
    await expect(classify(["ok", "boom"], ["a", "b"], { client: new Client({ client: fake }) })).rejects.toThrow("503");
    const jev = new Client({ client: fake, errors: "skip", onWarning: (m) => warnings.push(m) });
    expect(await classify(["ok", "boom", "fine"], ["a", "b"], { client: jev })).toEqual(["a", null, "a"]);
    expect(warnings[0]).toContain("1 of 3 requests failed");
  });

  it("dryRun counts requests without sending or caching", async () => {
    const { fake, jev } = client(firstOption);
    await classify(["cached"], ["a", "b"], { client: jev });
    let inside: unknown;
    const plan = await dryRun(async () => {
      inside = await classify(["cached", "new1", "new2", "new1"], ["a", "b"], { client: jev });
    });
    expect(inside).toEqual(["a", "a", "a", "a"]);
    expect(plan).toMatchObject({ requests: 2, questions: 2, items: 4 });
    expect(fake.calls).toHaveLength(1);
  });

  it("maxRps spaces requests and concurrency is capped", async () => {
    let live = 0;
    let peak = 0;
    const slow = {
      async systemOne({ state, questions }: any) {
        live += 1;
        peak = Math.max(peak, live);
        await new Promise((r) => setTimeout(r, 20));
        live -= 1;
        return { answers: firstOption(state, questions) };
      },
    };
    await classify(Array.from({ length: 12 }, (_, i) => `t${i}`), ["a", "b"], { client: new Client({ client: slow, maxConcurrency: 4 }) });
    expect(peak).toBe(4);
    const start = Date.now();
    await classify(Array.from({ length: 6 }, (_, i) => `u${i}`), ["a", "b"], { client: new Client({ client: slow, maxRps: 20 }) });
    expect(Date.now() - start).toBeGreaterThanOrEqual(220);
  });

  it("a file cache survives a new client", async () => {
    const dir = await mkdtemp(join(tmpdir(), "hunch-"));
    const fake = new FakeJev(firstOption);
    await classify("VP", ["a", "b"], { client: new Client({ client: fake, cache: dir }) });
    const later = new Client({ client: fake, cache: await fileCache(dir) });
    await classify("VP", ["a", "b"], { client: later });
    expect(fake.calls).toHaveLength(1);
    expect(later.usage.hits).toBe(1);
  });
});
