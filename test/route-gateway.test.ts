import { afterEach, describe, expect, it, vi } from "vitest";
import { Check, Classify, Client, Rate, ask, check, route } from "../src/index.js";
import { choice, client, noul, scored } from "./fakes.js";

const LEVELS = ["low", "medium", "high"];
const legend = Object.fromEntries(LEVELS.map((l, i) => [i, l]));

const tickets = (state: any) => {
  const text: string = state.input;
  const urgent = text.includes("down") ? 0.95 : 0.1;
  const topic = text.includes("charged") ? "billing" : text.includes("error") || text.includes("down") ? "bug" : "other";
  const tp = topic === "other" ? 0.4 : 0.9;
  return {
    urgent: noul(urgent),
    topic: choice(topic, Object.fromEntries(["billing", "bug", "other"].map((t) => [t, t === topic ? tp : (1 - tp) / 2]))),
    severity: scored(urgent > 0.5 ? 2 : 0.5, urgent > 0.5 ? { 2: 0.9, 1: 0.1 } : { 0: 0.5, 1: 0.5 }, legend),
  };
};

const TICKETS = ["Checkout is down for everyone", "I was charged twice", "How do I export?", "Weird error on login"];
const QUESTIONS = { urgent: Check("needs a human within the hour"), topic: Classify(["billing", "bug", "other"]), severity: Rate(LEVELS) };

describe("route", () => {
  it("picks the first rule whose conditions all hold", async () => {
    const { jev } = client(tickets);
    const answers = (await ask(TICKETS, QUESTIONS, { client: jev, detail: true })) as any[];
    const rules = {
      page: { urgent: 0.8 },
      billing: { topic: ["billing", 0.7] as [string, number] },
      review: { "topic.shape": "unsure" },
      engineering: { topic: ["bug"], severity: 0.4 },
    };
    expect(route(answers, rules, { default: "triage" })).toEqual(["page", "billing", "review", "engineering"]);
    expect(route(answers[2], rules, { default: "triage" })).toBe("review");
  });

  it("works on bare labels and booleans, and null never matches", () => {
    expect(route(["billing", "bug", "other"], { money: { _: "billing" }, eng: { _: ["bug"] } }, { default: "x" })).toEqual(["money", "eng", "x"]);
    expect(route({ spam: true, tier: "enterprise" }, { block: { spam: true, tier: "free" }, vip: { tier: "enterprise" } })).toBe("vip");
    expect(route({ spam: null }, { block: { spam: true } }, { default: "keep" })).toBe("keep");
  });

  it("takes functions and validates", async () => {
    const { jev } = client(tickets);
    const answers = (await ask(TICKETS.slice(0, 1), QUESTIONS, { client: jev, detail: true })) as any[];
    expect(route(answers, { hot: { severity: (r: any) => r.level === "high" } })).toEqual(["hot"]);
    expect(() => route(answers, {})).toThrow("at least one rule");
  });
});

describe("gateways", () => {
  afterEach(() => vi.unstubAllGlobals());

  const stubFetch = (json: unknown) => {
    const calls: { url: string; init: any }[] = [];
    vi.stubGlobal("fetch", async (url: string, init: any) => {
      calls.push({ url, init });
      return new Response(JSON.stringify(typeof json === "function" ? (json as any)(JSON.parse(init.body)) : json), { status: 200 });
    });
    return calls;
  };

  it("openrouter sends the systemone body to the decisions endpoint", async () => {
    const calls = stubFetch((body: any) => ({
      model: "~typesafe/jev-latest",
      answers: Object.fromEntries(Object.keys(body.questions).map((k) => [k, { type: "noul", noul: 0.92 }])),
      usage: { input_tokens: 312, output_tokens: 48 },
    }));
    const jev = new Client({ gateway: "openrouter", apiKey: "test-key" });
    expect(await check("payouts failing for 3 days", "conveys urgency", { client: jev })).toBe(true);
    expect(calls[0].url).toBe("https://openrouter.ai/api/alpha/decisions");
    expect(calls[0].init.headers.Authorization).toBe("Bearer test-key");
    const body = JSON.parse(calls[0].init.body);
    expect(body.model).toBe("~typesafe/jev-latest");
    expect(body.questions.check.type).toBe("noul");
    expect(jev.usage.inputTokens).toBe(312);
  });

  it("vercel translates boolean questions, legends, and confidence", async () => {
    const calls = stubFetch({
      answers: {
        urgent: { type: "boolean", probability: 0.91 },
        topic: { type: "choice", choice: "billing", probabilities: { billing: 0.8, bug: 0.15, other: 0.05 } },
        severity: { type: "score", score: 1.4, probabilities: { 0: 0.1, 1: 0.6, 2: 0.3 } },
      },
      usage: { inputTokens: 120, outputTokens: 3 },
      providerMetadata: { typesafe: { confidence: { topic: 0.72 } } },
    });
    const jev = new Client({ gateway: "vercel", apiKey: "test-key" });
    const out = (await ask("charged twice", QUESTIONS, { client: jev, detail: true })) as any;
    expect(calls[0].url).toBe("https://ai-gateway.vercel.sh/v4/ai/evaluation-model");
    expect(calls[0].init.headers["Ai-Model-Id"]).toBe("typesafe-ai/jev");
    expect(JSON.parse(calls[0].init.body).questions.urgent.type).toBe("boolean");
    expect(out.urgent.p).toBeCloseTo(0.91);
    expect(out.topic).toMatchObject({ label: "billing", confidence: 0.72 });
    expect(out.severity.level).toBe("medium");
    expect(out.severity.confidence).toBeCloseTo((3 * 0.6 - 1) / 2);
    expect(jev.usage.outputTokens).toBe(3);
  });

  it("retries a 503 and needs a key", async () => {
    let n = 0;
    vi.stubGlobal("fetch", async () => {
      n += 1;
      return n === 1 ? new Response("busy", { status: 503 }) : new Response(JSON.stringify({ answers: { check: { type: "noul", noul: 0.2 } } }), { status: 200 });
    });
    const jev = new Client({ gateway: "openrouter", apiKey: "k" });
    expect(await check("x", "y", { client: jev })).toBe(false);
    expect(n).toBe(2);
    const saved = process.env.AI_GATEWAY_API_KEY;
    delete process.env.AI_GATEWAY_API_KEY;
    expect(() => new Client({ gateway: "vercel" })).toThrow("AI_GATEWAY_API_KEY");
    if (saved) process.env.AI_GATEWAY_API_KEY = saved;
  });
});
