/** JSON with sorted keys, so equal values make equal cache and dedupe keys. */
export function stableStringify(value: unknown): string {
  return JSON.stringify(toJson(value));
}

/** Plain JSON for Jev state: Dates to ISO, Maps and Sets to objects and arrays, undefined dropped. */
export function toJson(value: unknown): unknown {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map(toJson);
  if (value instanceof Set) return [...value].map(toJson);
  if (value instanceof Map) return toJson(Object.fromEntries(value));
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as object).sort()) {
      const v = (value as Record<string, unknown>)[key];
      if (v !== undefined) out[key] = toJson(v);
    }
    return out;
  }
  if (typeof value === "bigint") return value.toString();
  return value;
}
