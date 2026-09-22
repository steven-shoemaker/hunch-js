/** Where raw Jev answers are kept. Any object with get / set works (Redis, a Map, a file store). */
export interface CacheStore {
  get(key: string): unknown | undefined | Promise<unknown | undefined>;
  set(key: string, value: unknown): void | Promise<void>;
  clear?(): void | Promise<void>;
}

export class MemoryCache implements CacheStore {
  #map = new Map<string, unknown>();
  get(key: string) {
    return this.#map.get(key);
  }
  set(key: string, value: unknown) {
    this.#map.set(key, value);
  }
  clear() {
    this.#map.clear();
  }
}

/** Hashed JSON files under a directory, so a rerun of the same script costs nothing. Node only. */
export async function fileCache(dir: string): Promise<CacheStore> {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const crypto = await import("node:crypto");
  const memory = new Map<string, unknown>();
  const file = (key: string) => {
    const digest = crypto.createHash("sha256").update(key).digest("hex");
    return path.join(dir, digest.slice(0, 2), `${digest}.json`);
  };
  return {
    async get(key) {
      if (memory.has(key)) return memory.get(key);
      try {
        const value = JSON.parse(await fs.readFile(file(key), "utf8"));
        memory.set(key, value);
        return value;
      } catch {
        return undefined;
      }
    },
    async set(key, value) {
      memory.set(key, value);
      const target = file(key);
      await fs.mkdir(path.dirname(target), { recursive: true });
      const tmp = `${target}.${process.pid}.tmp`;
      await fs.writeFile(tmp, JSON.stringify(value));
      await fs.rename(tmp, target);
    },
    async clear() {
      memory.clear();
      await fs.rm(dir, { recursive: true, force: true });
    },
  };
}
