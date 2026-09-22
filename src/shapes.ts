export type Shape = "sure" | "split" | "unsure";

export interface ShapePolicyOptions {
  surePeak?: number;
  unsurePeak?: number;
  splitMargin?: number;
  splitMass?: number;
}

/**
 * Sorts a distribution into sure / split / unsure from the probabilities alone.
 * Jev's confidence is derived from the top probability, so it adds nothing here.
 * These cutoffs are yours, and none of them says whether an answer is correct.
 */
export class ShapePolicy {
  readonly surePeak: number;
  readonly unsurePeak: number;
  readonly splitMargin: number;
  readonly splitMass: number;

  constructor({ surePeak = 0.8, unsurePeak = 0.5, splitMargin = 0.15, splitMass = 0.75 }: ShapePolicyOptions = {}) {
    if (!(0 <= unsurePeak && unsurePeak <= surePeak && surePeak <= 1)) {
      throw new RangeError("Need 0 <= unsurePeak <= surePeak <= 1.");
    }
    this.surePeak = surePeak;
    this.unsurePeak = unsurePeak;
    this.splitMargin = splitMargin;
    this.splitMass = splitMass;
  }

  classify(probabilities: Record<string, number>): Shape {
    const ranked = Object.values(probabilities).sort((a, b) => b - a);
    const top = ranked[0] ?? 0;
    const second = ranked[1] ?? 0;
    if (top - second < this.splitMargin && top + second >= this.splitMass) return "split";
    if (top < this.unsurePeak) return "unsure";
    if (top >= this.surePeak) return "sure";
    return "split";
  }
}
