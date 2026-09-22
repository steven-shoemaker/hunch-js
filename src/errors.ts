export class HunchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HunchError";
  }
}
