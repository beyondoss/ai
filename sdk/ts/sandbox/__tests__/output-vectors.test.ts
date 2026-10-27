import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fromBase64 } from "../src/bytes.js";
import { decodeOutput } from "../src/http.js";

const file = new URL(
  "../../../../crates/sandbox/tests/fixtures/exec/output.json",
  import.meta.url,
);
const vectors = JSON.parse(readFileSync(file, "utf8")) as {
  answers: {
    name: string;
    answer: Record<string, unknown>;
    expect: {
      invalid?: boolean;
      exit_code?: number;
      stdout_base64?: string;
      stderr_base64?: string;
    };
  }[];
};

describe("the one-shot answer vectors (output.json)", () => {
  for (const c of vectors.answers) {
    it(c.name, () => {
      if (c.expect.invalid) {
        expect(() => decodeOutput(c.answer, "u", 1 << 20)).toThrow();
        return;
      }
      const r = decodeOutput(c.answer, "u", 1 << 20);
      expect(r.exitCode).toBe(c.expect.exit_code);
      expect(r.stdout).toEqual(fromBase64(c.expect.stdout_base64 ?? ""));
      expect(r.stderr).toEqual(fromBase64(c.expect.stderr_base64 ?? ""));
    });
  }
});
