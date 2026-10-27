import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import * as scripts from "../src/scripts.js";

const here = dirname(fileURLToPath(import.meta.url));
const spec = readFileSync(
  join(here, "../../../conformance/FILES-OVER-EXEC.md"),
  "utf8",
);

/** The scripts are constants sent byte for byte: scripts.ts must be the spec's appendix, exactly. */
describe("the scripts are the spec's appendix", () => {
  const appendix = spec.slice(spec.indexOf("## 7. Appendix: the scripts"));
  const cases: Array<[string, keyof typeof scripts]> = [
    ["A", "STAT_SCRIPT"],
    ["B", "READ_SCRIPT"],
    ["C", "WRITE_STDIN_SCRIPT"],
    ["D", "WRITE_CHUNK_SCRIPT"],
    ["E", "REMOVE_TEMP_SCRIPT"],
    ["F", "GIT_SCRIPT"],
  ];
  for (const [letter, name] of cases) {
    it(`${letter}. ${name}`, () => {
      const m = appendix.match(
        new RegExp(
          `### ${letter}\\. [^\\n]*\\n\\n\`\`\`sh\\n([\\s\\S]*?)\\n\`\`\``,
        ),
      );
      expect(m, `script ${letter} in the spec`).not.toBeNull();
      expect(scripts[name]).toBe(m![1]);
    });
  }

  it("has the tab and newline the stat script prints", () => {
    expect(scripts.STAT_SCRIPT).toContain("printf '%s\t%s\t' \"$w\" \"$t\"");
    expect(scripts.STAT_SCRIPT).toContain("stat -L -c '%s\t%Y' --");
    expect(scripts.STAT_SCRIPT.endsWith("printf '\n'")).toBe(true);
  });

  it("carries the baseline jobs scripts, with the supervisor's kernel spliced in", () => {
    expect(scripts.JOB_SCRIPT.startsWith("umask 077\nop=$1 R=$2 S=$2/$3")).toBe(
      true,
    );
    expect(scripts.SUPERVISOR).not.toContain("#@kernel");
    expect(scripts.SUPERVISOR).toContain("kverdict()");
  });
});
