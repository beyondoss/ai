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
  const names: Array<keyof typeof scripts> = [
    "STAT_SCRIPT",
    "READ_SCRIPT",
    "WRITE_STDIN_SCRIPT",
    "REMOVE_TEMP_SCRIPT",
    "GIT_SCRIPT",
    "UPLOAD_SCRIPT",
    "DOWNLOAD_SCRIPT",
    "STAGE_SCRIPT",
  ];
  for (const name of names) {
    it(name, () => {
      const m = appendix.match(
        new RegExp(
          `### [A-Z]\\. ${name}[^\\n]*\\n\\n\`\`\`sh\\n([\\s\\S]*?)\\n\`\`\``,
        ),
      );
      expect(m, `script ${name} in the spec`).not.toBeNull();
      expect(scripts[name]).toBe(m![1]);
    });
  }

  it("has the tab and newline the stat script prints", () => {
    expect(scripts.STAT_SCRIPT).toContain("printf '%s\t%s\t' \"$w\" \"$t\"");
    expect(scripts.STAT_SCRIPT).toContain("stat -L -c '%s %Y %y' --");
    expect(scripts.STAT_SCRIPT).toContain(
      "printf '%s\t%s.%s' \"$1\" \"$2\" \"$f\"",
    );
    expect(scripts.STAT_SCRIPT.endsWith("printf '\n'")).toBe(true);
  });
});
