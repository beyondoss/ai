import { InvalidQueryError } from "./errors.js";

/**
 * A glob with the semantics of the Rust `globset` crate (default builder): `*` and `?` also cross
 * `/`; `**` as a whole path component matches any number of directories; `[..]` and `[!..]` /
 * `[^..]` classes; `{a,b}` alternation (nestable); `\` escapes. The match is anchored to the whole
 * path. Used for the host-side glob of `busybox-grep`, which has no `--include`.
 */
export function compileGlob(glob: string, ignoreCase: boolean): RegExp {
  const bad = (why: string) => new InvalidQueryError(`bad glob: ${why}`);
  const chars = Array.from(glob);
  let i = 0;
  let depth = 0;

  const esc = (c: string) => c.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
  const escClass = (c: string) => c.replace(/[\\\]\[^]/g, "\\$&");

  function seq(stopAtComma: boolean): string {
    let out = "";
    while (i < chars.length) {
      const c = chars[i]!;
      if (c === "}" && depth > 0) break;
      if (c === "," && stopAtComma) break;
      if (c === "\\") {
        i++;
        const n = chars[i];
        if (n === undefined) throw bad("dangling escape");
        out += esc(n);
        i++;
      } else if (c === "?") {
        out += "[^]";
        i++;
      } else if (c === "*") {
        let n = 0;
        const start = i;
        while (chars[i] === "*") {
          n++;
          i++;
        }
        const slashBefore = start > 0 && chars[start - 1] === "/";
        const after = chars[i];
        if (n === 1) {
          out += "[^]*";
        } else if (n === 2 && start === 0 && after === "/") {
          i++;
          out += "(?:/?|[^]*/)"; // `**/` prefix: zero or more directories
        } else if (n === 2 && start === 0 && after === undefined) {
          out += "[^]*";
        } else if (n === 2 && slashBefore && after === "/") {
          i++;
          out = out.slice(0, -1) + "(?:/|/[^]*/)"; // `/**/`
        } else if (n === 2 && slashBefore && after === undefined) {
          out = out.slice(0, -1) + "/[^]*"; // `/**` suffix
        } else {
          throw bad("invalid use of **; must be one path component");
        }
      } else if (c === "[") {
        i++;
        let neg = false;
        if (chars[i] === "!" || chars[i] === "^") {
          neg = true;
          i++;
        }
        let cls = "";
        let first = true;
        for (;;) {
          const d = chars[i];
          if (d === undefined) {
            throw bad("unclosed character class; missing ']'");
          }
          if (d === "]" && !first) break;
          first = false;
          if (d === "\\") {
            i++;
            const n = chars[i];
            if (n === undefined) throw bad("dangling escape");
            cls += escClass(n);
          } else if (d === "-") {
            cls += "-";
          } else {
            cls += escClass(d);
          }
          i++;
        }
        i++;
        out += `[${neg ? "^" : ""}${cls}]`;
      } else if (c === "{") {
        i++;
        depth++;
        const alts: string[] = [];
        for (;;) {
          alts.push(seq(true));
          const d = chars[i];
          if (d === ",") {
            i++;
            continue;
          }
          if (d === "}") {
            i++;
            break;
          }
          throw bad("unclosed alternate group; missing '}'");
        }
        depth--;
        out += `(?:${alts.join("|")})`;
      } else if (c === "}" && depth === 0) {
        throw bad("unopened alternate group; missing '{'");
      } else {
        out += esc(c);
        i++;
      }
    }
    return out;
  }

  const body = seq(false);
  try {
    return new RegExp(`^${body}$`, ignoreCase ? "iu" : "u");
  } catch (e) {
    throw bad(String(e));
  }
}
