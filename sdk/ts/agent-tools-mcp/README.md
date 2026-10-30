# @beyond.dev/agent-tools-mcp

Run **Eve / Agent-SDK-shaped `tools/*.ts`** as an MCP server **inside a Beyond sandbox**.

The Beyond agent cell is already an MCP client (`sandbox: true` connectors + bundle `.mcp-tools.json`). This package is the missing guest: load `defineTool` modules, speak a minimal MCP surface (`initialize` / `tools/list` / `tools/call`), stamp the manifest at pack time, and ship a **Bun AOT + `--trim` + `--smol`** executable.

```bash
# local
bun run src/cli.ts call --dir examples/weather/tools get_weather '{"city":"NYC"}'
bun run src/cli.ts manifest --dir examples/weather/tools --name tools -o /tmp/.mcp-tools.json
bun run src/cli.ts serve --dir examples/weather/tools --http 8931

# in-sandbox binary (build on the guest OS/arch)
npm run compile   # → bin/tools-mcp
./bin/tools-mcp serve --dir ./tools --http 127.0.0.1:8931
```

## Author shape

```ts
// tools/get_weather.ts
import { defineTool } from "eve/tools"; // alias to @beyond.dev/agent-tools-mcp/eve-tools

export default defineTool({
  description: "Get the weather for a city",
  inputSchema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  async execute({ city }, ctx) {
    return { city, condition: "sunny" };
  },
});
```

Agent `package.json`:

```json
{
  "imports": {
    "eve/tools": "@beyond.dev/agent-tools-mcp/eve-tools"
  }
}
```

## Bundle + grant

1. `beyond-agent-tools-mcp manifest --dir tools --name tools -o .mcp-tools.json`
2. Pack the agent folder (ustar) including `tools/`, the compiled `tools-mcp` binary, and `.mcp-tools.json`
3. Grant: `mcp: [{ name: "tools", url: "<exec-front>/mcp/<session>?…", sandbox: true }]`
4. Front on first MCP dial: seed bundle → `./tools-mcp serve --dir tools --http …` → forward

Harness (`beyond-agent`) needs no protocol changes.

## Compile flags (AOT with JIT fallback)

`scripts/compile.sh` is part of the **pack workflow**, not a separate experiment:

1. Try **AOT**: `bun compile --trim --aot-allow-code-from-disk --compile-exec-argv=--smol`
2. If AOT fails (types, experimental compiler, …) → **JIT-in-binary**: `bun compile --no-aot --trim --compile-exec-argv=--smol`
3. Writes `bin/tools-mcp` and `bin/tools-mcp.mode` (`aot` or `jit`)

`FORCE_NO_AOT=1` skips to JIT (used by `deploy/demo/tools_mcp_e2e.sh --jit` to prove the fallback artifact).

`--aot-allow-code-from-disk` is AOT-only (Bun requires `--aot`). JIT executables load `tools/*.ts` from disk without that flag.

End-to-end proof (host compile + E2B MCP round-trip):

```bash
./deploy/demo/tools_mcp_e2e.sh          # needs E2B_API_KEY
./deploy/demo/tools_mcp_e2e.sh --local  # compile + manifest + call only
./deploy/demo/tools_mcp_e2e.sh --jit --local
```

Measure RSS on a real Linux sandbox before tuning `--max-old-space-size`.
