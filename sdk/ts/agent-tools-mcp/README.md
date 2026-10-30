# @beyond.dev/agent-tools-mcp

**Eve / Agent-SDK-compatible `defineTool`** for Beyond sandboxes: Zod (or any Standard Schema, or plain JSON Schema), validate-before-execute, filename = tool name, MCP guest for `beyond-agent`.

```ts
import { defineTool } from "eve/tools"; // → @beyond.dev/agent-tools-mcp/eve-tools
import { z } from "zod";

export default defineTool({
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string().min(1) }),
  async execute({ city }, ctx) {
    return { city, condition: "sunny", temperatureF: 72 };
  },
});
```

## Eve parity — how we prove it

**`npm run parity`** runs `__tests__/eve-parity.test.ts` — one test per Eve docs claim ([eve.dev/docs/tools](https://eve.dev/docs/tools)). All must pass. That includes Zod/Standard Schema/JSON Schema, validate-before-execute, JSON Schema `default` fill, `deferred` omit from list, `endsTurn` model description append, and `outputSchema` validate.

## Commands

```bash
bun run src/cli.ts call --dir examples/weather/tools get_weather '{"city":"NYC"}'
bun run src/cli.ts manifest --dir examples/weather/tools --name tools -o /tmp/.mcp-tools.json
bun run src/cli.ts serve --dir examples/weather/tools --http 8931
npm run compile   # → bin/tools-mcp (AOT, JIT fallback)
```

## Agent project

`import { defineTool } from "eve/tools"` needs a package named `eve` (Node does not remap bare `eve/tools` via `imports`). Use the shipped shim:

```json
{
  "dependencies": {
    "@beyond.dev/agent-tools-mcp": "…",
    "eve": "file:…/agent-tools-mcp/eve-shim",
    "zod": "^4"
  }
}
```

## Bundle + grant

1. `beyond-agent-tools-mcp manifest --dir tools --name tools -o .mcp-tools.json`
2. Pack the agent folder (ustar) including `tools/`, deps, compiled `tools-mcp`, `.mcp-tools.json`
3. Grant: `mcp: [{ name: "tools", url: "<exec-front>/mcp/<session>?…", sandbox: true }]`
4. Front on first MCP dial: seed bundle → `tools-mcp serve --dir tools` → forward

## Proof

```bash
./deploy/demo/tools_mcp_e2e.sh --local
./deploy/demo/tools_mcp_e2e.sh          # E2B_API_KEY
./deploy/demo/tools_agent_e2e.sh        # agent → E2B → get_weather
```
