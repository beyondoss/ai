# @beyond.dev/tools

**Write Eve tools. Run them in a Beyond sandbox.** Same `defineTool` + Zod you would
ship on [eve.dev](https://eve.dev/docs/tools), served as MCP inside the guest so
`beyond-agent` dials them like any other `sandbox: true` connector.

```ts
import { defineTool } from "eve/tools";
import { z } from "zod";

export default defineTool({
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string().min(1) }),
  async execute({ city }, ctx) {
    const sb = await ctx.getSandbox();
    // …or just return data. Filename = tool name.
    return { city, condition: "sunny", temperatureF: 72 };
  },
});
```

Drop that in `tools/get_weather.ts`. The agent sees `mcp__tools__get_weather`. No
hand-written JSON Schema, no separate MCP server project, no keys in the VM for the
tools themselves — the replica holds the conversation; the sandbox only runs your code.

|                            | The fact                                                                                                                                                                                                                                                                                                                                                                                                                          | The proof                                                                      |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Eve-shaped author API**  | `defineTool`, Zod / Standard Schema / plain JSON Schema, validate-before-execute, `ctx.getSandbox()` rooted at the bundle (cwd / `--workspace`), `callId` / `toolName` / `session` / `abortSignal` (a `notifications/cancelled` aborts the call and its child process). `endsTurn: true` is in the cold-open menu and, on success, sets `_meta["beyond.dev/endsTurn"]`. `ctx.to()` throws (Eve channel host API; not this guest). | `npm run parity` — `__tests__/eve-parity.test.ts`, `__tests__/eve-ctx.test.ts` |
| **Cold open, hot call**    | Session open advertises tools from `.mcp-tools.json` with **no VM**. The first `tools/call` creates the sandbox, seeds the bundle, starts `tools-mcp`, and forwards.                                                                                                                                                                                                                                                              | `deploy/demo/demo.sh tools`, ARCHITECTURE “Agent tools as sandbox MCP”         |
| **One tar, two consumers** | The replica fetches the bundle for the **menu**; the exec front seeds the same bytes into the VM for **execution**.                                                                                                                                                                                                                                                                                                               | `tools_agent_e2e.sh`, grant `sandbox: true` + front `tools=` URL               |
| **Real agent path**        | `mcp::connect_granted` with `sandbox: true` — same client path as browser MCP — returns live `get_weather`.                                                                                                                                                                                                                                                                                                                       | `crates/agent/tests/agent_tools_mcp.rs`, `./deploy/demo/tools_agent_e2e.sh`    |

## Why this shape

Eve runs tools **in the app process**. Beyond runs the agent **outside** the sandbox
(keys, grants, session store stay on the replica) and tools **inside** it. So the
compatible thing to share is the **author API**, not the process model: you still
`import { defineTool } from "eve/tools"`, and Beyond hosts those modules as an MCP
server the grant already knows how to dial.

That is what makes it cool for production agents:

- **Menu without wake.** `.mcp-tools.json` is stamped at bundle build; open is cheap.
- **Code stays in the guest.** Tool TS and its deps live in the seeded workspace; the
  model never gets a shell just to “have tools.”
- **Same grant story as browser MCP.** `sandbox: true`, exec front, first dial creates
  the VM — one pattern operators already know.

## Try it

```sh
# Local: Zod validate + call (no sandbox)
bun run src/cli.ts call --dir examples/weather/tools get_weather '{"city":"NYC"}'

# Stamp the cold-open menu
bun run src/cli.ts manifest --dir examples/weather/tools --name tools -o .mcp-tools.json

# Serve MCP on loopback. Sandbox root is the cwd (or --workspace), not --dir.
bun run src/cli.ts serve --dir examples/weather/tools --workspace . --http 127.0.0.1:8931

# Compiled guest binary (bun build --compile; still imports tools/*.ts from disk)
npm run compile   # → bin/tools-mcp

# Proofs
npm test && npm run parity
./deploy/demo/tools_mcp_e2e.sh --local
./deploy/demo/tools_mcp_e2e.sh            # needs E2B_API_KEY
./deploy/demo/tools_agent_e2e.sh          # real agent → E2B → get_weather
```

## Agent folder layout

```text
my-agent/
  AGENTS.md                 # tools: mcp__tools__get_weather, …
  tools/
    get_weather.ts          # export default defineTool({…})
  package.json              # eve + zod (+ this package)
  .mcp-tools.json           # optional; stamped by `manifest`
  node_modules/…            # must resolve in the seeded guest
```

`import { defineTool } from "eve/tools"` needs a package named `eve` (Node/`imports`
cannot remap that bare path). Use the shipped shim:

```json
{
  "dependencies": {
    "@beyond.dev/tools": "…",
    "eve": "file:…/agent-tools-mcp/eve-shim",
    "zod": "^4"
  }
}
```

## Bundle + grant (production shape)

1. Stamp the menu: `beyond-agent-tools-mcp manifest --dir tools --name tools -o .mcp-tools.json`
2. Pack the agent folder (ustar): `tools/`, `node_modules` the guest needs, `.mcp-tools.json`
3. Publish a **linux** `tools-mcp` binary (`npm run compile` on linux/amd64, or Docker)
4. Grant a sandbox connector whose URL is the exec front with `tools=<https URL of that binary>`
5. On first MCP dial the front: creates the VM → seeds the bundle →
   `cd workspace && tools-mcp serve --dir tools --workspace "$workspace"` → bridges MCP.
   `--dir` is the tool modules. `--workspace` (default: the process cwd) is
   `ctx.workspaceRoot` and the root of `ctx.getSandbox()`.

Demo wiring: `./deploy/demo/demo.sh tools` (`CHECKS_tools`, `uc_prepare_tools`).

## Mental model

```text
           .mcp-tools.json
                │
     ┌──────────┴──────────┐
     ▼                     ▼
beyond-agent          exec front
(menu at open)     (first tools/call)
     │                     │
     │                     ├─ create VM
     │                     ├─ seed bundle
     │                     └─ start tools-mcp
     │                            │
     └──────── tools/call ───────┘
                in-guest
```

Same bytes in the bundle; two jobs. The replica must not wake the sandbox to list
tools; the front must not invent the tool list — it runs what you shipped.
