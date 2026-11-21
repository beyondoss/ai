// A minimal MCP Apps server built with the official SDK (`registerAppResource` / `registerAppTool`),
// because no shipped example puts `_meta.ui` on the resources/list entry alone. Over stdio.
//
//   ui://interop/listing  `_meta.ui` only on the listing  (the SDK's config `_meta`)
//   ui://interop/both     `_meta.ui` on the listing AND on the content item (content must win)
import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import { CONTENT_UI, LISTING_UI } from "./consts.mjs";

const server = new McpServer({ name: "interop-tiny", version: "1.0.0" });
const html = "<!doctype html><html><body>interop view</body></html>";
const text = (t) => ({ content: [{ type: "text", text: t }] });

for (
  const [name, uri, contentMeta] of [
    ["show_listing", "ui://interop/listing", undefined],
    ["show_both", "ui://interop/both", { ui: CONTENT_UI }],
  ]
) {
  registerAppResource(server, name, uri, {
    mimeType: RESOURCE_MIME_TYPE,
    _meta: { ui: LISTING_UI },
  }, async () => ({
    contents: [{
      uri,
      mimeType: RESOURCE_MIME_TYPE,
      text: html,
      ...(contentMeta && { _meta: contentMeta }),
    }],
  }));
  registerAppTool(server, name, {
    description: `open ${uri}`,
    inputSchema: z.object({}),
    _meta: { ui: { resourceUri: uri } },
  }, async () => text(`opened ${uri}`));
}
registerAppTool(server, "app_only", {
  description: "a view-only tool",
  inputSchema: z.object({ n: z.number().optional() }),
  _meta: { ui: { visibility: ["app"] } },
}, async ({ n }) => text(`app_only:${n ?? 0}`));
server.registerTool("model_only", {
  description: "a model-only tool",
  inputSchema: z.object({}),
  _meta: { ui: { visibility: ["model"] } },
}, async () => text("model_only"));

await server.connect(new StdioServerTransport());
