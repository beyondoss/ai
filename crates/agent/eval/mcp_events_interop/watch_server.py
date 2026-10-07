"""Independent MCP Events server: the `mcp-webhook-events` PyPI package (0.2.0) on the official
Python MCP SDK (mcp 2.3.0). Only test-harness bits are ours: a control route to publish, and
short TTLs so the client's refresh/rotation loop runs within seconds."""
import json
import sys

import uvicorn
from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import mcp_webhook_events as mwe
from mcp_webhook_events import EventDefinition, McpEvents, SafeHttp

port = int(sys.argv[1])
store = sys.argv[2]

# Let grants be seconds, not hours, so a refresh happens during the check.
mwe.MIN_TTL_MS = 1000

mcp = MCPServer("Ticket watcher")


@mcp.tool()
def echo(text: str) -> str:
    """Echo."""
    return text


events = McpEvents(
    mcp,
    store=store,
    definitions=[
        EventDefinition(
            name="ticket.updated",
            description="A support ticket you are watching was updated.",
            input_schema={"type": "object", "properties": {"ticket_id": {"type": "string"}},
                          "required": ["ticket_id"], "additionalProperties": False},
            payload_schema={"type": "object",
                            "properties": {"ticket_id": {"type": "string"}, "summary": {"type": "string"}},
                            "required": ["ticket_id", "summary"], "additionalProperties": False},
        ),
    ],
    http=SafeHttp(allow_insecure_for_tests=True),
    default_ttl_ms=3000,
)
events.install()


async def emit(request: Request):
    body = await request.json()
    queued = events.publish("ticket.updated", body["data"], event_id=body.get("event_id"))
    stats = events.deliver_pending()
    return JSONResponse({"queued": queued, "stats": stats})


async def state(request: Request):
    return JSONResponse({"subscriptions": events.active_subscriptions()}, )


if len(sys.argv) > 3 and sys.argv[3] == "stdio":
    # MCP over stdio; the control API on its own port, in a thread.
    import threading
    from starlette.applications import Starlette
    control = Starlette(routes=[Route("/control/emit", emit, methods=["POST"]),
                                Route("/control/state", state, methods=["GET"])])
    threading.Thread(target=lambda: uvicorn.run(control, host="127.0.0.1", port=port, log_level="warning"),
                     daemon=True).start()
    mcp.run(transport="stdio")
else:
    app = mcp.streamable_http_app(stateless_http=True, json_response=True)
    app.router.routes.append(Route("/control/emit", emit, methods=["POST"]))
    app.router.routes.append(Route("/control/state", state, methods=["GET"]))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
