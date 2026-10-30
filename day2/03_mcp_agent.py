import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent import MODEL, client
from tools import REGISTRY, TOOL_SCHEMAS


def to_groq_schema(t):
    return {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server_time"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            print("MCP server exposes:", [t.name for t in mcp_tools])

            schemas = TOOL_SCHEMAS + [to_groq_schema(t) for t in mcp_tools]
            local = set(REGISTRY)
            question = "What time is it right now in Tokyo? Also what is order A100's status?"
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
                {"role": "user", "content": question},
            ]
            for step in range(1, 7):
                resp = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=schemas, temperature=0, max_tokens=1024, reasoning_effort="low"
                )
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    print(f"[step {step}] final answer:\n{msg.content}")
                    return
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                        for c in msg.tool_calls
                    ],
                })
                for call in msg.tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    if call.function.name in local:
                        out, where = REGISTRY[call.function.name](**args), "local"
                    else:
                        res = await session.call_tool(call.function.name, args)
                        out, where = "".join(b.text for b in res.content if hasattr(b, "text")), "MCP"
                    print(f"[step {step}] ({where}) {call.function.name}({args}) -> {out}")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            print("Stopped: max steps reached")


asyncio.run(main())
