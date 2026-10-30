from agent import MODEL, client
from tools import TOOL_SCHEMAS

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Where is order A100, and what is 18 * 24?"}],
    tools=TOOL_SCHEMAS,
    temperature=0,
    reasoning_effort="low",
)
msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("text content:", repr(msg.content))
for c in msg.tool_calls or []:
    print(f"model REQUESTED: {c.function.name}({c.function.arguments})  <- nothing has run yet")
