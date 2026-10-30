import json
import os

import groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-20b"
client = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30, max_retries=5)


def run_agent(question, tool_schemas, registry, max_steps=6, system="You are a helpful assistant. Use tools when needed.", verbose=True):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tool_schemas, temperature=0, max_tokens=1024, reasoning_effort="low"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if verbose:
                print(f"[step {step}] final answer")
            return msg.content
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
            fn = registry.get(call.function.name)
            result = fn(**args) if fn else f"Error: unknown tool {call.function.name}"
            if verbose:
                print(f"[step {step}] {call.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return f"Stopped: reached max_steps={max_steps} without a final answer."
