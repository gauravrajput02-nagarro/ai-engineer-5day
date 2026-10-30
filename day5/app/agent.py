import ast
import json
import operator
import os
import re
from typing import Literal

import groq
from pydantic import BaseModel, Field, ValidationError

from .rag import Index
from .tracing import Trace

MODEL = os.environ.get("MODEL", "openai/gpt-oss-20b")
REASONING = os.environ.get("REASONING", "low")
MAX_STEPS = 6

TICKETS = {
    "T-1001": {"customer": "Maria", "order": "A100", "status": "open", "issue": "Error E-4021 after filter change"},
    "T-1002": {"customer": "Ken", "order": "A101", "status": "resolved", "issue": "Refund requested within 30 days"},
    "T-1003": {"customer": "Priya", "order": "A102", "status": "open", "issue": "Unit leaks from the bottom"},
}

SYSTEM = """You are a support agent for the AquaPure X200 water purifier.
Tools: search_docs for anything in the manual, lookup_ticket for support tickets (ids like T-1001), calculator for arithmetic.
Rules:
- Answer ONLY from tool results. Never invent facts. If search_docs does not contain the answer, say you don't know and give confidence "low" with no sources.
- If the question has several parts, search for each part separately.
- When done, call final_answer exactly once. Put the source file names (for example 03_error_codes.txt) or ticket ids you used in sources."""


class FinalAnswer(BaseModel):
    answer: str = Field(description="Concise answer for the customer")
    sources: list[str] = Field(description="File names or ticket ids that support the answer; empty if unknown")
    confidence: Literal["low", "medium", "high"]


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported")


def calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: use only numbers and + - * / ( )"


def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    schema("search_docs", "Search the AquaPure X200 manual (setup, maintenance, error codes, warranty, returns, prices, support contact). Use for any product question. Returns passages with their file names.",
           {"query": {"type": "string", "description": "What to look for, e.g. 'error E-4021' or 'refund policy'"}}, ["query"]),
    schema("lookup_ticket", "Look up a customer support ticket by id such as T-1001. Returns customer, order, status and issue.",
           {"ticket_id": {"type": "string", "description": "Ticket id like T-1001"}}, ["ticket_id"]),
    schema("calculator", "Evaluate an arithmetic expression such as '44 * 3'. Use for any math.",
           {"expression": {"type": "string", "description": "Arithmetic expression"}}, ["expression"]),
    {"type": "function", "function": {"name": "final_answer", "description": "Submit the final answer to the customer. Call exactly once when finished.",
                                      "parameters": FinalAnswer.model_json_schema()}},
]


class Agent:
    def __init__(self, index: Index):
        self.index = index
        self.client = groq.Groq(timeout=30, max_retries=5)

    def run_tool(self, name, args):
        if name == "search_docs":
            chunks = self.index.search(args.get("query", ""), 4)
            if not chunks:
                return "No relevant passages found."
            return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        if name == "lookup_ticket":
            t = TICKETS.get(str(args.get("ticket_id", "")).upper())
            return json.dumps(t) if t else "Error: ticket not found. Ids look like T-1001."
        if name == "calculator":
            return calculator(str(args.get("expression", "")))
        return f"Error: unknown tool {name}"

    def ask(self, question, session_id=None):
        trace = Trace("doc-qa-agent", question, session_id)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        tools_used, final, steps = [], None, 0
        for steps in range(1, MAX_STEPS + 1):
            with trace.observe(f"llm-step-{steps}", "generation", {"messages": len(messages)}) as obs:
                resp = self.client.chat.completions.create(
                    model=MODEL, messages=messages, tools=TOOLS, temperature=0, max_tokens=1500, reasoning_effort=REASONING
                )
                msg = resp.choices[0].message
                obs.update(model=MODEL, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens,
                           output=[c.function.name for c in msg.tool_calls or []] or msg.content)
            if not msg.tool_calls:
                final = FinalAnswer(answer=msg.content or "", sources=sorted(set(re.findall(r"\[([\w.\-]+\.txt)\]", msg.content or ""))),
                                    confidence="low")
                break
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in msg.tool_calls],
            })
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError:
                    args = {}
                if call.function.name == "final_answer":
                    try:
                        final = FinalAnswer.model_validate(args)
                        result = "ok"
                    except ValidationError as e:
                        result = f"Error: invalid final_answer: {e.errors()[0]['msg']}"
                else:
                    tools_used.append(call.function.name)
                    with trace.observe(call.function.name, "span", args) as obs:
                        result = self.run_tool(call.function.name, args)
                        obs["output"] = result[:300]
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if final:
                break
        if final is None:
            final = FinalAnswer(answer="I could not finish within the step limit.", sources=[], confidence="low")
        data = trace.finish(final.model_dump())
        return {
            **final.model_dump(),
            "tools_used": tools_used,
            "steps": steps,
            "trace_id": data["trace_id"],
            "cost_usd": round(data["total_cost_usd"], 6),
            "latency_s": data["duration_s"],
        }
