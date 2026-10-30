from agent import run_agent
from tools import REGISTRY, TOOL_SCHEMAS, always_fails

print("=== Multi-tool question ===")
print(run_agent("Order A100 and A101: which one has shipped? Also what is 18 * 24 + 7?", TOOL_SCHEMAS, REGISTRY))

print("\n=== Unknown order (error message guides the model) ===")
print(run_agent("What is the status of order Z999?", TOOL_SCHEMAS, REGISTRY))

print("\n=== Tool that always fails: max_steps stops the loop ===")
broken = {**REGISTRY, "get_order_status": always_fails}
print(run_agent(
    "Check order A100 status. Keep retrying the tool until it works, never give up.",
    TOOL_SCHEMAS, broken, max_steps=4,
))
