from agent import run_agent
from tools import ORDERS

REFUNDS = {"A100": "no refund requested", "A101": "refund approved", "A102": "refund paid"}


def get_order_status(id: str) -> str:
    return f"order {id}: {ORDERS[id]['status']}"


def get_refund_status(id: str) -> str:
    return f"order {id}: {REFUNDS[id]}"


def schema(name, description, param_desc):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": param_desc}},
                "required": ["id"],
            },
        },
    }


registry = {
    "get_order_status": get_order_status,
    "get_refund_status": get_refund_status,
    "fetch_a": get_order_status,
    "fetch_b": get_refund_status,
}

vague = [
    schema("fetch_a", "Gets data.", "id"),
    schema("fetch_b", "Gets data.", "id"),
]
fixed = [
    schema("get_order_status", "Shipping status of an order (processing, shipped, delivered). Use for 'where is my order' or 'has it shipped'. Not for refunds.", "Order id like A100"),
    schema("get_refund_status", "Refund status of an order (requested, approved, paid). Use only when the user asks about money back or refunds.", "Order id like A100"),
]

question = "Did I get my money back for order A101?"
for label, schemas in (("VAGUE descriptions", vague), ("FIXED descriptions", fixed)):
    print(f"=== {label} ===")
    print(run_agent(question, schemas, registry, max_steps=4))
    print()
