import ast
import operator

ORDERS = {
    "A100": {"status": "shipped", "eta": "2026-11-03", "item": "laptop"},
    "A101": {"status": "processing", "eta": None, "item": "headphones"},
    "A102": {"status": "delivered", "eta": None, "item": "keyboard"},
}

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: could not evaluate. Use only numbers and + - * / ** and parentheses."


def get_order_status(order_id: str) -> str:
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Error: no order with id '{order_id}'. Valid ids look like A100."
    return f"Order {order_id.upper()} ({order['item']}): {order['status']}, eta {order['eta']}"


def always_fails(**_) -> str:
    return "Error: service unavailable"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression such as '(12.5 * 4) + 3'. Use this for any math instead of calculating yourself.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 * (3 + 4)'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status of a customer order by its order id (format: letter plus 3 digits, e.g. A100). Use when the user asks where an order is or whether it shipped.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order id such as A100"}},
                "required": ["order_id"],
            },
        },
    },
]

REGISTRY = {"calculator": calculator, "get_order_status": get_order_status}
