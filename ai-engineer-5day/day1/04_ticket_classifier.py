import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from llm_client import LLMClient

EMAILS = json.loads((Path(__file__).parent / "sample_data" / "emails.json").read_text())


class Ticket(BaseModel):
    category: Literal["billing", "technical", "account", "feature_request", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str

    model_config = {"extra": "forbid"}


SYSTEM = """You are a support triage assistant. Classify the customer email.
Constraints: use only facts in the email; summary is one sentence under 20 words.
Urgency: high = money lost or service blocked now; medium = broken but workaround exists; low = question or suggestion.

Examples:
Email: "I can't log in, password reset email never arrives and I have a deadline today."
{"category": "account", "urgency": "high", "summary": "Customer cannot log in; reset email not arriving; deadline today."}
Email: "Is there a way to export my data as CSV?"
{"category": "feature_request", "urgency": "low", "summary": "Customer asks whether data can be exported as CSV."}"""

client = LLMClient()


def classify(email: str, strict: bool = True) -> Ticket:
    if strict:
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": "ticket", "strict": True, "schema": Ticket.model_json_schema()},
        }
    else:
        fmt = {"type": "json_object"}
    system = SYSTEM
    if not strict:
        system += "\nRespond with a JSON object."
    r = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": f'Email: "{email}"'}],
        response_format=fmt,
    )
    return Ticket.model_validate_json(r.text)


if __name__ == "__main__":
    print("--- Classify 5 sample emails ---")
    for e in EMAILS:
        t = classify(e["body"])
        print(f'#{e["id"]} {t.category:15} {t.urgency:6} | {t.summary}')

    print("\n--- 20x consistency check (same email, schema-enforced) ---")
    shapes, categories = Counter(), Counter()
    for _ in range(20):
        t = classify(EMAILS[0]["body"])
        shapes[tuple(sorted(t.model_dump()))] += 1
        categories[(t.category, t.urgency)] += 1
    print("Distinct key-sets:", dict(shapes))
    print("(category, urgency) values:", dict(categories))

    print("\n--- JSON mode only (no schema), 20 runs ---")
    ok = bad = 0
    for _ in range(20):
        try:
            classify(EMAILS[0]["body"], strict=False)
            ok += 1
        except ValidationError:
            bad += 1
    print(f"Valid against schema: {ok}, invalid: {bad}")
