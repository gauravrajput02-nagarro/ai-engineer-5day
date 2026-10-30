import json

from fastapi.testclient import TestClient

from app.main import app

QUESTIONS = [
    "Ticket T-1001 is open. What does the error in it mean and how do I fix it?",
    "How much would 3 packs of 2 filters cost, and what is the support phone number?",
    "Does the AquaPure X200 work with sparkling water?",
    "What is the status of ticket T-9999?",
]

with TestClient(app) as client:
    print("health:", client.get("/health").json())
    for q in QUESTIONS:
        r = client.post("/ask", json={"question": q, "session_id": "test"})
        print("\nQ:", q)
        print("status:", r.status_code)
        body = r.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))
    trace = client.get(f"/traces/{body['trace_id']}").json()
    print("\nlast trace observations:", [(o["name"], o["duration_s"]) for o in trace["observations"]])
    print("bad input status:", client.post("/ask", json={"question": "x"}).status_code)
