import re
import sys

from cases import CASES
from rag import Index, generate

size = int(sys.argv[1]) if len(sys.argv) > 1 else 120
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 30
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["vector", "hybrid"]

index = Index("debug", size=size, overlap=overlap)
print(f"chunk size={size} overlap={overlap} k={k} chunks={len(index.chunks)}\n")

totals = {m: {"retrieval": 0, "answer": 0} for m in modes}
for case in CASES:
    print("Q:", case["q"])
    for mode in modes:
        chunks = index.retrieve(case["q"], k, mode)
        retrieved_text = " ".join(c["text"] for c in chunks).lower()
        retrieval_ok = case["fact"] is None or case["fact"].lower() in retrieved_text
        answer = generate(case["q"], chunks)
        norm = re.sub(r"[‐-― ]", lambda m: " " if m.group() == " " else "-", answer.lower())
        answer_ok = any(a in norm for a in case["ans"])
        totals[mode]["retrieval"] += retrieval_ok
        totals[mode]["answer"] += answer_ok
        stage = "ok" if retrieval_ok and answer_ok else ("RETRIEVAL" if not retrieval_ok else "GENERATION")
        print(f"  [{mode:7}] retrieval={'ok' if retrieval_ok else 'MISS'} answer={'ok' if answer_ok else 'BAD'} -> {stage}")
        if stage != "ok":
            print(f"            chunks: {[c['id'] for c in chunks]}")
            print(f"            answer: {answer[:160]!r}")
    print()

print("Totals out of", len(CASES))
for m, t in totals.items():
    print(f"  {m:7} retrieval {t['retrieval']}  answer {t['answer']}")
