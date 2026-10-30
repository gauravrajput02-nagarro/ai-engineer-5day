import json
import re
import statistics
from pathlib import Path

from app import PROMPT_V1, PROMPT_V2, traced_ask
from rag import MODEL, REASONING, llm

DATASET = json.loads((Path(__file__).parent / "dataset.json").read_text())

JUDGE_SYSTEM = """You are a strict grader. Given a question, the context passages that were retrieved, and a model answer, decide whether the answer is faithful: every claim in the answer is supported by the context, or the answer correctly says it does not know when the context lacks the information.
Reply with JSON only: {"faithful": true or false, "reason": "short"}."""


def norm(text):
    return re.sub(r"[‐-―]", "-", text.replace(" ", " ")).lower()


def judge(question, chunks, answer):
    context = "\n\n".join(c["text"] for c in chunks)
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"},
        ],
        temperature=0,
        max_tokens=512,
        reasoning_effort=REASONING,
        response_format={"type": "json_object"},
    )
    try:
        return bool(json.loads(resp.choices[0].message.content)["faithful"])
    except (ValueError, KeyError, TypeError):
        return False


def run(label, system):
    rows = []
    for item in DATASET:
        answer, chunks, trace = traced_ask(item["q"], system=system, session_id=f"eval-{label}", finish=False)
        text = norm(answer)
        has_facts = all(any(alt in text for alt in norm(f).split("|")) for f in item["facts"])
        cites = item["source"] is None or item["source"] in answer
        faithful = judge(item["q"], chunks, answer)
        trace.score("has_expected_facts", float(has_facts))
        trace.score("cites_right_source", float(cites))
        trace.score("faithful_judge", float(faithful))
        trace.data["name"] = f"eval-{label}"
        trace.finish(answer)
        rows.append({"q": item["q"], "facts": has_facts, "cites": cites, "faithful": faithful,
                     "cost": trace.data["total_cost_usd"], "latency": trace.data["duration_s"], "answer": answer})
    return rows


def summarize(label, rows):
    n = len(rows)
    return {
        "label": label,
        "facts": sum(r["facts"] for r in rows) / n,
        "cites": sum(r["cites"] for r in rows) / n,
        "faithful": sum(r["faithful"] for r in rows) / n,
        "avg_cost": statistics.mean(r["cost"] for r in rows),
        "avg_latency": statistics.mean(r["latency"] for r in rows),
    }


if __name__ == "__main__":
    results = {}
    for label, system in (("prompt_v1", PROMPT_V1), ("prompt_v2", PROMPT_V2)):
        print(f"Running {label} on {len(DATASET)} questions...")
        results[label] = run(label, system)

    print(f"\n{'':12}{'facts':>8}{'cites':>8}{'faithful':>10}{'avg $':>12}{'avg s':>8}")
    for label, rows in results.items():
        s = summarize(label, rows)
        print(f"{label:12}{s['facts']:8.0%}{s['cites']:8.0%}{s['faithful']:10.0%}{s['avg_cost']:12.6f}{s['avg_latency']:8.2f}")

    print("\nQuestions where v2 differs from v1:")
    for a, b in zip(results["prompt_v1"], results["prompt_v2"]):
        if (a["facts"], a["cites"], a["faithful"]) != (b["facts"], b["cites"], b["faithful"]):
            print(f"- {a['q']}")
            print(f"    v1 facts={a['facts']} cites={a['cites']} faithful={a['faithful']}")
            print(f"    v2 facts={b['facts']} cites={b['cites']} faithful={b['faithful']}")

    fails = [r for r in results["prompt_v2"] if not (r["facts"] and r["cites"] and r["faithful"])]
    print(f"\nv2 still failing {len(fails)} question(s):")
    for r in fails:
        print(f"- {r['q']}  facts={r['facts']} cites={r['cites']} faithful={r['faithful']}")
        print(f"    answer: {r['answer'][:160]!r}")
