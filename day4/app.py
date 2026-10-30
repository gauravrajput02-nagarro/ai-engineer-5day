import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "day3"))

from rag import MODEL, REASONING, Index, build_prompt, llm
from tracing import Trace

PROMPT_V1 = "Answer the question using the context. Be helpful."

PROMPT_V2 = """You answer questions about the AquaPure X200 water purifier using ONLY the provided context.
Rules:
- If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
- Be concise.
- After each fact, cite its source in square brackets using the file name, e.g. [03_error_codes.txt]."""

_index = None


def get_index():
    global _index
    if _index is None:
        _index = Index("day4", size=120, overlap=30)
    return _index


def traced_ask(question, system=PROMPT_V2, session_id=None, k=4, mode="hybrid", finish=True):
    trace = Trace("rag-qa", question, session_id)
    index = get_index()
    with trace.observe("retrieve", "span", {"question": question, "k": k, "mode": mode}) as obs:
        chunks = index.retrieve(question, k, mode)
        obs.end(output=[c["id"] for c in chunks])
    with trace.observe("generate", "generation", {"system": system[:60], "question": question}) as obs:
        resp = llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": build_prompt(question, chunks)}],
            temperature=0,
            max_tokens=1024,
            reasoning_effort=REASONING,
        )
        answer = resp.choices[0].message.content
        obs.end(output=answer, model=MODEL, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)
    if finish:
        trace.finish(answer)
    return answer, chunks, trace
