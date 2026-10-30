import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

TRACE_FILE = Path(os.environ.get("TRACE_DIR", Path(__file__).parent.parent / "traces")) / "traces.jsonl"

PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "qwen/qwen3.8-27b": (0.29, 0.59)}


def cost_of(model, in_tok, out_tok):
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


class Trace:
    def __init__(self, name, input, session_id=None):
        self.data = {
            "trace_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "name": name,
            "input": input,
            "output": None,
            "observations": [],
        }
        self.start = time.perf_counter()

    @contextmanager
    def observe(self, name, kind="span", input=None):
        obs = {"name": name, "type": kind, "input": input, "output": None}
        start = time.perf_counter()
        try:
            yield obs
        finally:
            obs["duration_s"] = round(time.perf_counter() - start, 4)
            if obs.get("model"):
                obs["cost_usd"] = cost_of(obs["model"], obs.get("input_tokens", 0), obs.get("output_tokens", 0))
            self.data["observations"].append(obs)

    def finish(self, output):
        d = self.data
        d["output"] = output
        d["duration_s"] = round(time.perf_counter() - self.start, 4)
        d["total_cost_usd"] = sum(o.get("cost_usd", 0) for o in d["observations"])
        d["total_tokens"] = sum(o.get("input_tokens", 0) + o.get("output_tokens", 0) for o in d["observations"])
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        return d
