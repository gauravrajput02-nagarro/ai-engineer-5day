import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

TRACE_FILE = Path(__file__).parent / "traces" / "traces.jsonl"

PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "qwen/qwen3.8-27b": (0.29, 0.59)}


def cost_of(model, in_tok, out_tok):
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


class Observation:
    def __init__(self, name, kind, input):
        self.data = {"name": name, "type": kind, "input": input, "output": None}
        self.start = time.perf_counter()

    def end(self, output=None, model=None, input_tokens=None, output_tokens=None):
        self.data["output"] = output
        self.data["duration_s"] = round(time.perf_counter() - self.start, 4)
        if model:
            self.data.update(model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                             cost_usd=cost_of(model, input_tokens, output_tokens))


class Trace:
    def __init__(self, name, input, session_id=None):
        self.data = {
            "trace_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "name": name,
            "input": input,
            "output": None,
            "observations": [],
            "scores": [],
        }
        self.start = time.perf_counter()

    @contextmanager
    def observe(self, name, kind="span", input=None):
        obs = Observation(name, kind, input)
        try:
            yield obs
        finally:
            if "duration_s" not in obs.data:
                obs.end()
            self.data["observations"].append(obs.data)

    def score(self, name, value, comment=None):
        self.data["scores"].append({"name": name, "value": value, "comment": comment})

    def finish(self, output):
        self.data["output"] = output
        self.data["duration_s"] = round(time.perf_counter() - self.start, 4)
        self.data["total_cost_usd"] = sum(o.get("cost_usd", 0) for o in self.data["observations"])
        self.data["total_tokens"] = sum(
            (o.get("input_tokens") or 0) + (o.get("output_tokens") or 0) for o in self.data["observations"]
        )
        TRACE_FILE.parent.mkdir(exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.data, ensure_ascii=False) + "\n")


def load_traces():
    if not TRACE_FILE.exists():
        return []
    return [json.loads(line) for line in TRACE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
