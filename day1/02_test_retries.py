from types import SimpleNamespace

import groq
import httpx

from llm_client import LLMClient

req = httpx.Request("POST", "https://api.groq.com/x")


def fake_ok():
    msg = SimpleNamespace(content="ok")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def make_client(errors):
    c = LLMClient(base_delay=0.01)
    queue = list(errors)

    def create(**_):
        if queue:
            raise queue.pop(0)
        return fake_ok()

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c


msgs = [{"role": "user", "content": "hi"}]

rl = groq.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
r = make_client([rl, rl]).chat(msgs)
assert r.attempts == 3 and r.text == "ok", r
print("PASS: retried 2 rate-limit errors, succeeded on attempt 3")

to = groq.APITimeoutError(request=req)
try:
    make_client([to] * 10).chat(msgs)
    raise SystemExit("FAIL: should have raised")
except groq.APITimeoutError:
    print("PASS: gave up after max retries and re-raised")

auth = groq.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
try:
    make_client([auth]).chat(msgs)
    raise SystemExit("FAIL: should have raised")
except groq.AuthenticationError:
    print("PASS: 401 raised immediately, not retried")
