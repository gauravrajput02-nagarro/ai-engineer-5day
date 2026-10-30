import os
import random
import time
from dataclasses import dataclass

import groq
from dotenv import load_dotenv

load_dotenv()

PRICES = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

RETRYABLE = (
    groq.RateLimitError,
    groq.APITimeoutError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    attempts: int


class LLMClient:
    def __init__(self, model="openai/gpt-oss-20b", timeout=30.0, max_retries=4, base_delay=1.0):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=timeout, max_retries=0)

    def chat(self, messages, temperature=0.0, max_tokens=1024, **kwargs) -> LLMResult:
        if self.model.startswith("openai/gpt-oss"):
            kwargs.setdefault("reasoning_effort", "low")
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                break
            except RETRYABLE as e:
                if attempt > self.max_retries:
                    raise
                delay = self.base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.5)
                print(f"  [retry {attempt}/{self.max_retries}] {type(e).__name__}; sleeping {delay:.1f}s")
                time.sleep(delay)

        usage = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=self._cost(usage.prompt_tokens, usage.completion_tokens),
            latency_s=time.perf_counter() - start,
            attempts=attempt,
        )

    def _cost(self, in_tok, out_tok):
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        return (in_tok * p_in + out_tok * p_out) / 1_000_000
