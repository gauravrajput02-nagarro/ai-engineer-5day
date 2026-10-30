from llm_client import LLMClient

client = LLMClient()
result = client.chat(
    [{"role": "user", "content": "In two sentences, what is a token in an LLM?"}],
    temperature=0.2,
)

print("Reply:", result.text)
print(f"Model: {result.model}")
print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
print(f"Cost: ${result.cost_usd:.6f}  Latency: {result.latency_s:.2f}s")

print("\n--- No memory between calls ---")
r1 = client.chat([{"role": "user", "content": "My name is Gaurav. Remember it."}])
r2 = client.chat([{"role": "user", "content": "What is my name?"}])
print("Without history:", r2.text)
history = [
    {"role": "user", "content": "My name is Gaurav. Remember it."},
    {"role": "assistant", "content": r1.text},
    {"role": "user", "content": "What is my name?"},
]
print("With history resent:", client.chat(history).text)
