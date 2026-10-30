from pathlib import Path

from llm_client import LLMClient

complaint = (Path(__file__).parent / "sample_data" / "complaint.txt").read_text().strip()

bad = f"Summarize this customer complaint: {complaint}"

good = f"""Task: Summarize the customer complaint below for a support team lead.
Role: You are a senior support analyst at an online electronics retailer.
Context: The complaint is between <complaint> tags.
<complaint>
{complaint}
</complaint>
Constraints:
- Use only facts stated in the complaint; do not invent order numbers or dates.
- Maximum 3 bullet points, each under 20 words.
- If something is unclear, write "unclear" rather than guessing.
Output format:
Issue: <one line>
Customer wants: <one line>
Urgency: low | medium | high (with a 5-word reason)
"""

client = LLMClient()
bad_r = client.chat([{"role": "user", "content": bad}], max_tokens=400)
good_r = client.chat([{"role": "user", "content": good}], max_tokens=400)

line = "=" * 40
print(f"{line} BAD PROMPT {line}\n{bad}\n\n--> {bad_r.text}\n  ({bad_r.output_tokens} output tokens)")
print(f"\n{line} GOOD PROMPT {line}\n--> {good_r.text}\n  ({good_r.output_tokens} output tokens)")
