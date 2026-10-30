from app import traced_ask

questions = ["What should I do about error E-4021?", "How much does a pack of 4 filters cost?", "Can I use vinegar to clean the tank?"]
for q in questions:
    answer, chunks, trace = traced_ask(q, session_id="demo-session")
    d = trace.data
    print(f"Trace {d['trace_id']}  session={d['session_id']}  total {d['duration_s']:.2f}s  ${d['total_cost_usd']:.6f}  {d['total_tokens']} tokens")
    for o in d["observations"]:
        extra = f"  ${o['cost_usd']:.6f} {o['input_tokens']}+{o['output_tokens']} tok" if "cost_usd" in o else ""
        print(f"   - {o['type']:10} {o['name']:9} {o['duration_s']:.3f}s{extra}")
    slowest = max(d["observations"], key=lambda o: o["duration_s"])
    costly = max(d["observations"], key=lambda o: o.get("cost_usd", 0))
    print(f"   slowest step: {slowest['name']}   most expensive step: {costly['name']}\n")
