from rag import Index, ask

index = Index("manual")

questions = [
    "How often should I replace the filter?",
    "Can I get a refund if I change my mind?",
    "How long is the warranty?",
    "What should I do about error E-4021?",
]

for q in questions:
    answer, chunks = ask(index, q, k=4, mode="vector")
    print("Q:", q)
    print("A:", answer)
    print("Sources retrieved:", [c["id"] for c in chunks])
    print()
