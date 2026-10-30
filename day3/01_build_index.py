from rag import Index, load_docs

docs = load_docs()
print(f"Loaded {len(docs)} documents:", [s for s, _ in docs])

index = Index("manual", size=120, overlap=30)
print(f"Created {len(index.chunks)} chunks (120 words, 30 overlap)")
print(f"Stored in Chroma collection 'manual': {index.col.count()} vectors")

print("\nSample chunk:")
c = index.chunks[5]
print(c["id"], "->", c["text"][:300], "...")
