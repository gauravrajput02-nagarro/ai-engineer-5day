import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

DOCS_DIR = Path(__file__).parent.parent / "docs"


def chunk_text(text, size=120, overlap=30):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class Index:
    def __init__(self, docs_dir=DOCS_DIR, size=120, overlap=30):
        self.chunks = []
        for path in sorted(Path(docs_dir).glob("*.txt")):
            for i, c in enumerate(chunk_text(path.read_text(encoding="utf-8"), size, overlap)):
                self.chunks.append({"id": f"{path.name}#{i}", "source": path.name, "text": c})
        self.col = chromadb.EphemeralClient().get_or_create_collection("docs")
        if self.chunks:
            self.col.add(
                ids=[c["id"] for c in self.chunks],
                documents=[c["text"] for c in self.chunks],
                metadatas=[{"source": c["source"]} for c in self.chunks],
            )
            self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.by_id = {c["id"]: c for c in self.chunks}

    def search(self, query, k=4):
        if not self.chunks:
            return []
        vector_ids = self.col.query(query_texts=[query], n_results=min(k * 3, len(self.chunks)))["ids"][0]
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 3]
        keyword_ids = [self.chunks[i]["id"] for i in order if scores[i] > 0]
        fused = {}
        for ranking in (vector_ids, keyword_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        return [self.by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]
