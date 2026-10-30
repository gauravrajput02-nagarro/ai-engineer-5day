import json
import os
import re
from pathlib import Path

import chromadb
import groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"
DB_DIR = HERE / "chroma_db"
MODEL = os.environ.get("MODEL", "openai/gpt-oss-20b")
REASONING = os.environ.get("REASONING", "low")

llm = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30, max_retries=5)


def load_docs(docs_dir=DOCS_DIR):
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(Path(docs_dir).glob("*.txt"))]


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


def decompose(question):
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": 'Split the user question into independent search queries, one per distinct information need. Use identifiers like error codes only in queries about that identifier; a question about contacting support or warranty must not contain the error code. Reply with JSON only: {"queries": ["..."]}.'},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=512,
        reasoning_effort=REASONING,
        response_format={"type": "json_object"},
    )
    try:
        queries = json.loads(resp.choices[0].message.content)["queries"]
        return [q for q in queries if isinstance(q, str) and q.strip()] or [question]
    except (ValueError, KeyError, TypeError):
        return [question]


class Index:
    def __init__(self, name, size=120, overlap=30, docs=None):
        self.chunks = []
        for source, text in docs or load_docs():
            for i, c in enumerate(chunk_text(text, size, overlap)):
                self.chunks.append({"id": f"{source}#{i}", "source": source, "text": c})
        db = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            db.delete_collection(name)
        except Exception:
            pass
        self.col = db.create_collection(name)
        self.col.add(
            ids=[c["id"] for c in self.chunks],
            documents=[c["text"] for c in self.chunks],
            metadatas=[{"source": c["source"]} for c in self.chunks],
        )
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.by_id = {c["id"]: c for c in self.chunks}

    def vector_search(self, question, k):
        res = self.col.query(query_texts=[question], n_results=k)
        return res["ids"][0]

    def keyword_search(self, question, k):
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i]["id"] for i in order if scores[i] > 0]

    def retrieve(self, question, k=4, mode="vector"):
        if mode == "multi":
            seen, out = set(), []
            for sub in decompose(question):
                for chunk in self.retrieve(sub, max(2, k // 2), "hybrid"):
                    if chunk["id"] not in seen:
                        seen.add(chunk["id"])
                        out.append(chunk)
            return out
        if mode == "vector":
            ids = self.vector_search(question, k)
        elif mode == "keyword":
            ids = self.keyword_search(question, k)
        else:
            fused = {}
            for ranking in (self.vector_search(question, k * 3), self.keyword_search(question, k * 3)):
                for rank, cid in enumerate(ranking):
                    fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
            ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.by_id[i] for i in ids]


SYSTEM = """You answer questions about the AquaPure X200 water purifier using ONLY the provided context.
Rules:
- If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
- Be concise.
- After each fact, cite its source in square brackets using the file name, e.g. [03_error_codes.txt]."""


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question, chunks, system=SYSTEM):
    resp = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": build_prompt(question, chunks)}],
        temperature=0,
        max_tokens=1024,
        reasoning_effort=REASONING,
    )
    return resp.choices[0].message.content


def ask(index, question, k=4, mode="vector", system=SYSTEM):
    chunks = index.retrieve(question, k, mode)
    return generate(question, chunks, system), chunks
