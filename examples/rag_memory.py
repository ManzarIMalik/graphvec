"""Example: using graphvec as a RAG / agent memory store.

Demonstrates vector similarity search combined with graph traversal —
the key differentiator over plain vector stores.

Run with: python examples/rag_memory.py
(Requires: pip install graphvec[vector] — numpy)
"""

import math

from graphvec import GraphVec


# ---------------------------------------------------------------------------
# Toy embedding function (unit-circle projection on 4-dim space)
# In production replace with OpenAI / Cohere / sentence-transformers etc.
# ---------------------------------------------------------------------------

_WORD_VECS = {
    "python":     [1.0, 0.1, 0.0, 0.0],
    "database":   [0.1, 1.0, 0.0, 0.0],
    "graph":      [0.2, 0.8, 0.1, 0.0],
    "vector":     [0.0, 0.1, 1.0, 0.1],
    "embedding":  [0.0, 0.0, 0.9, 0.2],
    "search":     [0.1, 0.2, 0.8, 0.0],
    "machine":    [0.0, 0.0, 0.1, 1.0],
    "learning":   [0.1, 0.0, 0.2, 0.9],
}


def _normalise(v):
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v] if mag else v


def embed(text: str) -> list[float]:
    """Toy bag-of-words embedding."""
    words = text.lower().split()
    dim = 4
    vec = [0.0] * dim
    for w in words:
        wv = _WORD_VECS.get(w, [0.0] * dim)
        for i in range(dim):
            vec[i] += wv[i]
    return _normalise(vec)


# ---------------------------------------------------------------------------
# Build a knowledge graph of AI concepts
# ---------------------------------------------------------------------------

db = GraphVec(":memory:", embed_fn=embed, embed_field="summary")

# Add document nodes (auto-embedded via embed_field="summary")
docs = [
    ("d1", "Python is a high-level programming language"),
    ("d2", "Graph database stores nodes and edges"),
    ("d3", "Vector embedding enables semantic search"),
    ("d4", "Machine learning models learn from data"),
    ("d5", "Database indexing improves search performance"),
]
for did, summary in docs:
    db.add_node(did, label="Document", summary=summary)

# Add topic nodes
topics = [
    ("t_prog",  "Programming"),
    ("t_db",    "Databases"),
    ("t_ml",    "MachineLearning"),
    ("t_nlp",   "NLP"),
]
for tid, name in topics:
    db.add_node(tid, label="Topic", name=name)

# Link documents to topics
db.add_edges([
    {"src": "d1", "dst": "t_prog", "label": "ABOUT"},
    {"src": "d2", "dst": "t_db",   "label": "ABOUT"},
    {"src": "d3", "dst": "t_nlp",  "label": "ABOUT"},
    {"src": "d4", "dst": "t_ml",   "label": "ABOUT"},
    {"src": "d5", "dst": "t_db",   "label": "ABOUT"},
    {"src": "t_ml",  "dst": "t_nlp",  "label": "RELATED"},
    {"src": "t_nlp", "dst": "t_db",   "label": "RELATED"},
])

# ---------------------------------------------------------------------------
# RAG: find relevant documents then expand to related topics
# ---------------------------------------------------------------------------

query = "vector search database"
query_vec = embed(query)

print(f'Query: "{query}"\n')

print("=== Top-3 similar documents ===")
for r in db.search(query_vec, k=3, label="Document"):
    print(f"  [{r.score:.3f}] {r.node.id}: {r.node['summary']}")

print("\n=== Topics linked to top-2 similar documents ===")
topics_found = db.search(query_vec, k=2, label="Document").out("ABOUT").all()
for t in topics_found:
    print(f"  {t.id}: {t['name']}")

print("\n=== Related topics (2-hop) ===")
related = (
    db.search(query_vec, k=2, label="Document")
      .out("ABOUT")
      .out("RELATED")
      .all()
)
for t in related:
    print(f"  {t.id}: {t['name']}")

print("\n=== search_text convenience ===")
for r in db.search_text("graph database", k=2):
    print(f"  [{r.score:.3f}] {r.node.id}")
