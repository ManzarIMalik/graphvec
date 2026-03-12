"""Example: collaborative-filtering style recommendations using graphvec.

User → RATED → Item edges with weight = rating (1-5).
Find items liked by users similar to a target user via graph traversal.

Run with: python examples/recommendation.py
"""

from graphvec import GraphVec

db = GraphVec(":memory:")

# Users
users = ["alice", "bob", "carol", "dave"]
for u in users:
    db.add_node(u, label="User", name=u.title())

# Items
items = {
    "python_book": "Python Crash Course",
    "graph_book":  "Graph Databases",
    "ml_course":   "ML Engineering",
    "sql_guide":   "SQL for Beginners",
    "rust_book":   "Programming Rust",
}
for iid, title in items.items():
    db.add_node(iid, label="Item", title=title)

# Ratings (user → item, weight = 1-5)
ratings = [
    ("alice", "python_book", 5),
    ("alice", "graph_book",  4),
    ("alice", "ml_course",   3),
    ("bob",   "python_book", 4),
    ("bob",   "ml_course",   5),
    ("bob",   "rust_book",   3),
    ("carol", "graph_book",  5),
    ("carol", "sql_guide",   4),
    ("carol", "python_book", 3),
    ("dave",  "rust_book",   5),
    ("dave",  "sql_guide",   3),
]
for user, item, rating in ratings:
    db.add_edge(user, item, label="RATED", weight=float(rating))

# Find "co-rated" users: users who rated the same item as alice
print("=== Items Alice has rated ===")
alice_items = db.v("alice").out("RATED").all()
for item in alice_items:
    print(f"  {item['title']}")

print("\n=== Users who share taste with Alice (co-rated items) ===")
similar_users = (
    db.v("alice")
      .out("RATED")          # items alice rated
      .in_("RATED")          # users who rated those items
      .has_not(id="alice")   # exclude alice herself (uses property filter)
      .all()
)
# Filter out Alice manually since has_not works on properties not id field
similar_users = [u for u in similar_users if u.id != "alice"]
seen = set()
for u in similar_users:
    if u.id not in seen:
        print(f"  {u['name']}")
        seen.add(u.id)

print("\n=== Recommended items for Alice (items similar users rated that Alice hasn't) ===")
alice_item_ids = {n.id for n in alice_items}
recommendations: dict[str, int] = {}
for similar_user in similar_users:
    for item in db.v(similar_user.id).out("RATED").all():
        if item.id not in alice_item_ids:
            recommendations[item.id] = recommendations.get(item.id, 0) + 1

for item_id, votes in sorted(recommendations.items(), key=lambda x: -x[1]):
    item = db.get_node(item_id)
    print(f"  [{votes} votes] {item['title']}")

print("\n=== Graph stats ===")
print(f"  Nodes: {db.node_count()}")
print(f"  Edges: {db.edge_count()}")
pr = db.pagerank()
print("  Top PageRank nodes:")
for nid, score in sorted(pr.items(), key=lambda x: -x[1])[:3]:
    node = db.get_node(nid)
    label = node.label
    name = node.get("title") or node.get("name") or nid
    print(f"    {name} ({label}): {score:.4f}")
