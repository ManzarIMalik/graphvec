"""Example: building a simple knowledge graph with graphvec.

Demonstrates node/edge CRUD, traversal, path finding, and graph algorithms.
Run with: python examples/knowledge_graph.py
"""

from graphvec import GraphVec

# Create an in-memory graph (swap ":memory:" for a path to persist)
db = GraphVec(":memory:")

# Add people
people = [
    ("alice",   "Alice Cooper",   "Engineer"),
    ("bob",     "Bob Marley",     "Musician"),
    ("carol",   "Carol Danvers",  "Pilot"),
    ("dave",    "Dave Chapelle",  "Comedian"),
    ("eve",     "Eve Online",     "Gamer"),
]
for pid, name, role in people:
    db.add_node(pid, label="Person", name=name, role=role)

# Add relationships
db.add_edges([
    {"src": "alice", "dst": "bob",   "label": "KNOWS",      "since": 2018},
    {"src": "bob",   "dst": "carol", "label": "KNOWS",      "since": 2019},
    {"src": "carol", "dst": "dave",  "label": "KNOWS",      "since": 2020},
    {"src": "alice", "dst": "carol", "label": "WORKS_WITH", "years": 3},
    {"src": "dave",  "dst": "eve",   "label": "KNOWS",      "since": 2021},
])

print("=== Graph stats ===")
print(f"  Nodes : {db.node_count()}")
print(f"  Edges : {db.edge_count()}")
print(f"  Connected: {db.is_connected()}")

print("\n=== Alice's direct friends ===")
for node in db.v("alice").out("KNOWS").all():
    print(f"  {node['name']} ({node['role']})")

print("\n=== 2-hop friends of Alice ===")
for node in db.v("alice").out("KNOWS", hops=2).all():
    print(f"  {node['name']}")

print("\n=== Shortest path: alice → eve ===")
p = db.path("alice", "eve")
if p:
    print(f"  {' → '.join(n.id for n in p.nodes)}  ({p.length} hops)")

print("\n=== PageRank (top 3) ===")
pr = db.pagerank()
for nid, score in sorted(pr.items(), key=lambda x: -x[1])[:3]:
    node = db.get_node(nid)
    print(f"  {node['name']:<20} {score:.4f}")

print("\n=== BFS from alice ===")
order = db.bfs("alice")
print(f"  Visit order: {order}")
