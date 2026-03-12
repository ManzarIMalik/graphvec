"""Visualisation utilities for graphvec graphs.

Requires ``matplotlib`` and ``networkx`` (``graphvec[viz]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphvec.graph import Graph


def visualize(
    graph: Graph,
    output: str | None = None,
    highlight: list[str] | None = None,
    node_ids: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    title: str = "graphvec",
) -> None:
    """Render the graph (or a subgraph) using matplotlib + networkx.

    Args:
        graph: The source :class:`~graphvec.graph.Graph`.
        output: If given, save the figure to this file path instead of
                opening an interactive window.
        highlight: Node IDs to render with a contrasting colour.
        node_ids: If provided, render only nodes in this list (and edges
                  between them).
        figsize: Matplotlib figure size ``(width, height)`` in inches.
        title: Figure title.

    Raises:
        ImportError: If matplotlib or networkx are not installed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        import networkx as nx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required for visualisation. "
            "Install with: pip install graphvec[viz]"
        ) from exc

    highlight_set: set[str] = set(highlight or [])

    # Build the networkx graph (filtered if node_ids given)
    G = nx.DiGraph()
    nodes = graph.nodes()
    if node_ids is not None:
        id_set = set(node_ids)
        nodes = [n for n in nodes if n.id in id_set]

    for node in nodes:
        G.add_node(node.id, label=node.label)

    for edge in graph.edges():
        if node_ids is not None and (edge.src not in G or edge.dst not in G):
            continue
        G.add_edge(edge.src, edge.dst, label=edge.label)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)

    pos = nx.spring_layout(G, seed=42)

    # Node colours
    node_colors = [
        "#e74c3c" if nid in highlight_set else "#3498db" for nid in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={nid: f"{nid}\n{G.nodes[nid].get('label','')}" for nid in G.nodes()},
        font_size=7,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="#95a5a6", arrows=True, arrowsize=20, ax=ax
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    ax.axis("off")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
