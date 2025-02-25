import networkx as nx
import matplotlib.pyplot as plt


def list_to_obj(list_in):
    obj = {}

    for el in list_in:
        obj[el] = el

    return obj


actors = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$"]
fallbacks = [r"$f_2$"]


G = nx.DiGraph()

# G.add_nodes_from(actors)
# G.add_edges_from([(r"$\alpha_1$", r"$\alpha_2$")])
# G.add_edges_from([(r"$\alpha_2$", r"$\alpha_3$")])
# G.add_edges_from([(r"$\alpha_2$", r"$f_2$")])
# G.add_edges_from([(r"$f_2$", r"$\alpha_1$")])

options = {
    "edgecolors": "tab:gray",
    "node_size": 800,
    "alpha": 0.9,
}
pos_actors = {
    r"$\alpha_1$": (1, 0),
    r"$\alpha_2$": (3, 0),
    r"$\alpha_3$": (5, 0),
}
edge_actors = [
    (r"$\alpha_1$", r"$\alpha_2$"),
    (r"$\alpha_2$", r"$\alpha_3$"),
]
edge_fallbacks = [
    (r"$\alpha_2$", r"$f_2$"),
    (r"$f_2$", r"$\alpha_1$"),
]
pos_fallbacks = {
    r"$f_2$": (2, -1),
}

pos_actors_fallbacks = {
    r"$\alpha_1$": (1, 0),
    r"$\alpha_2$": (3, 0),
    r"$\alpha_3$": (3, 0),
    r"$f_2$": (2, -1),
}

nx.draw_networkx_nodes(G, pos_actors, nodelist=actors, node_color="tab:blue", **options)
nx.draw_networkx_nodes(
    G, pos_fallbacks, nodelist=fallbacks, node_color="tab:red", **options
)

nx.draw_networkx_labels(
    G, pos_actors, list_to_obj(actors), font_size=22, font_color="whitesmoke"
)
nx.draw_networkx_labels(
    G, pos_fallbacks, list_to_obj(fallbacks), font_size=22, font_color="whitesmoke"
)

nx.draw_networkx_edges(
    G,
    pos_actors,
    edgelist=edge_actors,
    width=3,
    alpha=1,
    edge_color="tab:blue",
)
nx.draw_networkx_edges(
    G,
    pos_actors_fallbacks,
    edgelist=edge_fallbacks,
    width=3,
    alpha=1,
    edge_color="tab:red",
)
# nx.draw_networkx(G, pos=pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()
