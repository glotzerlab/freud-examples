import freud
import numpy as np
from collections import defaultdict

# For simplicity, use a randomly generated set of points
box = freud.box.Box.cube(1)
points = box.wrap(np.random.rand(20, 3))
nl = freud.locality.LinkCell(
    box, 0.5).compute(box, points).nlist

# Get all sets of common neighbors.
common_neighbors = defaultdict(list)
for i, p in enumerate(points):
    for j in nl.index_j[nl.index_i == i]:
        for k in nl.index_j[nl.index_i == j]:
            if i != k:
                common_neighbors[(i, k)].append(j)


import networkx as nx

diagrams = defaultdict(list)
for (a, b), neighbors in common_neighbors.items():
    # Build up the graph of connections between the
    # common neighbors of a and b.
    g = nx.Graph()
    for i in neighbors:
        for j in set(nl.index_j[
            nl.index_i == i]).intersection(neighbors):
            g.add_edge(i, j)

    # Define the first 3 identifiers for a CNA diagram.
    are_neighbors = b in nl.index_j[nl.index_i == a]
    key = (are_neighbors, len(neighbors), g.number_of_edges())
    # If we've seen any neighborhood graphs with this signature,
    # we explicitly check if the two graphs are identical to
    # determine whether to save this one. Otherwise, we add
    # the new graph immediately.
    if key in diagrams:
        if all([not nx.is_isomorphic(g, h) for h in diagrams[key]]):
            diagrams[key].append(g)
    else:
        diagrams[key].append(g)
