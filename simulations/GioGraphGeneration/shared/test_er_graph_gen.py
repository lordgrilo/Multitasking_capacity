import os, sys
import numpy as np

import IGtools as igt
import networkx as nx

graph_layer_size = int(5)
overlap = float(1)
g = igt.create_ER_bipartite_graph(graph_layer_size,overlap/float(graph_layer_size-1));
print g.number_of_nodes(), g.number_of_edges(), np.mean(nx.degree(g).values()), overlap/float(graph_layer_size-1)