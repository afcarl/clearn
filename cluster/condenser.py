from collections import OrderedDict
from abc import ABC, abstractmethod

import networkx as nx


class Condenser(ABC):
    def __init__(self, size=None, depth=None, use_all_edges=None, **kwargs):
        self.size = size
        self.depth = depth
        self.use_all_edges = use_all_edges
        self.kwargs = kwargs

    def __call__(self, nodes, graph, size=None, depth=None, use_all_edges=None):
        size = size if size is not None else self.size
        depth = depth if depth is not None else self.depth
        use_all_edges = use_all_edges if use_all_edges is not None else self.use_all_edges
        if not use_all_edges:
            filter = None
            graph.remove_nodes_from(set(graph.nodes) - set(nodes))
        else:
            filter = set(graph.nodes) - set(nodes)
        graph_ = nx.empty_graph(nodes, create_using=nx.DiGraph())
        for node in nodes:
            nbrs = self.search(node, graph, size=size, depth=depth, filter=filter, **self.kwargs)
            nbrs = sorted(nbrs.keys(), key=lambda node: nbrs[node])
            graph_.add_weighted_edges_from([(node, nbr, idx) for idx, nbr in zip(range(size), nbrs)])
        return graph_

    @abstractmethod
    def search(self, node, graph, **kwargs):
        pass


class BFSCondenser(Condenser):
    def search(self, node, graph, size, filter, **kwargs):
        nbrs, tmp = OrderedDict(), []
        for idx, (_, nbr) in zip(range(size), nx.bfs_edges(graph, node)):
            nbrs[nbr] = idx
            if filter is not None and nbr in filter:
                tmp.append(nbr)
        [nbrs.pop(nbr) for nbr in tmp]
        return nbrs


class SharedNeighborCondenser(Condenser):
    def search(self, node, graph, size, depth, filter, **kwargs):
        iou = lambda u, v: - (graph.out_degree[u] + graph.out_degree[v] + 2) / len(set(graph.succ[u]) | set(graph.succ[v]) | {u, v})
        nbrs, tmp = OrderedDict(), []
        for _, nbr in nx.dfs_edges(graph, node, depth_limit=depth):
            nbrs[nbr] = iou(node, nbr)
            if filter is not None and nbr in filter:
                tmp.append(nbr)
        [nbrs.pop(nbr) for nbr in tmp]
        return nbrs


class StaticCondenser(Condenser):
    def __init__(self, matrix, size=None, depth=None, use_all_edges=None, **kwargs):
        super().__init__(size=size, depth=depth, use_all_edges=use_all_edges, matrix=matrix, **kwargs)

    def search(self, node, graph, size, depth, filter, matrix, **kwargs):
        nbrs, tmp = OrderedDict(), []
        for _, nbr in nx.dfs_edges(graph, node, depth_limit=depth):
            nbrs[nbr] = matrix(node, nbr)
            if filter is not None and nbr in filter:
                tmp.append(nbr)
        [nbrs.pop(nbr) for nbr in tmp]
        return nbrs
