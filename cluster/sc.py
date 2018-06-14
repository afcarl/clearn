import numpy as np
import networkx as nx

from .dendrogram import Dendrogram, Leaf


class SamplingClustering:
    def __init__(self, sampler, condenser, pre_condenser):
        self.sampler = sampler
        self.condenser = condenser
        self.pre_condenser = pre_condenser

    def fit(self, graph):
        if type(graph) is not nx.DiGraph:
            self.graph = nx.empty_graph(len(graph), create_using=nx.DiGraph())
            for node, nbrs in enumerate(graph):
                self.graph.add_weighted_edges_from([(node, nbr, idx) for idx, nbr in enumerate(nbrs)])
        else:
            self.graph = graph

        self.associations = nx.empty_graph(self.graph.nodes, create_using=nx.DiGraph())

        graph = self.pre_condenser(set(self.graph.nodes), graph=self.graph)
        subgraphs = [graph.subgraph(c).copy() for c in nx.strongly_connected_components(graph)]
        self.dendrogram = Dendrogram([self.cluster(graph=subgraph) for subgraph in subgraphs])
        return self

    def cluster(self, graph, **kwargs):
        removed = self.sampler(graph=graph)
        if len(removed) == 0 or len(removed) == len(graph.nodes):
            head = self.associate(list(graph.nodes))
            return Leaf([int(node) for node in list(nx.descendants(self.associations, head)) + [head]])
        else:
            self.associate(removed, graph)
            graph.remove_nodes_from(removed)
            graph = self.condenser(nodes=set(graph.nodes), graph=graph)
            subgraphs = [graph.subgraph(c).copy() for c in nx.strongly_connected_components(graph)]
            return Dendrogram([self.cluster(subgraph) for subgraph in subgraphs])

    def associate(self, nodes, graph=None):
        if graph is None:
            head = nodes.pop()
            self.associations.add_edges_from([(head, node) for node in nodes])
            return head
        else:
            for node in nodes:
                for u, v in nx.bfs_edges(graph, node):
                    if v not in nodes:
                        self.associations.add_edge(v, node)
                        break
                else:
                    print(f"error: failed to associate {node}")

    def smooth(self, dendrogram=None):
        return smooth((self.dendrogram if dendrogram is None else dendrogram), self.graph)


def smooth_labels(labels, graph):
    local_common_label = lambda node: np.bincount(labels[list(graph.succ[node]) + [node]]).argmax()
    return np.array([local_common_label(node) for node in range(len(labels))])


def smooth(dendrogram, graph):
    labels, leaves = dendrogram.flatten(return_leaves=True)
    labels = smooth_labels(labels, graph)
    for n, leaf in enumerate(leaves):
        leaf.clear()
        leaf += [int(node) for node in np.arange(len(labels))[labels == n]]
    dendrogram.format()
