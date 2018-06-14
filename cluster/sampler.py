from abc import ABC, abstractmethod

import numpy as np
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover


class Sampler(ABC):
    def __init__(self, rate=None, thres=None):
        self.rate = rate
        self.thres = thres
        if rate is not None and thres is not None:
            print('warning: both rate and thres are specified. only thres is used.')

    def __call__(self, graph, rate=None, thres=None):
        return self.select(self.measure(graph), rate=rate, thres=thres)

    def select(self, scores, rate=None, thres=None):
        if rate is None and thres is None:
            rate, thres = self.rate, self.thres
        return self.select_by_thres(scores, thres) if thres is not None else self.select_by_rate(scores, rate)

    def select_by_thres(self, scores, thres=None):
        thres = thres if thres is not None else self.thres
        return set(filter(lambda node: scores[node] < thres, scores.keys()))

    def select_by_rate(self, scores, rate=None):
        rate = rate if rate is not None else self.rate
        n = int(len(scores) * rate)
        thres = np.partition(np.array(list(scores.values())), n)[n]
        return self.select_by_thres(scores, thres)

    @abstractmethod
    def measure(self, graph):
        pass


class RandomSampler(Sampler):
    def measure(self, graph):
        return {node: np.random.random() for node in graph.nodes}


class IndegreeSampler(Sampler):
    def measure(self, graph):
        return dict(graph.in_degree)


class MutualNeighborSampler(Sampler):
    def measure(self, graph):
        return {node: len([nbr for nbr in graph.succ[node] if graph.has_edge(nbr, node)]) for node in graph.nodes}


class MinCoverSampler(Sampler):
    def __call__(self, graph, **kwargs):
        cover = min_weighted_vertex_cover(graph)
        return set(graph.nodes) - set(cover)

    def measure(self, graph):
        pass
