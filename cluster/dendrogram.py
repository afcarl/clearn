import collections
import queue

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class Leaf(collections.UserList):
    def descendants(self):
        return list(self)

    def __str__(self, prefix='', last=True):
        return f"{prefix}{'└' if last else '├'}── {len(self)}\n"

    def __lt__(self, other):
        return len(self) < len(other)


class Dendrogram(collections.UserList):
    def __init__(self, initlist=None):
        super().__init__(initlist)
        self.format()

    def __str__(self, prefix='', last=True):
        line = f"{prefix}{'└' if last else '├'}── {len(self.descendants())}\n"
        branches = [child for child in self if type(child) is Dendrogram or type(child) is Leaf]
        for idx, child in enumerate(branches):
            line += child.__str__(f"{prefix}{' ' if last else '│'}   ", last=idx == len(branches) - 1)
        return line

    def __lt__(self, other):
        return len(self.descendants()) < len(other.descendants())

    def all_children_are_leaves(self):
        return np.array([type(child) is Leaf for child in self]).all()

    def descendants(self):
        objects = []
        for child in self:
            objects += child.descendants()
        return objects

    def leaves(self, pred=False):
        leaves = [((child, self) if pred else child) for child in self if type(child) is Leaf]
        for child in self:
            leaves += child.leaves(pred=pred) if type(child) is Dendrogram else []
        return leaves

    def mergeables(self, pred=False):
        nodes = [((child, self) if pred else child) for child in self if type(child) is Dendrogram and child.all_children_are_leaves()]
        for child in self:
            nodes += child.mergeables(pred=pred) if type(child) is Dendrogram else []
        return nodes

    def format(self):
        leaves = [child for child in self if type(child) is Leaf]
        [self.remove(leaf) for leaf in leaves if len(leaf) is 0]
        branches = [child for child in self if type(child) is Dendrogram]
        for branch in branches:
            branch.format()
            if len(branch) < 2:
                self += branch
                self.remove(branch)

    def merge(self, n, balance=True):
        if balance:
            while len(self.leaves()) > n:
                node, pred = min(self.leaves(pred=True))
                if len(pred) > 1:
                    pred.remove(node)
                    brother = min(pred)
                    brother += node if type(brother) is Leaf else [node]
                self.format()
        else:
            tot = len(self.descendants())
            while True:
                nodes = self.mergeables(pred=True)
                if len(nodes) == 0:
                    break
                node, pred = min(nodes)
                leaves = sorted(self.leaves(), reverse=True)
                top_n = sum([len(leaf) for _, leaf in zip(range(n), leaves)])
                if  top_n < tot * 0.8 and len(leaves) + 1 - len(node) >= n:
                    pred.remove(node)
                    pred.append(Leaf(node.descendants()))
                else:
                    break
                self.format()
        return self

    def flatten(self, return_leaves=False):
        leaves = self.leaves()
        labels = -np.ones(sum([len(leaf) for leaf in leaves]), dtype=int)
        for label, leaf in enumerate(leaves):
            labels[leaf] = label
        return (labels, leaves) if return_leaves else labels

    def draw(self, pos, path):
        ims = []
        fig = plt.figure()
        plt.axis('off')
        fifo = queue.Queue()
        fifo.put(self)
        while not fifo.empty():
            node = fifo.get()
            if type(node) is Dendrogram:
                for child in node:
                    fifo.put(child)
                    objects = node.descendants()
                    plt.scatter(pos[objects, 0], pos[objects, 1], s=4)
                fig.canvas.draw()
                ims.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
        ims[0].save(path, save_all=True, append_images=ims, duration=1000)
        plt.close()
