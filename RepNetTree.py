
import numpy as np


class Node:
    def __init__(self):
        self.children = []
        self.depth = 0
        self.value = 0


class ORNode(Node):
    def __init__(self, s=None, AD=None, Img=None):
        Node.__init__(self)
        self.s = s
        self.AD = AD
        self.Img = Img

    def __str__(self):
        descendant = ""
        for c in self.children:
            c.depth += self.depth + 1
            descendant += str(c)

        ad = np.array2string(np.array(self.AD), precision=2).replace('\n', '')
        img = np.array2string(np.array(self.Img), precision=2).replace('\n', '')
        return self.depth * "---" + "OR(" + str(self.s) + ", " + ad + ", " + img + ")\n" + descendant


class ANDNode(Node):
    def __init__(self, a=None):
        Node.__init__(self)
        self.a = a

    def __str__(self):
        descendant = ""
        for c in self.children:
            c.depth += self.depth + 1
            descendant += str(c)

        return self.depth * "---" + "AND(" + str(self.a) + ")\n" + descendant
