
class Node:
    def __init__(self):
        self.children = []
        self.depth = 0
        self.value = 0


class ORNode(Node):
    def __init__(self, s=None):
        Node.__init__(self)
        self.s = s

    def __str__(self):
        descendant = ""
        for c in self.children:
            c.depth += self.depth + 1
            descendant += str(c)

        return self.depth * "---" + "OR(" + str(self.s) + ")\n" + descendant


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
