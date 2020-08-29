from abc import ABC

from core.membership_function import MembershipFunction


class Node(ABC):
    def __init__(self, label: str, editable: bool = False):
        self.label = label
        self.editable = editable

    def __str__(self):
        return self.label


class Operator(Node):
    def __init__(self, label: str, editable: bool = False):
        super(Operator, self).__init__(label, editable)
        self.children = []


class StateNode(Node):
    def __init__(self, label: str, cname: str, membership: MembershipFunction = None, editable: bool = False):
        super().__init__(label, editable)
        self.cname = cname
        self.membership = membership

    def __str__(self):
        return '["{}" "{}" {}]'.format(self.label, self.cname, "" if self.membership is None else self.membership)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    quality = StateNode('high quality', 'quality')
    print(quality)
