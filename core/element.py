import enum
from abc import ABC

from core.membership_function import MembershipFunction


class NodeType(enum.Enum):
    NOT = 0
    AND = 1
    OR = 2
    IMP = 3
    EQV = 4
    STATE = 5

    def __repr__(self):
        if self is NodeType.AND:
            return "AND"
        elif self is NodeType.NOT:
            return "NOT"
        elif self is NodeType.IMP:
            return "IMP"
        elif self is NodeType.EQV:
            return "EQV"
        elif self is NodeType.OR:
            return "OR"
        elif self is NodeType.STATE:
            return "STATE"
        raise Exception("Invalid")

    def __str__(self):
        return self.__repr__()


class Node(ABC):
    def __init__(self, label: str, editable: bool = False):
        self.label = label
        self.editable = editable
        self.type = None

    def __str__(self):
        return self.label


class Operator(Node):
    def __init__(self, type_: NodeType, editable: bool = False):
        super(Operator, self).__init__(str(type_), editable)
        self.children = []
        self.type = type

    def add_child(self, node: Node):
        if self.type is NodeType.AND:
            self.children.append(node)
        elif self.type is NodeType.NOT:
            if len(self.children) == 0:
                self.children.append(node)
            else:
                raise RuntimeError("arity must be one element.")
        elif self.type is NodeType.IMP:
            if len(self.children) < 2:
                self.children.append(node)
            else:
                raise RuntimeError("arity must be two element.")
        elif self.type is NodeType.EQV:
            if len(self.children) < 2:
                self.children.append(node)
            else:
                raise RuntimeError("arity must be two element.")
        elif self.type is NodeType.OR:
            self.children.append(node)


class StateNode(Node):
    def __init__(self, label: str, cname: str, membership: MembershipFunction = None, editable: bool = False):
        super().__init__(label, editable)
        self.type = NodeType.STATE
        self.cname = cname
        self.membership = membership

    def __str__(self):
        return '["{}" "{}" {}]'.format(self.label, self.cname, "" if self.membership is None else self.membership)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    quality = StateNode('high quality', 'quality')
    print(quality, quality.type)
