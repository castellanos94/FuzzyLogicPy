from __future__ import annotations

import enum
import json
from abc import ABC
from typing import List

from core.membership_function import MembershipFunction


class NodeType(str, enum.Enum):
    NOT = 'NOT'
    AND = 'AND'
    OR = 'OR'
    IMP = 'IMP'
    EQV = 'EQV'
    STATE = 'STATE'
    GENERATOR = 'GENERATOR'

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
        elif self is NodeType.GENERATOR:
            return "GENERATOR"
        raise RuntimeError("Invalid")

    def __str__(self):
        return self.__repr__()


class Node(ABC):
    def __init__(self, label: str, editable: bool = False):
        self.label = label
        self.editable = editable
        self.type = None

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)


class Operator(Node):
    def __init__(self, type_: NodeType, editable: bool = False):
        super(Operator, self).__init__(str(type_), editable)
        self.children = []
        self.type = type_
        self.fitness = None

    @staticmethod
    def dfs(root: Operator, node: Node, pos: int = 0):
        if root == node:
            return pos
        for child in root.children:
            if child == node:
                return pos + 1
            elif isinstance(child, Operator):
                v = Operator.dfs(child, node, pos + 1)
                if v != -1:
                    return v
        return -1

    @staticmethod
    def replace_node(root: Operator, old_value: Node, new_value: Node) -> bool:
        for idx, v in enumerate(root.children):
            if v == old_value:
                root.children[idx] = new_value
                return True
        return False

    @staticmethod
    def get_nodes_by_type(root: Operator, type_: NodeType) -> List[Node]:
        found = []
        for node in root.children:
            if node.type == type_:
                found.append(node)
            if isinstance(node, Operator):
                found += Operator.get_nodes_by_type(node, type_)
        return found

    @staticmethod
    def get_editable_nodes(root: Operator) -> List[Node]:
        editable = []
        for node in root.children:
            if node.editable:
                editable.append(node)
            if isinstance(node, Operator):
                editable += Operator.get_editable_nodes(node)
        return editable

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
        else:
            raise RuntimeError("error: " + str(self.type))

    def __str__(self):
        return "({} {})".format(str(self.type), " ".join([str(v) for v in self.children]))


class StateNode(Node):
    def __init__(self, label: str, cname: str, membership: MembershipFunction = None, editable: bool = False):
        super().__init__(label, editable)
        self.type = NodeType.STATE
        self.cname = cname
        self.membership = membership

    def __str__(self):
        return '"{}"'.format(self.label)

    def __repr__(self):
        return self.__str__()


class GeneratorNode(Node):
    def __init__(self, depth: int, labels: List[str], operators=List[NodeType]):
        super(GeneratorNode, self).__init__(str(NodeType.GENERATOR), editable=True)
        self.labels = labels
        self.operators = operators
        self.depth = depth

    def add_state(self, state: StateNode):
        self.labels.append(state.label)

    def __str__(self):
        return '{} {} {} {}'.format(self.label, self.labels, self.operators, self.depth)
