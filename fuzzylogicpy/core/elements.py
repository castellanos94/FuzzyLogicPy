from __future__ import annotations

import enum
import json
import random
from abc import ABC
from typing import List, Dict

from fuzzylogicpy.core.membership_function import MembershipFunction


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
        return '"{}"'.format(self.label)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False)


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
    def get_father(root: Operator, node: Node):
        if root == node:
            return root
        for child in root.children:
            if child == node:
                return root
            elif isinstance(child, Operator):
                v = Operator.get_father(child, node)
                if v is not None:
                    return v
        return None

    @staticmethod
    def replace_node(root: Operator, old_value: Node, new_value: Node) -> bool:
        for idx, v in enumerate(root.children):
            if v == old_value:
                root.children.__setitem__(idx, new_value)
                return True
            elif isinstance(v, Operator):
                if Operator.replace_node(v, old_value, new_value):
                    return True
        return False

    @staticmethod
    def get_grade(root: Node) -> int:
        if root.type == NodeType.STATE:
            return 0
        if isinstance(root, Operator):
            child = []
            for c in root.children:
                child.append(Operator.get_grade(c) + 1)
            return max(child)

    @staticmethod
    def get_nodes_by_type(root: Operator, type_: NodeType) -> List[Node]:
        found = []
        if type_ == NodeType.GENERATOR and root.type == type_:
            return [root]
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

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness


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
    def __init__(self, depth: int, label: str, labels: List[str], operators: List[NodeType],
                 max_child_number: int = None):
        super(GeneratorNode, self).__init__(label, editable=False)
        self.type = NodeType.GENERATOR
        self.labels = labels
        self.operators = operators
        self.depth = depth
        if max_child_number is not None and max_child_number >= 2:
            self.max_child_number = max_child_number
        else:
            self.max_child_number = int((len(self.labels) + len(self.operators)) / 2)

    def __generate_child(self, root: Operator, states: Dict, current_depth: int, balanced: bool = False):
        if current_depth < self.depth:
            if random.random() < 0.85 or balanced:
                tree = Operator(random.choice(self.operators), editable=True)
                tree.owner_generator = self.label
                if tree.type == NodeType.AND or tree.type == NodeType.OR:
                    for _ in range(int(random.uniform(2, self.max_child_number))):
                        tree.add_child(self.__generate_child(tree, states, current_depth + 1, balanced))
                    return tree
                elif tree.type == NodeType.IMP or tree.type == NodeType.EQV:
                    for _ in range(2):
                        tree.add_child(self.__generate_child(tree, states, current_depth + 1, balanced))
                    return tree
                elif tree.type == NodeType.NOT:
                    tree.add_child(self.__generate_child(tree, states, current_depth + 1, balanced))
                    return tree
                else:
                    raise RuntimeError("Invalid type: " + str(tree.type))
            else:
                return self.__generate_state(root, states)
        else:
            return self.__generate_state(root, states)

    def __generate_state(self, root: Operator, states: Dict) -> StateNode:
        choice = states[random.choice(self.labels)]
        intents = 0
        if root is not None:
            while any([child.label == choice.label for child in root.children]) and intents < self.max_child_number:
                choice = states[random.choice(self.labels)]
                intents += 1
        state = StateNode(choice.label, choice.cname, choice.membership, editable=True)
        state.owner_generator = self.label

        return state

    def generate(self, states: Dict, balanced: bool) -> Node:
        return self.__generate_child(None, states, 0, balanced)
