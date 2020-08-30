from typing import Dict

from lark import Lark, Tree, Transformer

from fuzzylogicpy.core.element import Node, NodeType, Operator, StateNode

base_parser = Lark("""
    expr: and_expr
        | or_expr
        | not_expr
        | imp_expr
        | eqv_expr
        
    not_expr: "(" not [token | expr] ")"
            
    imp_expr: "(" imp token token")"
            | "(" imp token expr")"
            | "(" imp expr token")"
            | "(" imp expr expr")"
            
    eqv_expr: "(" eqv token token")"
            | "(" eqv token expr")"
            | "(" eqv expr token")"
            | "(" eqv expr expr")"
    
    and_expr: "(" and [token | expr] [token | expr]+")"
            | "(" and expr expr")"
    or_expr:"(" or [token | expr] [token | expr]+")"
            | "(" or expr expr")"
    token: STRING
    and: "AND"
    or: "OR"
    not: "NOT"
    imp: "IMP"
    eqv: "EQV"
    %import common.ESCAPED_STRING   -> STRING
    %import common.WS
    %ignore WS
""", start="expr")


class ExpressionParser(Transformer):
    def __init__(self, expression: str, states: Dict, generators: Dict):
        self.expression = expression
        self.names = ["imp", "not", "eqv", "or", "and"]
        self.states = states
        self.generators = generators

    def expr(self, children):
        num_children = len(children)
        if num_children == 1:
            return children[0]
        else:
            raise RuntimeError()

    def and_expr(self, children):
        num_children = len(children)
        if num_children == 1:
            return children[0]
        else:
            return Tree(data="and_expr", children=children[1:])

    def imp_expr(self, children):
        num_children = len(children)
        if num_children == 1:
            return children[0]
        elif num_children == 3:
            first, middle, last = children
            return Tree(data="imp_expr", children=[middle, last])
        else:
            raise RuntimeError("Invalid number of children")

    def eqv_expr(self, children):
        num_children = len(children)
        if num_children == 1:
            return children[0]
        elif num_children == 3:
            first, middle, last = children
            return Tree(data="eqv_expr", children=[middle, last])
        else:
            raise RuntimeError("Invalid number of children")

    def not_expr(self, children):
        if len(children) > 2:
            raise RuntimeError("Invalid number of children")
        else:
            return Tree(data='not_expr', children=children[1])

    def or_expr(self, children):
        num_children = len(children)
        if num_children == 1:
            return children[0]
        else:
            return Tree(data="or_expr", children=children[1:])

    def __get_syntax_tree(self):
        return base_parser.parse(self.expression)

    def parser(self) -> Node:
        return self.__make_tree(self.__get_syntax_tree().children[0])

    def __make_tree(self, tree: Tree) -> Node:
        if tree.data == "expr":
            return self.__make_tree(tree.children[0])
        if tree.data == "and_expr":
            op = Operator(NodeType.AND)
            for child in tree.children:
                if len(child.children) > 0:
                    op.add_child(self.__make_tree(child))
            return op
        if tree.data == "or_expr":
            op = Operator(NodeType.OR)
            for child in tree.children:
                if len(child.children) > 0:
                    op.add_child(self.__make_tree(child))
            return op
        if tree.data == "imp_expr":
            op = Operator(NodeType.IMP)
            for child in tree.children:
                if child.data not in self.names:
                    op.add_child(self.__make_tree(child))
            return op
        if tree.data == "eqv_expr":
            op = Operator(NodeType.EQV)
            for child in tree.children:
                if len(child.children) > 0:
                    op.add_child(self.__make_tree(child))
            return op
        if tree.data == "not_expr":
            op = Operator(NodeType.NOT)
            for child in tree.children:
                if len(child.children) > 0:
                    op.add_child(self.__make_tree(child))
            return op

        if tree.data == "token":
            token = tree.children[0].replace('"', '')
            if token in self.states:
                return StateNode(self.states[token].label, self.states[token].cname, self.states[token].membership)
            if token in self.generators:
                return self.generators[token]
            raise RuntimeError("Label " + token + " not found")
