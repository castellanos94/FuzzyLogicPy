from lark import Lark, Tree, Transformer

from core.element import NodeType, Operator

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
    def __init__(self, expression: str):
        self.expression = expression

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

    def parser(self):
        self.make_tree(self.__get_syntax_tree().children[0])

    def make_tree(self, tree: Tree):
        if tree.data == "and_expr":
            op = Operator(NodeType.AND)
            for children in tree.children:
                print(children)
