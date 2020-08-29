import pandas as pd

from core.element import Operator, NodeType, Node
from core.logic import Logic


class ExpressionEvaluation:
    def __init__(self, data: dict, logic: Logic, tree: Operator):
        self.tree = tree
        self.logic = logic
        self.df = data
        self.data_fuzzy = pd.DataFrame({})

    def __fuzzy_data(self):
        header = self.df.head()
        for state in Operator.get_nodes_by_type(self.tree, NodeType.STATE):
            if state.cname in header:
                values = []
                for v in self.df[state.cname]:
                    values.append(state.membership.evaluate(v))
                self.data_fuzzy[state.label] = values

    def __fit_compute(self):
        values = []
        for row_idx in range(self.data_fuzzy.shape[0]):
            values.append(self.__fit_value(self.tree, row_idx))

        self.tree.fitness = self.logic.for_all(values)
        self.data_fuzzy['For ALL'] = self.tree.fitness
        self.data_fuzzy['Exist'] = self.logic.exist(values)
        self.data_fuzzy['Result'] = values

    def __fit_value(self, node: Node, index: int):
        values = []
        if node.type == NodeType.STATE:
            return self.data_fuzzy[node.label][index]
        elif node.type == NodeType.OR:
            for child in node.children:
                values.append(self.__fit_value(child, index))
            node.fitness = self.logic.or_(values)
            return node.fitness
        elif node.type == NodeType.AND:
            for child in node.children:
                values.append(self.__fit_value(child, index))
            node.fitness = self.logic.and_(values)
            return node.fitness
        elif node.type == NodeType.IMP:
            node.fitness = self.logic.imp_(self.__fit_value(node.children[0], index),
                                           self.__fit_value(node.children[1], index))
            return node.fitness
        elif node.type == NodeType.EQV:
            node.fitness = self.logic.eqv_(self.__fit_value(node.children[0], index),
                                           self.__fit_value(node.children[1], index))
            return node.fitness
        elif node.type == NodeType.NOT:
            node.fitness = self.logic.not_(self.__fit_value(node.childre[0], index))
            return node.fitness
        else:
            raise RuntimeError("Don't supported: " + node.type)

    def eval(self) -> Operator:
        self.__fuzzy_data()
        self.__fit_compute()
        return self.tree

    def export_data(self, output_path: str):
        if '.csv' in output_path:
            self.data_fuzzy.to_csv(output_path, index=False)
        elif '.xlsx' in output_path:
            self.data_fuzzy.to_excel(output_path, index=False)
        else:
            raise RuntimeError('Invalid output file format')
