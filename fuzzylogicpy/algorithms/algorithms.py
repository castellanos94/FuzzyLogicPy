import random

import pandas as pd

from fuzzylogicpy.core.elements import Operator, NodeType, Node
from fuzzylogicpy.core.impl.memberships import FPG
from fuzzylogicpy.core.logic import Logic


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
            node.fitness = self.logic.not_(self.__fit_value(node.children[0], index))
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


class MembershipFunctionOptimizer:

    def __init__(self, data: dict, logic, min_value: float = 0.5, population_size: int = 10, iteration: int = 2,
                 mutation_rate: float = 0.01):
        self.data = data
        self.logic = logic
        self.min_value = min_value
        self.population_size = population_size
        self.iteration = iteration
        self.mutation_rate = mutation_rate
        self.current_iteration = 0
        self.states = None

    def __random_function(self, cname: str) -> FPG:
        min_, max_ = self.data[cname].min(), self.data[cname].max()
        b = random.uniform(min_, max_)
        g = random.uniform(b, max_)
        return FPG(b, g, random.random())

    def __show(self, functions: dict, fitness: dict):
        print('Showing...')
        for k in functions.keys():
            print(fitness[k], functions[k])

    def __evaluate(self, tree: Operator, functions: dict, fitness: dict):
        for idx in range(3):
            for state in self.states:
                state.membership = functions[id(state)][idx]
                print(state.membership, state.membership.is_valid())
            f = ExpressionEvaluation(self.data, self.logic, tree).eval().fitness
            for v in fitness.values():
                v[idx] = f
        self.__show(functions, fitness)

    def optimizer(self, tree: Operator) -> Operator:
        functions = {}
        fitness = {}
        self.states = Operator.get_nodes_by_type(tree, NodeType.STATE)
        for state in self.states:
            if state.membership is None:
                functions[id(state)] = [self.__random_function(state.cname) for _ in range(3)]
                fitness[id(state)] = 3 * [0]
        self.current_iteration += 1
        if len(functions) > 0:
            self.__show(functions, fitness)
            self.__evaluate(tree, functions, fitness)
            while self.current_iteration < self.iteration and not any(self.min_value >= v for v in fitness):
                self.current_iteration += 1
                print("do something...")
            max_ = max(fitness[id(self.states[0])])
            idx = fitness[id(self.states[0])].index(max_)

            for state in self.states:
                state.membership = functions[id(state)][idx]
        f = ExpressionEvaluation(self.data, self.logic, tree).eval().fitness
        print(f)
        return tree
