import copy
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from fuzzylogicpy.core.elements import Operator, NodeType, Node, StateNode
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


def generate_membership_function(data: Dict, state: StateNode) -> Dict:
    min_, max_ = data[state.cname].min(), data[state.cname].max()
    b = random.uniform(min_, max_)
    g = random.uniform(b, max_)
    return {'F': FPG(b, g, random.random()), 'min': min_, 'max': max_}


def repair_membership_function(bundle: Dict):
    membership = bundle['F']
    if np.isnan(membership.beta):
        membership.beta = random.random(bundle['min'], bundle['max'])
    elif np.isnan(membership.gamma):
        membership.gamma = random.random(bundle['min'], bundle['max'])
    elif membership.beta > membership.gamma:
        membership.gamma += membership.beta
        membership.beta = membership.gamma - membership.beta
        membership.gamma -= membership.beta
    if membership.m > 1 or membership.m < 0 or np.isnan(membership.m):
        membership.m = random.random()


def crossover_membership_function(parent_a: FPG, parent_b: FPG) -> List[FPG]:
    a, b = copy.deepcopy(parent_a), copy.deepcopy(parent_b)

    return [a, b]


def mutation_membership_function(mutation_rate: float, eta: float, bundle: Dict) -> None:
    for key, item in bundle['F'].__dict__.items():
        if random.random() <= mutation_rate and key != 'type':
            x = bundle['F'].__dict__.get(key)
            xl = bundle['min']
            xu = bundle['max']
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - pow(val, mut_pow)

            x = x + delta_q * (xu - xl)
            bundle['F'].__dict__[key] = min(max(x, xl), xu)


class MembershipFunctionOptimizer:

    def __init__(self, data: Dict, logic, min_value: float = 0.5, population_size: int = 10, iteration: int = 2,
                 mutation_rate: float = 0.5,
                 operators=None):
        if operators is None:
            operators = {'repair': repair_membership_function, 'generate': generate_membership_function,
                         'crossover': crossover_membership_function, 'mutation': mutation_membership_function}
            self.data = data
            self.logic = logic
            self.min_value = min_value
            self.population_size = population_size
            self.iteration = iteration
            self.mutation_rate = mutation_rate
            self.current_iteration = 0
            self.states = None
            self.operators = operators
            while self.population_size % 2 != 0:
                self.population_size += 1
            if self.population_size == 0:
                self.population_size = 2

    def __show(self, functions: dict, fitness: dict):
        print('Showing...')
        for k in functions.keys():
            print(fitness[k], functions[k])

    def __evaluate(self, tree: Operator, functions: dict, fitness: dict):
        for idx in range(self.population_size):
            for state in self.states:
                state.membership = functions[id(state)][idx]['F']
            f = ExpressionEvaluation(self.data, self.logic, tree).eval().fitness
            for v in fitness.values():
                v[idx] = f

    def optimizer(self, tree: Operator) -> Operator:
        functions = {}
        fitness = {}
        self.states = Operator.get_nodes_by_type(tree, NodeType.STATE)
        for state in self.states:
            if state.membership is None:
                functions[id(state)] = [self.operators['generate'](self.data, state) for _ in
                                        range(self.population_size)]
                fitness[id(state)] = self.population_size * [0]
        self.current_iteration += 1
        if len(functions) > 0:
            self.__show(functions, fitness)
            self.__evaluate(tree, functions, fitness)

            while self.current_iteration < self.iteration and not any(
                    v >= self.min_value for v in fitness[id(self.states[0])]):
                self.current_iteration += 1
                print(self.current_iteration, "working...")
                for key, item in functions.items():
                    for v in item:
                        self.operators['mutation'](self.mutation_rate, 20, v)
                        self.operators['repair'](v)

                self.__evaluate(tree, functions, fitness)

            max_ = max(fitness[id(self.states[0])])
            idx = fitness[id(self.states[0])].index(max_)

            for state in self.states:
                state.membership = functions[id(state)][idx]['F']
        return ExpressionEvaluation(self.data, self.logic, tree).eval()
