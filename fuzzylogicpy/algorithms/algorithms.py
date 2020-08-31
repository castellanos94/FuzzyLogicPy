import copy
import pathlib
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from fuzzylogicpy.core.elements import Operator, NodeType, Node, StateNode
from fuzzylogicpy.core.impl.memberships import FPG
from fuzzylogicpy.core.logic import Logic

__EPS = 1.0e-14


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
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
    if np.isnan(membership.beta) or membership.beta < __EPS:
        membership.beta = random.random(bundle['min'], bundle['max'])
    elif np.isnan(membership.gamma) or membership.gamma < __EPS:
        membership.gamma = random.random(bundle['min'], bundle['max'])
    elif membership.beta > membership.gamma:
        membership.gamma += membership.beta
        membership.beta = membership.gamma - membership.beta
        membership.gamma -= membership.beta
    if membership.m > 1 or membership.m < 0 or np.isnan(membership.m) or membership.m < __EPS:
        membership.m = random.random()


def crossover_membership_function(parent_a: Dict, parent_b: Dict, probability: float = 1, ignore_key: List[str] = None,
                                  distribution_index: float = 20) -> List[Dict]:
    if ignore_key is None:
        ignore_key = ['type']
    elif 'type' not in ignore_key:
        ignore_key += 'type'
    a, b = copy.deepcopy(parent_a), copy.deepcopy(parent_b)
    rand = random.random()
    if rand <= probability:
        for key, item in a['F'].__dict__.items():
            if key not in ignore_key:
                value_x1, value_x2 = a['F'].__dict__.get(key), b['F'].__dict__.get(key)
                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > __EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1
                        lower_bound, upper_bound = a['min'], a['max']

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            a['F'].__dict__[key] = c2
                            b['F'].__dict__[key] = c1
                        else:
                            a['F'].__dict__[key] = c1
                            b['F'].__dict__[key] = c2

    return [a, b]


def mutation_membership_function(mutation_rate: float, eta: float, bundle: Dict,
                                 ignore_key=None) -> None:
    if ignore_key is None:
        ignore_key = ['type']
    elif 'type' not in ignore_key:
        ignore_key += 'type'
    for key, item in bundle['F'].__dict__.items():
        if random.random() <= mutation_rate and key not in ignore_key:
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
                val = val if val > __EPS else 0
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            bundle['F'].__dict__[key] = min(max(x, xl), xu)


class MembershipFunctionOptimizer:

    def __init__(self, data: Dict, logic, min_value: float = 0.5, population_size: int = 3, iteration: int = 2,
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
            if self.population_size < 4:
                self.population_size = 4

    def __show(self, functions: dict, fitness: List[float]):
        print('Showing...')
        for k in functions.keys():
            print(fitness, functions[k])

    def __evaluate(self, tree: Operator, functions: dict, fitness: List[float]):
        for idx in range(len(fitness)):
            for state in self.states:
                state.membership = functions[id(state)][idx]['F']
            fitness[idx] = ExpressionEvaluation(self.data, self.logic, tree).eval().fitness

    def optimizer(self, tree: Operator) -> Operator:
        functions = {}
        fitness = []
        self.states = Operator.get_nodes_by_type(tree, NodeType.STATE)
        for state in self.states:
            if state.membership is None:
                functions[id(state)] = [self.operators['generate'](self.data, state) for _ in
                                        range(self.population_size)]
                fitness = self.population_size * [0]
        self.current_iteration += 1
        if len(functions) > 0:
            self.__evaluate(tree, functions, fitness)
            while self.current_iteration < self.iteration and not any(v >= self.min_value for v in fitness):
                self.current_iteration += 1
                functions_, fitness_ = {}, None
                for key, item in functions.items():
                    parents = [random.choice(item) for _ in range(int(self.population_size / 2))]
                    idx = 0
                    children = []
                    while idx < len(parents):
                        a, b = parents[idx], parents[idx + 1 if idx + 1 < len(parents) else 0]
                        children += self.operators['crossover'](a, b)
                        idx += 2
                    for v in children:
                        self.operators['mutation'](self.mutation_rate, 20, v)
                        self.operators['repair'](v)
                    functions_[key] = children
                    if fitness_ is None:
                        fitness_ = len(children) * [0]
                self.__evaluate(tree, functions_, fitness_)
                for idx in range(len(fitness_)):
                    for jdx in range(len(fitness)):
                        if fitness_[idx] > fitness[jdx]:
                            for key in functions_.keys():
                                functions[key][jdx] = functions_[key][idx]
                                fitness[jdx] = fitness_[idx]
                            break
            max_ = max(fitness)
            idx = fitness.index(max_)

            for state in self.states:
                state.membership = functions[id(state)][idx]['F']
        return ExpressionEvaluation(self.data, self.logic, tree).eval()


class KDFLC:
    def __init__(self, data: Dict, tree: Operator, states: List[StateNode], logic: Logic, num_pop: int, num_iter: int,
                 num_result: int,
                 min_truth_value: float,
                 mut_percentage: float, adj_min_value: float = 0.0, adj_population_size: int = 3,
                 adj_iteration: int = 2, adj_mutation_rate: float = 0.1, adj_operators: Dict = None):
        self.data = data
        self.predicate = tree
        self.states = states
        self.logic = logic
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.num_result = num_result
        self.min_truth_value = min_truth_value
        self.mut_percentage = mut_percentage
        self.optimizer = MembershipFunctionOptimizer(data, logic, adj_min_value, adj_population_size, adj_iteration,
                                                     adj_mutation_rate, adj_operators)
        self.predicates = []
        self.generators = Operator.get_nodes_by_type(tree,
                                                     NodeType.GENERATOR) if tree.type != NodeType.GENERATOR else tree
        self.current_iteration = 0

    def __generate(self) -> Operator:
        predicate = copy.deepcopy(self.predicate)
        for gen in Operator.get_nodes_by_type(predicate, NodeType.GENERATOR):
            new_value = gen.generate(self.states)
            if gen != predicate:
                Operator.replace_node(predicate, gen, new_value)
            else:
                if new_value.type == NodeType.STATE:
                    root = Operator(NodeType.NOT)
                    root.add_child(new_value)
                    return root
                return new_value
        return predicate

    def mutation_predicate(self, predicate: Operator) -> None:
        pass

    def discovery(self) -> None:
        # Generate de population
        population = [self.__generate() for _ in range(self.num_pop)]
        # Evaluating population
        population = [self.optimizer.optimizer(individual) for individual in population]
        self.current_iteration = 1
        # Copying elements to result lists
        self.predicates = [individual for individual in population if individual.fitness >= self.min_truth_value]
        # Generational For
        while self.current_iteration < self.num_iter and len(self.predicates) < self.num_result:
            print('Iteration: ', self.current_iteration)
            self.current_iteration += 1
            population = [self.optimizer.optimizer(individual) for individual in population]
            self.predicates += [individual for individual in population if
                                individual.fitness >= self.min_truth_value and individual not in self.predicates]
            # TODO: For mutation operator is required generator id's

        self.predicates.sort(reverse=True)
        print('Num results: ', len(self.predicates), ',Max Value: ', self.predicates[0].fitness)

    def export_data(self, output_path: str) -> None:
        data = []
        for individual in self.predicates:
            data.append(
                {'truth_value': individual.fitness, 'predicate': str(individual),
                 'data': str(individual.to_json())})
        data_out = pd.DataFrame(data)
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if '.csv' in output_path:
            data_out.to_csv(output_path, index=False)
        elif '.xlsx' in output_path:
            data_out.to_excel(output_path, index=False)
        else:
            raise RuntimeError('Invalid output file format')
