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
__TOLERANCE = 0.01


class ExpressionEvaluation:
    def __init__(self, data: dict, logic: Logic, tree: Operator):
        self.tree = tree
        self.logic = logic
        self.df = data
        self.data_fuzzy = pd.DataFrame({})

    def __fuzzy_data(self):
        header = None
        try:
            header = self.df.head()
        except Exception as e:
            pass

        for state in Operator.get_nodes_by_type(self.tree, NodeType.STATE):
            if state.cname in header:
                values = []
                if isinstance(self.df, list):
                    for r in self.df:
                        values.append(state.membership.evaluate(r[state.cname]))
                else:
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
    min_, max_ = min(data[state.cname]), max(data[state.cname])
    g = random.uniform(min_, max_)
    b = random.uniform(min_, g)
    t = min(__TOLERANCE, abs(min_ - max_))
    while abs(b - g) <= t:
        b = random.uniform(min_, max_)
        g = random.uniform(b, max_)
    return {'F': FPG(b, g, random.random()), 'min': min_, 'max': max_, 'tolerance': t}


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
    if abs(membership.beta - membership.gamma) <= bundle['tolerance']:
        membership.gamma = max(membership.beta, membership.gamma)
        if abs(membership.gamma - bundle[min]) <= bundle['tolerance'] or membership.gamma == bundle['min']:
            membership.gamma = random.random([bundle['min'], bundle['max']])
            if membership.gamma < membership.beta:
                membership.gamma += membership.beta
                membership.beta = membership.gamma - membership.beta
                membership.gamma -= membership.beta
        while abs(membership.beta - membership.gamma) <= bundle['tolerance']:
            membership.beta = random.random(bundle['min'], membership.gamma)
    if membership.m > 1 or membership.m < 0 or np.isnan(membership.m) or membership.m < __EPS:
        membership.m = random.random()


def uniform_crossover_membership(parent_a: Dict, parent_b: Dict, probability: float = 1) -> List[Dict]:
    if random.random() <= probability:
        a, b = copy.deepcopy(parent_a), copy.deepcopy(parent_b)
        _a_values, _b_values = a['F'].get_values(), b['F'].get_values()
        for idx in range(len(_a_values)):
            value_x1, value_x2 = _a_values[idx], _b_values[idx]
            if random.random() <= 0.5:
                _a_values[idx] = value_x2
                _b_values[idx] = value_x1
        a['F'].set_values(_a_values)
        b['F'].set_values(_b_values)
        return [a, b]
    return [parent_a, parent_b]


def simple_mutation(bundle: Dict, mutation_rate: float) -> None:
    __values = bundle['F'].get_values()
    for idx, x in enumerate(__values):
        if random.random() <= mutation_rate:
            __values[idx] = random.uniform(bundle['min'], bundle['max'])
    bundle['F'].set_values(__values)


def crossover_membership_function(parent_a: Dict, parent_b: Dict, probability: float = 1,
                                  distribution_index: float = 20) -> List[Dict]:
    rand = random.random()
    if rand <= probability:
        a, b = copy.deepcopy(parent_a), copy.deepcopy(parent_b)
        __a_values = a['F'].get_values()
        __b_values = b['F'].get_values()
        for idx in range(len(__a_values)):
            value_x1, value_x2 = __a_values[idx], __b_values[idx]
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
                        __a_values[idx] = c2
                        __b_values[idx] = c1
                    else:
                        __a_values[idx] = c1
                        __b_values[idx] = c2
        a['F'].set_values(__a_values)
        b['F'].set_values(__b_values)
        return [a, b]
    return [parent_a, parent_b]


def mutation_membership_function(bundle: Dict, mutation_rate: float, eta: float = 20) -> None:
    __values = bundle['F'].get_values()
    for idx, x in enumerate(__values):
        if random.random() <= mutation_rate:
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
            __values[idx] = min(max(x, xl), xu)
    bundle['F'].set_values(__values)


class MembershipFunctionOptimizer:

    def __init__(self, data: Dict, logic, min_value: float = 0.5, population_size: int = 3, iteration: int = 2,
                 mutation_rate: float = 0.1,
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

    def __evaluate(self, tree: Operator, functions: dict, fitness: List[float]):
        for idx in range(len(fitness)):
            for state in self.states:
                if state.membership is None:
                    state.membership = functions[id(state)][idx]['F']
            fitness[idx] = ExpressionEvaluation(self.data, self.logic, tree).eval().fitness

    def optimize(self, tree: Operator) -> Operator:
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
                        self.operators['mutation'](v, self.mutation_rate)
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
                if state.membership is None:
                    state.membership = functions[id(state)][idx]['F']
        return ExpressionEvaluation(self.data, self.logic, tree).eval()


def is_valid(individual):
    if isinstance(individual, Operator):
        states_labels = [item.label for item in individual.children if item.type == NodeType.STATE]
        return len(states_labels) <= len(set(states_labels))
    else:
        return True


class KDFLC:
    def __init__(self, data: Dict, tree: Operator, states: Dict, logic: Logic, num_pop: int, num_iter: int,
                 num_result: int,
                 min_truth_value: float,
                 mut_percentage: float, **kwargs):
        self.data = data
        self.predicate = tree
        self.states = states
        self.logic = logic
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.num_result = num_result
        self.min_truth_value = min_truth_value
        self.mut_percentage = mut_percentage
        self.optimizer = MembershipFunctionOptimizer(data, logic, **{k: v for k, v in kwargs.items() if v is not None})
        self.predicates = []
        self.generators = Operator.get_nodes_by_type(tree,
                                                     NodeType.GENERATOR) if tree.type != NodeType.GENERATOR else [tree]
        self.current_iteration = 0

    def __generate(self) -> Operator:
        predicate = copy.deepcopy(self.predicate)
        genes = Operator.get_nodes_by_type(predicate, NodeType.GENERATOR)
        for idx, gen in enumerate(genes):
            new_value = gen.generate(self.states, True if idx < int(len(genes) / 2) else False)
            if gen != predicate:
                Operator.replace_node(predicate, gen, new_value)
            else:
                if new_value.type == NodeType.STATE:
                    root = Operator(NodeType.NOT)
                    root.add_child(new_value)
                    return root
                return new_value
        return predicate

    def mutation_predicate(self, predicate: Operator) -> Operator:
        if random.random() <= self.mut_percentage:
            editable = Operator.get_editable_nodes(predicate)
            if len(editable) > 0:
                candidate = random.choice(editable)
                owner = [gen for gen in self.generators if gen.label == candidate.owner_generator][0]
                if candidate.type == NodeType.AND:
                    candidate.type = NodeType.OR
                elif candidate.type == NodeType.OR:
                    candidate.type = NodeType.AND
                elif candidate.type == NodeType.IMP:
                    candidate.type = NodeType.EQV
                elif candidate.type == NodeType.EQV:
                    candidate.type = NodeType.IMP
                elif candidate.type == NodeType.STATE:
                    st = self.states[random.choice(owner.labels)]
                    if not any([st.label == item.label for item in Operator.get_father(predicate, candidate).children]):
                        st = copy.deepcopy(st)
                        st.editable = True
                        st.owner_generator = owner.label
                        Operator.replace_node(predicate, candidate, st)
                elif candidate.type == NodeType.NOT:
                    pass
                else:
                    raise RuntimeError('Unknown type: ' + str(candidate.type))
                self.optimizer.optimize(predicate)
        return predicate

    def crossover(self, a: Operator, b: Operator) -> List[Operator]:
        a_edit = Operator.get_editable_nodes(a)
        b_edit = Operator.get_editable_nodes(b)
        if len(a_edit) > 0 and len(b_edit) > 0:
            a_choice = random.choice(a_edit)

            b_choice = random.choice(b_edit)
            b_max_depth = [gen for gen in self.generators if gen.label == b_choice.owner_generator][0].depth

            b_grade = Operator.get_grade(
                b_choice) if b_choice.owner_generator == a_choice.owner_generator else b_max_depth
            father = Operator.get_father(a, a_choice)
            father_depth = Operator.dfs(a, father)
            intents = 1
            while father_depth + b_grade > b_max_depth and intents < len(b_edit):
                b_choice = random.choice(b_edit)
                b_grade = Operator.get_grade(
                    b_choice) if b_choice.owner_generator == a_choice.owner_generator else b_max_depth
                intents += 1
            Operator.replace_node(father, a_choice, b_choice)

            a_max_depth = [gen for gen in self.generators if gen.label == a_choice.owner_generator][0].depth
            grade = Operator.get_grade(
                a_choice) if a_choice.owner_generator == b_choice.owner_generator else a_max_depth
            father = Operator.get_father(b, b_choice)
            father_depth = Operator.dfs(b, father)
            intents = 1

            while father_depth + grade > a_max_depth and intents < len(a_edit):
                a_choice = random.choice(a_edit)
                Operator.get_grade(a_choice) if a_choice.owner_generator == b_choice.owner_generator else a_max_depth
                intents += 1
        return [self.optimizer.optimize(a), self.optimizer.optimize(b)]

    def discovery(self) -> None:
        # Generate de population
        population = [self.__generate() for _ in range(self.num_pop)]
        # Evaluating population
        population = [self.optimizer.optimize(individual) for individual in population]
        self.current_iteration = 1
        # Copying elements to result lists
        self.predicates = [individual for individual in population if individual.fitness >= self.min_truth_value]
        # removing elements from list
        for individual in self.predicates:
            population.remove(individual)
        # Incorporating new predicates
        for _ in range(int(self.num_pop - len(population))):
            population.append(self.optimizer.optimize(self.__generate()))
        # Checking diversity results
        self.ensure_diversity()
        # Generational For
        while self.current_iteration < self.num_iter and len(self.predicates) < self.num_result:
            print('Iteration: ', self.current_iteration, ', Results: ', len(self.predicates))
            self.current_iteration += 1
            # population = [self.optimizer.optimize(individual) for individual in population]
            # Random Selection and copy parents to modification
            # pt_ = [copy.deepcopy(random.choice(population)) for _ in range(int(len(population) / 2))]
            __pt = []
            n_ = int(len(population) / 2)
            for idx in range(n_):
                v = random.choice(population)
                __pt.append(v)
                population.remove(v)

            __qt = []
            idx = 0
            while idx < len(__pt):
                __qt += self.crossover(__pt[idx], __pt[idx + 1 if idx + 1 < len(__pt) else 0])
                idx += 2
            # Mutation  and Evaluation predicate
            __qt = [self.mutation_predicate(item) for item in __qt]
            population += [item for item in __qt if is_valid(item) and item not in self.predicates]

            self.predicates += [individual for individual in population if
                                individual.fitness >= self.min_truth_value and individual not in self.predicates
                                and is_valid(individual)]
            for individual in self.predicates:
                if individual in population:
                    population.remove(individual)
            self.ensure_diversity()
            if len(population) > self.num_pop:
                population = population[:self.num_pop]

            for idx in range(len(population)):
                if random.random() <= self.mut_percentage:
                    _child = self.__generate()
                    self.optimizer.optimize(_child)
                    population[idx] = _child
            # print('diversity: ', was_replaced)

        self.predicates.sort(reverse=True)
        if len(self.predicates) == 0:
            self.predicates.append(max(population))
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

    def ensure_diversity(self):
        # Looking for same elements for ensure not proximity between fitness
        _expressions = {}
        for item in self.predicates:
            if str(item) in _expressions.keys():
                _expressions[str(item)].append(item)
            else:
                _expressions[str(item)] = [item]
        self.predicates = []
        for k, v in _expressions.items():
            best_ = max(v)
            self.predicates += [_v for _v in v if best_ == _v or _v.fitness <= best_.fitness * 0.98]
