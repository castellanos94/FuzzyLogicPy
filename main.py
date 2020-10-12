import random
import time

import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer, KDFLC
from fuzzylogicpy.core.elements import StateNode, GeneratorNode, NodeType, Operator
from fuzzylogicpy.core.impl.logics import GMBC
from fuzzylogicpy.core.impl.memberships import Sigmoid, NSigmoid, Nominal
from fuzzylogicpy.parser.expression_parser import ExpressionParser
from fuzzylogicpy.parser.query import EvaluationQuery, query_to_json, LogicType, query_from_json, QueryExecutor


# Press the green button in the gutter to run the script.
def test_evaluation():
    data = pd.read_csv('datasets/tinto.csv')
    quality = StateNode('high quality', 'quality', Sigmoid(5.5, 4))
    alcohol = StateNode('high alcohol', 'alcohol', Sigmoid(11.65, 9))
    ph = StateNode('low pH', 'pH', NSigmoid(3.375, 2.93))
    states = {quality.label: quality, alcohol.label: alcohol, ph.label: ph}
    parser = ExpressionParser('(IMP (NOT (AND "high alcohol" "low pH")) "high quality")', states, dict())
    tree = parser.parser()

    evaluator = ExpressionEvaluation(data, GMBC(natural_imp=True), tree)
    print(GMBC())
    print(evaluator.eval(), tree.fitness)
    print(tree.to_json())
    evaluator.export_data('results/evaluation-natural-imp.xlsx')


def test_MembershipFunctionOptimizer():
    data = pd.read_csv('datasets/tinto.csv')
    states = {}
    for head in data.head():
        states[head] = StateNode(head, head)
    expression = '(IMP (AND {}) "quality")'.format(
        str([str(v) for v in states.keys() if 'quality' != v]).replace('\'', '"').replace(',', '').replace('[',
                                                                                                           '').replace(
            ']', ''))
    parser = ExpressionParser(expression, states, dict())
    root = parser.parser()
    mfo = MembershipFunctionOptimizer(data, GMBC())
    print(root, root.fitness)
    mfo.optimize(root)
    print(root, root.fitness)


def test_kdflc():
    data = pd.read_csv('datasets/tinto.csv')
    states = {}
    for head in data.head():
        states[head] = StateNode(head, head)

    props = GeneratorNode(2, 'properties', [v for v in states.keys() if 'quality' != v and 'alcohol' != v],
                          [NodeType.EQV, NodeType.AND, NodeType.NOT], 3)

    _all = GeneratorNode(2, 'columns', [v for v in states.keys()],
                         [NodeType.EQV, NodeType.AND, NodeType.NOT, NodeType.OR,NodeType.IMP])
    category = GeneratorNode(1, 'category', [v for v in states.keys() if 'quality' == v or 'alcohol' == v],
                             [NodeType.NOT])
    generators = {props.label: props, category.label: category, _all.label: _all}
    expression = '(IMP "{}" "{}")'.format(props.label, category.label)
    expression = '(IMP (AND {}) "quality")'.format(
        str([str(v) for v in states.keys() if 'quality' != v]).replace('\'', '"').replace(',', '').replace('[',
                                                                                                           '').replace(
            ']', ''))
    # expression = '(IMP "{}" "quality")'.format(props.label)
    # expression = '(IMP "alcohol" "quality")'
    # expression = '("properties")'
    expression = '("columns")'
    parser = ExpressionParser(expression, states, generators)
    root = parser.parser()
    algorithm = KDFLC(data, root, states, GMBC(), 100, 50, 15, 0.95, 0.1)
    algorithm.discovery()
    # for item in algorithm.predicates:
    #    print(item.fitness, item, 'Grade: ', Operator.get_grade(item))
    # Re evaluating
    # for item in algorithm.predicates:
    #    item = ExpressionEvaluation(data, GMBC(), item).eval()
    #    print(item.fitness, item, 'Grade: ', Operator.get_grade(item))
    algorithm.export_data('results/discovery.xlsx')


def test_classification():
    data = pd.read_csv('datasets/iris.csv')
    states = {}
    for head in data.head():
        states[head] = StateNode(head, head)

    props = GeneratorNode(1, 'properties', [v for v in states.keys() if 'variety' != v], [NodeType.OR, NodeType.AND])

    setosa = StateNode('Setosa', 'variety', Nominal(('Setosa', 1.0)))
    versicolor = StateNode('Versicolor', 'variety', Nominal(('Versicolor', 1.0)))
    virginica = StateNode('Virginica', 'variety', Nominal(('Virginica', 1.0)))
    states[setosa.label] = setosa
    states[versicolor.label] = versicolor
    states[virginica.label] = virginica
    category = GeneratorNode(0, 'category', [setosa.label, versicolor.label, virginica.label],
                             [NodeType.NOT, NodeType.AND])
    generators = {props.label: props, category.label: category}
    expression = '(IMP "{}" "{}")'.format(props.label, category.label)
    parser = ExpressionParser(expression, states, generators)
    root = parser.parser()
    algorithm = KDFLC(data, root, states, GMBC(), 150, 50, 30, 0.95, 0.1)
    algorithm.discovery()
    for item in algorithm.predicates:
        print(item.fitness, item, 'Grade: ', Operator.get_grade(item))
    algorithm.export_data('results/classification_iris.xlsx')


def test_parser():
    data = pd.read_csv('datasets/tinto.csv')
    quality = StateNode('high quality', 'quality', Sigmoid(5.5, 4))
    alcohol = StateNode('high alcohol', 'alcohol', Sigmoid(11.65, 9))
    states = {quality.label: quality, alcohol.label: alcohol}
    expression = '(IMP (NOT "high alcohol") "high quality")'
    query = EvaluationQuery('datasets/tinto.csv', 'results/evaluation.xlsx', states, LogicType.GMBC, expression)
    query_str = query_to_json(query)
    print(query.states['high alcohol'])
    evaluation = query_from_json(query_str)
    print(evaluation.name())
    print(evaluation.states['high alcohol'])
    executor = QueryExecutor(evaluation)
    executor.execute()


if __name__ == '__main__':
    start_time = time.time()
    test_evaluation()
    random.seed(1)
    # test_kdflc()
    # test_classification()

    print("--- %s seconds ---" % (time.time() - start_time))
