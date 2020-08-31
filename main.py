# This is a sample Python script.

import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer, KDFLC
from fuzzylogicpy.core.elements import StateNode, GeneratorNode, NodeType
from fuzzylogicpy.core.expression_parser import ExpressionParser
from fuzzylogicpy.core.impl.logics import GMBC
from fuzzylogicpy.core.impl.memberships import Sigmoid


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def test_evaluation():
    quality = StateNode('high quality', 'quality', Sigmoid(5.5, 4))
    alcohol = StateNode('high alcohol', 'alcohol', Sigmoid(11.65, 9))
    states = {quality.label: quality, alcohol.label: alcohol}
    parser = ExpressionParser('(IMP (NOT "high alcohol") "high quality")', states, dict())
    tree = parser.parser()

    evaluator = ExpressionEvaluation(data, GMBC(), tree)
    print(GMBC())
    print(evaluator.eval(), tree.fitness)
    print(tree.to_json())
    evaluator.export_data('results/evaluation.xlsx')


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
    mfo.optimizer(root)
    print(root, root.fitness)


if __name__ == '__main__':
    data = pd.read_csv('datasets/tinto.csv')
    states = {}
    for head in data.head():
        states[head] = StateNode(head, head)

    props = GeneratorNode(2, 'properties', [v for v in states.keys() if 'quality' != v],
                          [NodeType.AND, NodeType.OR, NodeType.IMP, NodeType.EQV, NodeType.NOT], 3)
    generators = {props.label: props}
    expression = '(IMP "{}" "quality")'.format(props.label)
    expression = '("properties")'
    parser = ExpressionParser(expression, states, generators)
    root = parser.parser()
    algorithm = KDFLC(data, root, states, GMBC(), 10, 10, 15, 0.8, 0.1)
    algorithm.discovery()
