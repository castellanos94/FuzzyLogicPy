# This is a sample Python script.

import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer
from fuzzylogicpy.core.elements import StateNode
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
    quality = StateNode('quality', 'quality')
    alcohol = StateNode('alcohol', 'alcohol')
    ph = StateNode("pH", "pH")
    citric = StateNode("citric_acid", "citric_acid")
    states = {quality.label: quality, alcohol.label: alcohol, ph.label: ph, citric.label: citric}
    parser = ExpressionParser('(EQV (AND "alcohol" "pH" "citric_acid") "quality")', states, dict())
    root = parser.parser()
    mfo = MembershipFunctionOptimizer(data, GMBC(), min_value=1, iteration=10,population_size=5)
    print(root, root.fitness)
    mfo.optimizer(root)

    print(root.to_json())
    print(root, root.fitness)


if __name__ == '__main__':
    test_MembershipFunctionOptimizer()
