# This is a sample Python script.

import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from core.algorithms import ExpressionEvaluation
from core.element import StateNode, Operator, NodeType
from core.expression_parser import ExpressionParser
from core.logic import GMBC
from core.membership_function import Sigmoid


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    express = '(OR (NOT (NOT "alcohol")) (NOT (AND "volatile_acidity" "density" "fixed_acidity" "alcohol" "total_sulfur_dioxide" "citric_acid" "chlorides" "residual_sugar" "free_sulfur_dioxide" "quality")))'
    express = '(IMP (NOT "sepal.width") (OR "petal.length" "sepal.length" "petal.width"))'
    express = '(IMP (IMP (EQV "sepal.length" "sepal.width") (AND "sepal.length" "sepal.width" "petal.length" "petal.width")) "variety")'
    parser = ExpressionParser(express, dict(), dict())
    # node = parser.parser()
    # print(node)
    quality = StateNode('high quality', 'quality', Sigmoid(5.5, 4))
    alcohol = StateNode('high alcohol', 'alcohol', Sigmoid(11.65, 9))
    tree = Operator(NodeType.OR)
    tree.add_child(alcohol)
    tree.add_child(quality)
    data = pd.read_csv('datasets/tinto.csv')
    evaluator = ExpressionEvaluation(data, GMBC(), tree)
    print(GMBC())
    print(evaluator.eval(), tree.fitness)
    evaluator.export_data('results/evaluation.xlsx')
