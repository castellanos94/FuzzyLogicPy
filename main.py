# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from core.element import StateNode, Operator, NodeType
from core.expression_parser import ExpressionParser


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
    quality = StateNode('high quality', 'quality')
    alcohol = StateNode('high alcohol', 'alcohol')
    imp = Operator(NodeType.IMP)
    imp.add_child(quality)
    imp.add_child(alcohol)
    print(imp, id(imp))
    print(id(quality), quality.parent_id)
    print(id(alcohol), alcohol.parent_id)
