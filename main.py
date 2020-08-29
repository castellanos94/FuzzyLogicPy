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
    parser = ExpressionParser('(AND "A1" (IMP "A3" "A4") "B4")')
    parser.parser()
    quality = StateNode('high quality', 'quality')
    alcohol = StateNode('high alcohol', 'alcohol')
    imp = Operator(NodeType.IMP)
    print(imp)
    imp.add_child(quality)
    imp.add_child(alcohol)
    print(imp)
