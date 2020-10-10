# -*- coding: utf-8 -*-
import time
from timeit import default_timer as timer

from fuzzylogicpy.algorithms.NeuralNetwork import NeuralNetwork as NN
from fuzzylogicpy.algorithms.NeuralNetwork import FuzzyRow as FR
from fuzzylogicpy.core.elements import StateNode, GeneratorNode, NodeType, Operator
from fuzzylogicpy.parser.expression_parser import ExpressionParser
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer, KDFLC
from fuzzylogicpy.core.impl.logics import GMBC

import random
import numpy as np
import pandas as pd

from fuzzylogicpy.core.impl.memberships import FPG

# for i in X[0]:
#    print(dictMF[0][0]['FPG'].evaluate(i))ANFANF

fileComplete = 'datasets/tinto.csv'
fileTrain = 'datasets/tinto.csv'
fileTest = 'datasets/tinto.csv'


class DataSet:

    def __init__(self, file):
        self.data_set = pd.read_csv(file)
        self.__arrayNumpy = self.data_set.to_numpy()

    def getTarget(self):
        return self.__arrayNumpy[:, -1]

    def getData(self):
        return self.__arrayNumpy[:, 0:-1]

    def getDataSet(self):
        return self.__arrayNumpy


def getParameters_GA(granulate=3):
    data = pd.read_csv(fileComplete)
    states = {}
    names = list(data.columns)
    for head in data.head():
        states[head] = StateNode(head, head)

    props = GeneratorNode(2, 'properties', [v for v in states.keys() if names[-1] != v],
                          [NodeType.EQV, NodeType.AND, NodeType.NOT], 3)
    category = GeneratorNode(1, 'category', [v for v in states.keys() if names[-1] == v],
                             [NodeType.NOT])
    generators = {props.label: props, category.label: category}

    a = '"' + '" "'.join([v for v in names[0:-1]]) + '"'
    expression = '(IMP (AND {}) "{}")'.format(a, names[-1])
    parser = ExpressionParser(expression, states, generators)
    root = parser.parser()
    algorithm = KDFLC(data, root, states, GMBC(), 50, 10, 30, 0.95, 0.1)
    algorithm.discovery()

    q = []

    a = list(range(len(algorithm.predicates)))

    for item in random.sample(a, granulate):
        d = Operator.get_nodes_by_type(algorithm.predicates[item], NodeType.STATE)
        q.append([i.membership for i in d])

    return q


start_time = time.time()

train = DataSet(fileTrain)
test = DataSet(fileTest)

granulate = 3
g = getParameters_GA(granulate)
print("--- %s seconds ---" % (time.time() - start_time))

X = train.getData()
X2 = test.getData()
Y = train.getTarget()
Y2 = test.getTarget()

# dictMF = [
# [FPG(2.147896350624007,3.5163722538044273, 0.1622362353227601),
# FPG(2.0003342426656783,4.0793004294307185,0.08176694727515899),
# FPG(2.124844064235014,3.223141968134696,0.6662550821458767)
# ],
# [FPG(4.361466988454315,6.761485878098842 ,0.22559186545856036),
# FPG(4.370821508416422,6.568426624826564 , 0.42907861013094606),
# FPG(4.637668382669777,6.265023278955064 ,0.5289049691800887)
# ],
# [FPG(1.4270656246097206,4.1086510499769,0.026421023757245865),
# FPG(1.0074706437695773,3.6296212308709777, 0.03956588467924682 ),
# FPG(1.3475025402809757,5.787178967737299,0.007490905551822413)
# ],
# [FPG(0.3241766868760386,1.422358561433751, 0.7576981912570984),
# FPG(0.23587772991825504,1.8716782266481184, 0.057583009540230545),
# FPG(0.15909362303435012,2.388266174289166, 0.254507451854)
# ]
# ]


asd = [[g[i][j] for i in range(granulate)] for j in range(len(g[0]) - 1)]

mf = FR(asd)
# m = mf.evaluateMF(X[0])
anf = NN(X, X2, Y, Y2, mf)

t1_start = timer()
anf.trainHybridJangOffLine(epochs=25)
results = anf.predict(anf.X, anf)
t1_stop = timer()

print("Salidas: \n")
for i, j in zip(results, Y2):
    print(i, j, "\n")

anf.plotErrors()
anf.plotResults()
print(anf.memFuncs)
print(anf.memFuncsByVariable)
pd.DataFrame(anf.layer4).to_csv("layer4.csv")
print("Elapsed time during the whole program in seconds:", t1_stop - t1_start)

# while True:
#    data = input("Entrada: \n").split()
#    if any(x == "exit" for x in data):
#        break
#    elif any(x == "test" for x in data):
results = anf.predict(X, anf)
results = np.append(results, anf.predict(X2, anf))
#
#    else:
#        data = [eval(x) for x in data]
#        results = anf.predict([data], anf)
#
#    print(results)

print("Salidas: \n")
for i in results:
    print(i, "\n")
# print(round(anf.consequents[-1][0],6))
# print(round(anf.consequents[-2][0],6))
# print(round(anf.consequents[9][0],6))