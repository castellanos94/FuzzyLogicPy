import random
import time
import numpy as np
import pandas as pd
from numba import jit, cuda

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer, KDFLC
from fuzzylogicpy.core.elements import StateNode, GeneratorNode, NodeType, Operator
from fuzzylogicpy.core.impl.logics import GMBC
from fuzzylogicpy.core.impl.memberships import Sigmoid, NSigmoid, Nominal
from fuzzylogicpy.parser.expression_parser import ExpressionParser
from fuzzylogicpy.parser.query import EvaluationQuery, query_to_json, LogicType, query_from_json, QueryExecutor



def func(atributos, granulacion):                                 
    predicates = [[x for x in range(granulacion)] for z in range(atributos)] 
    for a in range(atributos):
        for g in range(granulacion):
            if predicates[a][g] == 0:
                predicates[a][g] = np.inf
    return predicates

  
# function optimized to run on gpu  
@jit()                          
def func2(atributos, granulacion): 
    predicates = [[x for x in range(granulacion)] for z in range(atributos)] 
    for a in range(atributos):
        for g in range(granulacion):
            if predicates[a][g] == 0:
                predicates[a][g] = np.inf
    return predicates
    
if __name__ == '__main__':
    start_time = time.time()
    # test_evaluation()
    #random.seed(1)
    # test_kdflc()
    #data = pd.read_csv('datasets/tinto.csv')

    atributos = 4
    granulacion = 3
    predicates = [[x for x in range(granulacion)] for z in range(atributos)] 
    start = time.time() 
    func(atributos,granulacion) 
    print("without GPU:", time.time()-start)     
      
    start = time.time() 
    func2(atributos,granulacion) 
    print("with GPU:", time.time()-start) 
   