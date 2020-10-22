import pandas as pd
import random
from timeit import default_timer as timer
import matplotlib.pylab as plt
import numpy as np

from fuzzylogicpy.algorithms.NeuralNetwork import NeuralNetwork as NN
from fuzzylogicpy.algorithms.NeuralNetwork import FuzzyRow as FR
from fuzzylogicpy.core.elements import StateNode, GeneratorNode, NodeType, Operator
from fuzzylogicpy.parser.expression_parser import ExpressionParser
from fuzzylogicpy.algorithms.algorithms import ExpressionEvaluation, MembershipFunctionOptimizer, KDFLC
from fuzzylogicpy.core.impl.logics import GMBC

from sklearn.preprocessing import MinMaxScaler

PASOS=4

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[1]-1):
        x_test[0][i] = x_test[0][i+1]
    x_test[0][x_test.shape[1]-1]=nuevoValor
    return x_test

def getParameters_GA(data,granulate =  3 ):
    
    states = {}
    names = list(data.columns)
    for head in data.head():
        states[head] = StateNode(head, head)

    props = GeneratorNode(2, 'properties', [v for v in states.keys() if names[-1] != v ],
                          [NodeType.EQV, NodeType.AND, NodeType.NOT], 3)
    category = GeneratorNode(1, 'category', [v for v in states.keys() if names[-1] == v],
                             [NodeType.NOT])
    generators = {props.label: props, category.label: category}
    
    a = '"'+'" "'.join([v for v in names[0:-1]])+'"'
    expression = '(IMP (AND {}) "{}")'.format(a,names[-1])
    parser = ExpressionParser(expression, states, generators)
    root = parser.parser()
    algorithm = KDFLC(data, root, states, GMBC(), 50, 10, 30, 0.95, 0.1)
    algorithm.discovery()
    
    q = []
    
    a = list(range(len(algorithm.predicates)))

    for item in random.sample(a,granulate):
        
        d = Operator.get_nodes_by_type(algorithm.predicates[item],NodeType.STATE)
        q.append([i.membership for i in d])
        
    return q


datos = pd.read_csv('datasets/LineaTiempo2.csv',parse_dates=[0])
names = list(datos.columns)



for name in names[5:]:
    df = pd.DataFrame(data=datos[name].values, index=datos['tiempo'])
    # load dataset
    
   
    values = df.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, PASOS, 1)

    values = reframed.values
    n_train_days = 365 - (90+PASOS)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    # split into input and outputs
    x_train, y_train = np.array(train[:, :-1]), np.array(train[:, -1])
    x_val, y_val = np.array(test[:, :-1]), np.array(test[:, -1])
    # reshape input to be 3D [samples, timesteps, features]
    #x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    #x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


    granulate = 3
    g = getParameters_GA(reframed,granulate)

    asd= [[g[i][j] for i in range(granulate)] for j in range(len(g[0])-1)]

    mf = FR(asd)
    
    anf = NN(x_train, x_val, y_train, y_val, mf)

    t1_start = timer()
    anf.trainHybridJangOffLine(epochs=20)
    results = anf.predict(anf.X, anf)

    ## Stop learning 
    
    df = pd.DataFrame(data=datos[name].values, index=datos['tiempo'])
    #ultimosDias = df['2020-01-02':'2020-01-31']
    siguienteQuin = np.concatenate( df['2020-02-01':'2021-02-28'].values)
    #print(ultimosDias)
    values = df.values
    values = values.astype('float32')
    # normalize features
    values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, PASOS, 1)
    reframed.drop(reframed.columns[[PASOS]], axis=1, inplace=True)
    #print(reframed.head(7))
    values = reframed.values
    x_test = np.array(values[26:, :])
    #x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    

 
    results=[]
    days = len(x_test)
    for i in range(days):
        parcial= anf.predict(x_test,anf)
        results.append(parcial[0])
        #results.append(parcial)
        #print(x_test)
        x_test=agregarNuevoValor(x_test,parcial[0])
    
    adimen = [x for x in results]    
    inverted = np.concatenate(scaler.inverse_transform(adimen))

    
    print("para serie: ",name)
    for i,j in zip(inverted,siguienteQuin):
        print(i,j)

    t1_stop = timer()