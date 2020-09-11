# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from fuzzylogicpy.core.NeuralNetwork import NeuralNetwork as NN
import numpy as np
from fuzzylogicpy.core.impl.memberships import FPG
import pandas as pd
class FuzzyRow:
    'Common class for all membership functions'
    
    def __init__(self,MFList):
        self.MFList = MFList
    
    def evaluateMF(self, rowInput):
        if len(rowInput) != len(self.MFList):
            assert("Number of variables does not match number of rule sets")
        
        eMF = [[self.MFList[i][k].evaluate(rowInput[i]) for k in range(len(self.MFList[i]))] for i in range(len(rowInput))]
        return eMF  

#for i in X[0]:
#    print(dictMF[0][0]['FPG'].evaluate(i))ANFANF


class DataSet:
    
    def __init__(self,file):
        self.data_set = pd.read_csv(file)
        self.__arrayNumpy = self.data_set.to_numpy()
        
    def getTarget(self):
        return self.__arrayNumpy[:,-1]
    
    def getData(self):
        return self.__arrayNumpy[:,0:-1]
    
    def getDataSet(self):
        return self.__arrayNumpy

train = DataSet('datasets/IrisTrain.csv')
test = DataSet("datasets/IrisTest.csv")
#fpg = DataSet("r_src/FPGparameters.csv")

# training_data = numpy.loadtxt(file, usecols = [1,2,3])
X = train.getData()
X2 = test.getData()
Y = train.getTarget()
Y2 = test.getTarget()
#Z = fpg.getData()



dictMF = [
[FPG(2.147896350624007,3.5163722538044273, 0.1622362353227601),
FPG(2.0003342426656783,4.0793004294307185,0.08176694727515899),
FPG(2.124844064235014,3.223141968134696,0.6662550821458767)
],
[FPG(4.361466988454315,6.761485878098842 ,0.22559186545856036),
FPG(4.370821508416422,6.568426624826564 , 0.42907861013094606),
FPG(4.637668382669777,6.265023278955064 ,0.5289049691800887)
],
[FPG(1.4270656246097206,4.1086510499769,0.026421023757245865),
FPG(1.0074706437695773,3.6296212308709777, 0.03956588467924682 ),
FPG(1.3475025402809757,5.787178967737299,0.007490905551822413)
],
[FPG(0.3241766868760386,1.422358561433751, 0.7576981912570984),
FPG(0.23587772991825504,1.8716782266481184, 0.057583009540230545),
FPG(0.15909362303435012,2.388266174289166, 0.254507451854)
]
]


mf = FuzzyRow(dictMF)
m = mf.evaluateMF(X[0])
anf = NN(X, X2, Y, Y2, mf)

t1_start = timer()
anf.trainHybridJangOffLine(epochs=20)
results = anf.predict(anf.X2, anf)
t1_stop = timer()


x = [np.round(x) for x in results]
print("Salidas: \n")
for i, j in zip(x, Y2):
    print(i - j, "\n")

anf.plotErrors()
anf.plotResults()

print("Elapsed time during the whole program in seconds:", t1_stop - t1_start)


while True:
    data = input("Entrada: \n").split()
    if any(x == "exit" for x in data):
        break
    elif any(x == "test" for x in data):
            results = anf.predict(X, anf)
            results = np.append(results,anf.predict(X, anf))

    else:
        data = [eval(x) for x in data]
        results = anf.predict([data], anf)

    x = [np.round(x) for x in results]
    print("Salidas: \n")
    for i in x:
        print(i, "\n")
# print(round(anf.consequents[-1][0],6))
# print(round(anf.consequents[-2][0],6))
# print(round(anf.consequents[9][0],6))
