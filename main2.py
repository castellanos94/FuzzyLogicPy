# -*- coding: utf-8 -*-
from timeit import default_timer as timer
import numpy as np


train = ds("r_src/Train/IrisTrain.csv")
test = ds("r_src/Test/IrisTest.csv")
fpg = ds("r_src/FPGparameters.csv")

# training_data = numpy.loadtxt(file, usecols = [1,2,3])
X = train.getData()
X2 = test.getData()
Y = train.getTarget()
Y2 = test.getTarget()
Z = fpg.getData()
memFunc = "FPG"

#dictMF = [
#    [[membFunc, {"b": 0.5, "c": float(np.random.randint(-5, 5))}] for i in range(3)]
#    for x in range(X.shape[1])
#



dictMF = [
    [
        [memFunc, {"beta": 3.5163722538044273, "gamma": 2.147896350624007, "m": 0.1622362353227601 }],
        [memFunc, {"beta": 4.0793004294307185, "gamma": 2.0003342426656783, "m": 0.08176694727515899}],
        [memFunc, {"beta": 3.223141968134696, "gamma": 2.124844064235014, "m":0.6662550821458767 }]
    ],
    [
        [memFunc, {"beta": 6.761485878098842, "gamma": 4.361466988454315, "m": 0.22559186545856036}],
        [memFunc, {"beta": 6.568426624826564, "gamma": 4.370821508416422, "m": 0.42907861013094606}],
        [memFunc, {"beta": 6.265023278955064, "gamma": 4.637668382669777, "m": 0.5289049691800887}]
    ],
    [
        [memFunc, {"beta": 4.1086510499769, "gamma": 1.4270656246097206, "m": 0.026421023757245865}],
        [memFunc, {"beta": 3.6296212308709777, "gamma": 1.0074706437695773, "m":0.03956588467924682 }],
        [memFunc, {"beta": 5.787178967737299, "gamma": 1.3475025402809757, "m": 0.007490905551822413}]
    ],
    [
        [memFunc, {"beta": 1.422358561433751, "gamma": 0.3241766868760386, "m": 0.7576981912570984}],
        [memFunc, {"beta": 1.8716782266481184, "gamma": 0.23587772991825504, "m": 0.057583009540230545}],
        [memFunc, {"beta": 2.388266174289166, "gamma": 0.15909362303435012, "m":0.254507451854}]
    ]
]

mfc = mf(dictMF)
anf = ANF(X, X2, Y, Y2, mfc)

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
            results = anf.predict(X2, anf)

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
