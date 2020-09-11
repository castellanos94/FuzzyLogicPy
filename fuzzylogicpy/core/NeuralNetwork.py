import numpy as np
import itertools
import copy


class NeuralNetwork:
    """Class to implement an Adaptative Network Fuzzy Inference System: ANFIS";

    Attributes:
        'X' contains data to train the network;
        'Y' contains target to prove the data;
        'X2' is the variable data for multiple test;
        'Y2' is the variable data for multiple target;
        'XLen' length X variable, numbers of rows;
        'memClass' is a deep copy of memFunction;
        'memFuncs' List of list to Dictionaries 'MFList';
        'memFuncsByVariable' get an array of MF with index from 1 to n, where n is len(memFuncs);
        'rules' is a array of itertools of memFuncsByVariable, It is a permut N^N;
        'consequents' array with size of len(Y) * len(rules) * length_columns(X)+1);
        'errors' array empty;
        'memFuncsHomo' verify that all MF are Homogenous;
        'trainingType' Flag 'No trained yet/ trainend'.
    """

    def __init__(self, X, X2, Y, Y2, memFunction):
        self.X = np.array(copy.copy(X))
        self.X2 = np.array(copy.copy(X2))
        self.Y = np.array(copy.copy(Y))
        self.Y2 = np.array(copy.copy(Y2))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.memFuncsHomo = all(len(i)==len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'
        self.layer4 = None
        self.layer3 = None

    def setX2(self,X2):
        self.X2 = np.array(copy.copy(X2))
  
    def LSE(self, A, B, initialGamma = 1000.):
        coeffMat = A
        rhsMat = B
        S = np.eye(coeffMat.shape[1])*initialGamma
        x = np.zeros((coeffMat.shape[1],1)) # need to correct for multi-dim B
        for i in range(len(coeffMat[:,0])):
            a = coeffMat[i,:]
            bM = np.matrix(rhsMat[i])
            aM = np.matrix(a)
            S = S - (S.dot(aM.transpose()).dot(aM).dot(S))/ (1 + S.dot(a).dot(a))
            x = x + (S.dot(aM.transpose().dot(bM - aM.dot(x))))
        return x

    def firstLayer(self, ANFISObj, Xs, pattern):
        x = [[0] for i in range(len(ANFISObj.memFuncsByVariable))]  
        l1 =  ANFISObj.memClass.evaluateMF(Xs[pattern,:]) 
        #for i in range(len(x)):
        #    x[i] += l1[i]

        return l1
    
    def secondLayer(self,ANFISObj,layerOne):
        miAlloc = [[layerOne[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in range(len(ANFISObj.rules))]
        

        #tmp = []
        #for x in miAlloc:
        #    v = 0
            #x = [i for i in x if i != 0]
        #    for y in range(len(x)):
        #        if y == 0:
        #            v = x[y]
        #        else:
        #            v = GMBC()._and(v,x[y])
        #    tmp.append(v)
        #return np.array(tmp).T
        return np.array([np.product(x) for x in miAlloc]).T

    
    def fourthLayer(self,ANFISObj,Xs):
        layerFour = np.empty(0,) #create an empty array '[]'
        wSum = []
        for pattern in range(Xs.shape[0]):
            #layer one
            layerOne = self.firstLayer(ANFISObj,Xs,pattern)
            #layer two
            layerTwo = self.secondLayer(ANFISObj,layerOne)
            w = layerTwo if pattern == 0 else np.vstack((w,layerTwo))
            #layer three
            wSum.append(np.sum(layerTwo))
            wNormalized = layerTwo/wSum[pattern] if pattern == 0 else np.vstack((wNormalized,layerTwo/wSum[pattern]))
            #prep for return layer four  (bit of hack)
            layerThree = layerTwo/wSum[pattern]
            self.layer3 = layerThree
            rowHolder = np.concatenate([x*np.append(Xs[pattern,:],1) for x in layerThree])
            layerFour = np.append(layerFour,rowHolder)
        w = w.T #Transpose()
        wNormalized = wNormalized.T #Transpose
        layerFour = np.array(np.array_split(layerFour,pattern+1))
        return layerFour, wSum, w        

    def fifthLayer(self,layerFour,initialGamma):
        self.layer4 = layerFour
        layerFive = np.array(self.LSE(layerFour,self.Y,initialGamma))
        self.consequents = layerFive
        return np.dot(layerFour,layerFive)

    def trainHybridJangOffLine(self,epochs = 5, tolerance = 1e-5, initialGamma = 1000, k = 0.01):
        self.trainingType = "trainHybridJangOffLine"
        convergence = False
        epoch = 1

        while epoch < epochs and convergence is not True:
            #layer four: fordward half pass
            [layerFour, wSum, w] = self.fourthLayer(self, self.X)
            #layer five
            layerFive = self.fifthLayer(layerFour,initialGamma)
        
            #calc. error
            error = np.sum((self.Y - layerFive.T)**2)
            print('Current error: ',error)
            average_error = np.average(np.absolute(self.Y - layerFive.T))
            self.errors = np.append(self.errors,error)

            if len(self.errors) != 0:
                if self.errors[len(self.errors)-1] < tolerance:
                    convergence = True
            
            #back propagation
            
            if convergence is not True:
                cols = range(self.X.shape[1])
                dE_dAlpha = list(BackPropagation().backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.X.shape[1]))
            
            #Calc. of K
            #when mount errors are four or more
            if len(self.errors) >= 4:
                if (self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1
            #when mount errors are five or more
            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9

            ## handling of variables with a different number of MFs
            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / np.abs(np.sum(t))

            if(np.isinf(eta)):
                eta = k

            ## handling of variables with a different number of MFs
            dAlpha = copy.deepcopy(dE_dAlpha)
            if not(self.memFuncsHomo):
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)


            for varsWithMemFuncs in range(len(self.memFuncs)):
                for MFs in range(len(self.memFuncsByVariable[varsWithMemFuncs])):
                    paramList = self.memFuncs[varsWithMemFuncs][MFs].get_values()
                    for param in range(len(paramList)):
                        paramList[param] +=  dAlpha[varsWithMemFuncs][MFs][param]
            epoch = epoch + 1

    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print (self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.errors)),self.errors,'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()
    
    def plotMF(self, x, inputVar):
        import matplotlib.pyplot as plt
        from skfuzzy import gaussmf, gbellmf, sigmf

        for mf in range(len(self.memFuncs[inputVar])):
            if self.memFuncs[inputVar][mf][0] == 'gaussmf':
                y = gaussmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'gbellmf':
                y = gbellmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'sigmf':
                y = sigmf(x,**self.memClass.MFList[inputVar][mf][1])

            plt.plot(x,y,'r')

        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print (self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.fittedValues)),self.fittedValues,'r', label='trained')
            plt.plot(range(len(self.Y2)),self.Y2,'b', label='original')
            plt.legend(loc='upper left')
            plt.show()
    

    def predict(self, varsToTest,ANFISObj):
         #Evaluate test data
        self.setX2(varsToTest)
        self.fittedValues = self.__predict(ANFISObj,self.X2)
        return self.fittedValues
        


    def __predict(self,ANFISObj, varsToTest):

        [layerFour, wSum, w] = self.fourthLayer(ANFISObj, varsToTest)

        #layer five
        layerFive = np.dot(layerFour,ANFISObj.consequents)

        return layerFive

class BackPropagation:
    
    def backprop(self,ANFISObj, columnX, columns, theWSum, theW, theLayerFive):

        paramGrp = [0]* len(ANFISObj.memFuncs[columnX])
        for MF in range(len(ANFISObj.memFuncs[columnX])):

            parameters = np.empty(len(ANFISObj.memFuncs[columnX][MF].get_values()))
            timesThru = 0
            for alpha in ANFISObj.memFuncs[columnX][MF].get_values():

                bucket3 = np.empty(len(ANFISObj.X))
                for rowX in range(len(ANFISObj.X)):
                    varToTest = ANFISObj.X[rowX,columnX]
                    tmpRow = np.empty(len(ANFISObj.memFuncs))
                    tmpRow.fill(varToTest)

                    bucket2 = np.empty(ANFISObj.Y.ndim)
                    for colY in range(ANFISObj.Y.ndim):

                        rulesWithAlpha = np.array(np.where(ANFISObj.rules[:,columnX]==MF))[0]
                        adjCols = np.delete(columns,columnX)

                        senSit = ANFISObj.memFuncs[columnX][MF].derive(ANFISObj.X[rowX,columnX],alpha)
                        # produces d_ruleOutput/d_parameterWithinMF
                        dW_dAplha = senSit * np.array([np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha])

                        bucket1 = np.empty(len(ANFISObj.rules[:,0]))
                        for consequent in range(len(ANFISObj.rules[:,0])):
                            fConsequent = np.dot(np.append(ANFISObj.X[rowX,:],1.),ANFISObj.consequents[((ANFISObj.X.shape[1] + 1) * consequent):(((ANFISObj.X.shape[1] + 1) * consequent) + (ANFISObj.X.shape[1] + 1)),colY])
                            acum = 0
                            if consequent in rulesWithAlpha:
                                acum = dW_dAplha[np.where(rulesWithAlpha==consequent)] * theWSum[rowX]

                            acum = acum - theW[consequent,rowX] * np.sum(dW_dAplha)
                            acum = acum / theWSum[rowX]**2
                            bucket1[consequent] = fConsequent * acum

                        sum1 = np.sum(bucket1)

                        if ANFISObj.Y.ndim == 1:
                            bucket2[colY] = sum1 * (ANFISObj.Y[rowX]-theLayerFive[rowX,colY])*(-2)
                        else:
                            bucket2[colY] = sum1 * (ANFISObj.Y[rowX,colY]-theLayerFive[rowX,colY])*(-2)

                    sum2 = np.sum(bucket2)
                    bucket3[rowX] = sum2

                sum3 = np.sum(bucket3)
                parameters[timesThru] = sum3
                timesThru = timesThru + 1

            paramGrp[MF] = parameters

        return paramGrp