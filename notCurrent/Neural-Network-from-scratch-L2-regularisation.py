import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

class Neural_Network(object):
    def __init__(self, Lambda):        
        #Define Hyperparameters
        self.inputLayerSize = 10
        self.outputLayerSize = 1
        self.hiddenLayerSize = 100
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        #Make empty list to store costs:
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 400, 'disp' : True}
        #Train using Quasi-Newton method which iteratively computes an estimate of the inverse Hessian
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad



def main():
    #Training Data:
    TrainData = sio.loadmat('AllTrainData1.mat')
    trainX = np.array((TrainData['AllTrainData1']))
    TrainLabels = sio.loadmat('TrainLabels.mat')
    trainY = np.array((TrainLabels['TrainLabels']))

    #Testing Data:
    TestData = sio.loadmat('AllTestData1.mat')
    testX = np.array((TestData['AllTestData1']))
    TestLabels = sio.loadmat('TestLabels.mat')
    testY = np.array((TestLabels['TestLabels']))

    #Normalize:
    trainX = trainX/np.amax(trainX, axis=0)
    trainY = trainY
    
    #Normalize by max of training data:
    testX = testX/np.amax(trainX, axis=0)
    testY = testY 

    startTime = time.clock()
    NN = Neural_Network(Lambda = 0.0)
    cost1 = NN.costFunction(trainX,trainY)
    dJdW1, dJdW2 = NN.costFunctionPrime(trainX,trainY)

    scalar = 3
    NN.W1 = NN.W1 + scalar*dJdW1
    NN.W2 = NN.W2 + scalar*dJdW2
    cost2 = NN.costFunction(trainX,trainY)

    dJdW1, dJdW2 = NN.costFunctionPrime(trainX,trainY)
    NN.W1 = NN.W1 - scalar*dJdW1
    NN.W2 = NN.W2 - scalar*dJdW2
    cost3 = NN.costFunction(trainX,trainY)
    grad = NN.computeGradients(trainX,trainY)

    #Get rid of this for the time being as Numerical Gradient was only for 2D
    #numgrad = computeNumericalGradient(NN,trainX,trainY)
    #print "Perform numerical gradient computaton to ensure that our\n calculus is correct, the difference is:", np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

    T = trainer(NN)

    T.train(trainX, trainY, testX, testY)
    plt.plot(T.J)
    plt.plot(T.testJ)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    
    endTime = time.clock()
    print "Time taken to compute:", endTime
    yHat = NN.forward(trainX)
    print "True outputs:\n", trainY, "\n Estimated outputs for training data:\n", yHat

    np.savetxt("foo.csv", yHat, delimiter=",")
    plt.show()

	#Test network for various combinations of sleep/study:
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)
	#Normalize data (same way training data way normalized)
    hoursSleepNorm = hoursSleep/10.
    hoursStudyNorm = hoursStudy/5.
	#Contour plot
    yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
    xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

    a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()
    
    allOutputs = NN.forward(allInputs)

    fig = plt.figure()

    ax = fig.gca(projection='3d')    
    #surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), cmap=plt.cm.jet)

    #ax.set_xlabel('Electrode2')
    #ax.set_ylabel('Electrode7')
    #ax.set_zlabel('Probability of Finger Extension')

    #plt.show()

if __name__ == '__main__':
    main()
