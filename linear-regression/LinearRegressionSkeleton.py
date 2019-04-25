import numpy as np
import random
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, dataFilePath, outputPath, alpha=0.01, 
                 maxIter = 500, errorThreshold=0.001, 
                 performTest = False, normalize = False):

        self.dataFilePath = dataFilePath
        self.outputPath   = outputPath
        self.alpha        = alpha
        self.maxIter      = maxIter
        self.errorThreshold = errorThreshold
        self.performTest    = performTest
        self.normalize      = normalize

        self.loadDataFromFile()
        self.initWeights()


    def featureNormalize(self, X):
        X_Norm = x
        for i in range(len(X[a])):
            m = np.mean(X[:, i])
            s = np.std(X[:,i])
            X_Norm[:,i] = (X_Norm[:,i] - m) / s

        return X_Norm

    def loadDataFromFile(self):
        datasetLoaded = np.loadtxt(self.dataFilePath, delimiter=";", skiprows=1)

        if(self.normalize):
            datasetLoaded = self.featureNormalize(datasetLoaded)

        self.nExamples = datasetLoaded.shape[0]
        self.nAttributes = len(datasetLoaded[0])

        if(self.performTest):
            nExamplesTest = int(self.nExamples / 3.0)
            self.testData = np.ones(shape=(nExamplesTest, self.nAttributes))
            self.testTarget = np.zeros(shape=(nExamplesTest, 1))

            linesForTest = random.sample(range(0, self.nExamples), nExamplesTest)
            count = 0
            for line in linesForTest:
                self.testData[count,1] = datasetLoaded[line, :-1]
                self.testTarget[count] = datasetLoaded[line, -1]
                count += 1

            datasetLoaded = np.delete(datasetLoaded, linesForTest, 0) # 0 - linhas, 1 - Colunas
            self.nExamples -= nExamplesTest

        self.dataset = np.ones(shape=(self.nExamples, self.nAttributes))
        self.dataset[:,1:] = datasetLoaded[:,:-1]
        self.target = datasetLoaded[:, -1]
        self.target.shape = (self.nExamples, 1)

    def initWeights(self):
        #TODO: INICIAR OS PESOS
        self.weights = np.ones(shape=(self.nAttributes, 1))
        for i in range(0, self.nAttributes):
            self.weights[i][0] = random.random()

    def linearFunction(self, data):
        output = data.dot(self.weights)
        return  output

    def calculateError(self, data, target):
        output = self.linearFunction(data)
        error = output - target
        return error

    def squaredErrorCost(self, data, target):
        error = self.calculateError(data, target)
        squaredError = (1.0 / (2 * self.nExamples)) * (error.T.dot(error))
        return  squaredError

    def gradientDescent(self):
        cost = self.calculateError(self.dataset, self.target)
        for i in range (self.nAttributes):
            temp = self.dataset[:, i]
            temp.shape = (self.nExamples, 1)
            currentErros = cost * temp
            self.weights[i][0] = self.weights[i][0] - self.alpha * ((1.0 / self.nExamples) * currentErros.sum())


    def plotCostGraph(self, trainingErrorsList, testingErrorsList=None):
        xAxisValues = range(0, len(trainingErrorsList))
        line1 = plt.plot(xAxisValues, trainingErrorsList, label="Training error")
        if self.performTest:
            line2 = plt.plot(xAxisValues, testingErrorsList, linestyle="dashed", label="Testing error")

        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("LMS Error")
        plt.savefig(self.outputPath + "/lms_error.png")
        plt.show()
        plt.close()

    def plotLineGraph(self, weightsToPlot, iteration):

	    if self.performTest:
	        dataToPlot   = np.append(self.dataset, self.testData,0)
	        targetToPlot = np.append(self.target, self.testTarget,0)

	    else:
	        dataToPlot   = self.dataset
	        targetToPlot = self.target


	    xAxisValues = dataToPlot[:,1]
	    yAxisValues = targetToPlot

	    xMax = max(xAxisValues)
	    xMin = min(xAxisValues)
	    yMax = max(yAxisValues)

	    axes = plt.gca()
	    axes.set_xlim([0, xMax + 1])
	    axes.set_ylim([0, yMax + 1])

	    xLineValues = np.arange(xMin, xMax, 0.1)
	    yLineValues = weightsToPlot[0] + xLineValues * weightsToPlot[1]

	    plt.plot(xLineValues, yLineValues)
	    plt.plot(xAxisValues, yAxisValues, 'o')
	    plt.savefig(self.outputPath + "/line_" + str(iteration) + ".png")
	    plt.close()

    def run(self):
        lmsError = self.squaredErrorCost(self.dataset, self.target)
        count = 0
        trainingErrors = list()
        testingErros = list()
        trainingErros.append(lmsError[0])
        if(self.performTest):
            lmsTestError = self.squaredErrorCost(self.testData, self.testTarget)
            testingErros.append(lmsTestError[0])
        print "ERROR " + str(lmsError)
        print "weights" + str(self.weights)
        while lmsError > self.errorThreshold and count < self.maxIter:
            self.gradientDescent()
            count +=1

            lmsError = self.squaredErrorCost(self.dataset, self.target)
            trainingErrors.append(lmsError)

            if(self.performTest):
                lmsTestError = self.squaredErrorCost(self.testData, self.testTarget)
                testingErros.append(lmsTestError[0])

            if count % 100 == 0:
                print "ERROR: " + str(lmsError)
                print "weights: " + str(self.weights)
                self.plotLineGraph(self.weights, count)

        if self.performTest:
            self.plotCostGraph(trainingErrors, testingErros)
        else:
            self.plotCostGraph(trainingErrors)

if __name__ == '__main__':
        linReg = LinearRegression("/home/201719040800/Downloads/linear-regression/income.csv",
                              "/home/201719040800/Downloads/linear-regression/results",
                              normalize=True, performTest=True)
        linReg.run()
















































