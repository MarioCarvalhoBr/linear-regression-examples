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
        #TODO: NORMALIZAR OS DADOS

    def loadDataFromFile(self):
        #TODO: CARREGAR DADOS DO ARQUIVO

    def initWeights(self):
        #TODO: INICIAR OS PESOS

    def linearFunction(self, data):
        #TODO: SAÍDA DA FUNÇÃO LINEAR

    def calculateError(self, data, target):
        #TODO: CALCULAR O ERRO PARA UM PONTO

    def squaredErrorCost(self, data, target):
        #TODO: CALCULAR O ERRO PARA TODOS OS PONTOS

    def gradientDescent(self):
       #TODO: GRADIENTE DESCENDENTE

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
        #TODO: PRINCIPAL

if __name__ == '__main__':
        linReg = LinearRegression(CAMINHO-ARQUIVO-ENTRADA,
                              CAMINHO-PASTA-SAIDA, 
                              normalize=True, performTest=True)
        linReg.run()
















































