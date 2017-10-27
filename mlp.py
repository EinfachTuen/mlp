# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as numpy


input    = numpy.array([[0,0],[0,1],[1,0],[1,1]])
expected = numpy.array([[0],[1],[1],[0]])

trainingRuns= 5000
hiddenLayerNeuronen = 100
inputDimension = 2
outputDimension = 1
learnRate = 0.1

weightsInputToHiddenLayer = numpy.random.uniform(-0.1,0.1,(hiddenLayerNeuronen,inputDimension))
weightsHiddenToOutputLayer = numpy.random.uniform(-0.1,0.1,(outputDimension,hiddenLayerNeuronen))

biasHidden = numpy.random.uniform(-0.1,0.1,(hiddenLayerNeuronen))
biasOut = numpy.random.uniform(-0.1,0.1,(outputDimension))

for trainingRun in range(trainingRuns):
    for i in range(4):
        #Feedforward
        currentInput = numpy.copy(input[i])
        inputHiddenLayer = numpy.dot(weightsInputToHiddenLayer,currentInput) + biasHidden
        outputHiddenLayer = 1/(1+numpy.exp(-inputHiddenLayer)) 
        outputLayer = numpy.dot(weightsHiddenToOutputLayer,outputHiddenLayer) + biasOut
  
        
        #backprobab        
        error = expected[i] - outputLayer
        derivation = outputHiddenLayer*(1-outputHiddenLayer)       
        deltaHiddenLayer = derivation*numpy.dot(numpy.transpose(outputLayer),error)
        
        #learning
        weightsHiddenToOutputLayer += learnRate * numpy.outer(error,outputHiddenLayer)
        weightsInputToHiddenLayer += learnRate * numpy.outer(deltaHiddenLayer, currentInput)
        
        biasOut += learnRate * error
        biasHidden += learnRate * deltaHiddenLayer

        #if i == 0:
        print("error")
        print(error)

