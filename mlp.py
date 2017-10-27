# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as numpy


input    = numpy.array([[0,0],[0,1],[1,0],[1,1]])
expected = numpy.array([[0],[1],[1],[0]])

trainingRuns= 5000
hiddenLayerNeuronen = 10
inputDimension = 2
outputDimension = 1
learnRate = 0.1

weightsInputToHiddenLayer = numpy.random.uniform(-0.1,0.1,(hiddenLayerNeuronen,inputDimension))
weightsHiddenToOutputLayer = numpy.random.uniform(-0.1,0.1,(outputDimension,hiddenLayerNeuronen))

biasHidden = numpy.random.uniform(-0.1,0.1,(10))
biasOut = numpy.random.uniform(-0.1,0.1,(1))

for trainingRun in range(trainingRuns):
    for i in range(4):
        #Feedforward
        currentInput = numpy.copy(input[i])
        inputHiddenLayer = numpy.dot(weightsInputToHiddenLayer,currentInput) + biasHidden
        #print ("inputHiddenLayer") 
        #print (inputHiddenLayer)
        outputHiddenLayer = 1/(1+numpy.exp(-inputHiddenLayer)) 
        #print ("outputHiddenLayer") 
        #print(outputHiddenLayer)
        outputLayer = numpy.dot(weightsHiddenToOutputLayer,outputHiddenLayer) + biasOut
        #print("OutputLayer")
        #print(OutputLayer)
        
        #backprobab        
        error = expected[i] - outputLayer
        derivation = outputHiddenLayer*(1-outputHiddenLayer)       
        #print("derivation")
        #print(derivation)
        deltaHiddenLayer = derivation*numpy.dot(numpy.transpose(outputLayer),error)
        
        #learning
        weightsHiddenToOutputLayer += learnRate * numpy.outer(error,outputHiddenLayer)
        weightsInputToHiddenLayer += learnRate * numpy.outer(deltaHiddenLayer, currentInput)
        
        biasOut += learnRate * error
        biasHidden += learnRate * deltaHiddenLayer
        
        
        #print("weightsHiddenToOutputLayer")
        #print(weightsHiddenToOutputLayer)        
        
        #print("weightsInputToHiddenLayer")
        #print(weightsInputToHiddenLayer)

        #if i == 0:
        print("error")
        print(error)

#print(weightsInputToHiddenLayer)

#print(input)
#print(expected)