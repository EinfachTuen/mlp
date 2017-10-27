# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as numpy


input    = numpy.array([[0,0],[0,1],[1,0],[1,1]])
expected = numpy.array([[0],[1],[1],[0]])

trainingRuns= 20
hiddenLayerNeuronen = 10
inputDimension = 2
outputDimension = 1
lernRate = 0.1

weightsInputToHiddenLayer = numpy.random.uniform(-0.1,0.1,(hiddenLayerNeuronen,inputDimension))
weightsHiddenToOutputLayer = numpy.random.uniform(-0.1,0.1,(outputDimension,hiddenLayerNeuronen))

for trainingRun in range(trainingRuns):
    for i in range(4):
        #Feedforward
        currentInput = numpy.copy(input[i])
        inputHiddenLayer = numpy.dot(weightsInputToHiddenLayer,currentInput)  
        #print ("inputHiddenLayer") 
        #print (inputHiddenLayer)
        outputHiddenLayer = 1/(1+numpy.exp(-inputHiddenLayer)) 
        print ("outputHiddenLayer") 
        print(outputHiddenLayer)
        OutputLayer = numpy.dot(weightsHiddenToOutputLayer,outputHiddenLayer)
        #print("OutputLayer")
        #print(OutputLayer)
        
        #backprobab        
        error = expected[i] - OutputLayer
        derivation = outputHiddenLayer*(1-outputHiddenLayer)       
        print("derivation")
        print(derivation)
        deltaHiddenLayer = derivation*numpy.dot(numpy.transpose(OutputLayer),error)
        
        #learning
        
        

#print(weightsInputToHiddenLayer)

print(input)
print(expected)