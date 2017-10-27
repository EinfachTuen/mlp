# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as numpy
import KTimage as kt

def printKT(Q,i):
    kt.exporttiles(array=Q, height=1, width=len(Q[0]), outer_height=1, outer_width=len(Q), filename="results/obs_W_1_"+i+".pgm")
    
input    = numpy.array([[0,0],[0,1],[1,0],[1,1]])
expected = numpy.array([[0],[1],[1],[0]])

trainingRuns= 250000
hiddenLayerNeuronen = 10
inputDimension = 2
outputDimension = 1
learnRate = 0.1
printAmount =  10
j=0

moduloWert = trainingRuns/printAmount

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
        #print("error")
       # print(error)
        
       
        if trainingRun%moduloWert == 0 and i == 3:
            j += 1            
            print (moduloWert)
            print ("hiermal")
            print("error"+str(error))
            printKT(weightsInputToHiddenLayer, str(j))

