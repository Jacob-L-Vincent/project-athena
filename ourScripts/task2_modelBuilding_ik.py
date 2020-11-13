#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task 2 model building playground

Created on Fri Nov 13 14:26:26 2020

@author: isaac
"""

import os
import sys
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from matplotlib import pyplot as plt    
import tensorflow as tf
import tensorflow.keras as ker
import numpy as np
from utils.file import load_from_json
from ourFuncs_task2 import getMaxIndex

def trainNewModel(inputData, trueData, epochs=5, verbose=2):
    model = ker.models.Sequential([
      ker.layers.Flatten(input_shape=(28, 28)),
      ker.layers.Dense(128, activation='relu'),
      ker.layers.Dense(128, activation='relu'),
      ker.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=ker.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(inputData, trueData, epochs=5)
    
    return model



dataDir = "../ourDataFiles/ensembleOuts_demoWDs"
fileName = "ensemPredic_benign.npy"
filePath = os.path.join(dataDir,fileName)
dataPred = np.load(filePath)

data_configs = load_from_json("../src/configs/demo/data-mnist.json")
bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
data = np.load(bs_file)


# get index of max value, see src/scripts/ourFuncs_task2
dataTrain = data[0:200,:,:,:]
values = np.zeros(dataTrain.shape[0],dtype=np.int)
for i in range(dataTrain.shape[0]):
    values[i]=getMaxIndex(dataPred[i,:])

# run train model
model = trainNewModel(dataTrain, values)

# evaluate the model to get overall stats
dataTest = data[300:500,:,:,:]
values = np.zeros(dataTest.shape[0],dtype=np.int)
for i in range(300,500):
    values[i-300]=getMaxIndex(dataPred[i,:])
model.evaluate(dataTest,  values, verbose=2)


# iterate through each test data and record the output 1x10 probability array
#predictions = model(data[576,:,:,0])

##### ^^^ THIS DOESN'T WORK YET ^^^ ########







