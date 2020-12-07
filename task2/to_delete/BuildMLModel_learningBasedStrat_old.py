#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:20:26 2020
@author: jacob
"""

import os
import sys
import random
import numpy as np
import pandas as pd
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
from utils.file import load_from_json
import tensorflow.keras as ker
from misc_basicFuncs import getMaxIndex

# load configs
trans_configs = load_from_json("configs/athena-mnist.json")
model_configs = load_from_json("configs/model-mnist.json")
data_configs = load_from_json("configs/data-mnist.json")
verbose = 10
verModel = 0

################################################################
def trainNewModel(inputData, trueData, epochs=7, verbose=2):
    model = ker.models.Sequential([
      ker.layers.Flatten(input_shape=(28, 28)),
      ker.layers.Dense(128, activation='sigmoid'),
      ker.layers.Dense(128, activation='sigmoid'),
      ker.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=ker.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(inputData, trueData, epochs=7)
    
    return model

############################################################
# load data
cleanData = np.load(data_configs.get("bs_file"))
trueLabels = np.load(data_configs.get("label_file"))
ensPred = np.load("models/ensemPredic_benignInput_probs.npy")

ensPred_indexes = np.zeros(np.shape(ensPred)[0])
trueLabels_indexes = np.zeros(np.shape(trueLabels)[0])
for i in range(np.shape(ensPred)[0]):
    pred = getMaxIndex(ensPred[i])
    trueLab = getMaxIndex(ensPred[i])
    ensPred_indexes[i]=pred
    trueLabels_indexes[i]=trueLab

# Train ML Model
model = trainNewModel(cleanData[:8000,:,:,0], ensPred_indexes[:8000])
if(verbose>4): print("finished training model")

# create dataframe to save evaluation results
cols=["ae_type","label","accuracy","loss"]
results = pd.DataFrame(columns=cols)

#### Evaluate benign samples
evalOut = model.evaluate(cleanData[8000:],  trueLabels_indexes[8000:], verbose=verModel)
if(verbose>4): print("{} finished evaluating -- accuracy: {}".format("benigns",evalOut[1]))
# save to DataFrame
newRow = {cols[0]:"benign",cols[1]:"benign",cols[2]:evalOut[1],cols[3]:evalOut[0]}
results = results.append(newRow,ignore_index=True)

### Evaluate AE inputs that correspond to weak defenses
# load the filenames from configs
ae_dir = os.path.abspath(data_configs.get("ae_dir"))
ae_files = data_configs.get("ae_files_wds")
# iterate through each file, evaluate, and save to dataframe
for file in ae_files:
    # set filepaths
    filePath = os.path.join(ae_dir,file)
    aeLabel = file.replace(".npy","")
    # load data
    data = np.load(filePath)
    # evaluate
    evalOut = model.evaluate(data[8000:],  trueLabels_indexes[8000:], verbose=verModel)
    # save results
    if(verbose>9): print("{} finished evaluating -- accuracy: {}".format(aeLabel,evalOut[1]))
    newRow = {cols[0]:"aes_wds",cols[1]:aeLabel,cols[2]:evalOut[1],cols[3]:evalOut[0]}
    results = results.append(newRow,ignore_index=True)
if(verbose>4): print("finished evaluating ensemble defense AEs")
### Evaluate AE inputs generated in task 1
# load the filenames from configs
ae_files = data_configs.get("ae_files_task1")
# iterate through each file, evaluate, and save to dataframe
for file in ae_files:
    # set filepaths
    filePath = os.path.join(ae_dir,file)
    aeLabel = file.replace(".npy","")
    # load data
    data = np.load(filePath)
    # evaluate
    evalOut = model.evaluate(data[8000:],  trueLabels_indexes[8000:], verbose=verModel)
    # save results
    if(verbose>9): print("{} finished evaluating -- accuracy: {}".format(aeLabel,evalOut[1]))
    newRow = {cols[0]:"aes_task1",cols[1]:aeLabel,cols[2]:evalOut[1],cols[3]:evalOut[0]}
    results = results.append(newRow,ignore_index=True)
if(verbose>4): print("finished evaluating ensemble defense AEs")
    
# save data
results.to_csv("model_eval_results.csv")

if(verbose>0): print("finished running, saved evaluation output to model_eval_results.csv")




