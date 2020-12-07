#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:53:43 2020

@author: jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
dataFile = "model_eval_results.csv"
results = pd.read_csv(dataFile)
# clean AE labels
results["simpName"]=results['label'].str.split('-',3)
for i, row in results.iterrows():
    if(len(row["simpName"])>1):
        results.at[i,'simpName']=row["simpName"][3]
    else:
        results.at[i,'simpName']=row["simpName"][0]
#### Plotting

# select the AEs for corresponding weak defenses to plot
weakd = results.loc[(results['ae_type']=="aes_wds") | (results['ae_type']=="benign")]
# select the task 1 AEs to plot
task1 = results.loc[( (results['ae_type']=="aes_task1") | (results['ae_type']=="benign") ) &
                    (~results['label'].str.contains("fg"))]  


# plot the evaluation results for AEs for corresponding weak defenses
ax1 = weakd.plot.barh(x='simpName', y='accuracy',title = 'Accuracy of inndivudual ensmeble resuslts')
ax1.set_xlabel("Accuracy")
ax1.set_ylabel("Ensemble weak Defense")

# plot the eval. results for AEs from task 1
ax2 = task1.plot.barh(x='simpName', y='accuracy',title = 'Accuracy of AEs from Task1')
ax2.set_xlabel("Accuracy")
ax2.set_ylabel("Task1 AE's")