#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:53:43 2020

@author: jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataFile = "results/model_eval_results_relu.csv"
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
task1PGD = results.loc[( (results['ae_type']=="aes_task1") | (results['ae_type']=="benign") ) &
                    (~results['label'].str.contains("fg"))]  

task1FGSM = results.loc[( (results['ae_type']=="aes_task1") | (results['ae_type']=="benign") ) &
                    (results['label'].str.contains("fg"))] 

benign = results.loc[(results['ae_type']=="benign")]
task1FGSM = pd.concat([task1FGSM, benign])


wds = results.loc[(results['ae_type']=="aes_wds_overall") |
                        (results['ae_type']=="aes_task1_overall") |
                        (results['ae_type']=="benign") ]  

#Save .csv if you want
# wds.to_csv('Elu_AllAEs.csv') 
# task1PGD.to_csv('Task1AEsPGD.csv')   
# task1FGSM.to_csv('Task1AEsFGSM.csv')                     
# weakd.to_csv('EnsembleAEs.csv')     

                  

ax1 = weakd.plot.barh(x='label', y='accuracy',title = 'Accuracy of individual ensmeble results')
ax1.set_xlabel("Accuracy")
ax1.set_ylabel("Ensemble weak Defense")

ax2 = task1PGD.plot.barh(x='label', y='accuracy',title = 'Accuracy of PGD AEs from Task1')
ax2.set_xlabel("Accuracy")
ax2.set_ylabel("Task1 PGD AE's")

ax3 = task1FGSM.plot.barh(x='label', y='accuracy',title = 'Accuracy of FGSM AEs from Task1')
ax3.set_xlabel("Accuracy")
ax3.set_ylabel("Task1 FGSM AE's")

ax4 = wds.plot.barh(x='label', y='accuracy',title = 'Accuracy of Models using Sigmoid')
ax4.set_xlabel("Accuracy", fontweight='bold')
ax4.set_ylabel("Evaluations of models against different AE's", fontweight='bold')



