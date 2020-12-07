#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:53:43 2020

@author: jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pathway = r'/home/jacob/Desktop/project-athena/task2/model_eval_results.csv'

results = pd.read_csv(pathway)
resultssplt = results['label'].str.split('-',3)

task1 = results.loc[results['ae_type'] == 'aes_task1']
weakd = results.loc[results['ae_type'] == 'aes_wds']

for i in range(len(weakd)-1):
    weakd['label'][i+1]  = resultssplt[i+1][3]
    
    

    
task1PGD = task1[~task1['label'].str.contains('fg')]
task1FGSM = task1[task1['label'].str.contains('fg')]

benign = results.loc[results['ae_type'] == 'benign']
weakd = pd.concat([weakd,benign])
task1PGD = pd.concat([task1PGD,benign])
task1FGSM = pd.concat([task1FGSM,benign])


ax1 = weakd.plot.barh(x='label', y='accuracy',title = 'Accuracy of individual ensmeble results')
ax1.set_xlabel("Accuracy")
ax1.set_ylabel("Ensemble weak Defense")

ax2 = task1PGD.plot.barh(x='label', y='accuracy',title = 'Accuracy of PGD AEs from Task1')
ax2.set_xlabel("Accuracy")
ax2.set_ylabel("Task1 PGD AE's")

ax3 = task1FGSM.plot.barh(x='label', y='accuracy',title = 'Accuracy of FGSM AEs from Task1')
ax3.set_xlabel("Accuracy")
ax3.set_ylabel("Task1 FGSM AE's")


