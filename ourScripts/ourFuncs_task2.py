#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2 functions
Created on Thu Nov 12 15:55:23 2020

@author: isaac
"""

import random
import numpy as np
import os

# function to make a list of indices to subset the benign samples and get the
#  corresponding unused indices as well
def generate_subset(totalSize=4,number=2,doRandom=False,
                    opath='default',doSave=True):
    # gnerate a random subset
    if(doRandom):
        subset = random.sample(range(totalSize), number)
    # generate a subset from the top of the data
    else:
        subset = [i for i in range(number)]
    # create a list of the "other" indices
    subsetElse = [i for i in range(totalSize)]
    for i in subset:
        subsetElse.remove(i)
    # save the indices to .npy arrays so we can reference them later
    if(doSave):
        if(opath == 'default'):
            odir = os.path.abspath(os.path.join('../ourInfoSaves'))
            opath1 = odir+"/"+"subset.npy"
            opath2 = odir+"/"+"subsetElse.npy"
        else:
            opath1 = opath[0]
            opath2 = opath[1]
        
        np.save(opath1, subset)
        np.save(opath2, subsetElse)
        
    return subset, subsetElse 


def getMaxIndex(a):
    for i in range(len(a)):
        if a[i]==max(a):
            return i
    