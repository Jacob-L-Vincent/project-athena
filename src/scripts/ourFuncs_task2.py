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
                    odir='default',doSave=True):
    
    # set output path to /ourInfoSaves as default value
    if(odir=='default'):
        odir = os.path.abspath(os.path.join('../../ourInfoSaves'))
        print(odir)
    
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
        np.save(odir+"/"+"subset.npy",subset)
        np.save(odir+"/"+"subsetElse.npy",subsetElse)
        
    return subset, subsetElse 