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
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
from utils.file import load_from_json
import tensorflow.keras as ker




def getMaxIndex(a):
    for i in range(len(a)):
        if a[i]==max(a):
            return i
#Prj Athena dir is the path to your project Athena
prjAthenaDir = '/home/isaac/working_directory/misc/project-athena'
#Directory to ensemble predictions folder
predictDir = prjAthenaDir + "/ourDataFiles/ensembleOuts_noraw"
#directory to AE's used for ensemble predictions
aeDir = prjAthenaDir + '/data2_genAEs_weakD'
#Directory to the clean tst data asswers
clean = np.load(prjAthenaDir + '/data/test_Label-mnist-clean.npy')
randomnums = np.load(prjAthenaDir + '/ourInfoSaves/ensPred_subset.npy')
#weak has been set to Task1 AE's not weak defs
weakWefDir= prjAthenaDir + '/generated_aes'

#Create matrixes of generated AE's, AE predictions, velues of AE predictions, 
# and cooresponding correct lean predictions

dirs = os.listdir(predictDir)
results = []
results += [file for file in dirs] 



cleanmat = np.zeros((1,10))
values = np.zeros(1)
predictMat = np.zeros((1,10))
aeMat = np.zeros((1,28,28,1))



#Get rid of all the [;400]'s to get full data from AE's
for filename in results:
    tt = np.load(predictDir + '/' + filename)
    ww = np.load(aeDir + '/' + filename.split('_',1)[1])
    for i in range(len(randomnums)):
        v=getMaxIndex(tt[i,:])
        values = np.vstack((values,v))
        c = clean[randomnums[i]]
        cleanmat = np.vstack((cleanmat,c))
        w = ww[randomnums[i]]
        w = np.expand_dims(w, axis=0)
        aeMat = np.vstack((aeMat,w))
    predictMat = np.vstack((predictMat,tt))
    
    
    print(filename)

values = values[1:]
predictMat = predictMat[1:]
aeMat = aeMat[1:]
cleanmat =  cleanmat[1:] 

#filter out nans from matrixes
values = np.asarray(values).astype(np.float32)
filtnan = np.argwhere(np.isnan(values))

values = np.delete(values, filtnan[:,0],0)
aeMat = np.delete(aeMat, filtnan[:,0],0)
cleanmat = np.delete(cleanmat, filtnan[:,0],0)
################################Uncomment out below if not using random dataset
# cleanmat = np.zeros((1,10))
# values = np.zeros(1)
# predictMat = np.zeros((1,10))
# aeMat = np.zeros((1,28,28,1))



# #Get rid of all the [;400]'s to get full data from AE's
# for filename in results:
#     tt = np.load(predictDir + '/' + filename)
#     ww = np.load(aeDir + '/' + filename.split('_')[1])
#     for i in range(400):
#         v=getMaxIndex(tt[i,:])
#         values = np.vstack((values,v))
#     predictMat = np.vstack((predictMat,tt[:400]))
#     aeMat = np.vstack((aeMat,ww[:400]))
#     cleanmat = np.vstack((cleanmat,clean[:400]))
#     print(filename)

# values = values[1:]
# predictMat = predictMat[1:]
# aeMat = aeMat[1:]
# cleanmat =  cleanmat[1:] 

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

#Train ML Model on 

dataTrain = aeMat[0:4000,:,:,:]
model = trainNewModel(aeMat, values)

dataTest = aeMat[4001:len(aeMat),:,:,:]
selfeval = model.evaluate(aeMat,  values, verbose=2)

#import benign examples to test 
benign = np.load(prjAthenaDir + "/" + '/data/test_BS-mnist-clean.npy')

cvalues = np.zeros((len(clean),1))
for i in range(len(clean)):
    cvalues[i]=getMaxIndex(clean[i,:])
    
benigneval = model.evaluate(benign, cvalues, verbose=2)

wddirs = os.listdir(weakWefDir)
wresults = []
wresults += [file for file in wddirs] 
wdaeMat = np.zeros((1,28,28,1))
wdcvalues = np.zeros(1)
for filename in wresults:
    wd = np.load(weakWefDir + '/' + filename)
    for i in range(400):
        w = getMaxIndex(clean[i,:])
        wdcvalues = np.vstack((wdcvalues,w))
    wdaeMat = np.vstack((wdaeMat,wd[:400]))
    
wdaeMat = wdaeMat[1:]    
wdcvalues = wdcvalues[1:]

weakdef = model.evaluate(wdaeMat, wdcvalues, verbose=2)

print('The loss and accuracy of the Weak def model vs itself is')
print(selfeval[0],selfeval[1])

print('The loss and accuracy of the model vs benighn examples is')
print(benigneval[0],benigneval[1])

print('The loss and accuracy of the model vs Task 1 gen AEs examples is')
print(weakdef[0],weakdef[1])
# def generate_subset(totalSize=4,number=2,doRandom=False,
#                     opath='default',doSave=True):
#     # gnerate a random subset
#     if(doRandom):
#         subset = random.sample(range(totalSize), number)
#     # generate a subset from the top of the data
#     else:
#         subset = [i for i in range(number)]
#     # create a list of the "other" indices
#     subsetElse = [i for i in range(totalSize)]
#     for i in subset:
#         subsetElse.remove(i)
#     # save the indices to .npy arrays so we can reference them later
#     if(doSave):
#         if(opath == 'default'):
#             odir = os.path.abspath(os.path.join('../ourInfoSaves'))
#             opath1 = odir+"/"+"subset.npy"
#             opath2 = odir+"/"+"subsetElse.npy"
#         else:
#             opath1 = opath[0]
#             opath2 = opath[1]
        
#         np.save(opath1, subset)
#         np.save(opath2, subsetElse)
        
#     return subset, subsetElse 