#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:45:10 2020

@author: jvincent
"""
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter

path = r'/home/jvincent/Desktop/athena ipk/project-athena/dataOut_task1.csv'

df = pd.read_csv(path, header = None, index_col = None, parse_dates =  True)
df.columns = ['num','ae','UM','BL','Ensemble']
df = df.drop([0])
#eps fgsm['nes'].iloc[x]= "epsof"


fgsm = df.iloc[21::]
i = 0
temp = []
for x in fgsm['ae']:
    temp.append(float(x.split('f')[2].replace(',',"")))
    i+=0
fgsm['nes']=temp

pgdesp = df.iloc[0:10] 

i = 0
temp = []
for x in pgdesp['ae']:
    temp.append(float(x.split('f')[1].replace(',',"")))
    i+=0
pgdesp['nes']=temp

pgditer =df.iloc[11:21] 

i = 0
temp = []
for x in pgditer['ae']:
    temp.append(float(x.split('r')[1].replace(',',"")))
    i+=0
pgditer['nes']=temp

fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig.tight_layout()

ax1.plot(fgsm['nes'],np.array(fgsm['BL'],dtype=float))
ax1.plot(fgsm['nes'],np.array(fgsm['UM'],dtype=float))
ax1.plot(fgsm['nes'],np.array(fgsm['Ensemble'],dtype=float)) 
ax1.set_ylim(0,1)
ax1.set_ylabel("Error Rate")
ax1.set_xlabel('FGSM, Changing Epsilon Value')


ax2.plot(pgdesp['nes'],np.array(pgdesp['BL'],dtype=float))
ax2.plot(pgdesp['nes'],np.array(pgdesp['UM'],dtype=float))
ax2.plot(pgdesp['nes'],np.array(pgdesp['Ensemble'],dtype=float)) 
ax2.set_ylim(0,1)
ax2.set_ylabel('Error Rate')
ax2.set_xlabel('PGD, Changing Epsilon Value, Max Iteration set to 10')


ax3.plot(pgditer['nes'],np.array(pgditer['BL'],dtype=float))
ax3.plot(pgditer['nes'],np.array(pgditer['UM'],dtype=float))
ax3.plot(pgditer['nes'],np.array(pgditer['Ensemble'],dtype=float)) 
ax3.set_ylim(0,1)
ax3.set_ylabel('Error Rate')
ax3.set_xlabel('PGD, Changing Max Iteration Value, Epsilons set to 0.3')


#plt.xlim(dates.iloc[1],dates.iloc[-1])
#ax1.set_ylim(0,1)
#ax1.axhline(color='r')
#ax1.set_title("Correct Guess")
#ax1.set_ylabel('Velocity (m/yr)')

