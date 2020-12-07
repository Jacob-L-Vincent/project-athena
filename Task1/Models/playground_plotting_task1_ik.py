#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:16:47 2020

@author: isaac
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



data = pd.read_csv("/home/isaac/working_directory/misc/project-athena/dataOut_task1.csv",index_col=0)

expandedData = pd.DataFrame(columns=["AE","Error_rate","defense"])

for index, row in data.iterrows():
    newrows={
            "AE":[ row["ae"],row["ae"],row["ae"] ],
            "Error_rate":[ row["UM"],row["BL"],row["Ensemble"] ],
            "defense":[ "UM","BL","Ensemble" ]
            }
    expandedData = expandedData.append(pd.DataFrame(newrows),ignore_index=True)



plt.figure(figsize=(10,6))
sns.barplot(x="AE", hue="defense", y="Error_rate",data=expandedData)
plt.show()


