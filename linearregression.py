# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:11:35 2019

@author: Kapil
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import math
dataset = pd.read_csv("mod_kidney_disease.csv")
## print(dataset)
reg = linear_model.LinearRegression();
median_age = math.floor(dataset.age.median())
median_bp = math.floor(dataset.bp.median())
median_sg = math.floor(dataset.sg.median())
median_al = math.floor(dataset.al.median())
median_su = math.floor(dataset.su.median())

## print(median_age)
## print(median_bp)
## print(median_sg)
## print(median_al)
## print(median_su)
dataset.age = dataset.age.fillna(median_age)
dataset.bp = dataset.bp.fillna(median_bp)
dataset.sg = dataset.sg.fillna(median_sg)
dataset.al = dataset.al.fillna(median_al)
dataset.su = dataset.su.fillna(median_su)

##dataset.classification[dataset.classification == "ckd"] = 1;
##dataset.classification[dataset.classification == "notckd"] = 0;
##print(dataset)
reg.fit(dataset[['age','bp','sg','al','su']],dataset.classification)
print(reg.coef_)