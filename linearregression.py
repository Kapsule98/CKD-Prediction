# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:11:35 2019

@author: Kapil
"""

## importing libraries

import pandas as pd
from sklearn import linear_model
import math

## reading the dataset file
dataset = pd.read_csv("mod_kidney_disease.csv")
## print(dataset)

## initialising reg as LinearRegression model
reg = linear_model.LinearRegression();

##finding median of parameters
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

## Linear regression model can only be fitted to 
## complete data but we have some fields as NA 
## in the dataset so we need to correct it
## we use median value of all the values to
## replace NA in the dataset

dataset.age = dataset.age.fillna(median_age)
dataset.bp = dataset.bp.fillna(median_bp)
dataset.sg = dataset.sg.fillna(median_sg)
dataset.al = dataset.al.fillna(median_al)
dataset.su = dataset.su.fillna(median_su)


##dataset.classification[dataset.classification == "ckd"] = 1;
##dataset.classification[dataset.classification == "notckd"] = 0;
##print(dataset)

## Fitting the LR model on the dataset

reg.fit(dataset[['age','bp','sg','al','su']],dataset.classification)

## After we fit the LR model we get the 
## coefficients (weights) of the different parameters
print(reg.coef_)
