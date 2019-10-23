# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:34:12 2019

@author: Kapil
"""
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import math

df = pd.read_csv("kidney_disease.csv")

for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    df[i].fillna(df[i].mean(),inplace=True)

df['rbc'].fillna('normal',inplace=True)
df['pc'].fillna('normal',inplace=True)
df['pcc'].fillna('notpresent',inplace=True)
df['ba'].fillna('notpresent',inplace=True)
df['htn'].fillna('no',inplace=True)
df['dm'] = df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
df['dm'].fillna('no',inplace=True)
df['cad'] = df['cad'].replace(to_replace='\tno',value='no')
df['cad'].fillna('no',inplace=True)
df['appet'].fillna('good',inplace=True)
df['pe'].fillna('no',inplace=True)
df['ane'].fillna('no',inplace=True)
df['cad'] = df['cad'].replace(to_replace='ckd\t',value='ckd')
##df.dropna(how='any',inplace = True)

df.rbc.replace(to_replace=["normal","abnormal"],value=[0,1],inplace = True)
df.pc.replace(to_replace=["normal","abnormal"],value = [0,1],inplace = True)
df.pcc.replace(to_replace=["present","notpresent"],value=[1,0],inplace = True)
df.ba.replace(to_replace=["present","notpresent"],value=[1,0],inplace = True)
df.htn.replace(to_replace=["yes","no"],value=[1,0],inplace = True)
df.dm.replace(to_replace=["yes","no"],value=[1,0],inplace = True)
df.cad.replace(to_replace=["yes","no"],value=[1,0],inplace = True)
df.pe.replace(to_replace=["yes","no"],value=[1,0],inplace = True)
df.ane.replace(to_replace=["yes","no"],value=[1,0],inplace = True)
df.appet.replace(to_replace=["good","poor"],value=[1,0],inplace= True)

for i in df.columns:
    df[i] = MinMaxScaler().fit_transform(df[i].astype(float).values.reshape(-1, 1))

reg = linear_model.LinearRegression();
lreg = linear_model.LogisticRegression();
reg.fit(df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']],df.classification)
score = reg.score(df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']],df.classification)
lreg.fit(df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']],df.classification)
lscore = lreg.score(df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']],df.classification)

print(reg.coef_)
print(score)
print(lreg.coef_)
print(lscore)
##print(reg.predict([[58,80,1.025,0,0,0,0,0,0,131,18,	1.1,141,3.5,15.8,53,6800,6.1,0,0,0,1,0,0
##]]))
##print(lreg.predict([[58,80,1.025,0,0,0,0,0,0,131,18,	1.1,141,3.5,15.8,53,6800,6.1,0,0,0,1,0,0
##]]))
##print(lreg.predict([[65,100,1.015,0,0,NaN,	0,	0,	0,	90,	98	2.5	,NaN	,NaN,	9.1,	28,	5500,	3.6,	1,	0,	0,	1,	0,	0
##]]))



