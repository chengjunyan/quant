# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:30:17 2020

@author: skcjy
"""

import cx_Oracle
import numpy as np
import csv
import os
import pandas as pd
import sqlalchemy as db
from pandas import DataFrame
from scipy.optimize import minimize
from scipy.stats import norm
import copy
from WindPy import *
import xlsxwriter
import xlwt 

## data input and caculate cov and log return
os.environ['NLS_LANG']='SIMPLIFIED CHINESE_CHINA.UTF8'
pctchange=pd.DataFrame()
asset_data=pd.read_excel(r'D:\Risk Parity.xlsx')
asset_data.index=pd.to_datetime(asset_data['Date'],format='%Y-%m-%d')

asset_list=['H11017.CSI','H11074.CSI','H00300.CSI','H00905.CSI','HSI.HI','SPX.GI','NDX.GI','518880.OF']
ret=pd.DataFrame()

def get_ret(code):
  close=asset_data[code]
  close.name=code
  ret=np.log(close/close.shift(1)) #daily log return
  return ret
    
for code in asset_list:
    ret_=get_ret(code)
    ret=pd.concat([ret,ret_],axis=1)
ret=ret.dropna()
R_cov=ret.cov()  #calculate cov for multi asset
cov=np.array(R_cov)

### multi-asset portfolio
def risk_budget_objective(weights,cov):
    weights=np.array(weights)
    sigma=np.sqrt(np.dot(weights, np.dot(cov,weights)))
    MRC=np.dot(cov,weights)/sigma
    TRC=weights*MRC 
    delta_TRC=[sum((i-TRC)**2) for i in TRC]
    return sum(delta_TRC)

def total_weight_constraint(x):
    return np.sum(x)-1

x0=np.ones(cov.shape[0])/cov.shape[0]
bnds=tuple((0,None) for x in x0)
cons=({'type':'eq','fun':total_weight_constraint})
options={'disp':False,'maiter':1000,'ftol':1e-20}

solution= minimize(risk_budget_objective, x0,args=(cov),bounds=bnds,constraints=cons,method='SLSQP',options=options)

final_weights=solution.x

for i in range(len(final_weights)):
    print(f'{final_weights[i]:.2%} invest in {asset_list[i]}')
