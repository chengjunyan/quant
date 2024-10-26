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
#import datetime as dt
#import time
import copy
from WindPy import *
import xlsxwriter
import xlwt 
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdate

##  oracle access to Wind database
# ora_connect_str = 'oracle://wind:wind@10.23.153.15:21010/wind'
# dbengine = db.create_engine(ora_connect_str, poolclass=db.pool.NullPool)
# dbconn=dbengine.connect()
# sql="select s_info_windcode,trade_dt,s_dq_settle,s_dq_close,fs_info_type from CIndexFuturesEODPrices where FS_INFO_TYPE=1 and s_info_windcode like \'%IF.CFE%\'"
# IFZL=pd.DataFrame()
# IFZL=pd.read_sql(sql,dbconn)
# IFZL=IFZL.sort_values(by='trade_dt',ascending='False')
# IFZL.to_excel(r'E:\IF主力合约.xlsx')
# sql="select s_info_windcode,trade_dt,s_dq_close from AIndexEODPrices where s_info_windcode='000300.SH'"
# HS300=pd.DataFrame()
# HS300=pd.read_sql(sql,dbconn)
# HS300=HS300.sort_values(by='trade_dt',ascending='False')
# HS300.to_excel(r'E:\沪深300指数.xlsx')

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


    



## Calculation MACD for tick  
def calculate_macd(prices, fast=12, slow=26, signal=9):
     EMA12 = EMA26 = DEA = 0
     macd_list = []
     signal_list = []
     histogram_list = [] 
     for i in range(len(prices)):
         if i == 0:
             EMA12=prices[i]
             EMA26=prices[i]
         else:
             EMA12=(prices[i]*2+EMA12*(fast-1))/(fast+1)
             EMA26=(prices[i]*2+EMA26*(slow-1))/(slow+1)
         DIF=EMA12-EMA26
         DEA=(DIF*2+DEA*(signal-1))/(signal+1)
         MACD=2*(DIF-DEA)      
         macd_list.append(MACD)
         signal_list.append(DEA)
         histogram_list.append(MACD-DEA)
     return macd_list, signal_list, histogram_list

##call option pricing
sigma=0
def call_BS(S,K,Sigma,r,T):
  d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2=d1-sigma*np.sqrt(T)
  return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
## put option pricing
def put_BS(S,K,sigma,r,T):
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


#TrueRange=goldprice_daily['TrueRange'].tolist()
#ATR=np.mean(TrueRange[-20:])
#High=goldprice['High'].tolist()
#Low=goldprice['Low'].tolist()
#Close_daily=goldprice_daily['Close'].tolist()
#MACD,Signal,histogram=calculate_macd(Close_daily)


#MA5=np.mean(Close[-5:])
#MA20=np.mean(Close[-20:])
#MA60=np.mean(Close[-60:])

#Gold_Max20=np.max(High[-20:])
#Gold_Min20=np.min(Low[-20:])



#print("MACD:", macd)
#print("Signal:", signal)
#print("Histogram:", histogram)    
    
#     sql="select s_info_windcode,trade_dt,s_dq_close,discount_rate from ChinaClosedFundEODPrice where s_info_windcode='"+fundcode[i]+"'"
#     result=pd.DataFrame()    
#     result=pd.read_sql(sql,dbconn)
#     result=result.sort_values(by='trade_dt',ascending='False')
#     fundcode2=fundcode[i][0:-2]+"OF"
#     sql="select f_info_windcode,price_date,f_nav_unit,f_nav_accumulated from ChinaMutualFundNAV where f_info_windcode='"+fundcode2+"'"
#     filename="E:\\"
#     filename=filename+fundcode[i]+".xlsx"
#     result.to_excel(filename)
#     result2=pd.DataFrame()    
#     result2=pd.read_sql(sql,dbconn)
#     result2=result2.sort_values(by='price_date',ascending='False')    
#     filename="E:\\"
#     filename=filename+fundcode2+".xlsx"
#     result2.to_excel(filename)



# fundcode=ChinaClosedFundList['s_info_windcode'].tolist() 

# sql="select s_info_windcode,trade_dt,s_dq_close,s_dq_avgprice,discount_rate from ChinaClosedFundEODPrice where trade_dt='20211112'"
### 中信29个行业指数读取

#industry_code=industry_citics['Wind代码'].tolist()   
#total=pd.DataFrame()
# for i in range(len(fundcode)):
#     sql="select s_info_windcode,trade_dt,s_dq_close,discount_rate from ChinaClosedFundEODPrice where s_info_windcode='"+fundcode[i]+"'"
#     result=pd.DataFrame()    
#     result=pd.read_sql(sql,dbconn)
#     result=result.sort_values(by='trade_dt',ascending='False')
#     fundcode2=fundcode[i][0:-2]+"OF"
#     sql="select f_info_windcode,price_date,f_nav_unit,f_nav_accumulated from ChinaMutualFundNAV where f_info_windcode='"+fundcode2+"'"
#     filename="E:\\"
#     filename=filename+fundcode[i]+".xlsx"
#     result.to_excel(filename)
#     result2=pd.DataFrame()    
#     result2=pd.read_sql(sql,dbconn)
#     result2=result2.sort_values(by='price_date',ascending='False')    
#     filename="E:\\"
#     filename=filename+fundcode2+".xlsx"
#     result2.to_excel(filename)
    #total[industry_code[i]]=result['s_dq_pctchange']
### 6个Wind行业风格指数+4个宽基行业指数读取

#industry_style=pd.read_excel(r'D:\style index.xlsx')
#industry_code=industry_style['Wind代码'].tolist()   
#for i in range(len(industry_code)):
#    sql="select s_info_windcode,trade_dt,s_dq_pctchange from AIndexEODPrices where trade_dt>='20110101' and s_info_windcode='"+industry_code[i]+"'"
#    result=pd.DataFrame()
#    result=pd.read_sql(sql,dbconn,index_col='trade_dt')
#    result=result.sort_values(by='trade_dt',ascending='False')
#    total[industry_code[i]]=result['s_dq_pctchange']
###得到29+6+4个行业的协方差矩阵    
#total_corr=total.corr()
#total_corr.to_excel(r'D:\total_corr.xlsx')
#total.to_excel(r'D:\total.xlsx')
##sql= "select f_info_windcode,f_info_name,f_info_firstinvesttype,f_info_setupdate from ChinaMutualFundDescription where f_issue_totalunit>=50 and f_info_firstinvesttype not like'%货币市场型%' and f_info_firstinvesttype not like'%债券%' and f_info_status<>'101002000' and f_info_name not like '%ETF%' and f_info_name not like '%配售%' and f_info_setupdate>='20190101'"
###result=result.sort_index(by='s_info_windcode',ascending=False)
##
###select F_INFO_WINDCODE from ChinaInhouseFundManager)'
# #读取待清洗文件
# f=open(r'D:\20200816-2.txt','r',encoding= 'utf-8')
# text=f.read()
# f.close()
# result=list()


