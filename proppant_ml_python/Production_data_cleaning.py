# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:58:01 2018
Monthly Production Data and Well Header Data Pre-processing
@author: kourui
"""

import pandas as pd
#Read Raw Monthly Production file
Rp = pd.read_csv('Mountain Gap DSU test Production Time Series.csv',\
                          parse_dates = ['Production Date'])
#Read Raw Header file
Rh = pd.read_csv('Mountain Gap DSU test Production Headers.csv',\
                 parse_dates = ["First Prod Date"])
Rp['API/UWI']=Rp['API/UWI'].astype('int64')
#Create Desired Monthly Production DataFrame
Dp  = Rp.iloc[:,[1,3,4,5,6]].copy()
del Rp 
Dp.rename(columns={'API/UWI':'API','Production Date':'Date',\
'Liquid (bbl)':'Liquid','Gas (mcf)':'Gas','Water (bbl)':'Water'}, \
inplace=True)
Dp ['API']='A_'+Dp .API.astype(str)
Dp ['Year']=Dp ['Date'].dt.year
Dp ['Month']=Dp ['Date'].dt.strftime('%b')

#Create Desired Header DataFrame
Dh = Rh.iloc[:,[0,7]].copy()
Dh.rename(columns={'API/UWI':'API',\
'Production Type':'Primary Product'},inplace = True)
Dh['API']=Dh.API.astype('int64')
Dh['API']='A_'+Dh.API.astype(str)
Dh['Primary Product']=Dh['Primary Product'].astype(str).str[0]
Dh['Well Name']=Rh['Well/Lease Name'].copy()+' '+Rh['Well Number'].copy()
Dh['Surface Latitude']=Rh['Surface Latitude (WGS84)'].copy()
Dh['Surface Longitude']=Rh['Surface Longitude (WGS84)'].copy()
Dh['Total Proppant (lbs)']='1000000'
Dh['Total Fluid (gals)']='1000000'
Dh['Depth Total Driller']=Rh['Measured Depth (TD)'].copy()
Dh['Lat Len Gross Perf Intvl']=Rh['Gross Perforated Interval'].copy()
Dh['Hole Direction']=Rh['Drill Type'].copy().str.replace('H','HORIZONTAL')
Dh['Operator Name']=Rh['Reported Operator'].copy().str.replace(r","," ")
Dh['County Name']=Rh['County/Parish'].copy().str.replace(r"\(.*\)","")
Dh['Province State Name']=Rh['State'].copy()
Dh['Status Current Name']=Rh['Producing Status'].copy()
Dh['Formation Producing Name']=Rh['Reservoir'].str.replace(r"\(.*\)","")
Dh['Depth True Vertical']=Rh['True Vertical Depth'].copy()
Dh['Depth True Vertical']=Dh['Depth True Vertical'].fillna(10000)
Dh['Reserve Category']='1PDP'
Dh['Month']=Rh['First Prod Date'].dt.strftime('%b').str.upper()
Dh['Year']=Rh['First Prod Date'].dt.year
Dh['TypeCurveArea']='Default'

del Rh

# Based on which file has less API numbers, filter the other file.
key_dp=Dp['API'].unique().astype(str)
key_dh=Dh['API'].unique().astype(str)
num_dp = len(key_dp)
num_dh = len(key_dh)
if num_dp<num_dh:
    print("Well Number (API) in Monthly file is (",num_dp, ") less than number \
of API in headers file (", num_dh,")")
    key_dpdf=pd.DataFrame(key_dp,columns= ['API'])
    Dlj = pd.merge(key_dpdf,Dh,how = 'left', on = 'API')
    Dlj.to_csv('One Line Input Data.CSV', index = False)
    Dp.to_csv('Monthly Input Data.CSV',columns = ['API',\
    'Year','Month','Liquid','Gas','Water'], index = False)
elif num_dp > num_dh:
    print("Well Number (API) in Monthly file is (",num_dp, ") more than number \
of API in headers file(", num_dh,")")
    Dlj = pd.merge(Dh,Dp,how = 'left', on = 'API')
    Dlj.to_csv('Monthly Input Data.CSV',columns = ['API',\
    'Year','Month','Liquid','Gas','Water'], index = False)
    Dh.to_csv('One Line Input Data.CSV', index = False)
else:
     print("Well Number (API) in Monthly file is (",num_dp, ") equal to number \
of API in headers file (", num_dh,")")
     Dp.to_csv('Monthly Input Data.CSV',columns = ['API',\
    'Year','Month','Liquid','Gas','Water'], index = False)
     Dh.to_csv('One Line Input Data.CSV', index = False)
     