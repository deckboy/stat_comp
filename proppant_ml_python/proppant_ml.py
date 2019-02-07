# -*- coding: utf-8 -*-
"""
Proppant_ML
@author: kourui
"""
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestRegressor

#read Input Data file
df = pd.read_excel('One Line Input Data-model.xlsx').drop('MD-TVD',axis = 1)
d = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,\
     'OCT':10,'NOV':11,'DEC':12}
df.Month=df.Month.map(d)
df['Mon_int'] = (df.Year-2000)*12+df.Month

#numerical and catagorical feature
numcols = ['Surface Latitude','Surface Longitude','Total Proppant (lbs)',\
          'Total Fluid (gals)','Depth Total Driller',\
          'Lat Len Gross Perf Intvl','Depth True Vertical',\
          'Mon_int']
catcols = [x for x in df.columns if x not in numcols]


#missing value visualization (numerical features only)
missing_df = (df[numcols]==0).sum(axis=0).to_frame().reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')


#missing_df = missing_df[missing_df.column_name!='Surface Longitude']
ind = np.arange(missing_df.shape[0]) 
fig,ax = plt.subplots(figsize=(6,3))
rects = ax.barh(ind, missing_df.missing_count.values, color='yellow')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column, total = %s" \
             %(df.shape[0]))


#label encoding for operator name
le_op = preprocessing.LabelEncoder()
df['optor_le'] = le_op.fit_transform(df['Operator Name'])
list(le_op.inverse_transform(df['optor_le']))


#Formation Producing
F_filter = df['Formation Producing Name']==0
data_full =df[~F_filter]
data_x = data_full[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int']]
data_y = data_full['Formation Producing Name']
le_f = preprocessing.LabelEncoder()
data_y_le=le_f.fit_transform(data_y)
list(le_f.inverse_transform(data_y_le))

pred = df[F_filter]
pred_x = pred[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int']]

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y_le, test_size=0.33, random_state=42)

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
}


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
   
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models
 
 
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))
   
dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 7)
display_dict_models(dict_models)


rf=RandomForestClassifier()
rf=rf.fit(data_x,data_y_le)
pred_y = rf.predict(pred_x)
result=pd.Series(le_f.inverse_transform(pred_y))
result.index = df[F_filter].index
df['Formation Producing Name']=df['Formation Producing Name'][~F_filter].append([result])
df['Formation_le']=le_f.fit_transform(df['Formation Producing Name'])

# Regression for MD
MD_filter = df['Depth Total Driller']==0
data_full =df[~MD_filter]
data_x = data_full[['Surface Latitude','Surface Longitude','optor_le','Mon_int','Formation_le']]
data_y = data_full['Depth Total Driller']
pred = df[MD_filter]
pred_x = pred[['Surface Latitude','Surface Longitude','optor_le','Mon_int','Formation_le']]

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
rf.fit(X_train, Y_train)
rf.score(X_test,Y_test)
result_1=rf.predict(X_test)
result = pd.Series(rf.predict(pred_x))
result.index = df[MD_filter].index
df['Depth Total Driller']=df['Depth Total Driller'][~MD_filter].append([result])




# Regression for TVD
tvd_filter = df['Depth True Vertical']==0
data_full =df[~tvd_filter]
data_x = data_full[['Surface Latitude','Surface Longitude','optor_le','Mon_int','Formation_le','Depth Total Driller']]
data_y = data_full['Depth True Vertical']
pred = df[tvd_filter]
pred_x = pred[['Surface Latitude','Surface Longitude','optor_le','Mon_int','Formation_le','Depth Total Driller']]

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
rf.fit(X_train, Y_train)
rf.score(X_test,Y_test)
result = pd.Series(rf.predict(pred_x))
result.index = df[tvd_filter].index
df['Depth True Vertical']=df['Depth True Vertical'][~tvd_filter].append([result])

df['MD-TVD']=df['Depth Total Driller']-df['Depth True Vertical']


# Regression for Lateral length
LL_filter = df['Lat Len Gross Perf Intvl']==0
data_full =df[~LL_filter]
data_x = data_full[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int','Formation_le','Depth Total Driller','Depth True Vertical','MD-TVD']]
data_y = data_full['Lat Len Gross Perf Intvl']
pred = df[LL_filter]
pred_x = pred[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int','Formation_le','Depth Total Driller','Depth True Vertical','MD-TVD']]

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
rf.fit(X_train, Y_train)
rf.score(X_test,Y_test)
result = pd.Series(rf.predict(pred_x))
result.index = df[LL_filter].index
df['Lat Len Gross Perf Intvl']=df['Lat Len Gross Perf Intvl'][~LL_filter].append([result])

# Regression for Total Proppant
TP_filter = df['Total Proppant (lbs)']==0
data_full =df[~TP_filter]
data_x = data_full[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int','Formation_le','Depth Total Driller','Depth True Vertical','MD-TVD',\
'Lat Len Gross Perf Intvl']]
data_y = data_full['Total Proppant (lbs)']
pred = df[TP_filter]
pred_x = pred[['Surface Latitude','Surface Longitude','optor_le',\
'Mon_int','Formation_le','Depth Total Driller','Depth True Vertical','MD-TVD',\
'Lat Len Gross Perf Intvl']]

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
rf.fit(X_train, Y_train)
rf.score(X_test,Y_test)
result = pd.Series(rf.predict(pred_x))
result.index = df[TP_filter].index
df['Total Proppant (lbs)']=df['Total Proppant (lbs)'][~TP_filter].append([result])








'''

from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

## Works on regression
def fillna_knn_reg( df, base, target, n_neighbors = 5 ):
    cols = base + [target]
    X_train = df[cols]
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train[base].values.reshape(-1, 1))
    rescaledX = scaler.transform(X_train[base].values.reshape(-1, 1))

    X_train = rescaledX[df[target].notnull()]
    Y_train = df.loc[df[target].notnull(),target].values.reshape(-1, 1)

    knn = KNeighborsRegressor(n_neighbors, n_jobs = -1)    
    # fitting the model
    knn.fit(X_train, Y_train)
    # predict the response
    X_test = rescaledX[df[target].isnull()]
    pred = knn.predict(X_test)
    df.loc[df_train[target].isnull(),target] = pred
    
 '''   

















