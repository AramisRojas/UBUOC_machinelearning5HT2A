# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:44:47 2020

@author: usuario
"""

#*||Importing data||*
import os 
os.chdir(r"C:/Users/usuario/OneDrive/EstadisticaUOC/4-SEMESTRE/TFM/Datos_recuperados_ChEMBL")

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

import pandas as pd
df = pd.read_csv("5ht2a_definitive_nosalts.csv")


#*||Cbind of inchikeys to dataframe. Drop duplicates||*
import numpy as np

df_2 = np.genfromtxt(fname="inchikeys_2.txt", dtype="str", skip_header=1)

df_2df = pd.DataFrame(data=df_2, columns=["InChIKey_notation"])

df_final = pd.concat([df, df_2df], axis=1)

df_final.drop_duplicates(subset="InChIKey_notation",
                         keep = 'first', inplace = True)

df_final.to_csv("results_unique_p5.csv", index=False)

# =============================================================================
# *****FINGERPRINTS*****
# =============================================================================
#conda create -c rdkit -n my-rdkit-env rdkit
#conda activate my-rdkit-env
#conda deactivate

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator

#df_final.head()

#*Morgan fingerprints by default*
PandasTools.AddMoleculeColumnToFrame(df_final,smilesCol='canonical_smiles')

mfps = rdFingerprintGenerator.GetFPs(list(df_final['ROMol']))

df_final['MFPS'] = mfps

#*MACCS fingerprints*
from rdkit.Chem import MACCSkeys

romol_li = list(df_final['ROMol'])

maccs_fps = [MACCSkeys.GenMACCSKeys(x) for x in romol_li]

df_final['MACCS_FPS'] = maccs_fps

#*Morgan fingerprints with AllChem*
from rdkit.Chem import AllChem

morgan_fps2 = [AllChem.GetMorganFingerprint(x,2) for x in romol_li] #Very important: give the radius.
morgan_fps3 = [AllChem.GetMorganFingerprint(x,3) for x in romol_li]

df_final['MorganAllChem_FPS'] = morgan_fps2
df_final['Morgan3'] = morgan_fps3

#*Topological fingerprints*
topol_fps = [Chem.RDKFingerprint(x) for x in romol_li]

df_final['Topol_FPS'] = topol_fps

df_final.to_csv("results_fps_wp5.csv", index=False)

import pandas as pd
df_final_regr = df_final.dropna()
df_final_regr.reset_index(drop=True)


#New dataset with outliers to re-try models (from graphics_script.py)
#df_final_wo_outs = copy 

# =============================================================================
# REGRESSIONS
# MORGAN-2
# =============================================================================
# Original data with outliers: df_final_regr
#New dataset without outliers to re-try models (from graphics_script.py): df_final_wo_outs = copy
df_final_wo_outs = df_final_wo_outs.dropna()
m2_pch = df_final_wo_outs['MFPS']
y_pch = df_final_wo_outs['pchembl_value']

#*Split data into train and test*
from sklearn.model_selection import train_test_split

m2_pch_split = train_test_split(m2_pch, train_size=0.80, random_state=12345)
m2_pch_train = m2_pch_split[0]
m2_pch_test = m2_pch_split[1]

y_pch_split = train_test_split(y_pch, train_size=0.80, random_state=12345)

y_pch_train = y_pch_split[0]
y_pch_train = pd.DataFrame(y_pch_train)

y_pch_test = y_pch_split[1]
y_pch_test = pd.DataFrame(y_pch_test)

y_pch_train_array = np.ravel(np.array(y_pch_train))

#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
m2_pch_train_list = list(m2_pch_split[0])
m2_pch_train_strs = [BitVectToText(x) for x in m2_pch_train_list]

m2_pch_test_list = list(m2_pch_split[1])
m2_pch_test_strs = [BitVectToText(x) for x in m2_pch_test_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

m2_pch_train_strs_broken = breaksy(m2_pch_train_strs) #print(morg_2_train_strs_broken.dtypes)
m2_pch_test_strs_broken = breaksy(m2_pch_test_strs)


# =============================================================================
# ||*Linear regresions*||
# =============================================================================
def adjusted_r2 (r2, n, p):
    return(1-(1-r2))*((n-(1))/(n-p-(1)))
#it is a modification of R2 that adjusts for the number of explanatory
#terms in a model (p) relative to the number of data points (n).
    

# =============================================================================
# Simple
# =============================================================================
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(m2_pch_train_strs_broken, y_pch_train)
y_pch_pred_reg_m2 = regr.predict(m2_pch_test_strs_broken)

r2_linreg = r2_score(y_pch_test, y_pch_pred_reg_m2)
adjR2_linreg = adjusted_r2(r2_linreg, 2048, len(y_pch_test))
mse_linreg = mean_squared_error(y_pch_test, y_pch_pred_reg_m2)

print('R^2 score: ',r2_linreg)
print('Adjusted R^2 score: ',adjR2_linreg)
print('Mean squared error: ',mse_linreg)

# =============================================================================
# Ridge
# =============================================================================
from sklearn.linear_model import Ridge

def adjusted_r2 (r2, n, p):
    return(1-(1-r2))*((n-(1))/(n-p-(1))) #it is a modification of R2 that adjusts for the number of explanatory terms in a model (p) relative to the number of data points (n).

def ridger():
    coefs = [0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]
    for i in coefs:
        ridge = Ridge(alpha=i, fit_intercept=True, normalize=True)
        ridge.fit(m2_pch_train_strs_broken, y_pch_train)
        preds = ridge.predict(m2_pch_test_strs_broken)
        scores = r2_score(y_pch_test, preds)
        #adj_scores = adjusted_r2(scores, 2048, len(y_pch_test))
        print(round(scores, 3))
print(ridger())

ridger5 = Ridge(alpha=2, fit_intercept=True, normalize=True)
ridger5.fit(m2_pch_train_strs_broken, y_pch_train)
y_pch_pred_ridge5_m2 = ridger5.predict(m2_pch_test_strs_broken)

r2_ridgereg5 = r2_score(y_pch_test, y_pch_pred_ridge5_m2)
adjR2_ridgereg5 = adjusted_r2(r2_ridgereg5, 2048, len(y_pch_test))
mse_ridgereg5 = mean_squared_error(y_pch_test, y_pch_pred_ridge5_m2)

print('R^2 score: ',r2_ridgereg5)
print('Adjusted R^2 score: ',adjR2_ridgereg5)
print('Mean squared error: ',mse_ridgereg5)

#10-fold-CV
#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
m2_pch_list = list(m2_pch)
m2_pch_strs = [BitVectToText(x) for x in m2_pch_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

m2_pch_strs_broken = breaksy(m2_pch_strs) #print(morg_2_train_strs_broken.dtypes)

from sklearn.model_selection import KFold
scores = []
ridger5 = Ridge(alpha=2, fit_intercept=True, normalize=True)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_pch_strs_broken.iloc[train_index], m2_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    ridger5.fit(X_train, y_train)
    scores.append(ridger5.score(X_test, y_test))
    
print(np.mean(scores))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
ridger5 = Ridge(alpha=5)
visualizer = CVScores(ridger5, cv=cv, scoring='r2')
visualizer.fit(m2_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#Instantiate the linear model and visualizer
#from yellowbrick.regressor import ResidualsPlot

#visr5 = ResidualsPlot(ridger5)
#visr5.fit(m2_pch_train_strs_broken, y_pch_train)  # Fit the training data to the visualizer
#visr5.score(m2_pch_test_strs_broken, y_pch_test)  # Evaluate the model on the test data
#visr5.show() # Finalize and render the figure

#from yellowbrick.regressor import PredictionError
#ridger5 = Ridge(alpha=5)
#visualizer = PredictionError(ridger5)

#visualizer.fit(m2_pch_train_strs_broken, y_pch_train)  # Fit the training data to the visualizer
#visualizer.score(m2_pch_test_strs_broken, y_pch_test)  # Evaluate the model on the test data
#visualizer.show()                 # Finalize and render the figure

# =============================================================================
# Ridge-CrossValidation (Leave One-Out/Generalized)
# =============================================================================
from sklearn.linear_model import RidgeCV

ridgerCV = RidgeCV(alphas=np.array([0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]), store_cv_values=True) #RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=None, fit_intercept=False,
    #gcv_mode=None, normalize=False, scoring=None, store_cv_values=False)
ridgerCV.fit(m2_pch_train_strs_broken, y_pch_train)
y_pch_pred_ridgeCV_m2 = ridgerCV.predict(m2_pch_test_strs_broken)

r2_ridgeregCV = r2_score(y_pch_test, y_pch_pred_ridgeCV_m2)
adjR2_ridgeregCV = adjusted_r2(r2_ridgeregCV, 2048, len(y_pch_test))
mse_ridgeregCV = mean_squared_error(y_pch_test, y_pch_pred_ridgeCV_m2)

print('R^2 score: ',r2_ridgeregCV)
print('Adjusted R^2 score: ',adjR2_ridgeregCV)
print('Mean squared error: ',mse_ridgeregCV)

#W/cv=5
ridgerCV5 = RidgeCV(alphas=np.array([0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]), cv=5) #RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=None, fit_intercept=False,
    #gcv_mode=None, normalize=False, scoring=None, store_cv_values=False)
ridgerCV5.fit(m2_pch_train_strs_broken, y_pch_train)
y_pch_pred_ridgeCV5_m2 = ridgerCV5.predict(m2_pch_test_strs_broken)

r2_ridgeregCV5 = r2_score(y_pch_test, y_pch_pred_ridgeCV5_m2)
adjR2_ridgeregCV5 = adjusted_r2(r2_ridgeregCV5, 2048, len(y_pch_test))
mse_ridgeregCV5 = mean_squared_error(y_pch_test, y_pch_pred_ridgeCV5_m2)

print('R^2 score: ',r2_ridgeregCV5)
print('Adjusted R^2 score: ',adjR2_ridgeregCV5)
print('Mean squared error: ',mse_ridgeregCV5)

# =============================================================================
# Bayesian "ridge" regression
# =============================================================================
#https://stats.stackexchange.com/questions/328614/estimation-of-bayesian-ridge-regression
from sklearn.linear_model import BayesianRidge

ridger_bay = BayesianRidge(compute_score=True, verbose=True)
#BayesianRidge(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)[source]¶
ridger_bay.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_ridgebay_m2 = ridger_bay.predict(np.array(m2_pch_test_strs_broken))

r2_ridgerbay = r2_score(y_pch_test, y_pch_pred_ridgeCV5_m2)
adjR2_ridgerbay = adjusted_r2(r2_ridgeregCV5, 2048, len(y_pch_test))
mse_ridgerbay = mean_squared_error(y_pch_test, y_pch_pred_ridgeCV5_m2)

print('R^2 score: ',r2_ridgerbay)
print('Adjusted R^2 score: ',adjR2_ridgerbay)
print('Mean squared error: ',mse_ridgerbay)

print(ridger_bay.alpha_)
print(ridger_bay.lambda_)

def ridgerBayes():
    alphabay = [2,3,4,5,6]
    for i in alphabay:
        ridger_bay = BayesianRidge(alpha_1=i, compute_score=True, verbose=True)
        ridger_bay.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
        preds = ridger_bay.predict(m2_pch_test_strs_broken)
        scores = r2_score(y_pch_test, preds)
        adj_scores = adjusted_r2(scores, 2048, len(y_pch_test))
        print(round(adj_scores, 3))
print(ridgerBayes())

# =============================================================================
# ||*Support Vector Machine (regression)*||
# =============================================================================
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
from sklearn import svm

#RBF & C=1.0
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
#SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
svr.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_svr_c1_m2 = svr.predict(np.array(m2_pch_test_strs_broken))

r2_svr = r2_score(y_pch_test, y_pch_pred_svr_c1_m2)
adjR2_svr = adjusted_r2(r2_svr, 2048, len(y_pch_test))
mse_svr = mean_squared_error(y_pch_test, y_pch_pred_svr_c1_m2)

print('R^2 score: ',r2_svr)
print('Adjusted R^2 score: ',adjR2_svr)
print('Mean squared error: ',mse_svr)

#10-fold-CV
from sklearn.model_selection import KFold
scores2 = []
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_pch_strs_broken.iloc[train_index], m2_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    svr.fit(X_train, y_train)
    scores2.append(svr.score(X_test, y_test))
    
print(np.mean(scores2))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
visualizer = CVScores(svr, cv=cv, scoring='r2')
visualizer.fit(m2_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

from sklearn.model_selection import cross_val_score
cross_val_score(svr,  m2_pch_strs_broken, y_pch, cv=10)

#RBF & C=10.0
svr10 = svm.SVR(kernel='rbf', C=10.0, verbose=True)
#SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
svr10.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_svr10_m2 = svr.predict(np.array(m2_pch_test_strs_broken))

r2_svr10 = r2_score(y_pch_test, y_pch_pred_svr10_m2)
adjR2_svr10 = adjusted_r2(r2_svr10, 2048, len(y_pch_test))
mse_svr10 = mean_squared_error(y_pch_test, y_pch_pred_svr10_m2)

print('R^2 score: ',r2_svr10)
print('Adjusted R^2 score: ',adjR2_svr10)
print('Mean squared error: ',mse_svr10)

#Poly, degree=3, C=1.0
svr_poly = svm.SVR(kernel='poly', degree=3, C=1.0, verbose=True)
svr_poly.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_svrpoly_c1 =  svr_poly.predict(np.array(m2_pch_test_strs_broken))

r2_svrpoly = r2_score(y_pch_test, y_pch_pred_svrpoly_c1)
adjR2_svrpoly = adjusted_r2(r2_svrpoly, 2048, len(y_pch_test))
mse_svrpoly = mean_squared_error(y_pch_test, y_pch_pred_svrpoly_c1)

print('R^2 score: ',r2_svrpoly)
print('Adjusted R^2 score: ',adjR2_svrpoly)
print('Mean squared error: ',mse_svrpoly)

#Poly, degree=3, C=10.0
svr_poly4 = svm.SVR(kernel='poly', degree=4, C=10.0, verbose=True)
svr_poly4.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_svrpoly4_c1 =  svr_poly.predict(np.array(m2_pch_test_strs_broken))

r2_svrpoly4 = r2_score(y_pch_test, y_pch_pred_svrpoly4_c1)
adjR2_svrpoly4 = adjusted_r2(r2_svrpoly4, 2048, len(y_pch_test))
mse_svrpoly4 = mean_squared_error(y_pch_test, y_pch_pred_svrpoly4_c1)

print('R^2 score: ',r2_svrpoly4)
print('Adjusted R^2 score: ',adjR2_svrpoly4)
print('Mean squared error: ',mse_svrpoly4)

# =============================================================================
# ||*Gradient boosting regressor*||
# =============================================================================

#(*, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
#criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
#min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
#min_impurity_split=None, init=None, random_state=None, max_features=None,
#alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
#validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0

#lr=0.1
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_1.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_1 = GBR_pch_1.predict(np.array(m2_pch_test_strs_broken))

r2_GBR1 = r2_score(y_pch_test, y_pch_pred_GBR_1)
adjR2_GBR1 = adjusted_r2(r2_GBR1, 2048, len(y_pch_test))
mse_GBR1 = mean_squared_error(y_pch_test, y_pch_pred_GBR_1)

print('R^2 score: ',r2_GBR1)
print('Adjusted R^2 score: ',adjR2_GBR1)
print('Mean squared error: ',mse_GBR1)

#lr=0.25
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_25 = GradientBoostingRegressor(loss='ls', learning_rate=0.25, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_25.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_25 = GBR_pch_25.predict(np.array(m2_pch_test_strs_broken))

r2_GBR25 = r2_score(y_pch_test, y_pch_pred_GBR_25)
adjR2_GBR25 = adjusted_r2(r2_GBR25, 2048, len(y_pch_test))
mse_GBR25 = mean_squared_error(y_pch_test, y_pch_pred_GBR_25)

print('R^2 score: ',r2_GBR25)
print('Adjusted R^2 score: ',adjR2_GBR25)
print('Mean squared error: ',mse_GBR25)

#lr=0.5
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_5 = GradientBoostingRegressor(loss='ls', learning_rate=0.5, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_5.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_5 = GBR_pch_5.predict(np.array(m2_pch_test_strs_broken))

r2_GBR5 = r2_score(y_pch_test, y_pch_pred_GBR_5)
adjR2_GBR5 = adjusted_r2(r2_GBR5, 2048, len(y_pch_test))
mse_GBR5 = mean_squared_error(y_pch_test, y_pch_pred_GBR_5)

print('R^2 score: ',r2_GBR5)
print('Adjusted R^2 score: ',adjR2_GBR5)
print('Mean squared error: ',mse_GBR5)

#*10-fold-CV
from sklearn.model_selection import KFold
scores3 = []
GBR_pch_1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_pch_strs_broken.iloc[train_index], m2_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    GBR_pch_1.fit(X_train, y_train)
    scores3.append(GBR_pch_1.score(X_test, y_test))
    
print(np.mean(scores3))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
GBR_pch_5 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
visualizer = CVScores(GBR_pch_5, cv=cv, scoring='r2')
visualizer.fit(m2_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#LAD lr=0.75
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_lad1 = GradientBoostingRegressor(loss='lad', learning_rate=0.75, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_lad1.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_lad1 = GBR_pch_lad1.predict(np.array(m2_pch_test_strs_broken))

r2_GBRlad1 = r2_score(y_pch_test, y_pch_pred_GBR_lad1)
adjR2_GBRlad1 = adjusted_r2(r2_GBRlad1, 2048, len(y_pch_test))
mse_GBRlad1 = mean_squared_error(y_pch_test, y_pch_pred_GBR_lad1)

print('R^2 score: ',r2_GBRlad1)
print('Adjusted R^2 score: ',adjR2_GBRlad1)
print('Mean squared error: ',mse_GBRlad1)

# =============================================================================
# ||*Random Forest regression*||
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_regr.fit(np.array(m2_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_rf = rf_regr.predict(np.array(m2_pch_test_strs_broken))

r2_rf = r2_score(y_pch_test, y_pch_pred_rf)
adjR2_rf = adjusted_r2(r2_rf, 2048, len(y_pch_test))
mse_rf = mean_squared_error(y_pch_test, y_pch_pred_rf)

print('R^2 score: ',r2_rf)
print('Adjusted R^2 score: ',adjR2_rf)
print('Mean squared error: ',mse_rf)

#*10-fold-CV
from sklearn.model_selection import KFold
scores4 = []
rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_pch_strs_broken.iloc[train_index], m2_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    rf_regr.fit(X_train, y_train)
    scores4.append(rf_regr.score(X_test, y_test))
    
print(np.mean(scores4))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_regr, cv=cv, scoring='r2')
visualizer.fit(m2_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

# =============================================================================
# REGRESSIONS
# MACCS
# =============================================================================

maccs_pch = df_final_wo_outs['MACCS_FPS']
y_pch = df_final_wo_outs['pchembl_value']

#*Split data into train and test*
from sklearn.model_selection import train_test_split

maccs_pch_split = train_test_split(maccs_pch, train_size=0.80, random_state=12345)
maccs_pch_train = maccs_pch_split[0]
maccs_pch_test = maccs_pch_split[1]

y_pch_split = train_test_split(y_pch, train_size=0.80, random_state=12345)

y_pch_train = y_pch_split[0]
y_pch_train = pd.DataFrame(y_pch_train)

y_pch_test = y_pch_split[1]
y_pch_test = pd.DataFrame(y_pch_test)

y_pch_train_array = np.ravel(np.array(y_pch_train))

#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_pch_train_list = list(maccs_pch_split[0])
maccs_pch_train_strs = [BitVectToText(x) for x in maccs_pch_train_list]

maccs_pch_test_list = list(maccs_pch_split[1])
maccs_pch_test_strs = [BitVectToText(x) for x in maccs_pch_test_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

maccs_pch_train_strs_broken = breaksy(maccs_pch_train_strs) #print(morg_2_train_strs_broken.dtypes)
maccs_pch_test_strs_broken = breaksy(maccs_pch_test_strs)

# =============================================================================
# Ridge
# =============================================================================
from sklearn.linear_model import Ridge

def adjusted_r2 (r2, n, p):
    return(1-(1-r2))*((n-(1))/(n-p-(1))) #it is a modification of R2 that adjusts for the number of explanatory terms in a model (p) relative to the number of data points (n).

def ridger_two():
    coefs = [0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]
    for i in coefs:
        ridge = Ridge(alpha=i, fit_intercept=True, normalize=True)
        ridge.fit(maccs_pch_train_strs_broken, y_pch_train)
        preds = ridge.predict(maccs_pch_test_strs_broken)
        scores = r2_score(y_pch_test, preds)
        #adj_scores = adjusted_r2(scores, 2048, len(y_pch_test))
        print(round(scores, 3))
print(ridger_two())

ridger5 = Ridge(alpha=0.25, fit_intercept=True, normalize=True)
ridger5.fit(maccs_pch_train_strs_broken, y_pch_train)
y_pch_pred_ridge5_maccs = ridger5.predict(maccs_pch_test_strs_broken)

r2_ridgereg5_maccs = r2_score(y_pch_test, y_pch_pred_ridge5_maccs)
adjR2_ridgereg5_maccs = adjusted_r2(r2_ridgereg5_maccs, 2048, len(y_pch_test))
mse_ridgereg5_maccs = mean_squared_error(y_pch_test, y_pch_pred_ridge5_maccs)

print('R^2 score: ',r2_ridgereg5_maccs)
print('Adjusted R^2 score: ',adjR2_ridgereg5_maccs)
print('Mean squared error: ',mse_ridgereg5_maccs)

#10-fold-CV
#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_pch_list = list(maccs_pch)
maccs_pch_strs = [BitVectToText(x) for x in maccs_pch_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

maccs_pch_strs_broken = breaksy(maccs_pch_strs) #print(morg_2_train_strs_broken.dtypes)

from sklearn.model_selection import KFold
scores5 = []
ridger5 = Ridge(alpha=0.25, fit_intercept=True, normalize=True)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_pch_strs_broken.iloc[train_index], maccs_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    ridger5.fit(X_train, y_train)
    scores5.append(ridger5.score(X_test, y_test))
    
print(np.mean(scores5))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
ridger5 = Ridge(alpha=5)
visualizer = CVScores(ridger5, cv=cv, scoring='r2')
visualizer.fit(maccs_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure
# =============================================================================
# Ridge-CrossValidation (Leave One-Out/Generalized)
# =============================================================================
from sklearn.linear_model import RidgeCV

ridgerCV = RidgeCV(alphas=np.array([0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]), store_cv_values=True) #RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=None, fit_intercept=False,
    #gcv_mode=None, normalize=False, scoring=None, store_cv_values=False)
ridgerCV.fit(maccs_pch_train_strs_broken, y_pch_train)
y_pch_pred_ridgeCV_maccs = ridgerCV.predict(maccs_pch_test_strs_broken)

r2_ridgeregCV_maccs = r2_score(y_pch_test, y_pch_pred_ridgeCV_maccs)
adjR2_ridgeregCV_maccs = adjusted_r2(r2_ridgeregCV, 2048, len(y_pch_test))
mse_ridgeregCV_maccs = mean_squared_error(y_pch_test, y_pch_pred_ridgeCV_maccs)

print('R^2 score: ',r2_ridgeregCV_maccs)
print('Adjusted R^2 score: ',adjR2_ridgeregCV_maccs)
print('Mean squared error: ',mse_ridgeregCV_maccs)

# =============================================================================
# Bayesian "ridge" regression
# =============================================================================
#https://stats.stackexchange.com/questions/328614/estimation-of-bayesian-ridge-regression
from sklearn.linear_model import BayesianRidge

ridger_bay = BayesianRidge(compute_score=True, verbose=True)
#BayesianRidge(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)[source]¶
ridger_bay.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_ridgebay_maccs = ridger_bay.predict(np.array(maccs_pch_test_strs_broken))

r2_ridgerbay_maccs = r2_score(y_pch_test, y_pch_pred_ridgebay_maccs)
adjR2_ridgerbay_maccs = adjusted_r2(r2_ridgerbay_maccs, 2048, len(y_pch_test))
mse_ridgerbay_maccs = mean_squared_error(y_pch_test, y_pch_pred_ridgebay_maccs)

print('R^2 score: ',r2_ridgerbay_maccs)
print('Adjusted R^2 score: ',adjR2_ridgerbay_maccs)
print('Mean squared error: ',mse_ridgerbay_maccs)

# =============================================================================
# ||*Support Vector Machine (regression)*||
# =============================================================================
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
from sklearn import svm

#RBF & C=1.0
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
#SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
svr.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_svr_c1_maccs = svr.predict(np.array(maccs_pch_test_strs_broken))

r2_svr_maccs  = r2_score(y_pch_test, y_pch_pred_svr_c1_maccs)
adjR2_svr_maccs  = adjusted_r2(r2_svr_maccs, 2048, len(y_pch_test))
mse_svr_maccs = mean_squared_error(y_pch_test, y_pch_pred_svr_c1_maccs)

print('R^2 score: ',r2_svr_maccs)
print('Adjusted R^2 score: ',adjR2_svr_maccs)
print('Mean squared error: ',mse_svr_maccs)

from sklearn.model_selection import KFold
scores6 = []
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_pch_strs_broken.iloc[train_index], maccs_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    svr.fit(X_train, y_train)
    scores6.append(svr.score(X_test, y_test))
    
print(np.mean(scores6))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svr = svm.SVR(kernel='rbf', C=1.0, verbose=True)
visualizer = CVScores(svr, cv=cv, scoring='r2')
visualizer.fit(maccs_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


# =============================================================================
# ||*Gradient boosting regressor*||
# =============================================================================
#lr=0.1
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_1.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_1_maccs = GBR_pch_1.predict(np.array(maccs_pch_test_strs_broken))

r2_GBR1_maccs = r2_score(y_pch_test, y_pch_pred_GBR_1_maccs)
adjR2_GBR1_maccs = adjusted_r2(r2_GBR1_maccs, 2048, len(y_pch_test))
mse_GBR1_maccs = mean_squared_error(y_pch_test, y_pch_pred_GBR_1_maccs)

print('R^2 score: ',r2_GBR1_maccs)
print('Adjusted R^2 score: ',adjR2_GBR1_maccs)
print('Mean squared error: ',mse_GBR1_maccs)

#lr=0.25
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_25 = GradientBoostingRegressor(loss='ls', learning_rate=0.25, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_25.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_25_maccs = GBR_pch_25.predict(np.array(maccs_pch_test_strs_broken))

r2_GBR25_maccs = r2_score(y_pch_test, y_pch_pred_GBR_25_maccs)
adjR2_GBR25_maccs = adjusted_r2(r2_GBR25_maccs, 2048, len(y_pch_test))
mse_GBR25_maccs = mean_squared_error(y_pch_test, y_pch_pred_GBR_25_maccs)

print('R^2 score: ',r2_GBR25_maccs)
print('Adjusted R^2 score: ',adjR2_GBR25_maccs)
print('Mean squared error: ',mse_GBR25_maccs)

#lr=0.5
from sklearn.ensemble import GradientBoostingRegressor
GBR_pch_5 = GradientBoostingRegressor(loss='ls', learning_rate=0.5, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
GBR_pch_5.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_GBR_5_maccs = GBR_pch_5.predict(np.array(maccs_pch_test_strs_broken))

r2_GBR5_maccs = r2_score(y_pch_test, y_pch_pred_GBR_5_maccs)
adjR2_GBR5_maccs = adjusted_r2(r2_GBR5_maccs, 2048, len(y_pch_test))
mse_GBR5_maccs = mean_squared_error(y_pch_test, y_pch_pred_GBR_5_maccs)

print('R^2 score: ',r2_GBR5_maccs)
print('Adjusted R^2 score: ',adjR2_GBR5_maccs)
print('Mean squared error: ',mse_GBR5_maccs)

from sklearn.model_selection import KFold
scores7 = []
GBR_pch_1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_pch_strs_broken.iloc[train_index], maccs_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    GBR_pch_1.fit(X_train, y_train)
    scores7.append(GBR_pch_1.score(X_test, y_test))
    
print(np.mean(scores7))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
GBR_pch_1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, criterion='mse', 
                                    random_state=12345, verbose=1)
visualizer = CVScores(GBR_pch_1, cv=cv, scoring='r2')
visualizer.fit(maccs_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure
# =============================================================================
# ||*Random Forest regression*||
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_regr.fit(np.array(maccs_pch_train_strs_broken), y_pch_train_array)
y_pch_pred_rf_maccs  = rf_regr.predict(np.array(maccs_pch_test_strs_broken))

r2_rf_maccs = r2_score(y_pch_test, y_pch_pred_rf_maccs )
adjR2_rf_maccs  = adjusted_r2(r2_rf_maccs, 2048, len(y_pch_test))
mse_rf_maccs = mean_squared_error(y_pch_test, y_pch_pred_rf_maccs)

print('R^2 score: ',r2_rf_maccs)
print('Adjusted R^2 score: ',adjR2_rf_maccs)
print('Mean squared error: ',mse_rf_maccs)

from sklearn.model_selection import KFold
scores8 = []
rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs_pch):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_pch_strs_broken.iloc[train_index], maccs_pch_strs_broken.iloc[test_index], y_pch.iloc[train_index], y_pch.iloc[test_index]
    rf_regr.fit(X_train, y_train)
    scores8.append(rf_regr.score(X_test, y_test))
    
print(np.mean(scores8))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_regr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_regr, cv=cv, scoring='r2')
visualizer.fit(maccs_pch_strs_broken, y_pch)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

scores_regr = pd.DataFrame(np.array([[np.mean(scores),np.mean(scores5)],
                                        [np.mean(scores2),np.mean(scores6)],
                                        [np.mean(scores3),np.mean(scores7)],
                                        [np.mean(scores4),np.mean(scores8)]]),
    columns = ['MORGAN','MACCS'],
    index = ['Ridge', 'SVM', 'Gradient Boosting', 'Random Forest'])
    
scores_regr.to_csv("scores_regression.csv", index=True)