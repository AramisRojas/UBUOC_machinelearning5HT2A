# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:20:54 2020
@author: usuario
"""

# =============================================================================
# DBSCAN (full)
# =============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

f1 = df_final_regr["full_mwt"].values
f2 = df_final_regr["pchembl_value"].values
plt.scatter(f1,f2, s=15)
plt.xlabel("Weight mass")
plt.ylabel("pchembl value")
plt.title("Scatterplot weight Vs pchembl value (Ki)")
plt.show()

full_mwt = df_final_regr["full_mwt"]
pchembl_values = df_final_regr["pchembl_value"]

concatenation = [full_mwt, pchembl_values]
df_dbscan = pd.concat(concatenation, axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
df_dbscan_scal = min_max_scaler.fit_transform(df_dbscan)
df_dbscan_scal = pd.DataFrame(df_dbscan_scal)
df_dbscan_scal = df_dbscan_scal.rename(columns = {0: "full_mwt", 1: "pchembl_value"})

modelDB = DBSCAN(eps=0.1, min_samples=15).fit(df_dbscan_scal)
clusters = modelDB.fit_predict(df_dbscan_scal)
df_values = df_dbscan_scal.values
plt.scatter(df_values[:,1], df_values[:,0], c=clusters, cmap = "plasma")
plt.xlabel("pchembl_value")
plt.ylabel("full_mwt")

#Elbow with PCA
from sklearn.decomposition import PCA
import sklearn.neighbors
from sklearn.neighbors import kneighbors_graph

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(df_dbscan_scal)
dist = sklearn.neighbors.DistanceMetric.get_metric("euclidean")
matsim = dist.pairwise(X_pca)
minPts = 15
A = kneighbors_graph(X_pca, minPts, include_self=False)
Ar = A.toarray()
seq = []
for i,s in enumerate (X_pca):
    for j in range (len(X_pca)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
seq.sort()
plt.plot(seq)
plt.title("Best value for eps based on PCA elbow")
plt.show()

#Re-cluster
modelDB2 = DBSCAN(eps=0.045, min_samples=15).fit(df_dbscan_scal)
clusters2 = modelDB2.fit_predict(df_dbscan_scal)
df_values = df_dbscan_scal.values
plt.scatter(df_values[:,1], df_values[:,0], c=clusters2, cmap = "plasma")
plt.xlabel("pchembl_value")
plt.ylabel("full_mwt")

#Count of outliers
copy = pd.DataFrame()
copy["full_mwt"] = df_final_regr["full_mwt"].values
copy["pchembl_value"] = df_final_regr["pchembl_value"].values
copy["label"] = clusters2
copy["MFPS"] = df_final_regr["MFPS"]
copy["MACCS_FPS"] = df_final_regr["MACCS_FPS"]

cantidadGrupo = pd.DataFrame()
cantidadGrupo["cantidad"] = copy.groupby("label").size()
print(cantidadGrupo) #4.6%
#outs = df_values[:,1]
#2335 - 2232 = 103

#Drop outliers
copy = copy.drop(copy[copy['label'] == -1].index)
f11= copy["full_mwt"].values
f22= copy["pchembl_value"].values

plt.scatter(f11,f22, s=15)
plt.xlabel("Weight mass")
plt.ylabel("pchembl value")
plt.title("Scatterplot weight Vs pchembl value (Ki)")
plt.show()

#New dataset with outliers to re-try models
df_final_wo_outs = copy

# =============================================================================
# SEABORN - No balanced
# =============================================================================
df_final_wo_outs['activities'].value_counts()
means = df_final_wo_outs.groupby('activities').mean()
print(means['pchembl_value'])
std = df_final_wo_outs.groupby('activities').std()
print(std['pchembl_value'])

count_active = len(df_final_wo_outs[df_final_wo_outs['activities']==1])
count_inactive = len(df_final_wo_outs[df_final_wo_outs['activities']==0])
pct_of_active = count_active/(count_active+count_inactive)
pct_of_inactive = count_inactive/(count_active+count_inactive)

print("Percentage of active compounds is",pct_of_active*100)
print("Percentage of inactive compounds is",pct_of_inactive*100)

import pandas as pd
df_final_regr = df_final.dropna()
df_final_regr.reset_index(drop=True)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='activities', data=df_final_wo_outs, palette='hls')

sns.distplot(df_final_regr['pchembl_value'])
sns.violinplot('activities', 'pchembl_value', data=df_final_regr, palette=["lightblue", "lightpink"])
#------------------------------------------------------------------------------
# =============================================================================
# SEABORN - Balanced (use after executing balancing)
# =============================================================================
y_train_re_bal = y_train_bal.iloc[id_rus]
y_test_re_bal = y_test_bal.iloc[id_rus2]

ys = [y_train_re_bal, y_test_re_bal]
y_full_bal = pd.concat(ys, axis=0)
y_full_bal['activities'].value_counts()

count_active_bal2 = len(y_full_bal[y_full_bal['activities']==1])
count_inactive_bal2 = len(y_full_bal[y_full_bal['activities']==0])
pct_of_active_bal2 = count_active_bal2/(count_active_bal2+count_inactive_bal2)
pct_of_inactive_bal2 = count_inactive_bal2/(count_active_bal2+count_inactive_bal2)

print("Percentage of active compounds is",pct_of_active_bal2*100)
print("Percentage of inactive compounds is",pct_of_inactive_bal2*100)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='activities', data=y_full_bal, palette='hls')

# =============================================================================
# OUTLIERS (DO NOT USE. Draft)
# =============================================================================
import pandas as pd
import numpy as np
outliers=[]
def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for i in data:
        z_score = (i - mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

y_pch = df_final_regr['pchembl_value']
    
outlier_pts = detect_outliers(y_pch)
print(outlier_pts) #Indexes: 652,653

# =============================================================================
# DO NOT USE. Draft
# =============================================================================
#morg_2_train_strs_broken
#morg_2_test_strs_broken
#activ_inact_train
#activ_inact_test

frames = [morg_2_train_strs_broken, activ_inact_train]
import pandas as pd
dfrad = pd.concat(frames, axis=1)
dfrad = dfrad.dropna()
#dfrad.iloc[:,[2048]]
#dfrad.iloc[:,:100]

#CLASS BALANCE - No balanced
from yellowbrick.target import ClassBalance
visCB = ClassBalance(labels=[1,0])
visCB.fit(dfrad['activities'])#Fit the data to the visualizer
visCB.show()#Finalize and render the figure

#RANK 2D "Pearson correlation" -No balanced
from yellowbrick.features import Rank2D
visualizer = Rank2D(algorithm='pearson')
visualizer.fit(dfrad.iloc[:,:50], dfrad['activities'])# Fit the data to the visualizer
visualizer.transform(dfrad.iloc[:,:50])# Transform the data
visualizer.show() # Finalize and render the figure

#MANIFOLD - No balanced
from yellowbrick.features import Manifold
classes = [1, 0]
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() #label_encoder object knows how to understand word labels.
dfrad['activities']= label_encoder.fit_transform(dfrad['activities']) #Encode labels 
dfrad['activities'].unique() 
viz = Manifold(manifold="tsne", classes=classes)# Instantiate the visualizer
viz.fit_transform(dfrad.iloc[:,:100], dfrad['activities'])# Fit the data to the visualizer
viz.show()# Finalize and render the figure

# =============================================================================
# #CLASS BALANCE - Balanced (DO NOT USE. Draft)
# =============================================================================
#m2_train_s_bk_bal #dataframe
#m2_test_s_bk_bal #dataframe
#ai_train_rav_bal = np.ravel(y_rus)
#ai_test_rav_bal = np.ravel(y_rus2)
y_rus_df = pd.DataFrame(y_rus)

frames_bal = [m2_train_s_bk_bal, y_rus_df]
import pandas as pd
dfrad_bal = pd.concat(frames_bal, axis=1)
dfrad_bal = dfrad.dropna()


from yellowbrick.target import ClassBalance
visCB = ClassBalance(labels=[1,0])
visCB.fit(dfrad_bal['activities'])#Fit the data to the visualizer
visCB.show()#Finalize and render the figure