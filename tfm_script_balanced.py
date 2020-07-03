# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:09:42 2020

@author: usuario
"""
# =============================================================================
# *||Encode numeric data as categorical||*
# =============================================================================
import numpy as np
#df_final['activities'] = np.where(df_final['pchembl_value']>=6, 1, 0)
df_final_wo_outs['activities'] = np.where(df_final_wo_outs['pchembl_value']>=6, 1, 0)
df_final_wo_outs = df_final_wo_outs.dropna()

# =============================================================================
# *****MORGAN-2*****
# =============================================================================

#-----------------------------------------------------------------
m2 = df_final_wo_outs['MFPS']
y = df_final_wo_outs['activities']

#*||Split dataset into train/test||*
from sklearn.model_selection import train_test_split

morgan_2_bal = train_test_split(m2, train_size=0.80, random_state=12345)
morg_2_train_bal = morgan_2_bal[0]
#morg_2_train_bal = morg_2_train_bal.iloc
morg_2_test_bal = morgan_2_bal[1]

y_bal = train_test_split(y, train_size=0.80, random_state=12345) #list of series
y_train_bal = y_bal[0] #pandas.core.series.Series
y_train_bal = pd.DataFrame(y_train_bal) #df

y_test_bal = y_bal[1] #pandas.core.series.Series
y_test_bal = pd.DataFrame(y_test_bal) #df

#*||Fingerprints train to binary. Separate strings and append rows||*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
morg_2_train_list_bal = list(morgan_2_bal[0]) #list
morg_2_train_strs_bal = [BitVectToText(x) for x in morg_2_train_list_bal] #list of strings

morg_2_test_list_bal = list(morgan_2_bal[1]) #list
morg_2_test_strs_bal = [BitVectToText(x) for x in morg_2_test_list_bal] #list of strings

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

morg_2_train_strs_bk_bal = breaksy(morg_2_train_strs_bal) #print(morg_2_train_strs_broken.dtypes)
morg_2_test_strs_bk_bal = breaksy(morg_2_test_strs_bal)

#*||Balancing the dataset||*
#m2_arr_bal = list(m2)
#m2_arr_bal = [BitVectToText(x) for x in m2_arr_bal]
#m2_arr_bal = pd.DataFrame(m2_arr_bal)
#y_arr_bal = pd.DataFrame(y)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='majority', return_indices=True, random_state=12345, replacement=False)
#imblearn.under_sampling.RandomUnderSampler(sampling_strategy='auto',
    #return_indices=False,random_state=None, replacement=False,
    #ratio=None)

X_rus, y_rus, id_rus = rus.fit_sample(morg_2_train_strs_bk_bal, y_train_bal)
print('Removed indexes:',id_rus)
m2_train_s_bk_bal = morg_2_train_strs_bk_bal.iloc[id_rus]
print(y_rus)

X_rus2, y_rus2, id_rus2 = rus.fit_sample(morg_2_test_strs_bk_bal, y_test_bal)
print('Removed indexes:',id_rus2)
m2_test_s_bk_bal = morg_2_test_strs_bk_bal.iloc[id_rus2]
print(y_rus2)

#---
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score 
#accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
#accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
from sklearn.metrics import precision_score
#precision: The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
#precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#---

#*****ALGORITHMS with MORGAN-2*****
ai_train_rav_bal = np.ravel(y_rus)
ai_test_rav_bal = np.ravel(y_rus2)

#||*Logistic Regression*||
from sklearn.linear_model import LogisticRegression
import numpy as np
logreg_morgan2_bal = LogisticRegression(random_state=12345) #Defining object with function
logreg_morgan2_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal) #Fitting
ai_y_pred_lr_morgan2_bal = logreg_morgan2_bal.predict(m2_test_s_bk_bal) #Prediction

#Confusion matrix
conf_matr_LR_morgan2_bal = confusion_matrix(ai_test_rav_bal,ai_y_pred_lr_morgan2_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_lr_morg2_bal = conf_matr_LR_morgan2_bal[0,0]/(conf_matr_LR_morgan2_bal[0,0]+conf_matr_LR_morgan2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_lr_morg2_bal = conf_matr_LR_morgan2_bal[1,1]/(conf_matr_LR_morgan2_bal[1,0]+conf_matr_LR_morgan2_bal[1,1])
print('Sensitivity: ', sens_lr_morg2_bal)
print('Specificity: ', speci_lr_morg2_bal)
#F1 score = accuracy (precision + recall)
f1_lr_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_lr_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal),3)
#Accuracy: measuring how near to true value.
lr_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal),3) 
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
lr_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal),3)
#True -, False +, False -, True +
tn_lr_morg2_bal, fp_lr_morg2_bal, fn_lr_morg2_bal, tp_lr_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_lr_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal)
fpr_lr_m2_bal, tpr_lr_m2_bal, thresholds_lr_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_lr_morgan2_bal)
import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_lr_m2_bal, tpr_lr_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_lr_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
logreg_morgan2_bal = LogisticRegression()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(logreg_morgan2_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(LogisticRegression(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

#10-fold-CV
#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
m2_list = list(m2)
m2_strs = [BitVectToText(x) for x in m2_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

m2_strs_broken = breaksy(m2_strs)

from sklearn.model_selection import KFold
scoresA = []
logreg_morgan2_bal = LogisticRegression()
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    logreg_morgan2_bal.fit(X_train, y_train)
    scoresA.append(logreg_morgan2_bal.score(X_test, y_test))
    
print(np.mean(scoresA))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
logreg_morgan2_bal = LogisticRegression()
visualizer = CVScores(logreg_morgan2_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
logreg_morgan2_bal = LogisticRegression()
visualizer = CVScores(logreg_morgan2_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


#||*kNN neighbours*||
from sklearn.neighbors import KNeighborsClassifier
knn_morgan2_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') #Defining object with function
knn_morgan2_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal) #Fitting
import numpy as np
m2_test_s_bk_bal_arr = m2_test_s_bk_bal.to_numpy() #Change to an array
m2_test_s_bk_bal_arr = np.reshape(m2_test_s_bk_bal_arr,(84,2048)) #Reshaping the array
ai_y_pred_knn_morg2_bal = knn_morgan2_bal.predict(m2_test_s_bk_bal_arr) #Prediction

#Confusion matrix
conf_matr_knn_morgan2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_knn_morg2_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_knn_morg2_bal = conf_matr_knn_morgan2_bal[0,0]/(conf_matr_knn_morgan2_bal[0,0]+conf_matr_knn_morgan2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_knn_morg2_bal = conf_matr_knn_morgan2_bal[1,1]/(conf_matr_knn_morgan2_bal[1,0]+conf_matr_knn_morgan2_bal[1,1])
print('Sensitivity: ', sens_lr_morg2_bal)
print('Specificity: ', speci_lr_morg2_bal)
#F1 score
f1_knn_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_knn_morg2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_knn_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_knn_morg2_bal),3)
#Accuracy: measuring how near to true value.
knn_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_knn_morg2_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
knn_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_knn_morg2_bal),3)
#True -, False +, False -, True +
tn_knn_morg2_bal, fp_knn_morg2_bal, fn_knn_morg2_bal, tp_knn_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_knn_morg2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_knn_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_knn_morg2_bal)
fpr_knn_m2_bal, tpr_knn_m2_bal, thresholds_knn_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_knn_morg2_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_knn_m2_bal, tpr_knn_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_knn_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
knn_morgan2_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(knn_morgan2_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresB = []
knn_morgan2_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    knn_morgan2_bal.fit(X_train, y_train)
    scoresB.append(knn_morgan2_bal.score(X_test, y_test))
    
print(np.mean(scoresB))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
knn_morgan2_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
visualizer = CVScores(knn_morgan2_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
knn_morgan2_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
visualizer = CVScores(knn_morgan2_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Naive Bayes algorithm*||
from sklearn.naive_bayes import GaussianNB
gnb_morgan2_bal = GaussianNB() #Defining object with function
gnb_morgan2_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal) #Fitting
ai_y_pred_gnb_morgan2_bal = gnb_morgan2_bal.predict(m2_test_s_bk_bal) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (m2_test_s_bk_bal.shape[0], (ai_test_rav_bal != ai_y_pred_gnb_morgan2_bal).sum()))

conf_matr_gnb_morgan2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_gnb_morgan2_bal) #Confusion matrix
print(conf_matr_gnb_morgan2_bal)

from sklearn.naive_bayes import MultinomialNB #Better than gnb Naive Bayes
mnb_morgan2_bal = MultinomialNB() #Defining object with function
mnb_morgan2_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal) #Fitting
ai_y_pred_mnb_morgan2_bal = mnb_morgan2_bal.predict(m2_test_s_bk_bal) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (m2_test_s_bk_bal.shape[0], (ai_test_rav_bal != ai_y_pred_mnb_morgan2_bal).sum()))

#Confusion matrix
conf_matr_mnb_morgan2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_mnb_morg2_bal = conf_matr_mnb_morgan2_bal[0,0]/(conf_matr_mnb_morgan2_bal[0,0]+conf_matr_mnb_morgan2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_mnb_morg2_bal = conf_matr_mnb_morgan2_bal[1,1]/(conf_matr_mnb_morgan2_bal[1,0]+conf_matr_mnb_morgan2_bal[1,1])
print('Sensitivity: ', sens_mnb_morg2_bal)
print('Specificity: ', speci_mnb_morg2_bal)
#F1 score
f1_mnb_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_mnb_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal),3)
#Accuracy: measuring how near to true value.
mnb_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
mnb_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal),3)
#True -, False +, False -, True +
tn_mnb_morg2_bal, fp_mnb_morg2_bal, fn_mnb_morg2_bal, tp_mnb_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_mnb_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal)
fpr_mnb_m2_bal, tpr_mnb_m2_bal, thresholds_mnb_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_mnb_morgan2_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_mnb_m2_bal, tpr_mnb_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_mnb_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
mnb_morgan2_bal = MultinomialNB()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(mnb_morgan2_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(MultinomialNB(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresC = []
mnb_morgan2_bal = MultinomialNB()
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    mnb_morgan2_bal.fit(X_train, y_train)
    scoresC.append(mnb_morgan2_bal.score(X_test, y_test))
    
print(np.mean(scoresC))


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
mnb_morgan2_bal = MultinomialNB()
visualizer = CVScores(mnb_morgan2_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
mnb_morgan2_bal = MultinomialNB()
visualizer = CVScores(mnb_morgan2_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


#||*Gradient Boosting*|| (weak learner:decision tree)
#lr=0.1
from sklearn.ensemble import GradientBoostingClassifier
grad_morgan2_1_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.1, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_1_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_1_bal.score(m2_train_s_bk_bal, ai_train_rav_bal)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_1_bal.score(m2_test_s_bk_bal, ai_test_rav_bal)))
#lr=0.25
grad_morgan2_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_25_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_25_bal.score(m2_train_s_bk_bal, ai_train_rav_bal)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_25_bal.score(m2_test_s_bk_bal, ai_test_rav_bal)))
#lr=0.5
grad_morgan2_5_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.5, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_5_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_5_bal.score(m2_train_s_bk_bal, ai_train_rav_bal)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_5_bal.score(m2_test_s_bk_bal, ai_test_rav_bal)))
#lr=0.75
grad_morgan2_75_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_75_bal.fit(m2_train_s_bk_bal, ai_train_rav_bal)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_75_bal.score(m2_train_s_bk_bal, ai_train_rav_bal)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_75_bal.score(m2_test_s_bk_bal, ai_test_rav_bal)))

ai_y_pred_grad_morgan2_bal = grad_morgan2_25_bal.predict(m2_test_s_bk_bal) #Prediction

#Confusion matrix
conf_matr_grad_morgan2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_grad_morg2_bal = conf_matr_grad_morgan2_bal[0,0]/(conf_matr_grad_morgan2_bal[0,0]+conf_matr_grad_morgan2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_grad_morg2_bal = conf_matr_grad_morgan2_bal[1,1]/(conf_matr_grad_morgan2_bal[1,0]+conf_matr_grad_morgan2_bal[1,1])
print('Sensitivity: ', sens_grad_morg2_bal)
print('Specificity: ', speci_grad_morg2_bal)
#F1 score
f1_grad_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_grad_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal),3)
#Accuracy: measuring how near to true value.
grad_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
grad_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal),3)
#True -, False +, False -, True +
tn_grad_morg2_bal, fp_grad_morg2_bal, fn_grad_morg2_bal, tp_grad_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_grad_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal)
fpr_grad_m2_bal, tpr_grad_m2_bal, thresholds_grad_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_grad_morgan2_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_grad_m2_bal, tpr_grad_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_grad_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
grad_morgan2_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(grad_morgan2_25_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresD = []
grad_morgan2_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    grad_morgan2_25_bal.fit(X_train, y_train)
    scoresD.append(grad_morgan2_25_bal.score(X_test, y_test))
    
print(np.mean(scoresD))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
grad_morgan2_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
visualizer = CVScores(grad_morgan2_25_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
grad_morgan2_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
visualizer = CVScores(grad_morgan2_25_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


#||*Support Vector Machines*||
from sklearn import svm

m2_train_s_bk_bal_arr = np.asarray(m2_train_s_bk_bal) #Dataframe to array
print(ai_train_rav_bal)
svm_morgan2_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345) #Formula
svm_morgan2_bal.fit(m2_train_s_bk_bal_arr, ai_train_rav_bal) #Fitting

m2_test_s_bk_bal_arr = np.asarray(m2_test_s_bk_bal)
print(ai_test_rav_bal)
ai_y_pred_svm_morgan2_bal = svm_morgan2_bal.predict(m2_test_s_bk_bal_arr) #Prediction

#Confusion matrix
conf_matr_svm_morgan2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_svm_morg2_bal = conf_matr_svm_morgan2_bal[0,0]/(conf_matr_svm_morgan2_bal[0,0]+conf_matr_svm_morgan2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_svm_morg2_bal = conf_matr_svm_morgan2_bal[1,1]/(conf_matr_svm_morgan2_bal[1,0]+conf_matr_svm_morgan2_bal[1,1])
print('Sensitivity: ', sens_svm_morg2_bal)
print('Specificity: ', speci_svm_morg2_bal)
#F1 score
f1_svm_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_svm_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal),3)
#Accuracy: measuring how near to true value.
svm_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
svm_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal),3)
#True -, False +, False -, True +
tn_svm_morg2_bal, fp_svm_morg2_bal, fn_svm_morg2_bal, tp_svm_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_svm_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal)
fpr_svm_m2_bal, tpr_svm_m2_bal, thresholds_svm_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_svm_morgan2_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_svm_m2_bal, tpr_svm_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_svm_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
svm_morgan2_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(svm_morgan2_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresE = []
svm_morgan2_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    svm_morgan2_bal.fit(X_train, y_train)
    scoresE.append(svm_morgan2_bal.score(X_test, y_test))
    
print(np.mean(scoresE))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svm_morgan2_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
visualizer = CVScores(svm_morgan2_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svm_morgan2_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
visualizer = CVScores(svm_morgan2_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Random forest balanced*||
from sklearn.ensemble import RandomForestClassifier
rf_morg2_bal = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_morg2_bal.fit(m2_train_s_bk_bal_arr, ai_train_rav_bal)

ai_y_pred_rf_morg2_bal = rf_morg2_bal.predict(m2_test_s_bk_bal_arr)

#Confusion matrix
conf_matr_rf_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_rf_morg2_bal) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_rf_morg2_bal = conf_matr_rf_morg2_bal[0,0]/(conf_matr_rf_morg2_bal[0,0]+conf_matr_rf_morg2_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_rf_morg2_bal = conf_matr_rf_morg2_bal[1,1]/(conf_matr_rf_morg2_bal[1,0]+conf_matr_rf_morg2_bal[1,1])
print('Sensitivity: ', sens_rf_morg2_bal)
print('Specificity: ', speci_rf_morg2_bal)
#F1 score
f1_rf_morg2_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_rf_morg2_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_rf_morg2_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_rf_morg2_bal),3)
#Accuracy: measuring how near to true value.
rf_acc_morg2_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_rf_morg2_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
rf_prec_morg2_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_rf_morg2_bal),3)
#True -, False +, False -, True +
tn_rf_morg2_bal, fp_rf_morg2_bal, fn_rf_morg2_bal, tp_rf_morg2_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_rf_morg2_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_rf_morg2_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_rf_morg2_bal)
fpr_rf_m2_bal, tpr_rf_m2_bal, thresholds_rf_m2_bal = roc_curve(ai_test_rav_bal, ai_y_pred_rf_morg2_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_rf_m2_bal, tpr_rf_m2_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_rf_morg2_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
rf_morg2_bal = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(rf_morg2_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(m2_train_s_bk_bal_arr, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(m2_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(m2_train_s_bk_bal_arr, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(m2_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresF = []
rf_morg2_bal = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(m2):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = m2_strs_broken.iloc[train_index], m2_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    rf_morg2_bal.fit(X_train, y_train)
    scoresF.append(rf_morg2_bal.score(X_test, y_test))
    
print(np.mean(scoresF))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_morg2_bal = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_morg2_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_morg2_bal = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_morg2_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(m2_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#*Table with all the results*
final_table_morg2_bal = pd.DataFrame(np.array([[lr_acc_morg2_bal, lr_prec_morg2_bal],
                                     [knn_acc_morg2_bal, knn_prec_morg2_bal],
                                     [mnb_acc_morg2_bal, mnb_prec_morg2_bal],
                                     [grad_acc_morg2_bal,grad_prec_morg2_bal],
                                     [svm_acc_morg2_bal,svm_prec_morg2_bal],
                                     [rf_acc_morg2_bal,rf_prec_morg2_bal]]),
    columns=['Accuracy','Precision'], 
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
#Accuracy: measuring how near to true value.
#Precision: measuring how consistent results are. Precision is a good measure to determine, when the costs of False Positive is high.

final_trues_negs_morg2_bal = pd.DataFrame(np.array([[tn_lr_morg2_bal, fp_lr_morg2_bal, fn_lr_morg2_bal, tp_lr_morg2_bal],
                                                [tn_knn_morg2_bal, fp_knn_morg2_bal, fn_knn_morg2_bal, tp_knn_morg2_bal],
                                                [tn_mnb_morg2_bal, fp_mnb_morg2_bal, fn_mnb_morg2_bal, tp_mnb_morg2_bal],
                                                [tn_grad_morg2_bal, fp_grad_morg2_bal, fn_grad_morg2_bal, tp_grad_morg2_bal],
                                                [tn_svm_morg2_bal, fp_svm_morg2_bal, fn_svm_morg2_bal, tp_svm_morg2_bal],
                                                [tn_rf_morg2_bal, fp_rf_morg2_bal, fn_rf_morg2_bal, tp_rf_morg2_bal]]),
    columns=['True -','False +','False -','True +'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
   
final_F1_recall_morg2_bal = pd.DataFrame (np.array([[f1_lr_morg2_bal,rec_lr_morg2_bal],
                                                [f1_knn_morg2_bal,rec_knn_morg2_bal],
                                                [f1_mnb_morg2_bal,rec_mnb_morg2_bal],
                                                [f1_grad_morg2_bal,rec_grad_morg2_bal],
                                                [f1_svm_morg2_bal,rec_svm_morg2_bal],
                                                [f1_rf_morg2_bal,rec_rf_morg2_bal]]),
    columns=['F1 score', 'Recall'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
  
    
final_table_morg2_bal.to_csv("final_table_morg2_bal.csv", index=True)
final_trues_negs_morg2_bal.to_csv("final_trues_negs_morg2_bal.csv", index=True)
final_F1_recall_morg2_bal.to_csv("final_F1_recall_morg2_bal.csv", index=True)

# =============================================================================
# *****MACCS*****
# =============================================================================
#-----------------------------------------------------------------
df_final_wo_outs['activities'] = np.where(df_final_wo_outs['pchembl_value']>=6, 1, 0)
maccs = df_final_wo_outs['MACCS_FPS']
y = df_final_wo_outs['activities']

#*||Split dataset into train/test||*
from sklearn.model_selection import train_test_split

maccs_bal = train_test_split(maccs, train_size=0.80, random_state=12345)
maccs_train_bal = maccs_bal[0]
maccs_test_bal = maccs_bal[1]

y_bal = train_test_split(y, train_size=0.80, random_state=12345) #list of series
y_train_bal = y_bal[0] #pandas.core.series.Series
y_train_bal = pd.DataFrame(y_train_bal) #df

y_test_bal = y_bal[1] #pandas.core.series.Series
y_test_bal = pd.DataFrame(y_test_bal) #df

#*||Fingerprints train to binary. Separate strings and append rows||*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_train_list_bal = list(maccs_bal[0]) #list
maccs_train_strs_bal = [BitVectToText(x) for x in maccs_train_list_bal] #list of strings

maccs_test_list_bal = list(maccs_bal[1]) #list
maccs_test_strs_bal = [BitVectToText(x) for x in maccs_test_list_bal] #list of strings

maccs_train_strs_bk_bal = breaksy(maccs_train_strs_bal) #print(morg_2_train_strs_broken.dtypes)
maccs_test_strs_bk_bal = breaksy(maccs_test_strs_bal)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='majority', return_indices=True, random_state=12345, replacement=False)

X_rus_maccs, y_rus_maccs, id_rus_maccs = rus.fit_sample(maccs_train_strs_bk_bal, y_train_bal) #2102 rows
print('Removed indexes:',id_rus_maccs)
maccs_train_s_bk_bal = maccs_train_strs_bk_bal.iloc[id_rus_maccs] #932 rows
print(y_rus_maccs) #932 rows

X_rus2_maccs, y_rus2_maccs, id_rus2_maccs = rus.fit_sample(maccs_test_strs_bk_bal, y_test_bal) #526 rows
print('Removed indexes:',id_rus2_maccs)
maccs_test_s_bk_bal = maccs_test_strs_bk_bal.iloc[id_rus2_maccs] #212 rows
print(y_rus2_maccs) #212 rows

#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_list = list(maccs)
maccs_strs = [BitVectToText(x) for x in maccs_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

maccs_strs_broken = breaksy(maccs_strs)

import numpy as np
ai_train_rav_bal_maccs = np.ravel(y_rus_maccs)
ai_test_rav_bal_maccs = np.ravel(y_rus2_maccs)

#---
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score 
#accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
#accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
from sklearn.metrics import precision_score
#precision: The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
#precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#---

#||*Logistic Regression*||
from sklearn.linear_model import LogisticRegression
logreg_maccs_bal = LogisticRegression() #Defining object with function
logreg_maccs_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal_maccs) #Fitting
ai_y_pred_lr_maccs_bal = logreg_maccs_bal.predict(maccs_test_s_bk_bal) #Prediction

#Confusion matrix
conf_matr_LR_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs,ai_y_pred_lr_maccs_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_lr_maccs_bal = conf_matr_LR_maccs_bal[0,0]/(conf_matr_LR_maccs_bal[0,0]+conf_matr_LR_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_lr_maccs_bal = conf_matr_LR_maccs_bal[1,1]/(conf_matr_LR_maccs_bal[1,0]+conf_matr_LR_maccs_bal[1,1])
print('Sensitivity: ', sens_lr_maccs_bal)
print('Specificity: ', speci_lr_maccs_bal)
#F1 score
f1_lr_maccs_bal = round(f1_score(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_lr_maccs_bal = round(recall_score(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal),3)
#Accuracy: measuring how near to true value.
lr_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal),3) 
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
lr_prec_maccs_bal = round(precision_score(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal),3)
#True -, False +, False -, True +
tn_lr_maccs_bal, fp_lr_maccs_bal, fn_lr_maccs_bal, tp_lr_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_lr_maccs_bal = roc_auc_score(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal)
fpr_lr_maccs_bal, tpr_lr_maccs_bal, thresholds_lr_maccs_bal = roc_curve(ai_test_rav_bal_maccs, ai_y_pred_lr_maccs_bal)
import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_lr_maccs_bal, tpr_lr_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_lr_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
logreg_maccs_bal = LogisticRegression()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(logreg_maccs_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(LogisticRegression(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

#10-fold-CV
#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_list = list(maccs)
maccs_strs = [BitVectToText(x) for x in maccs_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

maccs_strs_broken = breaksy(maccs_strs)

from sklearn.model_selection import KFold
scoresAA = []
logreg_maccs_bal = LogisticRegression(random_state=12345)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    logreg_maccs_bal.fit(X_train, y_train)
    scoresAA.append(logreg_maccs_bal.score(X_test, y_test))
    
print(np.mean(scoresAA))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
logreg_maccs_bal = LogisticRegression(random_state=12345)
visualizer = CVScores(logreg_maccs_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
logreg_maccs_bal = LogisticRegression(random_state=12345)
visualizer = CVScores(logreg_maccs_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*kNN neighbours*||
from sklearn.neighbors import KNeighborsClassifier
knn_maccs_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') #Defining object with function
knn_maccs_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal_maccs) #Fitting
import numpy as np
maccs_test_s_bk_bal_arr = maccs_test_s_bk_bal.to_numpy() #Change to an array
maccs_test_s_bk_bal_arr = np.reshape(maccs_test_s_bk_bal_arr,(84,167)) #Reshaping the array
ai_y_pred_knn_maccs_bal = knn_maccs_bal.predict(maccs_test_s_bk_bal_arr) #Prediction

#Confusion matrix
conf_matr_knn_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_knn_maccs_bal = conf_matr_knn_maccs_bal[0,0]/(conf_matr_knn_maccs_bal[0,0]+conf_matr_knn_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_knn_maccs_bal = conf_matr_knn_maccs_bal[1,1]/(conf_matr_knn_maccs_bal[1,0]+conf_matr_knn_maccs_bal[1,1])
print('Sensitivity: ', sens_knn_maccs_bal)
print('Specificity: ', speci_knn_maccs_bal)
#F1 score
f1_knn_maccs_bal = round(f1_score(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_knn_maccs_bal = round(recall_score(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal),3)
#Accuracy: measuring how near to true value.
knn_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
knn_prec_maccs_bal = round(precision_score(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal),3)
#True -, False +, False -, True +
tn_knn_maccs_bal, fp_knn_maccs_bal, fn_knn_maccs_bal, tp_knn_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_knn_maccs_bal = roc_auc_score(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal)
fpr_knn_maccs_bal, tpr_knn_maccs_bal, thresholds_knn_maccs_bal = roc_curve(ai_test_rav_bal_maccs, ai_y_pred_knn_maccs_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_knn_maccs_bal, tpr_knn_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_knn_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
knn_maccs_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(knn_maccs_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresBB = []
knn_maccs_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    knn_maccs_bal.fit(X_train, y_train)
    scoresBB.append(knn_maccs_bal.score(X_test, y_test))
    
print(np.mean(scoresBB))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
knn_maccs_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
visualizer = CVScores(knn_maccs_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
knn_maccs_bal = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
visualizer = CVScores(knn_maccs_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Naive Bayes algorithm*||
from sklearn.naive_bayes import GaussianNB
gnb_maccs_bal = GaussianNB() #Defining object with function
gnb_maccs_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal_maccs) #Fitting
ai_y_pred_gnb_maccs_bal = gnb_maccs_bal.predict(maccs_test_s_bk_bal) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (maccs_test_s_bk_bal.shape[0], (ai_test_rav_bal_maccs != ai_y_pred_gnb_maccs_bal).sum()))

conf_matr_gnb_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_gnb_maccs_bal) #Confusion matrix
print(conf_matr_gnb_maccs_bal)

from sklearn.naive_bayes import MultinomialNB #Better than gnb Naive Bayes
mnb_maccs_bal = MultinomialNB() #Defining object with function
mnb_maccs_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal_maccs) #Fitting
ai_y_pred_mnb_maccs_bal = mnb_maccs_bal.predict(maccs_test_s_bk_bal) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (maccs_test_s_bk_bal.shape[0], (ai_test_rav_bal_maccs != ai_y_pred_mnb_maccs_bal).sum()))

#Confusion matrix
conf_matr_mnb_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_mnb_maccs_bal = conf_matr_mnb_maccs_bal[0,0]/(conf_matr_mnb_maccs_bal[0,0]+conf_matr_mnb_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_mnb_maccs_bal = conf_matr_mnb_maccs_bal[1,1]/(conf_matr_mnb_maccs_bal[1,0]+conf_matr_mnb_maccs_bal[1,1])
print('Sensitivity: ', sens_mnb_maccs_bal)
print('Specificity: ', speci_mnb_maccs_bal)
#F1 score
f1_mnb_maccs_bal = round(f1_score(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_mnb_maccs_bal = round(recall_score(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal),3)
#Accuracy: measuring how near to true value.
mnb_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
mnb_prec_maccs_bal = round(precision_score(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal),3)
#True -, False +, False -, True +
tn_mnb_maccs_bal, fp_mnb_maccs_bal, fn_mnb_maccs_bal, tp_mnb_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_mnb_maccs_bal = roc_auc_score(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal)
fpr_mnb_maccs_bal, tpr_mnb_maccs_bal, thresholds_mnb_maccs_bal = roc_curve(ai_test_rav_bal_maccs, ai_y_pred_mnb_maccs_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_mnb_maccs_bal, tpr_mnb_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_mnb_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
mnb_maccs_bal = MultinomialNB()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(mnb_maccs_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(MultinomialNB(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresCC = []
mnb_maccs_bal = MultinomialNB()
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    mnb_maccs_bal.fit(X_train, y_train)
    scoresCC.append(mnb_maccs_bal.score(X_test, y_test))
    
print(np.mean(scoresCC))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
mnb_maccs_bal = MultinomialNB()
visualizer = CVScores(mnb_maccs_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
mnb_maccs_bal = MultinomialNB()
visualizer = CVScores(mnb_maccs_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Gradient Boosting*|| (weak learner:decision tree)
from sklearn.ensemble import GradientBoostingClassifier
#lr=0.25
grad_maccs_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_maccs_25_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal_maccs)
print("Accuracy score (training): {0:.3f}".format(grad_maccs_25_bal.score(maccs_train_s_bk_bal, ai_train_rav_bal_maccs)))
print("Accuracy score (validation): {0:.3f}".format(grad_maccs_25_bal.score(maccs_test_s_bk_bal, ai_test_rav_bal_maccs)))

ai_y_pred_grad_maccs_bal = grad_maccs_25_bal.predict(maccs_test_s_bk_bal) #Prediction

#Confusion matrix
conf_matr_grad_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_grad_maccs_bal = conf_matr_grad_maccs_bal[0,0]/(conf_matr_grad_maccs_bal[0,0]+conf_matr_grad_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_grad_maccs_bal = conf_matr_grad_maccs_bal[1,1]/(conf_matr_grad_maccs_bal[1,0]+conf_matr_grad_maccs_bal[1,1])
print('Sensitivity: ', sens_grad_maccs_bal)
print('Specificity: ', speci_grad_maccs_bal)
#F1 score
f1_grad_maccs_bal = round(f1_score(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_grad_maccs_bal = round(recall_score(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal),3)
#Accuracy: measuring how near to true value.
grad_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
grad_prec_maccs_bal = round(precision_score(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal),3)
#True -, False +, False -, True +
tn_grad_maccs_bal, fp_grad_maccs_bal, fn_grad_maccs_bal, tp_grad_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_grad_maccs_bal = roc_auc_score(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal)
fpr_grad_maccs_bal, tpr_grad_maccs_bal, thresholds_grad_maccs_bal = roc_curve(ai_test_rav_bal_maccs, ai_y_pred_grad_maccs_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_grad_maccs_bal, tpr_grad_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_grad_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
grad_maccs_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(grad_maccs_25_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresDD = []
grad_maccs_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    grad_maccs_25_bal.fit(X_train, y_train)
    scoresDD.append(grad_maccs_25_bal.score(X_test, y_test))
    
print(np.mean(scoresDD))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
grad_maccs_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
visualizer = CVScores(grad_maccs_25_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
grad_maccs_25_bal = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
visualizer = CVScores(grad_maccs_25_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Support Vector Machines*||
from sklearn import svm
maccs_train_s_bk_bal_arr = np.asarray(maccs_train_s_bk_bal) #Dataframe to array
print(ai_train_rav_bal_maccs)
svm_maccs_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345) #Formula
svm_maccs_bal.fit(maccs_train_s_bk_bal_arr, ai_train_rav_bal_maccs) #Fitting

maccs_test_s_bk_bal_arr = np.asarray(maccs_test_s_bk_bal)
print(ai_test_rav_bal_maccs)
ai_y_pred_svm_maccs_bal = svm_maccs_bal.predict(maccs_test_s_bk_bal_arr) #Prediction

#Confusion matrix
conf_matr_svm_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_svm_maccs_bal = conf_matr_svm_maccs_bal[0,0]/(conf_matr_svm_maccs_bal[0,0]+conf_matr_svm_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_svm_maccs_bal = conf_matr_svm_maccs_bal[1,1]/(conf_matr_svm_maccs_bal[1,0]+conf_matr_svm_maccs_bal[1,1])
print('Sensitivity: ', sens_svm_maccs_bal)
print('Specificity: ', speci_svm_maccs_bal)
#F1 score
f1_svm_maccs_bal = round(f1_score(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_svm_maccs_bal = round(recall_score(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal),3)
#Accuracy: measuring how near to true value.
svm_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
svm_prec_maccs_bal = round(precision_score(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal),3)
#True -, False +, False -, True +
tn_svm_maccs_bal, fp_svm_maccs_bal, fn_svm_maccs_bal, tp_svm_maccs_bal = confusion_matrix(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_svm_maccs_bal = roc_auc_score(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal)
fpr_svm_maccs_bal, tpr_svm_maccs_bal, thresholds_svm_maccs_bal = roc_curve(ai_test_rav_bal_maccs, ai_y_pred_svm_maccs_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_svm_maccs_bal, tpr_svm_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_svm_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
svm_maccs_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(svm_maccs_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresEE = []
svm_maccs_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    svm_maccs_bal.fit(X_train, y_train)
    scoresEE.append(svm_maccs_bal.score(X_test, y_test))
    
print(np.mean(scoresEE))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svm_maccs_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
visualizer = CVScores(svm_maccs_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
svm_maccs_bal = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
visualizer = CVScores(svm_maccs_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#||*Random forest balanced*||
from sklearn.ensemble import RandomForestClassifier
rf_maccs_bal = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_maccs_bal.fit(maccs_train_s_bk_bal, ai_train_rav_bal)

ai_y_pred_rf_maccs_bal = rf_maccs_bal.predict(maccs_test_s_bk_bal)

#Confusion matrix
conf_matr_rf_maccs_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_rf_maccs_bal) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_rf_maccs_bal = conf_matr_rf_maccs_bal[0,0]/(conf_matr_rf_maccs_bal[0,0]+conf_matr_rf_maccs_bal[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_rf_maccs_bal = conf_matr_rf_maccs_bal[1,1]/(conf_matr_rf_maccs_bal[1,0]+conf_matr_rf_maccs_bal[1,1])
print('Sensitivity: ', sens_rf_maccs_bal)
print('Specificity: ', speci_rf_maccs_bal)
#F1 score
f1_rf_maccs_bal = round(f1_score(ai_test_rav_bal, ai_y_pred_rf_maccs_bal),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_rf_maccs_bal = round(recall_score(ai_test_rav_bal, ai_y_pred_rf_maccs_bal),3)
#Accuracy: measuring how near to true value.
rf_acc_maccs_bal = round(accuracy_score(ai_test_rav_bal, ai_y_pred_rf_maccs_bal),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
rf_prec_maccs_bal = round(precision_score(ai_test_rav_bal, ai_y_pred_rf_maccs_bal),3)
#True -, False +, False -, True +
tn_rf_maccs_bal, fp_rf_maccs_bal, fn_rf_maccs_bal, tp_rf_maccs_bal = confusion_matrix(ai_test_rav_bal, ai_y_pred_rf_maccs_bal).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_rf_maccs_bal = roc_auc_score(ai_test_rav_bal, ai_y_pred_rf_maccs_bal)
fpr_rf_maccs_bal, tpr_rf_maccs_bal, thresholds_rf_maccs_bal = roc_curve(ai_test_rav_bal, ai_y_pred_rf_maccs_bal)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_rf_maccs_bal, tpr_rf_maccs_bal, color='mediumturquoise',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_rf_maccs_bal)
plt.plot([0, 1], [0, 1], color='violet', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
rf_maccs_bal = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(rf_maccs_bal, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_s_bk_bal_arr, ai_train_rav_bal)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_s_bk_bal_arr, ai_train_rav_bal)
# Evaluate the model on the test data
visualizer.score(maccs_test_s_bk_bal, ai_test_rav_bal)
# Draw visualization
visualizer.show()

from sklearn.model_selection import KFold
scoresFF = []
rf_maccs_bal = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
cv = KFold(n_splits=10, random_state=12345, shuffle=False)

for train_index, test_index in cv.split(maccs):
    print("Train index: ", train_index, "\n")
    print("Test index: ", test_index)
    
    X_train, X_test, y_train, y_test = maccs_strs_broken.iloc[train_index], maccs_strs_broken.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    rf_maccs_bal.fit(X_train, y_train)
    scoresFF.append(rf_maccs_bal.score(X_test, y_test))
    
print(np.mean(scoresFF))

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_maccs_bal = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_maccs_bal, cv=cv, scoring='accuracy',color='violet')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

from yellowbrick.model_selection import CVScores
cv = KFold(n_splits=10, random_state=12345, shuffle=False)
rf_maccs_bal = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
visualizer = CVScores(rf_maccs_bal, cv=cv, scoring='f1',color='mediumturquoise')
visualizer.fit(maccs_strs_broken, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

#*Table with all the results*
final_table_maccs_bal = pd.DataFrame(np.array([[lr_acc_maccs_bal, lr_prec_maccs_bal],
                                     [knn_acc_maccs_bal, knn_prec_maccs_bal],
                                     [mnb_acc_maccs_bal, mnb_prec_maccs_bal],
                                     [grad_acc_maccs_bal,grad_prec_maccs_bal],
                                     [svm_acc_maccs_bal,svm_prec_maccs_bal],
                                     [rf_acc_maccs_bal,rf_prec_maccs_bal]]),
    columns=['Accuracy','Precision'], 
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
#Accuracy: measuring how near to true value.
#Precision: measuring how consistent results are. Precision is a good measure to determine, when the costs of False Positive is high.

final_trues_negs_maccs_bal = pd.DataFrame(np.array([[tn_lr_maccs_bal, fp_lr_maccs_bal, fn_lr_maccs_bal, tp_lr_maccs_bal],
                                                [tn_knn_maccs_bal, fp_knn_maccs_bal, fn_knn_maccs_bal, tp_knn_maccs_bal],
                                                [tn_mnb_maccs_bal, fp_mnb_maccs_bal, fn_mnb_maccs_bal, tp_mnb_maccs_bal],
                                                [tn_grad_maccs_bal, fp_grad_maccs_bal, fn_grad_maccs_bal, tp_grad_maccs_bal],
                                                [tn_svm_maccs_bal, fp_svm_maccs_bal, fn_svm_maccs_bal, tp_svm_maccs_bal],
                                                [tn_rf_maccs_bal, fp_rf_maccs_bal, fn_rf_maccs_bal, tp_rf_maccs_bal]]),
    columns=['True -','False +','False -','True +'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
   
final_F1_recall_maccs_bal = pd.DataFrame(np.array([[f1_lr_maccs_bal,rec_lr_maccs_bal],
                                                [f1_knn_maccs_bal,rec_knn_maccs_bal],
                                                [f1_mnb_maccs_bal,rec_mnb_maccs_bal],
                                                [f1_grad_maccs_bal,rec_grad_maccs_bal],
                                                [f1_svm_maccs_bal,rec_svm_maccs_bal],
                                                [f1_rf_maccs_bal,rec_rf_maccs_bal]]),
    columns=['F1 score', 'Recall'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])
    
final_table_maccs_bal.to_csv("final_table_maccs_bal.csv", index=True)
final_trues_negs_maccs_bal.to_csv("final_trues_negs_maccs_bal.csv", index=True)
final_F1_recall_maccs_bal.to_csv("final_F1_recall_maccs_bal.csv", index=True)

    
scores_classif = pd.DataFrame(np.array([[np.mean(scoresA),np.mean(scoresAA)],
                                        [np.mean(scoresB),np.mean(scoresBB)],
                                        [np.mean(scoresC),np.mean(scoresCC)],
                                        [np.mean(scoresD),np.mean(scoresDD)],
                                        [np.mean(scoresE),np.mean(scoresEE)],
                                        [np.mean(scoresF),np.mean(scoresFF)]]),
    columns = ['MORGAN','MACCS'],
    index = ['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50'])

scores_classif.to_csv("scores_classification.csv", index=True)