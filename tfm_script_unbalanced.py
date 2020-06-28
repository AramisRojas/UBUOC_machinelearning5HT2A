# -*- coding: utf-8 -*-
#import pathlib
#pathlib.Path().absolute()

#*||Importing data||*
import os 
os.chdir(r"C:/Users/usuario/OneDrive/EstadisticaUOC/4-SEMESTRE/TFM/Datos_recuperados_ChEMBL")

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
df = pd.read_csv("5ht2a_definitive_nosalts.csv")

#print(df.head(5))

#*||Cbind of inchikeys to dataframe. Drop duplicates||*
import numpy as np

df_2 = np.genfromtxt(fname="inchikeys_2.txt", dtype="str", skip_header=1)

df_2df = pd.DataFrame(data=df_2, columns=["InChIKey_notation"])

df_final = pd.concat([df, df_2df], axis=1)

df_final.drop_duplicates(subset="InChIKey_notation",
                         keep = 'first', inplace = True)

df_final.to_csv("results_unique_p5.csv", index=False)

#df_final.head()

#*****FINGERPRINTS*****
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

#*Encode numeric data as categorical*
import numpy as np
#df_final['activities'] = np.where(df_final['pchembl_value']>=6, 'active', 'inactive')
df_final['activities'] = np.where(df_final['pchembl_value']>=6, 1, 0)


#*****ALGORITHMS with MORGAN-2*****
#In Anaconda3 shell, enter this: conda install scikit-learn
#conda list scikit-learn # to see which scikit-learn version is installed
#conda list # to see all packages installed in the active conda environment
#python -c "import sklearn; sklearn.show_versions()"

m2 = df_final['MFPS']
y = df_final['activities']

#*Split data into train and test*
from sklearn.model_selection import train_test_split

morgan_2 = train_test_split(m2, train_size=0.80, random_state=12345)
morgan_2_train = morgan_2[0]
morgan_2_test = morgan_2[1]

activ_inact = train_test_split(y, train_size=0.80, random_state=12345)

activ_inact_train = activ_inact[0]
activ_inact_train = pd.DataFrame(activ_inact_train)

activ_inact_test = activ_inact[1]
activ_inact_test = pd.DataFrame(activ_inact_test)


#*Fingerprints to strings. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
morgan2_train_list = list(morgan_2[0])
morgan2_train_strs = [BitVectToText(x) for x in morgan2_train_list]

morgan2_test_list = list(morgan_2[1])
morgan2_test_strs = [BitVectToText(x) for x in morgan2_test_list]

import pandas as pd
def breaksy(inputdf):
    acumulador = pd.DataFrame()
    for x in inputdf:
        x_partido = [int(i) for i in x]
        dt_x = pd.DataFrame(x_partido)
        x_trans = dt_x.transpose()
        acumulador = acumulador.append(pd.DataFrame(data = x_trans), ignore_index=True)
    return acumulador

morg_2_train_strs_broken = breaksy(morgan2_train_strs) #print(morg_2_train_strs_broken.dtypes)
morg_2_test_strs_broken = breaksy(morgan2_test_strs)
#---

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
import numpy as np
logreg_morgan2 = LogisticRegression() #Defining object with function
ai_train_rav = np.ravel(activ_inact_train)
ai_test_rav = np.ravel(activ_inact_test)

logreg_morgan2.fit(morg_2_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_lr_morgan2 = logreg_morgan2.predict(morg_2_test_strs_broken) #Prediction

#Confusion matrix
conf_matr_LR_morgan2 = confusion_matrix(activ_inact_test,ai_y_pred_lr_morgan2)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_lr_morg2 = conf_matr_LR_morgan2[0,0]/(conf_matr_LR_morgan2[0,0]+conf_matr_LR_morgan2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_lr_morg2 = conf_matr_LR_morgan2[1,1]/(conf_matr_LR_morgan2[1,0]+conf_matr_LR_morgan2[1,1])
print('Sensitivity: ', sens_lr_morg2)
print('Specificity: ', speci_lr_morg2)
#F1 score
f1_lr_morg2 = round(f1_score(ai_test_rav, ai_y_pred_lr_morgan2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_lr_morg2 = round(recall_score(ai_test_rav, ai_y_pred_lr_morgan2),3)
#Accuracy: measuring how near to true value.
lr_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_lr_morgan2),3) 
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
lr_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_lr_morgan2),3)
#True -, False +, False -, True +
tn_lr_morg2, fp_lr_morg2, fn_lr_morg2, tp_lr_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_lr_morgan2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_lr_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_lr_morgan2)
fpr_lr_m2, tpr_lr_m2, thresholds_lr_m2 = roc_curve(ai_test_rav, ai_y_pred_lr_morgan2)
import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_lr_m2, tpr_lr_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_lr_morg2)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
logreg_morgan2 = LogisticRegression()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(logreg_morgan2, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(LogisticRegression(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()


#||*kNN neighbours*||
from sklearn.neighbors import KNeighborsClassifier
knn_morgan2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') #Defining object with function
knn_morgan2.fit(morg_2_train_strs_broken, ai_train_rav) #Fitting
import numpy as np
morg_2_test_strs_br_arr = morg_2_test_strs_broken.to_numpy() #Change to an array
morg_2_test_strs_br_arr = np.reshape(morg_2_test_strs_br_arr,(526,2048)) #Reshaping the array
ai_y_pred_knn_morg2 = knn_morgan2.predict(morg_2_test_strs_br_arr) #Prediction

#Confusion matrix
conf_matr_knn_morgan2 = confusion_matrix(activ_inact_test, ai_y_pred_knn_morg2)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_knn_morg2 = conf_matr_knn_morgan2[0,0]/(conf_matr_knn_morgan2[0,0]+conf_matr_knn_morgan2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_knn_morg2 = conf_matr_knn_morgan2[1,1]/(conf_matr_knn_morgan2[1,0]+conf_matr_knn_morgan2[1,1])
print('Sensitivity: ', sens_lr_morg2)
print('Specificity: ', speci_lr_morg2)
#F1 score
f1_knn_morg2 = round(f1_score(ai_test_rav, ai_y_pred_knn_morg2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_knn_morg2 = round(recall_score(ai_test_rav, ai_y_pred_knn_morg2),3)
#Accuracy: measuring how near to true value.
knn_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_knn_morg2),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
knn_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_knn_morg2),3)
#True -, False +, False -, True +
tn_knn_morg2, fp_knn_morg2, fn_knn_morg2, tp_knn_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_knn_morg2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_knn_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_knn_morg2)
fpr_knn_m2, tpr_knn_m2, thresholds_knn_m2 = roc_curve(ai_test_rav, ai_y_pred_knn_morg2)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_knn_m2, tpr_knn_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_knn_morg2)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
knn_morgan2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(knn_morgan2, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Naive Bayes algorithm*||
from sklearn.naive_bayes import GaussianNB
gnb_morgan2 = GaussianNB() #Defining object with function
gnb_morgan2.fit(morg_2_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_gnb_morgan2 = gnb_morgan2.predict(morg_2_test_strs_broken) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (morg_2_test_strs_broken.shape[0], (ai_test_rav != ai_y_pred_gnb_morgan2).sum()))

conf_matr_gnb_morgan2 = confusion_matrix(activ_inact_test, ai_y_pred_gnb_morgan2) #Confusion matrix
print(conf_matr_gnb_morgan2)

from sklearn.naive_bayes import MultinomialNB #Better than gnb Naive Bayes
mnb_morgan2 = MultinomialNB() #Defining object with function
mnb_morgan2.fit(morg_2_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_mnb_morgan2 = mnb_morgan2.predict(morg_2_test_strs_broken) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (morg_2_test_strs_broken.shape[0], (ai_test_rav != ai_y_pred_mnb_morgan2).sum()))

#Confusion matrix
conf_matr_mnb_morgan2 = confusion_matrix(activ_inact_test, ai_y_pred_mnb_morgan2)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_mnb_morg2 = conf_matr_mnb_morgan2[0,0]/(conf_matr_mnb_morgan2[0,0]+conf_matr_mnb_morgan2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_mnb_morg2 = conf_matr_mnb_morgan2[1,1]/(conf_matr_mnb_morgan2[1,0]+conf_matr_mnb_morgan2[1,1])
print('Sensitivity: ', sens_mnb_morg2)
print('Specificity: ', speci_mnb_morg2)
#F1 score
f1_mnb_morg2 = round(f1_score(ai_test_rav, ai_y_pred_mnb_morgan2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_mnb_morg2 = round(recall_score(ai_test_rav, ai_y_pred_mnb_morgan2),3)
#Accuracy: measuring how near to true value.
mnb_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_mnb_morgan2),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
mnb_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_mnb_morgan2),3)
#True -, False +, False -, True +
tn_mnb_morg2, fp_mnb_morg2, fn_mnb_morg2, tp_mnb_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_mnb_morgan2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_mnb_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_mnb_morgan2)
fpr_mnb_m2, tpr_mnb_m2, thresholds_mnb_m2 = roc_curve(ai_test_rav, ai_y_pred_mnb_morgan2)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_mnb_m2, tpr_mnb_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_mnb_morg2)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
mnb_morgan2 = MultinomialNB()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(mnb_morgan2, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(MultinomialNB(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Gradient Boosting*|| (weak learner:decision tree)
#lr=0.1
from sklearn.ensemble import GradientBoostingClassifier
grad_morgan2_1 = GradientBoostingClassifier(loss='exponential',learning_rate=0.1, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_1.fit(morg_2_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_1.score(morg_2_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_1.score(morg_2_test_strs_broken, ai_test_rav)))
#lr=0.25
grad_morgan2_25 = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_25.fit(morg_2_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_25.score(morg_2_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_25.score(morg_2_test_strs_broken, ai_test_rav)))
#lr=0.5
grad_morgan2_5 = GradientBoostingClassifier(loss='exponential',learning_rate=0.5, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_5.fit(morg_2_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_5.score(morg_2_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_5.score(morg_2_test_strs_broken, ai_test_rav)))
#lr=0.75
grad_morgan2_75 = GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_morgan2_75.fit(morg_2_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_morgan2_75.score(morg_2_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_morgan2_75.score(morg_2_test_strs_broken, ai_test_rav)))

ai_y_pred_grad_morgan2 = grad_morgan2_75.predict(morg_2_test_strs_broken) #Prediction

#Confusion matrix
conf_matr_grad_morgan2 = confusion_matrix(activ_inact_test, ai_y_pred_grad_morgan2)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_grad_morg2 = conf_matr_grad_morgan2[0,0]/(conf_matr_grad_morgan2[0,0]+conf_matr_grad_morgan2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_grad_morg2 = conf_matr_grad_morgan2[1,1]/(conf_matr_grad_morgan2[1,0]+conf_matr_grad_morgan2[1,1])
print('Sensitivity: ', sens_grad_morg2)
print('Specificity: ', speci_grad_morg2)
#F1 score
f1_grad_morg2 = round(f1_score(ai_test_rav, ai_y_pred_grad_morgan2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_grad_morg2 = round(recall_score(ai_test_rav, ai_y_pred_grad_morgan2),3)
#Accuracy: measuring how near to true value.
grad_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_grad_morgan2),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
grad_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_grad_morgan2),3)
#True -, False +, False -, True +
tn_grad_morg2, fp_grad_morg2, fn_grad_morg2, tp_grad_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_grad_morgan2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_grad_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_grad_morgan2)
fpr_grad_m2, tpr_grad_m2, thresholds_grad_m2 = roc_curve(ai_test_rav, ai_y_pred_grad_morgan2)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_grad_m2, tpr_grad_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_grad_morg2)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
grad_morgan2_75 = GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(grad_morgan2_75, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Support Vector Machines*||
from sklearn import svm

morg_2_train_st_bro_arr = np.asarray(morg_2_train_strs_broken) #Dataframe to array
print(ai_train_rav)
svm_morgan2 = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345) #Formula
svm_morgan2.fit(morg_2_train_st_bro_arr, ai_train_rav) #Fitting

morg_2_test_st_bro_arr = np.asarray(morg_2_test_strs_broken)
print(ai_test_rav)
ai_y_pred_svm_morgan2 = svm_morgan2.predict(morg_2_test_st_bro_arr) #Prediction

#Confusion matrix
conf_matr_svm_morgan2 = confusion_matrix(activ_inact_test, ai_y_pred_svm_morgan2) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_svm_morg2 = conf_matr_svm_morgan2[0,0]/(conf_matr_svm_morgan2[0,0]+conf_matr_svm_morgan2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_svm_morg2 = conf_matr_svm_morgan2[1,1]/(conf_matr_svm_morgan2[1,0]+conf_matr_svm_morgan2[1,1])
print('Sensitivity: ', sens_svm_morg2)
print('Specificity: ', speci_svm_morg2)
#F1 score
f1_svm_morg2 = round(f1_score(ai_test_rav, ai_y_pred_svm_morgan2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_svm_morg2 = round(recall_score(ai_test_rav, ai_y_pred_svm_morgan2),3)
#Accuracy: measuring how near to true value.
svm_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_svm_morgan2),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
svm_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_svm_morgan2),3)
#True -, False +, False -, True +
tn_svm_morg2, fp_svm_morg2, fn_svm_morg2, tp_svm_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_svm_morgan2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_svm_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_svm_morgan2)
fpr_svm_m2, tpr_svm_m2, thresholds_svm_m2 = roc_curve(ai_test_rav, ai_y_pred_svm_morgan2)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_svm_m2, tpr_svm_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_svm_morg2)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
svm_morgan2 = svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(svm_morgan2, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(svm.SVC(C=1.0, kernel='rbf', verbose=True,
                      random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Random forest unbalanced*||
from sklearn.ensemble import RandomForestClassifier
rf_morg2 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_morg2.fit(morg_2_train_st_bro_arr, ai_train_rav)

ai_y_pred_rf_morg2 = rf_morg2.predict(morg_2_test_st_bro_arr)

#Confusion matrix
conf_matr_rf_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_rf_morg2) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_rf_morg2 = conf_matr_rf_morg2[0,0]/(conf_matr_rf_morg2[0,0]+conf_matr_rf_morg2[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_rf_morg2 = conf_matr_rf_morg2[1,1]/(conf_matr_rf_morg2[1,0]+conf_matr_rf_morg2[1,1])
print('Sensitivity: ', sens_rf_morg2)
print('Specificity: ', speci_rf_morg2)
#F1 score
f1_rf_morg2 = round(f1_score(ai_test_rav, ai_y_pred_rf_morg2),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_rf_morg2 = round(recall_score(ai_test_rav, ai_y_pred_rf_morg2),3)
#Accuracy: measuring how near to true value.
rf_acc_morg2 = round(accuracy_score(ai_test_rav, ai_y_pred_rf_morg2),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
rf_prec_morg2 = round(precision_score(ai_test_rav, ai_y_pred_rf_morg2),3)
#True -, False +, False -, True +
tn_rf_morg2, fp_rf_morg2, fn_rf_morg2, tp_rf_morg2 = confusion_matrix(activ_inact_test, ai_y_pred_rf_morg2).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_rf_morg2 = roc_auc_score(ai_test_rav, ai_y_pred_rf_morg2)
fpr_rf_m2, tpr_rf_m2, thresholds_rf_m2 = roc_curve(ai_test_rav, ai_y_pred_rf_morg2)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_rf_m2, tpr_rf_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_rf_morg2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
rf_morg2 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(rf_morg2, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(morg_2_train_st_bro_arr, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(morg_2_test_st_bro_arr,ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(morg_2_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(morg_2_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#*Table with all the results*
final_table_morg2 = pd.DataFrame(np.array([[lr_acc_morg2, lr_prec_morg2],
                                     [knn_acc_morg2, knn_prec_morg2],
                                     [mnb_acc_morg2, mnb_prec_morg2],
                                     [grad_acc_morg2,grad_prec_morg2],
                                     [svm_acc_morg2,svm_prec_morg2],
                                     [rf_acc_morg2,rf_prec_morg2]]),
    columns=['Accuracy','Precision'], 
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])
#Accuracy: measuring how near to true value.
#Precision: measuring how consistent results are. Precision is a good measure to determine, when the costs of False Positive is high.

final_trues_negs_morg2 = pd.DataFrame(np.array([[tn_lr_morg2, fp_lr_morg2, fn_lr_morg2, tp_lr_morg2],
                                                [tn_knn_morg2, fp_knn_morg2, fn_knn_morg2, tp_knn_morg2],
                                                [tn_mnb_morg2, fp_mnb_morg2, fn_mnb_morg2, tp_mnb_morg2],
                                                [tn_grad_morg2, fp_grad_morg2, fn_grad_morg2, tp_grad_morg2],
                                                [tn_svm_morg2, fp_svm_morg2, fn_svm_morg2, tp_svm_morg2],
                                                [tn_rf_morg2, fp_rf_morg2, fn_rf_morg2, tp_rf_morg2]]),
                                                
    columns=['True -','False +','False -','True +'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])
   
final_F1_recall_morg2 = pd.DataFrame (np.array([[f1_lr_morg2,rec_lr_morg2],
                                                [f1_knn_morg2,rec_knn_morg2],
                                                [f1_mnb_morg2,rec_mnb_morg2],
                                                [f1_grad_morg2,rec_grad_morg2],
                                                [f1_svm_morg2,rec_svm_morg2],
                                                [f1_rf_morg2,rec_rf_morg2]]),
    columns=['F1 score', 'Recall'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])
    
final_table_morg2.to_csv("final_table_morg2.csv", index=True)
final_trues_negs_morg2.to_csv("final_trues_negs_morg2.csv", index=True)
final_F1_recall_morg2.to_csv("final_F1_recall_morg2.csv", index=True)
    
#Recall: the model metric we use to select our best model when there is a high cost associated with False Negative.
     #The ability of the classifier to find all the positive samples
#F1-score: a measure of a test's accuracy. It considers both the precision p and the recall r. Best value at 1.

#*****ALGORITHMS with MACCS*****
maccs = df_final['MACCS_FPS']
y = df_final['activities']

#*Split data into train and test*
from sklearn.model_selection import train_test_split

maccs_split = train_test_split(maccs, train_size=0.80, random_state=12345)
maccs_train = maccs_split[0]
maccs_test = maccs_split[1]

#*Split activities into train and test*
activ_inact = train_test_split(y, train_size=0.80, random_state=12345)

activ_inact_train = activ_inact[0]
activ_inact_train = pd.DataFrame(activ_inact_train)

activ_inact_test = activ_inact[1]
activ_inact_test = pd.DataFrame(activ_inact_test)

#*Fingerprints train to binary. Separate strings and append rows*
from rdkit.DataStructs.cDataStructs import BitVectToText #rdkit.DataStructs.cDataStructs.BitVectToText((SparseBitVect)arg1) → str
maccs_train_list = list(maccs_split[0])
maccs_train_strs = [BitVectToText(x) for x in maccs_train_list]

maccs_test_list = list(maccs_split[1])
maccs_test_strs = [BitVectToText(x) for x in maccs_test_list]

maccs_train_strs_broken = breaksy(maccs_train_strs)
maccs_test_strs_broken = breaksy(maccs_test_strs)

#*Libraries required*
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score

#||*Logistic Regression*||
from sklearn.linear_model import LogisticRegression
import numpy as np
ai_train_rav = np.ravel(activ_inact_train)
ai_test_rav = np.ravel(activ_inact_test)

logreg_maccs = LogisticRegression() #Defining object with function
logreg_maccs.fit(maccs_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_lr_maccs = logreg_maccs.predict(maccs_test_strs_broken) #Prediction

#Confusion matrix
conf_matr_lr_maccs = confusion_matrix(activ_inact_test,ai_y_pred_lr_maccs)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_lr_maccs = conf_matr_lr_maccs[0,0]/(conf_matr_lr_maccs[0,0]+conf_matr_lr_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_lr_maccs = conf_matr_lr_maccs[1,1]/(conf_matr_lr_maccs[1,0]+conf_matr_lr_maccs[1,1])
print('Sensitivity: ', sens_lr_maccs)
print('Specificity: ', speci_lr_maccs)
#F1 score
f1_lr_maccs = round(f1_score(ai_test_rav, ai_y_pred_lr_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_lr_maccs = round(recall_score(ai_test_rav, ai_y_pred_lr_maccs),3)
#Accuracy: measuring how near to true value.
lr_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_lr_maccs),3) 
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
lr_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_lr_maccs),3)
#True -, False +, False -, True +
tn_lr_maccs, fp_lr_maccs, fn_lr_maccs, tp_lr_maccs = confusion_matrix(activ_inact_test, ai_y_pred_lr_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_lr_maccs = roc_auc_score(ai_test_rav, ai_y_pred_lr_maccs)
fpr_lr_m2, tpr_lr_m2, thresholds_lr_m2 = roc_curve(ai_test_rav, ai_y_pred_lr_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_lr_m2, tpr_lr_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_lr_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
logreg_maccs = LogisticRegression()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(logreg_maccs, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(LogisticRegression(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*kNN neighbours*||
from sklearn.neighbors import KNeighborsClassifier
knn_maccs = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') #Defining object with function
knn_maccs.fit(maccs_train_strs_broken, ai_train_rav) #Fitting
import numpy as np
maccs_test_strs_br_arr = maccs_test_strs_broken.to_numpy() #Change to an array
maccs_test_strs_br_arr = np.reshape(maccs_test_strs_br_arr,(526,167)) #Reshaping the array
ai_y_pred_knn_maccs = knn_maccs.predict(maccs_test_strs_br_arr) #Prediction

#Confusion matrix
conf_matr_knn_maccs = confusion_matrix(activ_inact_test, ai_y_pred_knn_maccs)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_knn_maccs = conf_matr_knn_maccs[0,0]/(conf_matr_knn_maccs[0,0]+conf_matr_knn_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_knn_maccs = conf_matr_knn_maccs[1,1]/(conf_matr_knn_maccs[1,0]+conf_matr_knn_maccs[1,1])
print('Sensitivity: ', speci_knn_maccs)
print('Specificity: ', speci_knn_maccs)
#F1 score
f1_knn_maccs = round(f1_score(ai_test_rav, ai_y_pred_knn_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_knn_maccs = round(recall_score(ai_test_rav, ai_y_pred_knn_maccs),3)
#Accuracy: measuring how near to true value.
knn_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_knn_maccs),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
knn_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_knn_maccs),3)
#True -, False +, False -, True +
tn_knn_maccs, fp_knn_maccs, fn_knn_maccs, tp_knn_maccs = confusion_matrix(activ_inact_test, ai_y_pred_knn_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_knn_maccs = roc_auc_score(ai_test_rav, ai_y_pred_knn_maccs)
fpr_knn_m2, tpr_knn_m2, thresholds_knn_m2 = roc_curve(ai_test_rav, ai_y_pred_knn_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_knn_m2, tpr_knn_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_knn_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
knn_maccs = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(knn_maccs, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Naive Bayes algorithm*||
from sklearn.naive_bayes import GaussianNB
gnb_maccs = GaussianNB() #Defining object with function
gnb_maccs.fit(maccs_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_gnb_maccs = gnb_maccs.predict(maccs_test_strs_broken) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (maccs_test_strs_broken.shape[0], (ai_test_rav != ai_y_pred_gnb_maccs).sum()))

conf_matr_gnb_maccs = confusion_matrix(activ_inact_test, ai_y_pred_gnb_maccs) #Confusion matrix
print(conf_matr_gnb_maccs)

from sklearn.naive_bayes import MultinomialNB #Better than gnb Naive Bayes
mnb_maccs = MultinomialNB() #Defining object with function
mnb_maccs.fit(maccs_train_strs_broken, ai_train_rav) #Fitting
ai_y_pred_mnb_maccs = mnb_maccs.predict(maccs_test_strs_broken) #Prediction
print("Number of mislabeled points out of a total %d points : %d"
       % (maccs_test_strs_broken.shape[0], (ai_test_rav != ai_y_pred_mnb_maccs).sum()))

#Confusion matrix
conf_matr_mnb_maccs = confusion_matrix(activ_inact_test, ai_y_pred_mnb_maccs)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_mnb_maccs = conf_matr_mnb_maccs[0,0]/(conf_matr_mnb_maccs[0,0]+conf_matr_mnb_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_mnb_maccs = conf_matr_mnb_maccs[1,1]/(conf_matr_mnb_maccs[1,0]+conf_matr_mnb_maccs[1,1])
print('Sensitivity: ', sens_mnb_maccs)
print('Specificity: ', speci_mnb_maccs)
#F1 score
f1_mnb_maccs = round(f1_score(ai_test_rav, ai_y_pred_mnb_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_mnb_maccs = round(recall_score(ai_test_rav, ai_y_pred_mnb_maccs),3)
#Accuracy: measuring how near to true value.
mnb_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_mnb_maccs),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative
mnb_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_mnb_maccs),3)
#True -, False +, False -, True +
tn_mnb_maccs, fp_mnb_maccs, fn_mnb_maccs, tp_mnb_maccs = confusion_matrix(activ_inact_test, ai_y_pred_mnb_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_mnb_maccs = roc_auc_score(ai_test_rav, ai_y_pred_mnb_maccs)
fpr_mnb_m2, tpr_mnb_m2, thresholds_mnb_m2 = roc_curve(ai_test_rav, ai_y_pred_mnb_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_mnb_m2, tpr_mnb_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_mnb_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
mnb_maccs = MultinomialNB()
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(mnb_maccs, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(MultinomialNB(), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Gradient Boosting*|| (weak learner:decision tree)
#lr=0.1
from sklearn.ensemble import GradientBoostingClassifier
grad_maccs_1 = GradientBoostingClassifier(loss='exponential',learning_rate=0.1, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_maccs_1.fit(maccs_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_maccs_1.score(maccs_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_maccs_1.score(maccs_test_strs_broken, ai_test_rav)))
#lr=0.25
grad_maccs_25 = GradientBoostingClassifier(loss='exponential',learning_rate=0.25, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_maccs_25.fit(maccs_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_maccs_25.score(maccs_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_maccs_25.score(maccs_test_strs_broken, ai_test_rav)))
#lr=0.5
grad_maccs_5 = GradientBoostingClassifier(loss='exponential',learning_rate=0.5, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_maccs_5.fit(maccs_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_maccs_5.score(maccs_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_maccs_5.score(maccs_test_strs_broken, ai_test_rav)))
#lr=0.75
grad_maccs_75 = GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
grad_maccs_75.fit(maccs_train_strs_broken, ai_train_rav)
print("Accuracy score (training): {0:.3f}".format(grad_maccs_75.score(maccs_train_strs_broken, ai_train_rav)))
print("Accuracy score (validation): {0:.3f}".format(grad_maccs_75.score(maccs_test_strs_broken, ai_test_rav)))

ai_y_pred_grad_maccs = grad_maccs_75.predict(maccs_test_strs_broken) #Prediction

#Confusion matrix
conf_matr_grad_maccs = confusion_matrix(activ_inact_test, ai_y_pred_grad_maccs)
#Sensitivity: quantifies the ability to avoid false negatives.
sens_grad_maccs = conf_matr_grad_maccs[0,0]/(conf_matr_grad_maccs[0,0]+conf_matr_grad_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_grad_maccs = conf_matr_grad_maccs[1,1]/(conf_matr_grad_maccs[1,0]+conf_matr_grad_maccs[1,1])
print('Sensitivity: ', sens_grad_maccs)
print('Specificity: ', speci_grad_maccs)
#F1 score
f1_grad_maccs = round(f1_score(ai_test_rav, ai_y_pred_grad_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_grad_maccs = round(recall_score(ai_test_rav, ai_y_pred_grad_maccs),3)
#Accuracy: measuring how near to true value.
grad_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_grad_maccs),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
grad_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_grad_maccs),3)
#True -, False +, False -, True +
tn_grad_maccs, fp_grad_maccs, fn_grad_maccs, tp_grad_maccs = confusion_matrix(activ_inact_test, ai_y_pred_grad_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_grad_maccs = roc_auc_score(ai_test_rav, ai_y_pred_grad_maccs)
fpr_grad_m2, tpr_grad_m2, thresholds_grad_m2 = roc_curve(ai_test_rav, ai_y_pred_grad_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_grad_m2, tpr_grad_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_grad_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
grad_maccs_75 = GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(grad_maccs_75, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(GradientBoostingClassifier(loss='exponential',learning_rate=0.75, n_estimators=100, max_depth=3, verbose=1, random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Support Vector Machines*||
from sklearn import svm

maccs_train_st_bro_arr = np.asarray(maccs_train_strs_broken) #Dataframe to array
print(ai_train_rav)
svm_maccs = svm.SVC(C=1.0, kernel='rbf', class_weight={0:1.0}, verbose=True,
                      random_state=12345) #Formula
svm_maccs.fit(maccs_train_st_bro_arr, ai_train_rav) #Fitting

maccs_test_st_bro_arr = np.asarray(maccs_test_strs_broken)
print(ai_test_rav)
ai_y_pred_svm_maccs = svm_maccs.predict(maccs_test_st_bro_arr) #Prediction

#Confusion matrix
conf_matr_svm_maccs = confusion_matrix(activ_inact_test, ai_y_pred_svm_maccs) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_svm_maccs = conf_matr_svm_maccs[0,0]/(conf_matr_svm_maccs[0,0]+conf_matr_svm_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_svm_maccs = conf_matr_svm_maccs[1,1]/(conf_matr_svm_maccs[1,0]+conf_matr_svm_maccs[1,1])
print('Sensitivity: ', sens_svm_maccs)
print('Specificity: ', speci_svm_maccs)
#F1 score
f1_svm_maccs = round(f1_score(ai_test_rav, ai_y_pred_svm_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_svm_maccs = round(recall_score(ai_test_rav, ai_y_pred_svm_maccs),3)
#Accuracy: measuring how near to true value.
svm_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_svm_maccs),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
svm_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_svm_maccs),3)
#True -, False +, False -, True +
tn_svm_maccs, fp_svm_maccs, fn_svm_maccs, tp_svm_maccs = confusion_matrix(activ_inact_test, ai_y_pred_svm_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_svm_maccs = roc_auc_score(ai_test_rav, ai_y_pred_svm_maccs)
fpr_svm_m2, tpr_svm_m2, thresholds_svm_m2 = roc_curve(ai_test_rav, ai_y_pred_svm_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_svm_m2, tpr_svm_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_svm_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
svm_maccs = svm.SVC(C=1.0, kernel='rbf', class_weight={0:1.0}, verbose=True,
                      random_state=12345)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(svm_maccs, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(svm.SVC(C=1.0, kernel='rbf', class_weight={0:1.0}, verbose=True,
                      random_state=12345), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#||*Random forest unbalanced*||
from sklearn.ensemble import RandomForestClassifier
rf_maccs = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
rf_maccs.fit(maccs_train_st_bro_arr, ai_train_rav)

ai_y_pred_rf_maccs = rf_maccs.predict(maccs_test_st_bro_arr)

#Confusion matrix
conf_matr_rf_maccs = confusion_matrix(activ_inact_test, ai_y_pred_rf_maccs) 
#Sensitivity: quantifies the ability to avoid false negatives.
sens_rf_maccs = conf_matr_rf_maccs[0,0]/(conf_matr_rf_maccs[0,0]+conf_matr_rf_maccs[0,1])
#Specificity: quantifies the ability to avoid false positives.
speci_rf_maccs = conf_matr_rf_maccs[1,1]/(conf_matr_rf_maccs[1,0]+conf_matr_rf_maccs[1,1])
print('Sensitivity: ', sens_rf_maccs)
print('Specificity: ', speci_rf_maccs)
#F1 score
f1_rf_maccs = round(f1_score(ai_test_rav, ai_y_pred_rf_maccs),3)
#Recall: the ability of the classifier to find all the positive samples
    #Same as specificity: the ability to avoid false positives
rec_rf_maccs = round(recall_score(ai_test_rav, ai_y_pred_rf_maccs),3)
#Accuracy: measuring how near to true value.
rf_acc_maccs = round(accuracy_score(ai_test_rav, ai_y_pred_rf_maccs),3)
#Precision: measuring how consistent results are.
    #The ability of the classifier not to label as positive a sample that is negative.
rf_prec_maccs = round(precision_score(ai_test_rav, ai_y_pred_rf_maccs),3)
#True -, False +, False -, True +
tn_rf_maccs, fp_rf_maccs, fn_rf_maccs, tp_rf_maccs = confusion_matrix(activ_inact_test, ai_y_pred_rf_maccs).ravel()

#ROC score & curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc_rf_maccs = roc_auc_score(ai_test_rav, ai_y_pred_rf_maccs)
fpr_rf_m2, tpr_rf_m2, thresholds_rf_m2 = roc_curve(ai_test_rav, ai_y_pred_rf_maccs)

import matplotlib.pyplot as plt
plt.figure()
lw=2
plt.plot(fpr_rf_m2, tpr_rf_m2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_rf_maccs)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ROC')
plt.legend(loc="lower right")
plt.show()

from yellowbrick.classifier import ConfusionMatrix
rf_maccs = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1)
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(rf_maccs, classes=[1,0])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(maccs_train_strs_broken, ai_train_rav)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(maccs_test_strs_broken, ai_test_rav)
# How did we do?
cm.show()

from yellowbrick.classifier import ClassPredictionError
# Instantiate the classification model and visualizer
classes = [1,0]
visualizer = ClassPredictionError(RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, bootstrap=True,
                                random_state=12345, verbose=1), classes=classes)
# Fit the training data to the visualizer
visualizer.fit(maccs_train_strs_broken, ai_train_rav)
# Evaluate the model on the test data
visualizer.score(maccs_test_strs_broken,ai_test_rav)
# Draw visualization
visualizer.show()

#*Table with all the results*
final_table_maccs = pd.DataFrame(np.array([[lr_acc_maccs, lr_prec_maccs],
                                     [knn_acc_maccs, knn_prec_maccs],
                                     [mnb_acc_maccs, mnb_prec_maccs],
                                     [grad_acc_maccs,grad_prec_maccs],
                                     [svm_acc_maccs,svm_prec_maccs],
                                     [rf_acc_maccs,rf_prec_maccs]]),
    columns=['Accuracy','Precision'], 
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])
#Accuracy: measuring how near to true value.
#Precision: measuring how consistent results are. Precision is a good measure to determine, when the costs of False Positive is high.

final_trues_negs_maccs = pd.DataFrame(np.array([[tn_lr_maccs, fp_lr_maccs, fn_lr_maccs, tp_lr_maccs],
                                                [tn_knn_maccs, fp_knn_maccs, fn_knn_maccs, tp_knn_maccs],
                                                [tn_mnb_maccs, fp_mnb_maccs, fn_mnb_maccs, tp_mnb_maccs],
                                                [tn_grad_maccs, fp_grad_maccs, fn_grad_maccs, tp_grad_maccs],
                                                [tn_svm_maccs, fp_svm_maccs, fn_svm_maccs, tp_svm_maccs],
                                                [tn_rf_maccs, fp_rf_maccs, fn_rf_maccs, tp_rf_maccs]]),
    columns=['True -','False +','False -','True +'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])
   
final_F1_recall_maccs = pd.DataFrame (np.array([[f1_lr_maccs,rec_lr_maccs],
                                                [f1_knn_maccs,rec_knn_maccs],
                                                [f1_mnb_maccs,rec_mnb_maccs],
                                                [f1_grad_maccs,rec_grad_maccs],
                                                [f1_svm_maccs,rec_svm_maccs],
                                                [f1_rf_maccs,rec_rf_maccs]]),
                                                
    columns=['F1 score', 'Recall'],
    index=['Logistic','kNN','Naive Bayes','Gradient Boosting','SVM','Random Forest 50 trees'])


final_table_maccs.to_csv("final_table_maccs.csv", index=True)
final_trues_negs_maccs.to_csv("final_trues_negs_maccs.csv", index=True)
final_F1_recall_maccs.to_csv("final_F1_recall_maccs.csv", index=True)
    
#Recall: the model metric we use to select our best model when there is a high cost associated with False Negative.
     #The ability of the classifier to find all the positive samples
#F1-score: a measure of a test's accuracy. It considers both the precision p and the recall r. Best value at 1.
