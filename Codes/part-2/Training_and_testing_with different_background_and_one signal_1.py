
# coding: utf-8

# Do training on 
# Background
# File1= '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
# File2 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root'
# File3 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
# File4 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_THQ_ctcvcp_HToGG_M125_13TeV-madgraph-pythia8.root'
# File5 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8.root'
# Signal
# File6 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TprimeBToTH_Hgg_M-600_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
# 
# 
# Make HDF5 file and import it.
# Do testing on 
# File7 ='/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TprimeBToTH_Hgg_M-1200_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'

# In[1]:


import pandas as pd
import math
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# from keras.optimizers import adam
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import pandas, numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# this wrapper makes it possible to train on subset of features
import sklearn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# import xgboost as xgb
from itertools import tee, islice
from ROOT import TFile, TCanvas, TPad, TPaveLabel, TPaveText, TTree, TH1F, TF1
from root_numpy import root2array, tree2array, array2tree, array2root
import sys
from ROOT import gROOT, AddressOf
from root_numpy import root2array, rec2array
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.ticker as ticker
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree


# In[2]:


signal = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TprimeBToTH_Hgg_M-600_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'
back_1 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8.root'
back_01= '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
back_02 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root'
back_03 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_VHToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8.root'
back_04 = '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_THQ_ctcvcp_HToGG_M125_13TeV-madgraph-pythia8.root'
back_05= '/eos/home-s/sraj/M.Sc._Thesis/data_files/output_GluGluHToGG_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8.root'


# In[3]:


treeName_back_1 ="tagsDumper/trees/ttgg_13TeV_THQLeptonicTag" 
treeName_signal = "tagsDumper/trees/Tprime_600_13TeV_THQLeptonicTag"
treeName_back_05 = "tagsDumper/trees/ggh_125_13TeV_THQLeptonicTag"
treeName_back_04 = "tagsDumper/trees/thq_125_13TeV_THQLeptonicTag"
treeName_back_03 = "tagsDumper/trees/vh_125_13TeV_THQLeptonicTag"
treeName_back_02 = 'tagsDumper/trees/vbf_125_13TeV_THQLeptonicTag'
treeName_back_01 = "tagsDumper/trees/tth_125_13TeV_THQLeptonicTag"


# In[4]:


columns = ['dipho_pt', 
           'dipho_phi',
           'dipho_eta',
           'dipho_e',
           'dipho_mass',
           'dipho_leadPt',
           'dipho_leadEt',
           'dipho_leadEta',
           'dipho_leadPhi',
           'dipho_subleadEta',
           'bjet1_pt',
           'bjet2_pt',
           'bjet1_eta',
           'bjet2_eta',
           'jet1_pt',
           'jet2_pt',
           'jet1_eta',
           'n_jets',
           'n_bjets',
           'bjet2_phi',
           'bjet3_phi',
#            'bjet3_eta',
#            'bjet1_discr',
#            'bjet2_discr',
#            'bjet3_discr',
#            'jet3_pt',
# #           'jet1_phi',
# #          'jet2_phi' ,
#           'jet3_phi',
#           'jet1_e',
#           'jet2_e',
#           'jet3_e',
#            'CMS_hgg_mass',
#             'sigmaMoM_decorr',
#             'dipho_sumpt',
#             'dipho_cosphi',
# #             'dipho_mass',
#             'dipho_lead_ptoM',
#             'ele1_pt',
#             'ele2_pt',
#             'ele1_eta',
#             'ele2_eta',
#             'ele1_phi',
#             'ele2_phi',
#             'ele1_e',
#             'ele2_e',
#             'ele1_ch',
#             'ele2_ch',
          ]


# In[5]:


signal = root2array(signal, treeName_signal, columns)   #Signal TPrime at 600TeV
back_ttgg = root2array(back_1, treeName_back_1, columns)     # ttgg background(Not using this)
back_tth = root2array(back_01, treeName_back_01, columns)      
back_vbf = root2array(back_02, treeName_back_02, columns)
back_vh = root2array(back_03, treeName_back_03, columns)
back_thq = root2array(back_04, treeName_back_04, columns)
back_ggh = root2array(back_05, treeName_back_05, columns)


# In[6]:


#Now do training with each beackground seperately
# further combine all the background and do training
#Again do training seperately
# again take output as HDF5 file and then Use training Output HDF5 file and compare with testing file 
# File7 ='/eos/home-s/sraj/M.Sc._Thesis/data_files/output_TprimeBToTH_Hgg_M-1200_LH_TuneCP5_PSweights_13TeV-madgraph_pythia8.root'


# In[7]:


signal_1 = pd.DataFrame(signal)            #Signal for the testing
back_tth = pd.DataFrame(back_tth)          #tth background dataframe 
back_vbf = pd.DataFrame(back_vbf)             #vbf background dataframe 
back_vh = pd.DataFrame(back_vh)             #vh background dataframe 
back_thq = pd.DataFrame(back_thq) #thq background dataframe 
back_ggh = pd.DataFrame(back_ggh) #ggh background dataframe 
back_ttgg = pd.DataFrame(back_ttgg)


# # Training and Testing of tth as background and TPrime_600  as signal

# In[8]:


X = np.concatenate((signal_1, back_tth))
Y = np.concatenate((np.ones(signal_1.shape[0]),
                    np.zeros(back_tth.shape[0])))


# In[9]:


X.shape, Y.shape


# In[43]:


X_train,X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state =5)


# In[45]:


X_test.shape, Y_test.shape


# In[46]:


X_train.shape, Y_train.shape


# In[13]:


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM


# In[47]:


# define model for training

clf = Sequential()
# clf.add(LSTM(1, return_sequences=True ))
clf.add(Dropout(0.3,input_shape = (21,)))
clf.add(Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform',name = 'dense_1'))
clf.add(Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_2'))
clf.add(Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_3'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',name = 'dense_4'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_5'))
#Output
clf.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output'))
#compile model
clf.compile(loss = 'binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])
print(clf.summary())
# plot_model(clf, to_file='clf_plot____.png', show_shapes=True, show_layer_names=True)


# In[48]:


h = clf.fit(X_train, Y_train, epochs = 10, batch_size= 900, validation_split = 0.25)


# In[49]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = clf.evaluate(Xtest, Ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[50]:


from sklearn.metrics import roc_curve, auc

decisions = clf.predict(Xtest)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(Ytest, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[51]:


plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = clf.predict(Xtest)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg = tTest[Ytest==0]
tSig = tTest[Ytest==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig, bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime')
n, bins, patches = plt.hist(tBkg, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'tth')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
# plt.savefig('/eos/home-s/sraj/M.Sc._Thesis/Plot/''output_TPrime_ttgg.png')
plt.show()


# # Training and Testing of ttgg as background and TPrime_600  as signal

# In[33]:


X_ttgg = np.concatenate((signal_1, back_ttgg))
Y_ttgg = np.concatenate((np.ones(signal_1.shape[0]),
                    np.zeros(back_ttgg.shape[0])))


# In[34]:


Xtrain,Xtest, Ytrain,Ytest = train_test_split(X_ttgg, Y_ttgg, test_size=0.33, random_state =5)


# In[35]:


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM


# In[36]:


# define model for training

clf = Sequential()
# clf.add(LSTM(1, return_sequences=True ))
clf.add(Dropout(0.3,input_shape = (21,)))
clf.add(Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform',name = 'dense_1'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_2'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_3'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',name = 'dense_4'))
clf.add(Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_5'))
#Output
clf.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output'))
#compile model
clf.compile(loss = 'binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])
print(clf.summary())
# plot_model(clf, to_file='clf_plot____.png', show_shapes=True, show_layer_names=True)


# In[38]:


h = clf.fit(Xtrain, Ytrain, epochs = 100, batch_size= 900, validation_split = 0.25)


# In[39]:


# Final evaluation of the model for DNN
# Testing Outputs
scores = clf.evaluate(Xtest, Ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[40]:


from sklearn.metrics import roc_curve, auc

decisions = clf.predict(Xtest)

# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(Ytest, decisions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("ROC_curve.png")
plt.show()


# In[41]:


plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = clf.predict(Xtest)
# if hasattr(clf, "decision_function"):
#     tTest = clf.decision_function(X_test)        # if available use decision_function
# else:
#     tTest = clf.predict_proba(X_test)[:,1]       # for e.g. MLP need to use predict_proba
tBkg = tTest[Ytest==0]
tSig = tTest[Ytest==1]
nBins = 20
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
# plt.title('Multilayer perceptron')
plt.xlabel(' $DNN$', labelpad=3)
plt.ylabel('$Probability density$', labelpad=40)
n, bins, patches = plt.hist(tSig, bins=bins, density=True, histtype='step', fill=False, color ='dodgerblue' ,edgecolor = 'blue', hatch = 'XX',label='Tprime')
n, bins, patches = plt.hist(tBkg, bins=bins, density=True, histtype='step', fill=False,color = 'red' ,alpha=0.5, edgecolor = 'green', hatch='++', label = 'ttgg')
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed')
plt.legend(loc='center')
# plt.savefig('/eos/home-s/sraj/M.Sc._Thesis/Plot/''output_TPrime_ttgg.png')
plt.show()


# # Training and Testing of vbf as background and TPrime_600  as signal

# In[54]:


X_vbf = np.concatenate((signal_1, back_vbf))
Y_vbf = np.concatenate((np.ones(signal_1.shape[0]),
                    np.zeros(back_vbf.shape[0])))


# In[55]:


X_train_vbf,X_test_vbf, Y_train_vbf,Y_test_vbf = train_test_split(X_vbf, Y_vbf, test_size=0.33, random_state =5)


# In[56]:


h = clf.fit(X_train_vbf, Y_train_vbf, epochs = 100, batch_size= 900, validation_split = 0.25)

