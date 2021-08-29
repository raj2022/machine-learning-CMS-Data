
# coding: utf-8

# In[43]:


#How to read data from a .root file with having lots of 
#trees. Here I have import data from a root file at lxplus. 


# In[44]:


import ROOT as root   
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.utils import shuffle
import xgboost as xgb
from itertools import tee, islice
from ROOT import TFile, TCanvas, TPad, TPaveLabel, TPaveText, TTree, TH1F, TF1
from root_numpy import root2array, tree2array, array2tree, array2root
import sys
from ROOT import gROOT, AddressOf


# In[45]:


back = '/afs/cern.ch/user/s/sraj/public/back.root'
signal = '/afs/cern.ch/user/s/sraj/public/signal.root'
out_dir = '/afs/cern.ch/user/s/sraj/public/plot'
treeName_back = "tagsDumper/trees/tth_125_13TeV_THQLeptonicTag"
treeName_signal = "tagsDumper/trees/thq_125_13TeV_THQLeptonicTag"
columns = ['dipho_pt','dipho_phi','dipho_eta','dipho_e','dipho_leadPt','dipho_leadEt','dipho_leadEta','dipho_leadPhi']


# In[46]:


# back_arr = root2array(background, treeName, columns)


# In[47]:


f = root.TFile("/afs/cern.ch/user/s/sraj/public/back.root")
f.ls()


# In[48]:


mc_arr = root2array(back, treeName_back, columns)
data_arr = root2array(signal, treeName_signal, columns)


# In[50]:


mc_arr = shuffle(mc_arr)
data_arr = shuffle(data_arr)
df_mc_ = pd.DataFrame(mc_arr)
df_data_ = pd.DataFrame(data_arr)
# print(df_mc_['dipho_pt'],df_mc_['dipho_phi'])
df_mc_['dipho_pt'].plot.hist(bins = 40, alpha =0.5)
df_data_['dipho_pt'].plot.hist(bins = 40, alpha =0.5)


plt.show()


# In[51]:


df_mc_['dipho_phi'].plot.hist(bins = 500, alpha =0.5)

