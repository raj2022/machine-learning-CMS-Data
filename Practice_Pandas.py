
# coding: utf-8

# In[24]:


import root_numpy
events = root_numpy.root2array("/afs/cern.ch/user/s/sraj/public/data_folder/back.root","tagsDumper/trees/thq_125_13TeV_THQLeptonicTag",stop=10)


# In[16]:


import pandas as pd
import numpy as np


# The main data structure is a data frame. It can be created from a ndarray of named fields
# 
# 

# In[25]:


events_pd = pd.DataFrame(events)
events_pd ##REAding of all the file from the root file "."


# In[32]:


random_data = np.random.normal(size=(10,18))
random_data.shape


# In[33]:


pd.DataFrame(random_data, columns=('dipho_pt','dipho_phi','dipho_eta','dipho_e','dipho_mass','dipho_leadPt','dipho_leadEt','dipho_leadEta','dipho_leadPhi','dipho_subleadEta','bjet1_pt',
          'bjet2_pt','bjet1_eta','bjet2_eta','jet1_pt','jet2_pt','jet1_eta','n_jets'))


# In[43]:


h=events_pd.dipho_150pt
h


# In[39]:


events[0]


# columns can be nicely accessed
# 
# 

# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
h.hist(figsize=(5,5));


# In[48]:


events_pd.hist(figsize=(15,15));

