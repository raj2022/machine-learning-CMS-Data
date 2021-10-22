import training_utils as utils
import os
import numpy as np
import pandas as pd
import root_pandas as rpd
from ROOT import TFile, TH1F
import copy 

def scale_weight(dataframe, sf):
    print 'Weighting with SF : '
    dataframe['weight'] *= sf
    
    
def define_process_weight(df,proc,name,treename='bbggSelectionTree',cleanSignal=True,cleanOverlap=False):
    df['proc'] = ( np.ones_like(df.index)*proc ).astype(np.int8)
    if treename=='bbggSelectionTree':
        df['weight'] = ( np.ones_like(df.index)).astype(np.float32)
        input_df=rpd.read_root(name,treename, columns = ['genTotalWeight', 'lumiFactor','isSignal','puweight'])
        w = np.multiply(input_df[['lumiFactor']],input_df[['genTotalWeight']])
        w = np.multiply(w,input_df[['puweight']])
        df['lumiFactor'] = input_df[['lumiFactor']]
        df['genTotalWeight'] = input_df[['genTotalWeight']]
        df['isSignal'] = input_df[['isSignal']]
        if cleanSignal:#some trees include also the control region,select only good events
            df['weight']= np.multiply(w,input_df[['isSignal']])
        else:
            df['weight']=w

    df['overlapSave']  = np.ones_like(df.index).astype(np.int8)
    if cleanOverlap : cleanOverlapDiphotons(name,df)

def scale_lumi(dataframe):
    print 'Weighting with lumi : '
    dataframe['weight'] *= 41.5/35.9  #scale with lumi 2017
    
def normalize_process_weights_split_all(w,y):
    sum_weights_b = 0
    sum_weights_s = 0
    proc_considered = []
    for proc in np.unique(y):
        if proc!=1:  #fist bkg
            w_proc = np.asarray(w[np.asarray(y) == proc])
            sum_weights_b += float(np.sum(w_proc))
        else : 
            w_proc = np.asarray(w[np.asarray(y) == proc])
            sum_weights_s += float(np.sum(w_proc))
        proc_considered.append(proc)
    w[np.where(y==1)] = np.divide(w[np.where(y==1)],sum_weights_s)
    w[np.where(y!=1)] = np.divide(w[np.where(y!=1)],sum_weights_b)

    return w

def get_training_sample(x,splitting=0.5):
    halfSample = int((x.size/len(x.columns))*splitting)
    return np.split(x,[halfSample])[0]


def get_test_sample(x,splitting=0.5):
    halfSample = int((x.size/len(x.columns))*splitting)
    return np.split(x,[halfSample])[1]

    
def get_total_training_sample(x_sig,x_bkg,splitting=0.5):
    x_s=pd.DataFrame(x_sig)
    x_b=pd.DataFrame(x_bkg)
    halfSample_s = int((x_s.size/len(x_s.columns))*splitting)
    halfSample_b = int((x_b.size/len(x_b.columns))*splitting)
    return np.concatenate([np.split(x_s,[halfSample_s])[0],np.split(x_b,[halfSample_b])[0]])

    
def get_total_test_sample(x_sig,x_bkg,splitting=0.5):
    x_s=pd.DataFrame(x_sig)
    x_b=pd.DataFrame(x_bkg)
    halfSample_s = int((x_s.size/len(x_s.columns))*splitting)
    halfSample_b = int((x_b.size/len(x_b.columns))*splitting)
    return np.concatenate([np.split(x_s,[halfSample_s])[1],np.split(x_b,[halfSample_b])[1]])

def get_total_test_sample_event_num(x_sig,x_bkg,event_sig,event_bkg,sig_frac=4,bkg_frac=5):
    x_s = x_sig[np.where(event_sig%sig_frac!=0)]
    x_b = x_bkg[np.where(event_bkg%bkg_frac==0)]
    return np.concatenate((x_s,x_b))

def get_total_training_sample_event_num(x_sig,x_bkg,event_sig,event_bkg,sig_frac=4,bkg_frac=5):
    x_s = x_sig[np.where(event_sig%sig_frac==0)]
    x_b = x_bkg[np.where(event_bkg%bkg_frac!=0)]
    return np.concatenate((x_s,x_b))
            
#         adjust_and_compress(utils.IO.signal_df[i]).to_hdf('/tmp/micheli/signal.hd5','sig',compression=9,complib='bzip2',mode='a')

def set_signals(branch_names,shuffle,cuts='event>=0'):
    for i in range(utils.IO.nSig):
        treeName = utils.IO.signalTreeName[i]
        print "using tree:"+treeName
	utils.IO.signal_df.append((rpd.read_root(utils.IO.signalName[i],treeName, columns = branch_names)).query(cuts))
	define_process_weight(utils.IO.signal_df[i],utils.IO.sigProc[i],utils.IO.signalName[i],treeName)
	utils.IO.signal_df[i]['year'] = (np.ones_like(utils.IO.signal_df[i].index)*utils.IO.sigYear[i] ).astype(np.int8)
    
    for i in range(utils.IO.nSig):
        utils.IO.signal_df[i] = drop_nan(utils.IO.signal_df[i])
        if shuffle:
            utils.IO.signal_df[i]['random_index'] = np.random.permutation(range(utils.IO.signal_df[i].index.size))
            utils.IO.signal_df[i].sort_values(by='random_index',inplace=True)

def  set_backgrounds(branch_names,shuffle,cuts='event>=0'):
    for i in range(utils.IO.nBkg):
           treeName = utils.IO.bkgTreeName[i]
           print "using tree: "+str(i)+" : "+treeName
           print "using tree(Prafulla): "+str(i)+" : "+treeName
           utils.IO.background_df.append((rpd.read_root(utils.IO.backgroundName[i],treeName, columns = branch_names)).query(cuts))
           define_process_weight(utils.IO.background_df[i],utils.IO.bkgProc[i],utils.IO.backgroundName[i],treeName)
           utils.IO.background_df[i]['year'] = (np.ones_like(utils.IO.background_df[i].index)*utils.IO.bkgYear[i] ).astype(np.int8)
           utils.IO.background_df[i] = drop_nan(utils.IO.background_df[i])
           if shuffle:
              utils.IO.background_df[i]['random_index'] = np.random.permutation(range(utils.IO.background_df[i].index.size))
              utils.IO.background_df[i].sort_values(by='random_index',inplace=True)

#def set_data(branch_names,cuts='event>=0'):
#    treeName = utils.IO.dataTreeName[0]
#    utils.IO.data_df.append((rpd.read_root(utils.IO.dataName[0],treeName, columns = branch_names)).query(cuts))
#    utils.IO.data_df[0]['proc'] =  ( np.ones_like(utils.IO.data_df[0].index)*utils.IO.dataProc[0] ).astype(np.int8)
#    utils.IO.data_df[0]['year'] = (np.ones_like(utils.IO.data_df[0].index)*utils.IO.dataYear[0] ).astype(np.int8)

#    if treeName=='bbggSelectionTree':
#       input_df=rpd.read_root(utils.IO.dataName[0],treeName, columns = ['isSignal'])
#       w = (np.ones_like(utils.IO.data_df[0].index)).astype(np.int8)
#       utils.IO.data_df[0]['weight'] = np.multiply(w,input_df['isSignal'])


#def set_variables_data(branch_names):
#    y_data = utils.IO.data_df[0][['proc']]
#    w_data = utils.IO.data_df[0][['weight']]
#    for j in range(len(branch_names)):
#        if j == 0:
#            X_data = utils.IO.data_df[0][[branch_names[j].replace('noexpand:','')]]
#        else:
#            X_data = np.concatenate([X_data,utils.IO.data_df[0][[branch_names[j].replace('noexpand:','')]]],axis=1)
    
#    return np.round(X_data,5),y_data,w_data

def set_signals_and_backgrounds(branch_names,shuffle=True,cuts='event>=0'):
    #signals will have positive process number while bkg negative ones
    set_signals(branch_names,shuffle,cuts)
    set_backgrounds(branch_names,shuffle,cuts)

    
def set_signals_and_backgrounds_drop(branch_names,shuffle=True,cuts='event>=0'):
    #signals will have positive process number while bkg negative ones
    set_signals_drop(branch_names,shuffle,cuts)
    set_backgrounds(branch_names,shuffle,cuts)
    
    

def randomize(X,y,w,event_num=None,seed=0):
    randomize=np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(randomize)
    X = X[randomize]
    y = np.asarray(y)[randomize]
    w = np.asarray(w)[randomize]
    if event_num is None :
        event_num = np.asarray(event_num)[randomize]
        return X,y,w
    else : 
        return X,y,w,event_num
    
    
    
def set_variables(branch_names,use_event_num=False):
    for i in range(utils.IO.nSig):
        if i ==0:
            y_sig = utils.IO.signal_df[i][['proc']]
            #w_sig = utils.IO.signal_df[i][['weight']]
            sig_weight = np.sum(utils.IO.signal_df[i]['weight'])
            print sig_weight
            w_sig = utils.IO.signal_df[i][['weight']]/sig_weight
            if use_event_num :  event_sig = utils.IO.signal_df[i][['event']]
            print branch_names
            for j in range(len(branch_names)):
                if j == 0:
                    X_sig = utils.IO.signal_df[i][[branch_names[j].replace('noexpand:','')]]
                else:
                    X_sig = np.concatenate([X_sig,utils.IO.signal_df[i][[branch_names[j].replace('noexpand:','')]]],axis=1)
        else:
            y_sig = np.concatenate((y_sig,utils.IO.signal_df[i][['proc']]))
            sig_weight = np.sum(utils.IO.signal_df[i]['weight'])
            print "multi signal inclusion"
            print sig_weight
            w_sig = np.concatenate((w_sig,utils.IO.signal_df[i][['weight']]/sig_weight))
            if use_event_num : event_sig = np.concatenate((event_sig,utils.IO.signal_df[i][['event']]))
            print branch_names
            for j in range(len(branch_names)):
                if j == 0:
                    X_sig_2 = utils.IO.signal_df[i][[branch_names[j].replace('noexpand:','')]]
                else:
                    X_sig_2 = np.concatenate([X_sig_2,utils.IO.signal_df[i][[branch_names[j].replace('noexpand:','')]]],axis=1)
            X_sig=np.concatenate((X_sig,X_sig_2))
    for i in range(utils.IO.nBkg):
        if i ==0:
            y_bkg = utils.IO.background_df[i][['proc']]
            bkg_weight = np.sum(utils.IO.background_df[i]['weight'])
            print bkg_weight
            w_bkg = utils.IO.background_df[i][['weight']]/bkg_weight
            if use_event_num : event_bkg = utils.IO.background_df[i][['event']]
            print branch_names
            for j in range(len(branch_names)):
                if j == 0:
                    X_bkg = utils.IO.background_df[i][[branch_names[j].replace('noexpand:','')]]
                else:
                    X_bkg = np.concatenate([X_bkg,utils.IO.background_df[i][[branch_names[j].replace('noexpand:','')]]],axis=1)
        else:
            y_bkg = np.concatenate((y_bkg,utils.IO.background_df[i][['proc']]))
            bkg_weight = np.sum(utils.IO.background_df[i]['weight'])
            print bkg_weight
            w_bkg = np.concatenate((w_bkg,utils.IO.background_df[i][['weight']]/bkg_weight))
            if use_event_num : event_bkg = np.concatenate((event_bkg,utils.IO.background_df[i][['event']]))
            print branch_names
            for j in range(len(branch_names)):
                if j == 0:
                    X_bkg_2 = utils.IO.background_df[i][[branch_names[j].replace('noexpand:','')]]
                else:
                    X_bkg_2 = np.concatenate([X_bkg_2,utils.IO.background_df[i][[branch_names[j].replace('noexpand:','')]]],axis=1)
            X_bkg=np.concatenate((X_bkg,X_bkg_2))

    if not use_event_num :  return np.round(X_bkg,5),y_bkg,w_bkg,np.round(X_sig,5),y_sig,w_sig
    else :   return np.round(X_bkg,5),y_bkg,w_bkg,event_bkg,np.round(X_sig,5),y_sig,w_sig,event_sig

   

def check_for_nan(df,branch_name='event'):
    print df.isnull().sum()
    index = df[branch_name].index[df[branch_name].apply(np.isnan)]
    print 'event numbers for nan events : ', df['event'][index]
    new_df = df.drop(df.index[index])
    return new_df


    
def drop_from_df(df,index):
    return df.drop(df.index[index])

def drop_nan(df):
    return df.dropna()

def profile(target,xvar,bins=10,range=None,uniform=False,moments=True,
            quantiles=np.array([0.25,0.5,0.75])):

    if range is None:
        if type(bins) is not int:
            xmin, xmax = bins.min(), bins.max()
        else:
            xmin, xmax = xvar.min(),xvar.max()
    else:
        xmin, xmax = range
    mask = ( xvar >= xmin ) & ( xvar <= xmax )
    xvar = xvar[mask]
    target = target[mask]
    if type(bins) == int:
        if uniform:
            bins = np.linspace(xmin,xmax,num=bins+1)
        else:
            ## print(xmin,xmax)
            ## xvar = np.clip( xvar, xmin, xmax )
            bins = np.percentile( xvar, np.linspace(0,100.,num=bins+1) )
            bins[0] = xmin
            bins[-1] = xmax
    print bins.shape 
    ibins = np.digitize(xvar,bins)-1
    categories = np.eye(np.max( ibins ) + 1)[ibins]

    ret = [bins]
    if moments:
        mtarget = target.reshape(-1,1) * categories
        weights = categories
        mean = np.average(mtarget,weights=categories,axis=0)
        mean2 = np.average(mtarget**2,weights=categories,axis=0)
        ret.extend( [mean, np.sqrt( mean2 - mean**2)] )
    if quantiles is not None:
        values = []
        print(categories.shape[1])
        for ibin in np.arange(categories.shape[1],dtype=int):
            values.append( np.percentile(target[categories[:,ibin].astype(np.bool)],quantiles*100.,axis=0).reshape(-1,1) )
            ## print(values)
        ret.append( np.concatenate(values,axis=-1) )
    return tuple(ret)
