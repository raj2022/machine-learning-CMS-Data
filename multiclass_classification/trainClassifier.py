import os
#import sys; sys.path.append("/work/nchernya//HHbbgg_ETH_devel/Training/python") # to load packages
import sys; sys.path.append("/afs/cern.ch/work/p/prsaha/public/plotting_macro/CMSSW_9_4_9/src/Tprime/Training_XGBoost_v2/Higgs_NISER/Training/python")
import sys
import matplotlib
matplotlib.use('Agg')

import training_utils as utils
import numpy as np
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
import plotting_utils as plotting
reload(plotting)
#import optimization_utils as optimization
#reload(optimization)
#import postprocessing_utils as postprocessing
#reload(postprocessing)
import pandas as pd
import root_pandas as rpd
import matplotlib.pyplot as plt
import json
from ROOT import TLorentzVector
from optparse import OptionParser, make_option
from  pprint import pprint
import commands
import time
import datetime
start_time = time.time()



def main(options,args):
    year=options.year
    #please specify which year you want 
#    Y = options.year
    Y = 2017
#    if '28i16' in Y :
#        year  =0
#    if '2017' in Y :
#        year  =1
#    if '2018' in Y :
#        year  =2
#    outstr = "%s_MX_gt_500_ttHkiller_0p26_NLO_reweight_sigfrac_2_gghh_0p3"%Y
    outstr = "mixed_Tprime600_700_Vs_HiggsBkg"
    doRhoReweight = False
    dirs = ['']
    ntuples = dirs[year]
    print Y
    print year
    print ntuples

    THQname = 'thq_125_13TeV_THQLeptonicTag'
    TTHname = 'tth_125_13TeV_THQLeptonicTag'
    VHname = 'vh_125_13TeV_THQLeptonicTag'
    Tprime_600 = 'Tprime_600_13TeV_THQLeptonicTag'
    Tprime_625 = 'Tprime_625_13TeV_THQLeptonicTag'
    Tprime_650 = 'Tprime_650_13TeV_THQLeptonicTag'
    Tprime_675 = 'Tprime_675_13TeV_THQLeptonicTag'
    Tprime_700 = 'Tprime_700_13TeV_THQLeptonicTag'
    Tprime_800 = 'Tprime_800_13TeV_THQLeptonicTag'
    Tprime_900 = 'Tprime_900_13TeV_THQLeptonicTag'
    Tprime_1000 = 'Tprime_1000_13TeV_THQLeptonicTag'
    Tprime_1100 = 'Tprime_1100_13TeV_THQLeptonicTag'
    Tprime_1200 = 'Tprime_1200_13TeV_THQLeptonicTag'

    ttggname = 'ttgg_13TeV_THQLeptonicTag'
    ttgjetsname = 'ttgjets_13TeV_THQLeptonicTag'
    DiPhotonJetsBoxname = 'DiPhotonJetsBox_13TeV_THQLeptonicTag'
    gjetsname = 'gjets_13TeV_THQLeptonicTag'
    DiPhotonJetsBox2BJetsname = 'DiPhotonJetsBox2BJets_13TeV_THQLeptonicTag'    

    NodesNormalizationFile = '/afs/cern.ch/user/n/nchernya/public/Soumya/reweighting_normalization_26_11_2019.json'
#ps    useMixOfNodes = True
    useMixOfNodes = False
    whichNodes = ['SM']
    ggHHMixOfNodesNormalizations = json.loads(open(NodesNormalizationFile).read())
    # "%" sign allows to interpret the rest as a system command
    get_ipython().magic(u'env data=$utils.IO.ldata$ntuples')
    status,files = commands.getstatusoutput('! ls $data | sort -t_ -k 3 -n')
    files=files.split('\n')   
    print files    
    signal_600 = [s for s in files if ("output_TprimeBToTH_Hgg_M-600" in s) ]
    signal_625 = [s for s in files if ("output_TprimeBToTH_Hgg_M-625" in s) ]
    signal_650 = [s for s in files if ("output_TprimeBToTH_Hgg_M-650" in s) ]
    signal_675 = [s for s in files if ("output_TprimeBToTH_Hgg_M-675" in s) ]
    signal_700 = [s for s in files if ("output_TprimeBToTH_Hgg_M-700" in s) ]
    signal_800 = [s for s in files if ("output_TprimeBToTH_Hgg_M-800" in s) ]
    signal_900 = [s for s in files if ("output_TprimeBToTH_Hgg_M-900" in s) ]
    signal_1000 = [s for s in files if ("output_TprimeBToTH_Hgg_M-1000" in s) ]
    signal_1100 = [s for s in files if ("output_TprimeBToTH_Hgg_M-1100" in s) ]
    signal_1200 = [s for s in files if ("output_TprimeBToTH_Hgg_M-1200" in s) ]
    thq = [s for s in files if ("output_THQ" in s) ]    
    tth = [s for s in files if ("output_ttHJetToGG_M125" in s) ]
    vh = [s for s in files if ("output_VHToGG_M125" in s) ]

    print signal_600

    utils.IO.reweightVBFHH = False
#************************************************************************************************************************************************
    
#    for sig in signal:
    utils.IO.add_signal(ntuples,signal_600,1,'tagsDumper/trees/%s'%Tprime_600)
    utils.IO.add_signal(ntuples,signal_625,1,'tagsDumper/trees/%s'%Tprime_625)
#    utils.IO.add_signal(ntuples,signal_650,1,'tagsDumper/trees/%s'%Tprime_650)
#    utils.IO.add_signal(ntuples,signal_675,1,'tagsDumper/trees/%s'%Tprime_675)
#    utils.IO.add_signal(ntuples,signal_700,1,'tagsDumper/trees/%s'%Tprime_700)
#    utils.IO.add_signal(ntuples,signal_800,1,'tagsDumper/trees/%s'%Tprime_800)
#    utils.IO.add_signal(ntuples,signal_900,1,'tagsDumper/trees/%s'%Tprime_900)
#    utils.IO.add_signal(ntuples,signal_1000,1,'tagsDumper/trees/%s'%Tprime_1000)
#    utils.IO.add_signal(ntuples,signal_1100,1,'tagsDumper/trees/%s'%Tprime_1100)
#    utils.IO.add_signal(ntuples,signal_1200,1,'tagsDumper/trees/%s'%Tprime_1200)


    utils.IO.add_background(ntuples,thq,-1, 'tagsDumper/trees/%s'%THQname)
    utils.IO.add_background(ntuples,tth,-2, 'tagsDumper/trees/%s'%TTHname)
    utils.IO.add_background(ntuples,vh,-3, 'tagsDumper/trees/%s'%VHname)    
#    utils.IO.add_background(ntuples,ttgjets,-1, 'tagsDumper/trees/%s'%ttgjetsname)
#    utils.IO.add_background(ntuples,DiPhotonJetsBox,-1, 'tagsDumper/trees/%s'%DiPhotonJetsBoxname)
#    utils.IO.add_background(ntuples,gjets,-1, 'tagsDumper/trees/%s'%gjetsname)
#    utils.IO.add_background(ntuples,DiPhotonJetsBox2BJets,-1, 'tagsDumper/trees/%s'%DiPhotonJetsBox2BJetsname)
#******************************for three class single Higgs bkgs*********************************************************************************
    #utils.IO.add_background(ntuples,ggh,-3, 'tagsDumper/trees/%s'%gghname[year],year)    
    #utils.IO.add_background(ntuples,vh,-3, 'tagsDumper/trees/%s'%vhname[year],year)
    #utils.IO.add_background(ntuples,qqh,-3, 'tagsDumper/trees/%s'%qqhname[year],year)
    #utils.IO.add_background(ntuples,tth,-3, 'tagsDumper/trees/%s'%tthname[year],year)
#************************************************************************************************************************************************

    for i in range(len(utils.IO.backgroundName)):        
        print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]
    for i in range(len(utils.IO.signalName)):    
        print "using signal file n."+str(i)+": "+utils.IO.signalName[i]


    utils.IO.plotFolder = '/eos/user/p/prsaha/www/xgboost_outout/plots/%s/'%outstr
    if not os.path.exists(utils.IO.plotFolder):
        print utils.IO.plotFolder, "doesn't exist, creating it..."
        os.makedirs(utils.IO.plotFolder)

    #use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
    ########################new code branches############################
#    branch_names = 'noexpand:(dipho_leadPt/dipho_mass),noexpand:(dipho_subleadPt/dipho_mass),dipho_leadEta,dipho_subleadEta,dipho_leadIDMVA,dipho_subleadIDMVA,dipho_lead_haspixelseed,dipho_sublead_haspixelseed,n_jets,n_bjets,n_centraljets,lepton_charge,lepton_leadPt,lepton_leadEta,fwdjet1_pt,fwdjet1_eta,fwdjet1_discr,top_mt,dr_tHchainfwdjet,dr_leptonbjet,dr_leptonfwdjet,dr_bjetfwdjet,dr_leadphofwdjet,dr_subleadphofwdjet,bjet1_pt,bjet2_pt,bjet3_pt,bjet1_eta,bjet2_eta,bjet3_eta,bjet1_discr,bjet2_discr,bjet3_discr,jet1_pt,jet2_pt,jet3_pt,jet1_eta,jet2_eta,jet3_eta,jet1_discr,jet2_discr,jet3_discr'.split(",")    
#ps    branch_cuts = 'leadingJet_pt,subleadingJet_pt,leadingJet_bRegNNCorr,subleadingJet_bRegNNCorr,noexpand:(leadingJet_pt/leadingJet_bRegNNCorr),noexpand:(subleadingJet_pt/subleadingJet_bRegNNCorr)'.split(',')
    branch_names = 'noexpand:(dipho_leadPt/dipho_mass),noexpand:(dipho_subleadPt/dipho_mass),dipho_leadEta,dipho_subleadEta,dipho_leadIDMVA,dipho_subleadIDMVA,dipho_lead_haspixelseed,dipho_sublead_haspixelseed,n_jets,n_bjets,n_centraljets,lepton_charge,lepton_leadPt,lepton_leadEta,fwdjet1_pt,fwdjet1_discr,dr_tHchainfwdjet,dr_leptonbjet,dr_leptonfwdjet,dr_bjetfwdjet,dr_leadphofwdjet,dr_subleadphofwdjet,bjet1_pt,bjet1_eta,bjet1_discr,jet1_pt,jet2_pt,jet1_eta,jet2_eta,jet1_discr,jet2_discr,noexpand:((solvedMET_pt + dipho_pt + lepton_leadPt + bjet1_pt)/(HT + recoMET_pt + lepton_leadPt)),recoMET_pt'.split(",")
    nodesWeightBranches=[]
    if utils.IO.signalMixOfNodes : nodesWeightBranches=[ 'benchmark_reweight_%s'%i for i in whichNodes ] 
    #cuts = 'subleadingJet_pt>25'
    ######################
    print (nodesWeightBranches)
    event_branches = ['event','weight','lepton_leadPt'] 
#ps    event_branches+=['leadingJet_phi','leadingJet_eta','subleadingJet_phi','subleadingJet_eta']
#ps    event_branches+=['leadingPhoton_eta','leadingPhoton_phi','subleadingPhoton_eta','subleadingPhoton_phi','ttHScore']
    #cuts = 'ttHScore > 0.26'
    cuts = 'weight > 0 & lepton_leadPt > 10'

    resolution_weighting = 'ggbb' # None, gg or ggbb
    doOverlapRemoval=False   #diphotons overlap removal if using b-enriched samples


    branch_names = [c.strip() for c in branch_names]
    print branch_names

    event_bkg,event_sig = None,None
    preprocessing.set_signals(branch_names+event_branches,True,cuts)
    preprocessing.set_backgrounds(branch_names+event_branches,True,cuts)

    info_file = open(utils.IO.plotFolder+"info_%s.txt"%outstr,"w") 
    info_file.write("\n".join(branch_names))
    info_file.write("Resolution weighting : %s\n"%resolution_weighting)
    info_file.write("Cuts : %s\n"%cuts)
    info_file.write("Signal weighted Events Sum before inverse resolution weighting : \n")
    info_file.write("%.4f \n"%(np.sum(utils.IO.signal_df[0]['weight']))) 
    info_file.write("Background weighted Events Sum : \n")
    sum_bkg_weights = 0
    for bkg_type in range(utils.IO.nBkg):
        bkg_weight = np.sum(utils.IO.background_df[bkg_type]['weight'])
        sum_bkg_weights+=bkg_weight
        info_file.write("proc %d : %.4f \n"%( utils.IO.bkgProc[bkg_type],bkg_weight)) 
    info_file.write("Background weighted Events Sum Total : %.4f \n"%(sum_bkg_weights)) 
    info_file.close()


    X_bkg,y_bkg,weights_bkg,event_bkg,X_sig,y_sig,weights_sig,event_sig=preprocessing.set_variables(branch_names,use_event_num=True)

    X_bkg,y_bkg,weights_bkg,event_bkg = preprocessing.randomize(X_bkg,y_bkg,weights_bkg,event_num = np.asarray(event_bkg))
    X_sig,y_sig,weights_sig,event_sig = preprocessing.randomize(X_sig,y_sig,weights_sig,event_num = np.asarray(event_sig))

    #Get training and test samples based on event number : even/odd or %5, set in the function for now
    y_total_train = preprocessing.get_total_training_sample_event_num(y_sig.reshape(-1,1),y_bkg,event_sig.reshape(-1,1),event_bkg).ravel()
    X_total_train = preprocessing.get_total_training_sample_event_num(X_sig,X_bkg,event_sig.reshape(-1,),event_bkg.reshape(-1,))

    y_total_test = preprocessing.get_total_test_sample_event_num(y_sig.reshape(-1,1),y_bkg,event_sig.reshape(-1,1),event_bkg).ravel()
    X_total_test = preprocessing.get_total_test_sample_event_num(X_sig,X_bkg,event_sig.reshape(-1,),event_bkg.reshape(-1,))

    w_total_train = preprocessing.get_total_training_sample_event_num(weights_sig.reshape(-1,1),weights_bkg.reshape(-1,1),event_sig.reshape(-1,1),event_bkg).ravel()
    w_total_test = preprocessing.get_total_test_sample_event_num(weights_sig.reshape(-1,1),weights_bkg.reshape(-1,1),event_sig.reshape(-1,1),event_bkg).ravel()


    ##########Normalize weights for training and testing. Sum(signal)=Sum(bkg)=1. But keep relative normalization
    # between bkg classes
    w_total_train = preprocessing.normalize_process_weights_split_all(w_total_train,y_total_train)
    w_total_test = preprocessing.normalize_process_weights_split_all(w_total_test,y_total_test)


    print "Starting the training now : "
    now = str(datetime.datetime.now())
    print(now)
    
    ################Training a classifier###############
    ########final optimization with all fixed#######
    from sklearn.externals import joblib
    import xgboost as xgb
    n_threads=10

    #optimized parameters with Mjj for 2016 done by Francesco
#Optimized for the 2017 C2V_2 training 
    clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
           min_child_weight=1e-6,  n_estimators=1000,
           nthread=n_threads, objective='binary:logistic', reg_alpha=0.0,
           reg_lambda=0.05, scale_pos_weight=1, seed=None, silent=True,
           subsample=1)

#    clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
#           min_child_weight=1e-06,  n_estimators=400,
#           nthread=n_threads, objective='multi:softprob', reg_alpha=0.0,
#           reg_lambda=0.05, scale_pos_weight=1, seed=None, silent=True,
#           subsample=1)


    clf.fit(X_total_train,y_total_train, sample_weight=w_total_train)
##########################################
    y_test_predict = clf.predict(X_total_test)
    y_train_predict = clf.predict(X_total_train)
    from sklearn.metrics import accuracy_score
    print('Train accuracy', accuracy_score(y_total_train, y_train_predict))
    print('Test accuracy', accuracy_score(y_total_test, y_test_predict))
##########################################    
    
    print 'Training is done. It took', time.time()-start_time, 'seconds.'
    #print(clf.feature_importances_)    
    #from xgboost import plot_importance
    #from matplotlib import pyplot
    #plot_importance(clf)
    #pyplot.rcParams['figure.figsize'] = [5, 5]
    #pyplot.savefig('graph_2018_setII.png')

#ps    _,_,_ = plt.hist(utils.IO.signal_df[0]['PhoJetMinDr'], np.linspace(0,3,30), facecolor='b',weights=utils.IO.signal_df[0]['weight'], alpha=0.5,normed=False,label='2018')
#ps    plt.xlabel('rho [GeV]')
#ps    plt.ylabel('A.U.')
#ps    plt.savefig('%s_2018.png'%Y)

    #_,_,_ = plt.hist(utils.IO.background_df[5]['MX'], np.linspace(0,2500,245), facecolor='b',weights=utils.IO.background_df[5]['weight'], alpha=0.5,normed=False,label='2016')
    #plt.xlabel('MX [GeV]')
    #plt.ylabel('A.U.')
    #plt.savefig('MX_ggHH_reweightedSM_2017.png') 



    joblib.dump(clf, os.path.expanduser('/afs/cern.ch/work/p/prsaha/public/XGBoost_train/output/plots/training_with_%s.pkl'%outstr), compress=9)

    plot_classifier = plotting.plot_classifier_output(clf,-1,X_total_train,X_total_test,y_total_train,y_total_test,w_total_train,w_total_test,outString=outstr)
    plot_classifier_gghh = plotting.plot_classifier_output(clf,-2,X_total_train,X_total_test,y_total_train,y_total_test,w_total_train,w_total_test,outString=outstr) 

#ps    plot_classifier_weight = plotting.plot_classifier_output_weight(clf,-1,X_total_train,X_total_test,y_total_train,y_total_test,w_total_train,w_total_test,outString=outstr)
#ps    plot_classifier_gghh_weight = plotting.plot_classifier_output_weight(clf,-2,X_total_train,X_total_test,y_total_train,y_total_test,w_total_train,w_total_test,outString=outstr) 




    fpr_dipho,tpr_dipho = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-1,outString=outstr,weights=w_total_test)
    fpr_gJets,tpr_gJets = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-2,outString=outstr,weights=w_total_test)
    #fpr_singleH,tpr_singleH = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-3,outString=outstr,weights=w_total_test)


#ps    roc_df_dipho = pd.DataFrame({"fpr_dipho": (fpr_dipho).tolist(),"tpr_dipho": (tpr_dipho).tolist()})
#ps    roc_df_gJets = pd.DataFrame({"fpr_gJets": (fpr_gJets).tolist(),"tpr_gJets": (tpr_gJets).tolist()})
    #roc_df_singleH = pd.DataFrame({"fpr_singleH": (fpr_singleH).tolist(),"tpr_singleH": (tpr_singleH).tolist()})
#ps    roc_df_dipho.to_hdf(utils.IO.plotFolder+"roc_curves_dipho_%s.h5"%outstr, key='df', mode='w')
#ps    roc_df_gJets.to_hdf(utils.IO.plotFolder+"roc_curves_gJets_%s.h5"%outstr, key='df', mode='w')
    #roc_df_singleH.to_hdf(utils.IO.plotFolder+"roc_curves_singleH_%s.h5"%outstr, key='df', mode='w')




if __name__ == "__main__":

    parser = OptionParser(option_list=[
            make_option("-y","--year",
                        action="store",type=int,dest="year",default=0,
                        help="which year : 2016-0,2017-1,2018-2",
                        )
            ])

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    pprint(options.__dict__)
    
    main(options,args)
        
