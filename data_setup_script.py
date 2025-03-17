#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import optimize as scopt
import scipy.stats as stats
import scipy.integrate as scint
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
## Plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import seaborn as sns

import pyabc

# Custom imports
import pyEnceladus.universal_htv as uhtv
import pyEnceladus.data_htv as dhtv
import pyEnceladus.physical as phc
import pyEnceladus.plot_tools as cplt
from pyEnceladus.biosims import makePriors, makeOnePrior
from pyEnceladus.plot_abc import confusion_format
import pyEnceladus.simulation_qstar as smqs
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import sys, traceback
import argparse
import os

# Data from Waite et al. 2017
h2_low   = 0.4
h2_high  = 1.4
co2_low  = 0.3
co2_high = 0.8
ch4_low  = 0.1
ch4_high = 0.3
water_ejec = 200 #kg.s-1
timec = 60*60*24*365
flx_obs = lambda mc : mc*water_ejec*3600*24*365
observations = pd.DataFrame(data= {'R1':np.array([h2_low/ch4_high,h2_high/ch4_low]),
                                   'FH2':flx_obs(np.array([h2_low,h2_high])),
                                   'FCH4':flx_obs(np.array([ch4_low,ch4_high]))})
logobservations = pd.DataFrame(data={'R1':np.log10(observations['R1']),
                                     'FH2':np.log10(observations['FH2']),
                                     'FCH4':np.log10(observations['FCH4'])})
observation = {'R1':np.mean(observations['R1']),'FH2':np.mean(observations['FH2']),'FCH4':np.mean(observations['FCH4'])}

observations.to_csv('data/observations/observations.csv',sep=';')
logobservations.to_csv('data/observations/logobservations.csv',sep=';')
np.save('data/observations/observation.npy',observation)

# functions to compute observables

def R1fun(out,I):
    if I:
        return(out['H2']/out['CH4'])
    else:
        return(out['H2_ab']/out['CH4_ab'])

def R2fun(out,I):
    if I:
        return(out['H2']/out['CO2'])
    else:
        return(out['H2_ab']/out['CO2_ab'])

def Ffun(out,var,I):
    timec = 60*60*24*365
    if I:
        return(out[var]*timec)
    else:
        return(out[var+'_ab']*timec)
    
def modelfun(I,H):
    if I:
        return('inhabited')
    elif H:
        return('habitable')
    else:
        return('uninhabitable')

def tablegen(Ntrain,simulations,PIH,bmass_cutoff,seed=None):
    "bmass cutoff is a percentage eg 5% is 0.05"
    if seed is not None:
        np.random.seed(seed)
    
    minbmass = np.sort(simulations['totB'][simulations['totB']>0])[int(len(simulations)*bmass_cutoff)]
        
    Ntest  = len(simulations) - Ntrain
    keys = ['R1','FH2','FCH4']
    sim_hab = simulations['H2_ab'] != simulations['H2']
    reftable_data = {}
    reftable_ref  = []

    testtable_data = {}
    testtable_ref  = []

    
    for key in keys:
        reftable_data[key] = []
        testtable_data[key] = []
    for i in range(Ntrain):
        out = simulations.loc[i]
        I = False
        H = False
        if sim_hab[i]:
            H = True
            if np.random.rand()<PIH:
                if out['totB'] > minbmass:
                    I = True
        reftable_data['FH2'].append(Ffun(out,'H2',I))   
        reftable_data['FCH4'].append(Ffun(out,'CH4',I))
        reftable_data['R1'].append(R1fun(out,I))
        reftable_ref.append(modelfun(I,H))
        
    for i in range(Ntest):
        out = simulations.loc[Ntrain+i]
        I = False
        H = False
        if sim_hab[Ntrain+i]:
            H = True
            if np.random.rand()<PIH:
                if out['totB'] > minbmass:
                    I = True
        testtable_data['FH2'].append(Ffun(out,'H2',I))   
        testtable_data['FCH4'].append(Ffun(out,'CH4',I))
        testtable_data['R1'].append(R1fun(out,I))
        testtable_ref.append(modelfun(I,H))
        
    for key in keys:
        reftable_data[key] = np.array(reftable_data[key])
        testtable_data[key] = np.array(testtable_data[key])
    reftable_ref = np.array(reftable_ref)
    testtable_ref = np.array(testtable_ref)
    return(reftable_data,reftable_ref,testtable_data,testtable_ref)


def BoostrapSimulations(K,seeds,outcomes,observations,Ntrain,bmass_cutoff):
    probs = np.linspace(0.05,0.95,K)
    bootstrap = {'confusion':[],'PIH':[],'PI':[],'score':[],'prediction':[],'probabilities':[]}
    for k in range(K):
        reftable_data,reftable_ref,testtable_data,testtable_ref = tablegen(Ntrain = Ntrain,simulations=outcomes,PIH=probs[k],seed=seeds[k],bmass_cutoff=bmass_cutoff)
        training_df = pd.DataFrame(data=reftable_data)
        test_df = pd.DataFrame(data=testtable_data)
        rfcl = RandomForestClassifier(random_state=seeds[k],n_estimators=500,criterion='gini',max_depth=20,
                                      max_features=3,warm_start=False,min_samples_leaf=10,min_samples_split=100,
                                      bootstrap=True,max_leaf_nodes=None,oob_score=True)
        rfcl.fit(training_df, reftable_ref)
        crossvalid = rfcl.predict(test_df)
        matrix = metrics.confusion_matrix(testtable_ref,crossvalid)
        confusion     = np.round(matrix/np.repeat(np.sum(matrix,axis=1),3).reshape(3,3),2)
        score         = rfcl.score(test_df,testtable_ref)
        prediction    = rfcl.predict(observations)
        probabilities = rfcl.predict_proba(observations)

        bootstrap['confusion'].append(confusion)
        bootstrap['score'].append(score)
        bootstrap['prediction'].append(prediction)
        bootstrap['probabilities'].append(probabilities)
        bootstrap['PIH'].append(probs[k])
        bootstrap['PI'].append(np.sum(reftable_ref=='inhabited')/len(reftable_ref))
    return(bootstrap)


def main(dirname,cutoff=0.0,K=10,Ntrain=40000,PIH=0.5,btsrp_seeds=None,tbgen_seed=None):
    ## load simulations
    rootpath = os.path.join('data',dirname)
    priorfile_name = os.path.join(rootpath,dirname+'.prior.csv')
    priors_std = pd.read_csv(priorfile_name,delimiter=';',index_col=0)
    outcome_name = os.path.join(rootpath,'000','core_batch_000raw.csv')
    outcomes_std = pd.read_csv(outcome_name,delimiter=';',index_col=0)

    ## turn raw model outputs into observables
    reftable_data_std,reftable_ref_std,testtable_data_std,testtable_ref_std = tablegen(Ntrain = Ntrain,simulations=outcomes_std,PIH=PIH,seed=tbgen_seed,bmass_cutoff=cutoff)
    ## Save data
    
    np.save(os.path.join(rootpath,"reftable_data_standard.npy"),reftable_data_std)
    np.save(os.path.join(rootpath,"reftable_ref_standard.npy"),reftable_ref_std)
    np.save(os.path.join(rootpath,"testtable_data_standard.npy"),testtable_data_std)
    np.save(os.path.join(rootpath,"testtable_ref_standard.npy"),testtable_ref_std)
    total_table_std = np.log10(pd.DataFrame(data=reftable_data_std))
    total_table_std['models'] = reftable_ref_std
    translate = {'uninhabitable':0,'habitable':1,'inhabited':2}
    total_table_std['model'] = np.array([translate[total_table_std['models'].loc[i]] for i in range(len(total_table_std))])
    total_table_std.to_csv(os.path.join(rootpath,"total_table.csv"),sep=';')
    
    if btsrp_seeds==None:
        seeds = np.arange(K)
    else:
        seeds = btsrp_seeds
    bootstrap_std = BoostrapSimulations(K,seeds,outcomes_std,observations,Ntrain=Ntrain,bmass_cutoff=cutoff)
    np.save(os.path.join(rootpath,"bootstrap_standard.npy"),bootstrap_std)
    print('done')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='data_setup_script.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    First layer of analyses raw model outputs 
                                     ''')
    
    # Required arguments
    parser.add_argument("--dirname", help="dirname of outputs",type=str,action="store",required=True)
    #Optional Arguments
    parser.add_argument('--cutoff',type=float,action='store',help='Cutoff percentage for biomass')
    parser.add_argument('-K','--nbootstrap',type=int,action='store',help='Number of bootstraps')
    parser.add_argument('-N','--Ntrain',type=int,action='store',help='Number of training data points')
    parser.add_argument('-P','--pih', type=float, action='store',help='P(I|H)')
    
    # Set optional
    parser.set_defaults(cutoff=0.0,
                        nbootstrap=10,
                        Ntrain=40000,
                        pih=0.5)
    args = parser.parse_args()
    path = os.getcwd()
    dirname=args.dirname
    # ------------------------------------------------------------------------------
    # Other parameters (simulation properties)
    # ------------------------------------------------------------------------------
    K = args.nbootstrap
    Ntrain = args.Ntrain
    PIH = args.pih
    cutoff=args.cutoff
    main(dirname=dirname,K=K,Ntrain=Ntrain,PIH=PIH,cutoff=cutoff)
    