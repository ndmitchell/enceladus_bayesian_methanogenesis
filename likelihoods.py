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

# Custom imports
import pyEnceladus.universal_htv as uhtv
import pyEnceladus.data_htv as dhtv
import pyEnceladus.physical as phc
import pyEnceladus.plot_tools as cplt
from pyEnceladus.biosims import makePriors, makeOnePrior
from pyEnceladus.plot_abc import confusion_format
import pyEnceladus.simulation_qstar as smqs
import pyEnceladus.stats_analysis as homestats
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import os

mpl.rc('axes', labelsize=7)
mpl.rc('legend', fontsize=7)
mpl.rc('xtick', labelsize=5.5)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=5.5)

labelsDict = {'FH2':r'$\Phi_{\mathregular{H_2}}$','FCO2':r'$\Phi_{\mathregular{CO_2}}$','FCH4':r'$\Phi_{\mathregular{CH_4}}$','R1':r'$\mathregular{H_2}:\mathregular{CH_4}$','R2':r'$R_2$'}

palette = sns.color_palette('colorblind')

def Likelihood(outputs,observations,observables,scale):
    """
    Returns the so-called normalized likelihoods (an estimate)
    of the different models given observations
    """
    Likelyhood = []
    for var in observables:
        distr = np.array(outputs[var])
        Lfun = stats.gaussian_kde(distr)
        Lfun_weighed = lambda x : Lfun(x)*scale
        xobsmin = np.min(observations[var])
        xobsmax = np.max(observations[var])
        e = (xobsmax-xobsmin)/2
        xobs = np.mean(observations[var])
        NLi  = scint.quad(Lfun_weighed,xobs-e,xobs+e)[0]
        if NLi >1:
            print('warning : N Likelyhood greater than 1 ({})! This may be due to numerical integration'.format(NLi))
            NLi = 1.0
        Likelyhood.append(NLi)
    return(Likelyhood)

def autolabel(rects,bottom,fontsize,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()+bottom
        if height < 0.01 and height > 0.001:
            text = '< 1%'
        elif height < 0.001 and height > 0.0001:
            text = r'$<1\perthousand$'
        elif height < 0.0001:
            text = r'$\approx 0$'
        else:
            text = np.round(rect.get_height(),2)
        ax.annotate('{}'.format(text),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1.5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=fontsize)
        
        
        
# Importing data from the standard scenario

dirname = 'standard_thermocorr_taucorr2'
rootdir = os.path.join('data',dirname)
priors = pd.read_csv(os.path.join(rootdir,dirname+'.prior.csv'),delimiter=';',index_col=0)
outcomes = pd.read_csv(os.path.join(rootdir,'000/core_batch_000raw.csv'),delimiter=';',index_col=0)

observations    = pd.read_csv('data/observations/observations.csv',delimiter=';',index_col=0)
logobservations = pd.read_csv('data/observations/logobservations.csv',delimiter=';',index_col=0)
observation     = np.load('data/observations/observation.npy',allow_pickle=True)

reftable_data = np.load(os.path.join(rootdir,"reftable_data_standard.npy"),allow_pickle=True).item()
reftable_ref = np.load(os.path.join(rootdir,"reftable_ref_standard.npy"),allow_pickle=True)
testtable_data = np.load(os.path.join(rootdir,"testtable_data_standard.npy"),allow_pickle=True).item()
testtable_ref = np.load(os.path.join(rootdir,"testtable_ref_standard.npy"),allow_pickle=True)

total_table = pd.read_csv(os.path.join(rootdir,"total_table.csv"),delimiter=';')
training_df = pd.DataFrame(data=reftable_data)
test_df = pd.DataFrame(data=testtable_data)

bootstrapS = np.load(os.path.join(rootdir,"bootstrap_standard.npy"),allow_pickle=True).item()

reftable_data_005 = np.load(os.path.join(rootdir,"reftable_data_005.npy"),allow_pickle=True).item()
reftable_ref_005 = np.load(os.path.join(rootdir,"reftable_ref_005.npy"),allow_pickle=True)
testtable_data_005 = np.load(os.path.join(rootdir,"testtable_data_005.npy"),allow_pickle=True).item()
testtable_ref_005 = np.load(os.path.join(rootdir,"testtable_ref_005.npy"),allow_pickle=True)

total_table_005 = pd.read_csv(os.path.join(rootdir,"total_table.csv"),delimiter=';')
training_df_005 = pd.DataFrame(data=reftable_data)
test_df_005 = pd.DataFrame(data=testtable_data)

total_table_005 = np.log10(pd.DataFrame(data=reftable_data_005))
total_table_005['models'] = reftable_ref_005
translate = {'uninhabitable':0,'habitable':1,'inhabited':2}
total_table_005['model'] = np.array([translate[total_table_005['models'].loc[i]] for i in range(len(total_table_005))])

# Importing data fromm the higher methane scenario

dirname_M = 'highermethane_thermocorr_taucorr2'
rootdir_M = os.path.join('data',dirname_M)
priors = pd.read_csv(os.path.join(rootdir_M,dirname_M+'.prior.csv'),delimiter=';',index_col=0)
outcomes = pd.read_csv(os.path.join(rootdir_M,'000/core_batch_000raw.csv'),delimiter=';',index_col=0)

reftable_data_M = np.load(os.path.join(rootdir_M,"reftable_data_standard.npy"),allow_pickle=True).item()
reftable_ref_M = np.load(os.path.join(rootdir_M,"reftable_ref_standard.npy"),allow_pickle=True)
testtable_data_M = np.load(os.path.join(rootdir_M,"testtable_data_standard.npy"),allow_pickle=True).item()
testtable_ref_M = np.load(os.path.join(rootdir_M,"testtable_ref_standard.npy"),allow_pickle=True)

total_table_M = pd.read_csv(os.path.join(rootdir_M,"total_table.csv"),delimiter=';')
training_df_M = pd.DataFrame(data=reftable_data_M)
test_df_M = pd.DataFrame(data=testtable_data_M)

bootstrapM = np.load(os.path.join(rootdir_M,"bootstrap_standard.npy"),allow_pickle=True).item()

reftable_data_M005 = np.load(os.path.join(rootdir_M,"reftable_data_005.npy"),allow_pickle=True).item()
reftable_ref_M005 = np.load(os.path.join(rootdir_M,"reftable_ref_005.npy"),allow_pickle=True)
testtable_data_M005 = np.load(os.path.join(rootdir_M,"testtable_data_005.npy"),allow_pickle=True).item()
testtable_ref_M005 = np.load(os.path.join(rootdir_M,"testtable_ref_005.npy"),allow_pickle=True)

total_table_M005 = pd.read_csv(os.path.join(rootdir_M,"total_table.csv"),delimiter=';')
training_df_M005 = pd.DataFrame(data=reftable_data_M005)
test_df_M005 = pd.DataFrame(data=testtable_data_M005)

total_table_M005 = np.log10(pd.DataFrame(data=reftable_data_M005))
total_table_M005['models'] = reftable_ref_M005
translate = {'uninhabitable':0,'habitable':1,'inhabited':2}
total_table_M005['model'] = np.array([translate[total_table_M005['models'].loc[i]] for i in range(len(total_table_M005))])


var_order = np.array(['FH2','FCH4','R1'])
#var_order = list(logobservations.keys())
PI_S05 = len(total_table[total_table['model']==2])/len(total_table)
PH_S05 = len(total_table[total_table['model']==1])/len(total_table)
PU_S05 = len(total_table[total_table['model']==0])/len(total_table)

inh_LS05 = np.array(Likelihood(total_table[total_table['model']==2],logobservations,observables=var_order,scale=PI_S05))
hab_LS05 = np.array(Likelihood(total_table[total_table['model']==1],logobservations,observables=var_order,scale=PH_S05))
uni_LS05 = np.array(Likelihood(total_table[total_table['model']==0],logobservations,observables=var_order,scale=PU_S05))

PI_S005 = len(total_table_005[total_table_005['model']==2])/len(total_table_005)
PH_S005 = len(total_table_005[total_table_005['model']==1])/len(total_table_005)
PU_S005 = len(total_table_005[total_table_005['model']==0])/len(total_table_005)
inh_LS005 = np.array(Likelihood(total_table_005[total_table_005['model']==2],logobservations,observables=var_order,scale=PI_S005))
hab_LS005 = np.array(Likelihood(total_table_005[total_table_005['model']==1],logobservations,observables=var_order,scale=PH_S005))
uni_LS005 = np.array(Likelihood(total_table_005[total_table_005['model']==0],logobservations,observables=var_order,scale=PU_S005))

## For the higher methane

PI_M05 = len(total_table_M[total_table_M['model']==2])/len(total_table_M)
PH_M05 = len(total_table_M[total_table_M['model']==1])/len(total_table_M)
PU_M05 = len(total_table_M[total_table_M['model']==0])/len(total_table_M)

inh_LM05 = np.array(Likelihood(total_table_M[total_table_M['model']==2],logobservations,observables=var_order,scale=PI_M05))
hab_LM05 = np.array(Likelihood(total_table_M[total_table_M['model']==1],logobservations,observables=var_order,scale=PH_M05))
uni_LM05 = np.array(Likelihood(total_table_M[total_table_M['model']==0],logobservations,observables=var_order,scale=PU_M05))

PI_M005 = len(total_table_M005[total_table_M005['model']==2])/len(total_table_M005)
PH_M005 = len(total_table_M005[total_table_M005['model']==1])/len(total_table_M005)
PU_M005 = len(total_table_M005[total_table_M005['model']==0])/len(total_table_M005)

inh_LM005 = np.array(Likelihood(total_table_M005[total_table_M005['model']==2],logobservations,observables=var_order,scale=PI_M005))
hab_LM005 = np.array(Likelihood(total_table_M005[total_table_M005['model']==1],logobservations,observables=var_order,scale=PH_M005))
uni_LM005 = np.array(Likelihood(total_table_M005[total_table_M005['model']==0],logobservations,observables=var_order,scale=PU_M005))



labelsDict =  {'FH2' :'$\mathregular{H_2} $ escape rate \n $\Phi_{\mathregular{H_2} }$',
           'FCO2':'$\mathregular{CO_2}$ escape rate \n $\Phi_{\mathregular{CO_2}}$',
           'FCH4':'$\mathregular{CH_4}$ escape rate \n $\Phi_{\mathregular{CH_4}}$',
           'R1':'gas ratio \n $\mathregular{H_2}:\mathregular{CH_4}$',
           'R2':r'$R_2$',
           'H2f':'HF $\mathregular{H_2}$ \n $[\mathregular{H_2}]_f$',
           'CO2f':'HF $\mathregular{CO_2}$ \n $[\mathregular{CO_2}]_f$ \n ($\log_{10} \mathregular{mol}~\mathregular{kg}^{-1}$)',
           'CH4f':'HF $\mathregular{CH_4}$ \n $[\mathregular{CH_4}]_f$ \n ($\log_{10} \mathregular{mol}~\mathregular{kg}^{-1}$)',
           'H2o' :'Ocean $\mathregular{H_2}$ \n $[\mathregular{H_2}]_o$ \n ($\log_{10} \mathregular{mol}~\mathregular{kg}^{-1}$)',
           'CO2o':'Ocean $\mathregular{CO_2}$ \n $[\mathregular{CO_2}]_o$ \n ($\log_{10} \mathregular{mol}~\mathregular{kg}^{-1}$)',
           'CH4o':'Ocean $\mathregular{CH_4}$ \n $[\mathregular{CH_4}]_o$ \n ($\log_{10} \mathregular{mol}~\mathregular{kg}^{-1}$)',
           'Tf':'HF temperature \n $T_f$ (K)'}


mpl.rc('axes', labelsize=9)
mpl.rc('xtick', labelsize=8)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=8)
fig,axes =plt.subplots(figsize=(8,4),ncols=2,nrows=2)
Xticks = [labelsDict[k] for k in var_order]
X = np.array([1,2,3])
width=0.3
bottom = -0.001
f = 6

##First pannel (a) P=0.5 M=standard
Urect_S05 = axes[0,0].bar(X-width,uni_LS05-bottom,width=width,color=palette[0],bottom=bottom)
Hrect_S05 = axes[0,0].bar(X,hab_LS05-bottom,width=width,color=palette[1],bottom=bottom)
Irect_S05 = axes[0,0].bar(X+width,inh_LS05-bottom,width=width,color=palette[2],bottom=bottom)

autolabel(Urect_S05,bottom=bottom,fontsize=f,ax=axes[0,0])
autolabel(Hrect_S05,bottom=bottom,fontsize=f,ax=axes[0,0])
autolabel(Irect_S05,bottom=bottom,fontsize=f,ax=axes[0,0])
axes[0,0].set_xticks([1,2,3])
axes[0,0].set_xticklabels(Xticks,fontsize=7)
axes[0,0].set_ylim(-0.01,.3)
sns.despine(ax=axes[0,0],trim=True,left=False)
axes[0,0].set_ylabel('Likelihood')

## Second standard P=0.05
Urect_S005 = axes[1,0].bar(X-width,uni_LS005-bottom,width=width,color=palette[0],bottom=bottom)
Hrect_S005 = axes[1,0].bar(X,hab_LS005-bottom,width=width,color=palette[1],bottom=bottom)
Irect_S005 = axes[1,0].bar(X+width,inh_LS005-bottom,width=width,color=palette[2],bottom=bottom)

autolabel(Urect_S005,bottom=bottom,fontsize=f,ax=axes[1,0])
autolabel(Hrect_S005,bottom=bottom,fontsize=f,ax=axes[1,0])
autolabel(Irect_S005,bottom=bottom,fontsize=f,ax=axes[1,0])
axes[1,0].set_xticks([1,2,3])
axes[1,0].set_xticklabels(Xticks,fontsize=7)
axes[1,0].set_ylim(-0.01,.3)
sns.despine(ax=axes[1,0],trim=True,left=False)
axes[1,0].set_ylabel('Likelihood')

## Third higher 0.5
Urect_M05 = axes[0,1].bar(X-width,uni_LM05-bottom,width=width,color=palette[0],bottom=bottom)
Hrect_M05 = axes[0,1].bar(X,hab_LM05-bottom,width=width,color=palette[1],bottom=bottom)
Irect_M05 = axes[0,1].bar(X+width,inh_LM05-bottom,width=width,color=palette[2],bottom=bottom)

autolabel(Urect_M05,bottom=bottom,fontsize=f,ax=axes[0,1])
autolabel(Hrect_M05,bottom=bottom,fontsize=f,ax=axes[0,1])
autolabel(Irect_M05,bottom=bottom,fontsize=f,ax=axes[0,1])
axes[0,1].set_xticks([1,2,3])
axes[0,1].set_xticklabels(Xticks,fontsize=7)
axes[0,1].set_ylim(-0.01,.3)
sns.despine(ax=axes[0,1],trim=True,left=False)
#axes[1,0].set_ylabel('Likelihood')

Urect_M005 = axes[1,1].bar(X-width,uni_LM005-bottom,width=width,color=palette[0],bottom=bottom)
Hrect_M005 = axes[1,1].bar(X,hab_LM005-bottom,width=width,color=palette[1],bottom=bottom)
Irect_M005 = axes[1,1].bar(X+width,inh_LM005-bottom,width=width,color=palette[2],bottom=bottom)

autolabel(Urect_M005,bottom=bottom,fontsize=f,ax=axes[1,1])
autolabel(Hrect_M005,bottom=bottom,fontsize=f,ax=axes[1,1])
autolabel(Irect_M005,bottom=bottom,fontsize=f,ax=axes[1,1])
axes[1,1].set_xticks([1,2,3])
axes[1,1].set_xticklabels(Xticks,fontsize=7)
axes[1,1].set_ylim(-0.01,.3)
sns.despine(ax=axes[1,1],trim=True,left=False)
#plt.savefig(os.path.join(rootdir,'figures','likelihood05.svg'),dpi=400)

axes[0,0].text(.1,0.25,'a',fontsize=8,weight='bold')
axes[0,1].text(.1,0.25,'b',fontsize=8,weight='bold')
axes[1,0].text(.1,0.25,'c',fontsize=8,weight='bold')
axes[1,1].text(.1,0.25,'d',fontsize=8,weight='bold')

axes[0,0].text(2,0.27,'$\mathregular{CH}_4$ from serpentinization only',fontsize=8,weight='bold',ha='center')
axes[0,1].text(2,0.27,'Increased upper bound of $[\mathregular{CH}_4]_f$',fontsize=8,weight='bold',ha='center')

axes[0,0].text(.5,0.15,'$P(B|H)=0.5$',fontsize=8,ha='left')
axes[1,0].text(.5,0.15,'$P(B|H)=0.05$',fontsize=8,ha='left')

axes[0,1].text(.5,0.15,'$P(B|H)=0.5$',fontsize=8,ha='left')
axes[1,1].text(.5,0.15,'$P(B|H)=0.05$',fontsize=8,ha='left')

plt.subplots_adjust(hspace=0.25)

plt.savefig('figures/likelihood_pannel.png',dpi=400,bbox_inches = 'tight',transparent=True)
plt.show()