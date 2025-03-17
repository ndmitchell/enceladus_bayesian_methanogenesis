#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, traceback
import argparse
import os

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

mpl.rc('axes', labelsize=7)
mpl.rc('legend', fontsize=7)
mpl.rc('xtick', labelsize=5.5)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=5.5)

labelsDict = {'FH2':r'$\Phi_{\mathregular{H_2}}$','FCO2':r'$\Phi_{\mathregular{CO_2}}$','FCH4':r'$\Phi_{\mathregular{CH_4}}$','R1':r'$\mathregular{H_2}:\mathregular{CH_4}$','R2':r'$R_2$'}

palette = sns.color_palette('colorblind')


def scatters(figpath,priors,observables,logobservations,labels,colorvector):
    keys1 = list(priors.keys())
    keys2 = list(observables.keys())
    rasterize_scatter = True
    fig,axes = plt.subplots(ncols=len(keys1),nrows=len(keys2),figsize=(7,4))
    for i in range(len(keys2)):
        for j in range(len(keys1)):
            if keys1[j] == 'Tf':
                axes[i,j].plot([300,500],[logobservations[keys2[i]]]*2,linestyle='--',color='magenta')
                axes[i,j].scatter(priors[keys1[j]],np.log10(observables[keys2[i]]),s=0.5,c=colorvector,rasterized=rasterize_scatter)
                #axes[i,j].set_xlim(300,500)
                #axes[i,j].set_xticks([300,400,500])
                #axes[i,j].set_xticklabels(['300','400','500'],rotation=-45)
            else:
                axes[i,j].plot([-8,0],[logobservations[keys2[i]]]*2,linestyle='--',color='magenta')
                axes[i,j].scatter(np.log10(priors[keys1[j]]),np.log10(observables[keys2[i]]),s=0.5,c=colorvector,rasterized=rasterize_scatter)
                #if keys1[j] in ['H2o','CO2f','CH4o']:
                    #axes[i,j].set_xticks([-8,-7,-6])
                    #axes[i,j].set_xticklabels(['-8','-7','-6'])
                    #axes[i,j].set_xlim(-8,-6)
                #elif keys1[j] == 'CH4f':
                    #axes[i,j].set_xticks([-8,-6,-4])
                    #axes[i,j].set_xticklabels(['-8','-6','-4'])
                    #axes[i,j].set_xlim(-8,-4)
                #elif keys1[j] == 'CO2o':
                    #axes[i,j].set_xticks([-2,-1])
                    #axes[i,j].set_xticklabels(['-2','-1'])
                    #axes[i,j].set_xlim(-2.7,-0.7)
                #else:
                    #axes[i,j].set_xticks([-8,-4,0])
                    #axes[i,j].set_xticklabels(['-8','-4','0'])
                    #axes[i,j].set_xlim(-8,0)

            if j > 0:
                axes[i,j].set_yticklabels([])
            if i < len(keys2)-1:
                axes[i,j].set_xticklabels([])
            if j == 0:
                axes[i,j].set_ylabel(labels[keys2[i]])
            if i == len(keys2)-1:
                axes[i,j].set_xlabel(labels[keys1[j]])
            axes[i,j].xaxis.set_minor_locator(AutoMinorLocator())
            axes[i,j].yaxis.set_minor_locator(AutoMinorLocator())
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    fig.align_labels(axes)
    plt.savefig(figpath,dpi=400,bbox_inches='tight')
    #plt.savefig('figures/inflow.svg',dpi=400)
    #plt.show()
    plt.close()
    
def plotconfusion(testtable_ref,crossvalid,figpath):
    fig, ax = plt.subplots(figsize=(3,3))
    matrix = metrics.confusion_matrix(testtable_ref,crossvalid)
    ticks = [r'$H$',r'$I$',r'$\bar{H}$']
    sns.heatmap(np.round(matrix/np.repeat(np.sum(matrix,axis=1),3).reshape(3,3),2),annot=True,square=True,cmap='YlOrBr',
                linecolor='white',linewidths=2,cbar=False,xticklabels=ticks,yticklabels=ticks,ax=ax,annot_kws={'fontsize':5.5})
    ax.set_ylabel('True model')
    ax.set_xlabel('Predicted model')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(figpath,dpi=400,bbox_inches='tight')
    plt.close()
    
def plotbootstraps(bootstrap,palette,outcomes,figpath):
    fig  = plt.figure(figsize=(6,3.5))
    #ax   = plt.axes([0.1,0,1,0.8])
    ax = fig.add_subplot(1,1,1)
    MinP = np.min(np.array(bootstrap['probabilities'])[:,:,1],axis=1) # lower values of posterior probabilities
    MaxP = np.max(np.array(bootstrap['probabilities'])[:,:,1],axis=1) # higher values
    ax.plot(bootstrap['PIH'],MinP,'.',color=palette[2])
    ax.plot(bootstrap['PIH'],MaxP,'.',color=palette[2])
    ax.fill_between(bootstrap['PIH'],MaxP,MinP,color=palette[2],alpha=0.5) # plot the area
    ax.plot(bootstrap['PIH'],bootstrap['PI'],'--',color=palette[2])

    MinPh = np.min(np.array(bootstrap['probabilities'])[:,:,0],axis=1)
    MaxPh = np.max(np.array(bootstrap['probabilities'])[:,:,0],axis=1)
    ax.plot(bootstrap['PIH'],MinPh,'.',color=palette[1])
    ax.plot(bootstrap['PIH'],MaxPh,'.',color=palette[1])
    ax.fill_between(bootstrap['PIH'],MaxPh,MinPh,color=palette[1],alpha=0.5)


    PH = np.sum(outcomes['CH4_ab'] != outcomes['CH4'])/len(outcomes)
    meanPI = np.mean(np.array(bootstrap['probabilities'])[:,:,1])

    ax.plot(bootstrap['PIH'],PH-np.array(bootstrap['PI']),'--',color=palette[1])

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.plot(bootstrap['PIH'],bootstrap['score'],'--k',linewidth=0.75)
    ax.plot([0.03],[meanPI],marker='<',color=palette[2],markersize=10)


    ax.set_ylim(0,1)
    ax.set_xlim(0,1)


    handles = [mpatches.Patch(color=palette[2]),mpatches.Patch(color=palette[1])]
    handles.append(mlines.Line2D([], [], color=palette[2],linestyle='--', linewidth=0.75))
    handles.append(mlines.Line2D([], [], color=palette[1],linestyle='--', linewidth=0.75))

    handles.append(mlines.Line2D([], [], color='k',linestyle='--', linewidth=0.75))
    ax.legend(handles=handles,labels=[r'$P(I|x^0)$',r'$P(H|x^0)$',r'$P_{prior}(I)$',r'$P_{prior}(H)$','score'],
              loc='center left', bbox_to_anchor=(1, 0.5),frameon=False)
    ax.set_xlabel(r'$P(I|H)$')
    ax.set_ylabel('Probability')


    #plt.tight_layout()
    plt.savefig(figpath,dpi=400,bbox_inches = 'tight')

    plt.close()
    
def plotarray(total_table,logobservations,uni_L,inh_L,hab_L,figpath):
    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(6,6),constrained_layout=False)
    gs = fig.add_gridspec(3, 3)

    variables = ['FH2','FCH4','R1']
    barlabels = np.array([r'$H$',r'$I$',r'$\bar{H}$'])
    palette = sns.color_palette('colorblind')
    colors=[palette[total_table['model'].iloc[k]] for k in range(len(total_table))]

    for i in range(3):
        for j in range(3):
            var = variables[i]
            var2 = variables[j]
            if var2 == 'R1':
                axes[i,j].set_xlim(-5,7)
            else:
                axes[i,j].set_xlim(2,12)
            if i==j:
                cplt.DistriPlot(ax=axes[i,j],data=total_table,var=var,Nbins=20,alpha=1,histtype='step',linewidth=1.45,palette=palette)
                axes[i,j].set_xlabel('log '+labelsDict[var],labelpad=1)
                axes[i,j].plot([logobservations[var]]*2,[0,0.085],linestyle='--',color='magenta')
            elif i < j:
                axes[i,j].scatter(total_table[var2],total_table[var],c=colors,marker='d',s=0.01,rasterized=True)
                axes[i,j].plot(logobservations[var2],logobservations[var],marker='*',linestyle='',color='magenta')

            axes[i,j].xaxis.set_minor_locator(AutoMinorLocator())

    axes[2,0].axis('off')
    axes[2,1].axis('off')
    axes[1,0].axis('off')

    axes[0,0].set_ylabel('Density')
    axes[0,1].yaxis.tick_right()
    axes[0,2].yaxis.tick_right()
    axes[0,2].yaxis.set_label_position("right")
    axes[0,2].set_ylabel('log '+labelsDict[variables[0]])
    axes[1,1].set_ylabel('Density')
    axes[1,2].yaxis.tick_right()
    axes[1,2].yaxis.set_label_position("right")
    axes[1,2].set_ylabel('log '+labelsDict[variables[1]])

    axes[2,2].yaxis.tick_right()
    axes[2,2].yaxis.set_label_position("right")
    axes[2,2].set_ylabel('Density')



    cols  = [0,1,2]*2+ [0,2]
    xlabc = [0.135,0.41,0.69]*2+[0.13,0.69]
    rows  = np.repeat([0,1,2],[3,3,2])
    ylabc = np.repeat([0.855,0.58,0.32,0.31],[3,3,1,1])

    labels = ['{0}'.format(letter) for letter in ['a','b','c','','d','e','f','g']]

    ax_boot = fig.add_subplot(gs[2,:2])
    axes[2,1] = ax_boot
    width=0.3
    X = np.array([1,2,3])
    bottom = -0.1
    Xticks = [labelsDict[k] for k in logobservations.keys()]
    Urect = ax_boot.bar(X-width,uni_L-bottom,width=width,color=palette[0],bottom=bottom)
    Hrect = ax_boot.bar(X,hab_L-bottom,width=width,color=palette[1],bottom=bottom)
    Irect = ax_boot.bar(X+width,inh_L-bottom,width=width,color=palette[2],bottom=bottom)
    cplt.autolabel(Urect,ax=ax_boot,bottom=bottom)
    cplt.autolabel(Hrect,ax=ax_boot,bottom=bottom)
    cplt.autolabel(Irect,ax=ax_boot,bottom=bottom)

    ax_boot.set_ylim(-0.1,1.3)
    ax_boot.set_xticks(X)
    ax_boot.set_xticklabels(Xticks,rotation=0,ha='center',fontsize=7)
    ax_boot.tick_params(axis='x', which='major', pad=12)
    ax_boot.set_ylabel('Normalized Likelihood')


    sns.despine(ax=ax_boot,trim=True,left=False)

    for i,j,z in [(0,0,0),(0,1,1),(0,2,2),(1,1,4),(1,2,5),(2,1,6),(2,2,7)]:
        axes[i,j].text(0.03,0.89,labels[z],transform=axes[i,j].transAxes,fontsize=8,weight='bold')

    axes[0,1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0,2].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1,2].yaxis.set_minor_locator(AutoMinorLocator())

    uninh_patch = mpatches.Patch(color=palette[0], label=r'Uninhabitable ($\bar{H}$)')
    habit_patch = mpatches.Patch(color=palette[1], label=r'Habitable ($H$)')
    inhab_patch = mpatches.Patch(color=palette[2], label=r'Inhabited ($I$)')
    datap_line  = mlines.Line2D([], [], color='magenta', marker='*', linestyle='--', markersize=8, label='Observation')

    axes[1,0].legend(handles = [uninh_patch, habit_patch, inhab_patch, datap_line],
                     fontsize=7,loc='center left', borderaxespad=0, frameon=False)

    fig.align_labels(axes[:,2])

    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.savefig(figpath,dpi=400,bbox_inches = 'tight')
    plt.close()

def main(dirname):
    rootdir = os.path.join('data',dirname)
    
    if not os.path.isdir(os.path.join(rootdir,'figures')):
        os.mkdir(os.path.join(rootdir,'figures'))
    
    priors = pd.read_csv(os.path.join(rootdir,dirname+'.prior.csv'),delimiter=';',index_col=0)
    outcomes = pd.read_csv(os.path.join(rootdir,'000','core_batch_000raw.csv'),delimiter=';',index_col=0)

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

    bootstrap = np.load(os.path.join(rootdir,"bootstrap_standard.npy"),allow_pickle=True).item()
    
    timec = 60*60*24*365 # They have to be converted from s-1 into yr-1
    observables_abiotic = pd.DataFrame(data={'FH2':outcomes['H2_ab']*timec,
                                             'FCH4':outcomes['CH4_ab']*timec,
                                             'R1':outcomes['H2_ab']/outcomes['CH4_ab']})
    model = np.array(['uninhabitable','habitable'],dtype='str')[np.array(outcomes['H2']!=outcomes['H2_ab'],dtype=int)]
    colors = {'uninhabitable':palette[0],
              'habitable':palette[1],
              'inhabited':palette[2]}
    c1 = [colors[k] for k in model]
    labels =  {'FH2':r'$\Phi_{\mathregular{H_2}}$','FCO2':r'$\Phi_{\mathregular{CO_2}}$','FCH4':r'$\Phi_{\mathregular{CH_4}}$','R1':r'$\mathregular{H_2}:\mathregular{CH_4}$','R2':r'$R_2$',
               'H2f':r'$[\mathregular{H_2}]_f$','CO2f':r'$[\mathregular{CO_2}]_f$','CH4f':r'$[\mathregular{CH_4}]_f$',
               'H2o':r'$[\mathregular{H_2}]_o$','CO2o':r'$[\mathregular{CO_2}]_o$','CH4o':r'$[\mathregular{CH_4}]_o$',
               'Tf':r'$T_f$'}
    # Scatter 1
    unin_scatterfile = os.path.join(rootdir,'figures','scatter_unin.png')
    if not os.path.exists(unin_scatterfile):
        scatters(unin_scatterfile,priors,observables_abiotic,logobservations,labels,colorvector=c1)
    
    observables_biotic = pd.DataFrame(data={'FH2':outcomes['H2']*timec,
                                        'FCH4':outcomes['CH4']*timec,
                                        'R1':outcomes['H2']/outcomes['CH4']})
    model2 = np.array(['uninhabitable','inhabited'],dtype='str')[np.array(outcomes['H2']!=outcomes['H2_ab'],dtype=int)]
    c2 = [colors[k] for k in model2]
    
    in_scatterfile = os.path.join(rootdir,'figures','scatter_in.png')
    if not os.path.exists(in_scatterfile):
        scatters(in_scatterfile,priors,observables_biotic,logobservations,labels,colorvector=c2)
    
    rfcl = RandomForestClassifier(random_state=0,n_estimators=500,criterion='gini',max_depth=20,
                              max_features=3,warm_start=False,min_samples_leaf=10,min_samples_split=100,
                              bootstrap=True,max_leaf_nodes=None,oob_score=True)
    
    rfcl.fit(training_df, reftable_ref)
    
    # To dump in a text file
    prediction = rfcl.predict(observations)
    PH = np.sum(model=='habitable')/len(model)
    probs = rfcl.predict_proba(observations)
    mprob = np.mean(probs,axis=0)
    
    crossvalid = rfcl.predict(test_df)
    
    confusionfile = os.path.join(rootdir,'figures','confusion05.png')
    if not os.path.exists(confusionfile):
        plotconfusion(testtable_ref,crossvalid,figpath=confusionfile)
    
    bootstrapfile = os.path.join(rootdir,'figures','bootstrap.png')
    if not os.path.exists(bootstrapfile):
        plotbootstraps(bootstrap,palette=palette,outcomes=outcomes,figpath=bootstrapfile)
    
    inh_L = np.array(homestats.NLikelihood(total_table[total_table['model']==2],logobservations))
    hab_L = np.array(homestats.NLikelihood(total_table[total_table['model']==1],logobservations))
    uni_L = np.array(homestats.NLikelihood(total_table[total_table['model']==0],logobservations))
    
    arrayfile = os.path.join(rootdir,'figures','array_fig.png')
    if not os.path.exists(arrayfile):
        plotarray(total_table,logobservations,uni_L,inh_L,hab_L,figpath=arrayfile)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='makefigures.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    Plots the essential figs
                                     ''')
    
    # Required arguments
    parser.add_argument("--dirname", help="dirname of inputs",type=str,action="store",required=True)
    args = parser.parse_args()
    path = os.getcwd()
    dirname=args.dirname
    main(dirname=dirname)
    print('done')