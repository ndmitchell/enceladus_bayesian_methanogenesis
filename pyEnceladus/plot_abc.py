#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Plotting functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import sys
import argparse
import os
import re

from pyEnceladus.plot_tools import *

def confusion_format(cmat_name):
        cmat = pd.read_csv(os.path.join('confusion_matrixes',cmat_name),sep=';',index_col=0)
        # Handling that a scenario may be missing in case P=0 or 1
        # This is actually a nightmare
        potential = np.array(['X0','X1','X2'],dtype=str)
        cmkeys = [k for k in cmat.keys()]
        del cmkeys[-1]
        cmat_complete = {}
        if len(cmkeys) < len(potential):
            missing_key = np.array([k not in cmkeys for k in potential])
            missing_idx = np.where(missing_key)[0][0]
            other_idx   = np.where(~missing_key)[0]
            cmat[potential[missing_idx]] = np.zeros(2)
            for key in potential:
                fill = np.zeros(3)
                fill[other_idx] = cmat[key]
                #fill[missing_idx] = 0
                cmat_complete[key] = fill
        else :
            cmat_complete = cmat
        confmat = np.transpose(np.array([np.array(cmat_complete[key]) for key in potential]))
        dmat = np.zeros_like(confmat,dtype=float)
        for i in range(len(potential)):
            if np.sum(confmat[i,:]):
                dmat[i,:] = confmat[i,:]/np.sum(confmat[i,:])
            else:
                dmat[i,:] = 0
        return(dmat)

def main(directory,datapoints):
    path = os.getcwd()
    os.chdir(directory)

    index_path = os.path.join(path,directory,'simulation_index.csv')
    index_data = pd.read_csv(index_path,delimiter=';')

    abc_res_path = os.path.join(path,directory,'abcrf_analysis.csv')
    abc_res      = pd.read_csv(abc_res_path,delimiter=';')
    batches = [ name for name in os.listdir() if os.path.isdir(name) ]

    os.mkdir('plots')

    for batch in batches:
        os.chdir(os.path.join(path,directory,batch,'sumstats'))
        sumstats = [ name for name in os.listdir() if os.path.isfile(name) ]
        for simfile in sumstats:
            # Extract simulations info
            sim = pd.read_csv(simfile,delimiter=';')
            logsim = pd.DataFrame({"R1":np.log10(sim['R1']),"R2":np.log10(sim['R2']),"Qp":np.log10(sim['Qp'])})
            logsim['model'] = np.array(sim['model'],dtype=int)

            # Extract confusion matrix info
            rootname = re.findall('(.+[^stat])stat(\.csv)',simfile)[0][0]
            cmat_name = rootname+'.confusion.csv'
            dmat      = confusion_format(cmat_name)

            # Extract abc inference results info
            dth  = np.float(index_data[index_data['file']==simfile]['death_rate'])
            PIH  = np.float(index_data[index_data['file']==simfile]['P(IH)'])
            data = abc_res[(abc_res['prior_inhab']==PIH)&(abc_res['death_rate']==dth)]
            sel1 = np.int(data['allocation_dp1'])
            sel2 = np.int(data['allocation_dp2'])
            pprob1 = np.float(data['post_prob1'])
            pprob2 = np.float(data['post_prob2'])
            plot_name = rootname+'.abcrf.png'

            plot_inference(logsim,dmat,datapoints,PIH,dth,sel1,sel2,pprob1,pprob2,os.path.join(path,directory,'plots',plot_name))
        os.chdir(os.path.join(path,directory))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='plot_abc.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    Plots abc results for every simulation
                                     ''')
    parser.add_argument("directory", help="batch directory",type=str,action="store")
    args = parser.parse_args()
    dirname = args.directory
    # Data points, to move in a source file to use also in R
    minR1 = 0.4/0.6
    maxR1 = 1.4/0.3
    minR2 = 0.4/0.3
    maxR2 = 1.4/0.1
    minR3 = 0.4/(0.6+0.3)
    maxR3 = 1.4/(0.3+0.1)
    H2pn = 1e-4
    CO2pn = 7e-5
    CH4pn = 3e-5
    H2pb = 2e-7
    CO2pb = 1e-7
    CH4pb = 4e-8
    Qpb = (CH4pb**0.25)/(H2pb*(CO2pb**0.25))
    Qpn = (CH4pn**0.25)/(H2pn*(CO2pn**0.25))
    datapoints = [(minR1,maxR1),(minR2,maxR2),(Qpn,Qpb)]
    # action
    main(dirname,datapoints)
