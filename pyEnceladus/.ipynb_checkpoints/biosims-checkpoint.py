#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pyEnceladus.universal_htv as uhtv
import pyEnceladus.data_htv as dhtv
import pyEnceladus.physical as phc
from tqdm import tqdm
import pandas as pd
from scipy import optimize
import pyEnceladus.simulation_qstar as smqs

import sys
import argparse

def makeOnePrior(bdries,rndgen=None):
    """
    Draws one set of parameters
    bdries is a dictionnary {H2fp,H2op,CO2fp,CO2op,CH4fp,CH4op,Tfp}
    in which each key is associated with a list of len 2
    """
    if rndgen is None:
        rndgen = np.random.RandomState()
    H2f  = np.exp(rndgen.uniform(np.log(bdries['H2fp'][0]), np.log(bdries['H2fp'][1])))
    H2o  = np.exp(rndgen.uniform(np.log(bdries['H2op'][0]), np.log(bdries['H2op'][1])))
    CO2f = np.exp(rndgen.uniform(np.log(bdries['CO2fp'][0]), np.log(bdries['CO2fp'][1])))
    CO2o = np.exp(rndgen.uniform(np.log(bdries['CO2op'][0]), np.log(bdries['CO2op'][1])))
    CH4f = np.exp(rndgen.uniform(np.log(bdries['CH4fp'][0]), np.log(bdries['CH4fp'][1])))
    CH4o = np.exp(rndgen.uniform(np.log(bdries['CH4op'][0]), np.log(bdries['CH4op'][1])))
    Tf   = rndgen.uniform(bdries['Tfp'][0], bdries['Tfp'][1])
    return({'H2f':H2f,'H2o':H2o,'CO2f':CO2f,'CO2o':CO2o,'CH4f':CO2o,'CH4o':CH4o,'Tf':Tf})

def makePriors(prior_bdries,nsim,outfile,rndgen=None):
    """
    Takes the csv file with the boundaries and generates a file with values taken
    in log-uniform distributions.
    """
    bdries    = np.genfromtxt(prior_bdries,delimiter=';')
    # Extracting parameters
    #   Concentrations
    H2fp  = bdries[0]
    H2op  = bdries[1]
    CO2fp = bdries[2]
    CO2op = bdries[3]
    CH4fp = bdries[4]
    CH4op = bdries[5]
    Tfp   = bdries[6]
    ## Parameters Distributions (Uniform only, no covariance)
    # ==============================================================================
    # Remarks
    # ------------------------------------------------------------------------------
    # There is discrepancy in parameter exploration if distributions have different
    # spans... If one wants to have a parameter vary across multiple orders of ma-
    # -gnitude, they should look into drawing from log uniform laws
    # Perhaps it would be better to give the file containing the priors instead of
    # generating them here
    # ------------------------------------------------------------------------------

    # Drawing values
    ## Setting random generator
    if rndgen is None:
        rndgen = np.random.RandomState()
    # ------------------------------------------------------------------------------
    H2f  = np.exp(rndgen.uniform(np.log(H2fp[0]), np.log(H2fp[1]), nsim))
    H2o  = np.exp(rndgen.uniform(np.log(H2op[0]), np.log(H2op[1]), nsim))
    CO2f = np.exp(rndgen.uniform(np.log(CO2fp[0]), np.log(CO2fp[1]), nsim))
    CO2o = np.exp(rndgen.uniform(np.log(CO2op[0]), np.log(CO2op[1]), nsim))
    CH4f = np.exp(rndgen.uniform(np.log(CH4fp[0]), np.log(CH4fp[1]), nsim))
    CH4o = np.exp(rndgen.uniform(np.log(CH4op[0]), np.log(CH4op[1]), nsim))
    Tf   = rndgen.uniform(Tfp[0], Tfp[1], nsim)

    prior = pd.DataFrame(data={'H2o':H2o,'H2f':H2f,'CO2o':CO2o,'CO2f':CO2f,'CH4o':CH4o,'CH4f':CH4f,'Tf':Tf})
    prior.to_csv(outfile+'.prior.csv',sep=';')
    Run_oconc = []
    Run_fconc = []
    for i in range(nsim):
        Run_oconc.append({'H2':prior['H2o'][i],'CO2':prior['CO2o'][i],'CH4':prior['CH4o'][i]})
        Run_fconc.append({'H2':prior['H2f'][i],'CO2':prior['CO2f'][i],'CH4':prior['CH4f'][i]})
    return(Tf,Run_oconc,Run_fconc)

# ==============================================================================
def runOneBatch(Tf,Run_oconc,Run_fconc,nsim,outfile,dth,To=275,tau=dhtv.BioRunMeth['tau'],pbar=False):

    # Arrays
    #-------------------------------------------------------------------------------
    # Building arrays for concentrations
    H2ab    = np.zeros_like(Tf, dtype=np.float32)
    H2bi    = np.zeros_like(Tf, dtype=np.float32)
    CO2ab   = np.zeros_like(Tf, dtype=np.float32)
    CO2bi   = np.zeros_like(Tf, dtype=np.float32)
    CH4ab   = np.zeros_like(Tf, dtype=np.float32)
    CH4bi   = np.zeros_like(Tf, dtype=np.float32)
    totalF  = np.zeros_like(Tf, dtype=np.float32)
    totalB  = np.zeros_like(Tf, dtype=np.float32)
    

    # Setting jmax to filling the whole local mixing layer
    for i in tqdm(range(nsim),disable= not pbar):
        cplb,cplab,totF,Btot,s = smqs.sim_meth_enc(F=5e9,Tf=Tf[i],To=To,
                                                   oconc=Run_oconc[i],fconc=Run_fconc[i],
                                                   dth=dth,tau=tau,
                                                   **dhtv.methanogens,**smqs.defsimpar)
        H2ab[i]    = cplab['H2']
        H2bi[i]    = cplb['H2']
        CO2ab[i]   = cplab['CO2']
        CO2bi[i]   = cplb['CO2']
        CH4ab[i]   = cplab['CH4']
        CH4bi[i]   = cplb['CH4']
        totalF[i]  = totF
        totalB[i]  = Btot
        
    raw_out = {'H2_ab':H2ab,'H2':H2bi,'CO2_ab':CO2ab,'CO2':CO2bi,'CH4_ab':CH4ab,'CH4':CH4bi,'totF':totalF,'totB':totalB}
    df_raw = pd.DataFrame(data=raw_out)
    df_raw.to_csv(outfile+'raw.csv',sep=';')
    return()

if __name__ == "__main__":
    ## Parsing Arguments
    # =============================================================================

    # Input file
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='biosims2.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    Runs a pool of simulations with parameters drawn in prior distributions
                                     ''')
    # Optional arguments
    parser.add_argument("-n","--nsims", type=int, help="Number of simulations",action="store")
    parser.add_argument("-d","--deathrate", type=float, help="basal death rate",action="store")
    parser.add_argument('-b','--progress-bar', dest='pbar', action='store_true',help='displays a progress bar')
    parser.add_argument('--no-pbar', dest='pbar', action='store_false')
    # Positional Arguments
    parser.add_argument("prior", help="filename of priors",type=str,action="store")
    parser.add_argument("output", type=str, help="output filename without extension ",action="store")
    parser.set_defaults(nsims  = 20000,
                        deathrate=dhtv.BioRunMeth['dth'],
                        pbar=False)
    args = parser.parse_args()
    priorfile = args.prior
    # ------------------------------------------------------------------------------
    # Other parameters (simulation properties)
    # ------------------------------------------------------------------------------
    nsim    = args.nsims                  # Number of simulations
    outfile = args.output
    dth     = args.deathrate
    pbar    = args.pbar
    Tf,Run_oconc,Run_fconc = makePriors(priorfile,nsim,outfile)
    runOneBatch(Tf,Run_oconc,Run_fconc,nsim,outfile,dth,pbar)
