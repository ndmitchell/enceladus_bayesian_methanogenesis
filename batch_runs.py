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
from pyEnceladus.biosims import runOneBatch, makePriors


import sys, traceback
import argparse
import os
import logging

from pyEnceladus.plot_tools import plot_priors
import pickle


def simulation_summary(filename,pinh,delimiter=';'):
    simfile = pd.read_csv(filename,delimiter=delimiter)
    habitable = simfile.loc[simfile['R1']!=simfile['R1_ab']]
    ninh = int(np.round(pinh*len(habitable)))
    return(len(simfile),len(habitable),ninh)

def sumstats_gen(filename,pinh,outname,delimiter=';',saveraw=False):
    pd.set_option('mode.chained_assignment',None)
    simfile = pd.read_csv(filename,delimiter=delimiter)
    # Generate sumstats
    R1    = np.array(simfile['H2']/simfile['CH4'])
    R1_ab = np.array(simfile['H2_ab']/simfile['CH4_ab'])
    R2    = np.array(simfile['H2']/simfile['CO2'])
    R2_ab = np.array(simfile['H2_ab']/simfile['CO2_ab'])
    Qp    = np.array((simfile['CH4']**0.25)/(simfile['H2']*simfile['CO2']**0.25))
    Qp_ab = np.array((simfile['CH4_ab']**0.25)/(simfile['H2_ab']*simfile['CO2_ab']**0.25))
    sumstats_raw = pd.DataFrame(data={'ranks':simfile.loc[:,'Unnamed: 0'],
                                      'R1':R1,'R1_ab':R1_ab,
                                      'R2':R2,'R2_ab':R2_ab,
                                      'Qp':Qp,'Qp_ab':Qp_ab})

    uninhabitable = sumstats_raw.loc[sumstats_raw['R1']==sumstats_raw['R1_ab']]
    habitable = sumstats_raw.loc[sumstats_raw['R1']!=sumstats_raw['R1_ab']]
    ranks = np.array(habitable.loc[:,'ranks'])
    ninh = int(np.round(pinh*len(ranks)))
    inh_ranks = ranks[:ninh]
    uninh_ranks = ranks[ninh:]
    inhabited = sumstats_raw.iloc[inh_ranks]
    uninhabited = sumstats_raw.iloc[uninh_ranks]
    frames = []
    if len(uninhabitable):
        uninhabitable.loc[:,'model']=str(0)
        frames.append(uninhabitable)
    if len(uninhabited):
        uninhabited.loc[:,'model']=str(1)
        # Replacing effective values by abiotic
        uninhabited['R1'] = uninhabited['R1_ab']
        uninhabited['R2'] = uninhabited['R2_ab']
        uninhabited['Qp'] = uninhabited['Qp_ab']
        frames.append(uninhabited)
    if len(inhabited):
        inhabited.loc[:,'model']=str(2)
        frames.append(inhabited)
    sumstats = pd.concat(frames)
    if saveraw:
        sumstats.to_csv(outname+'_raw_.csv',sep=';')
    # Removing useless columns
    #del sumstats['Unnamed: 0']
    del sumstats['R1_ab']
    del sumstats['R2_ab']
    del sumstats['Qp_ab']
    #del sumstats['R3']
    #del sumstats['R3_ab']

    warn = False
    if len(sumstats) != len(simfile):
        warn = True
    sumstats.to_csv(outname+'.csv',sep=';')
    total_sim = len(simfile)
    nuni      = len(uninh_ranks)
    nhab      = len(ranks)
    nunibted  = len(uninh_ranks)
    return(total_sim,nuni,nhab,ninh,nunibted,warn)


# The idea is to launch several batches of simulation for different values of d and P(I|H)
def runSomeBatches(dlist,plist,dirname,priorfile,seed=None,nsim=20000,ndigit=3,pbar=False,saveraw=False):
    """
    dlist, plist : lists containing the values of d and P(I|H) to run sim with
    dirname : project folder name
    priorfile: name of the file containing priors
    nsim : number of simulations per batch
    seed can be an integer or a filename
    """
    path = dirname
    os.makedirs(dirname,0o755,exist_ok=True)
    os.chdir(dirname)
    logging.captureWarnings(False)
    logging.basicConfig(filename=dirname+'.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    logger.info(str(__name__)+'was asked to perform simulated datasets with the following parameters : \n')
    logger.info('Values of basal death rate : %s \n' %dlist)
    logger.info('Values of P(I|H) : %s' %plist)

    ## Set the random generator
    # Allow seed to be int or a seed object stored in a pickle file
    isfile = False
    if seed is None:
        RndGen = np.random.RandomState()
        seed_state = RndGen.get_state()
        seedfilename = dirname+'seed.pck'
        with open(seedfilename,'wb') as seedfile:
            pickle.dump(seed_state,file=seedfile)
        logger.info('No specified seed or state, dumping state in {0}'.format(seedfilename))
    else:
        try :
            seed_obj = np.int(seed)
            logger.info('Seed was specified as an integer with value {0}'.format(seed_obj))
        except ValueError :
            seed_obj = str(seed)
            isfile = True
            logger.info('Seed was specified as state file : {0}'.format(seed_obj))
        if isfile:
            with open(seed_obj,'rb') as seedfile:
                seed_state = pickle.load(seedfile)
            RndGen = np.random.RandomState()
            RndGen.set_state(seed_state)
        else:
            RndGen = np.random.RandomState(seed=seed_obj)

    # Create index of simulation files
    list_names = []
    d_index = []
    p_index = []
    fine = True
    sim_name = os.path.split(dirname)[-1]
    outfile_root = os.path.join(dirname,sim_name)
    Tf,Run_oconc,Run_fconc = makePriors(priorfile,nsim,outfile_root,rndgen = RndGen)
    try :
        logger.info('plotting priors')
        plot_priors(outfile_root+'.prior.csv',outfile_root+'.prior.pdf')
    except Exception as err:
        logger.info('Something went wrong plotting priors from batch #%s' %digit)
        logger.error(err)
        logger.info('Aborting')
        logger.info('now in %s' %os.getcwd())
        fine = False
        pass
    for i in tqdm(range(len(dlist)),disable= not pbar):
        # Define the filename
        # Should contain d and p like batch%d%p
        digit = str(i)
        while len(digit) < ndigit:
            digit = '0' + digit
        logger.info('making directory '+digit)
        os.makedirs(digit,0o755,exist_ok=True)
        os.chdir(digit)
        logger.info('now in %s' %os.getcwd())
        logger.info('Simulation batch '+digit)
        filename = 'core_batch_'+digit
        logger.info('Writing in '+filename)
        deathrate = float(dlist[i])
        try :
            logger.info('Run batch with d='+str(dlist[i]))
            runOneBatch(Tf,Run_oconc,Run_fconc,nsim,outfile=filename,dth=deathrate,pbar=False)
        except Exception as err:
            logger.info('Something went wrong running batch #%s' %digit)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            msg = str(exc_value)+"\n"+u"\n".join(traceback.format_tb(exc_traceback))
            logger.error(msg)
            logger.info('Aborting')
            os.chdir('../')
            logger.info('now in %s' %os.getcwd())
            fine = False
            continue
        try :
            logger.info('now generating sumstats files for different P(I|H)')
            logger.info('Writing in directory sumstats')
            os.mkdir('sumstats',0o755)
            os.chdir('./sumstats')
            logger.info('now in %s' %os.getcwd())
            for j in tqdm(range(len(plist)),disable = not pbar):
                pinh = float(plist[j])
                outname = filename+'_P'+str(np.format_float_scientific(pinh))+'stat'
                logger.info('writing in %s' %outname)
                total_sim,nuni,nhab,ninh,nunibted,warn = sumstats_gen('../'+filename+'raw.csv',pinh,outname,delimiter=';',saveraw=saveraw)
                list_names.append(outname)
                d_index.append(deathrate)
                p_index.append(pinh)
                logger.info('{0} with {1} total data points of which {2} habitable and {3} inhabited'.format(outname,total_sim,nhab,ninh))
                if warn:
                    logger.warning('The total number of data points in sumstats is different from the number of simulations')
            os.chdir('../')
        except Exception as err:
            logger.info('Something went wrong while generating sumstats in batch #%s' %digit)
            logger.info('Error occured at iteration #%s' %j)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            msg = str(exc_value)+"\n"+u"\n".join(traceback.format_tb(exc_traceback))
            logger.error(msg)
            logger.info('Attempting to give some more info')
            try :
                tsim,hab,inh = simulation_summary('../'+filename+'.csv',pinh,delimiter=';')
                logger.info('Problem occured with P(I|H)={0} giving a set with {1} total data points of which {2} were found habitable and {3} inhabited'.format(pinh,tsim,hab,inh))
            except Exception as err:
                logger.info('`simulation_summary` Failed to obtain more information with argument {0}'.format(pinh))
                logger.error(err)
                pass
            logger.info('Aborting')
            os.chdir(dirname)
            logger.info('now in %s' %os.getcwd())
            fine = False
            continue

        os.chdir('../')
        logger.info('now in %s' %os.getcwd())


    index_files = pd.DataFrame(data={'file':list_names,'death_rate':d_index,'P(IH)':p_index})
    index_files.to_csv('simulation_index.csv',sep=';')
    logger.info('done')
    if fine:
        tqdm.write('All went well !')
    else:
        tqdm.write('Something went wrong somewhere. See log for details')

if __name__ == "__main__":
    ## Parsing Arguments
    # =============================================================================

    # Input file
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog='batch_runs.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    Runs simulations in a batch for various values of the death rate
                                     ''')
    # Optional arguments
    parser.add_argument('-n','--nsim',type=int,action='store')
    parser.add_argument('-s','--seed',type=str,action='store')
    parser.add_argument('-b','--progress-bar', dest='pbar', action='store_true',help='displays a progress bar')
    parser.add_argument('--no-pbar', dest='pbar', action='store_false')
    parser.add_argument('--save-raw',dest='svaraw',action='store_true',help='save raw sumstats')
    # Required arguments
    parser.add_argument("--prior", help="filename of priors",type=str,action="store")
    parser.add_argument("--dirname", type=str, help="output dirname",action="store")
    parser.add_argument('-d','--list-d', nargs='+', help='<Required> list of d values', required=True,dest='dlist')
    parser.add_argument('-p','--list-p', nargs='+', help='<Required> list of d values', required=True,dest='plist')
    # Set optional
    parser.set_defaults(nsims  = 20000,
                        pbar=False,
                        seed = None,
                        saveraw = False)
    args = parser.parse_args()
    path = os.getcwd()
    priorfile = os.path.abspath(args.prior)
    # ------------------------------------------------------------------------------
    # Other parameters (simulation properties)
    # ------------------------------------------------------------------------------
    dirname = os.path.abspath(args.dirname)
    dlist   = args.dlist
    plist   = args.plist
    pbar    = args.pbar
    nsim    = args.nsim
    seed    = args.seed
    saveraw = args.saveraw
    runSomeBatches(dlist,plist,dirname,priorfile,seed=seed,nsim=nsim,ndigit=3,pbar=pbar,saveraw=saveraw)
