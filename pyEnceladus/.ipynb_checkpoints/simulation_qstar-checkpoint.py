#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyEnceladus.universal_htv as uhtv
import pyEnceladus.physical as phc
import pyEnceladus.data_htv as dhtv
import numpy as np

defsimpar = {'epsilon':1,'dr':0.1,'Rmax':100}

def sim_meth_enc(F,Tf,To,oconc,fconc,dth,T_eq,dgacat,dhaeq,tau,epsilon,dr,Rmax,**kwargs):
    """
    Runs a simulation (computes spatial composition, then steady-state, then mixes)
    ----------------------------------------------------------------------------
    args :
        - F       : float
                    hotspot total power (W)
        - Tf      : float
                    Hydrothermal fluid temperature
        - To      : float
                    Ocean temperature
        - oconc   : dict
                    Contains concentrations in the ocean
        - fconc   : dict
                    Contains concentrations in the hydrothermal fluid
        - dth     : float
                    Basal mortality rate
        - T_eq    : float
                    Temperature midpoint of the quilibrium Eact/Einact (K)
        - dgacat  : float
                    Activation energy of the catalysed reaction (J/mol)
        - dhaeq   : float
                    change in enthalpy associated with the Eact/Einact equilibrium (J/mol)
        - tau     : float
                    Scaling factor : moles of catalytic enzymes per mole of biomass
        - epsilon : float
                    Mixing layer thickness
        - dr      : float
                    Discrete spatial interval size
        - Rmax    : float
                    Patch size (m)
    ----------------------------------------------------------------------------
    Returns :
        - inhabited_plume   : dict
                              Composition of the inhabited plume
        - uninhabited_plume : dict
                              Composition of the uninhabited plume
        - success           : bool
                              Success of numerical root-finding
    """
    ## non varying parameters
    r = np.arange(0,Rmax,dr)
    dgdiss = uhtv.Dgdiss(**dhtv.methanogens)
    Stoi   = dhtv.methanogens['St_cata']

    qmaxfun = uhtv.Vmax(T_eq,dgacat,dhaeq)
    Jmax = phc.buoyflux(epsilon,phc.enceladus['g'],Tf,To)
    c = phc.t_grad_par(F,Tf,To,Jmax=Jmax)
    # Computing the gradients
    jf = phc.thermflux(c,Jmax,r)
    T = phc.localtempss(jf,Tf,To,epsilon,phc.enceladus['g'])
    jc = phc.buoyflux(epsilon,phc.enceladus['g'],T,To)
    Conc0 = phc.abiotic_conc(oconc,fconc,jf,jc)
    # Computing generic biological parameters

    qcat = qmaxfun(T)*tau
    em   = uhtv.MaintenanceE(T)
    dg0vec = uhtv.DeltaG0(T,dhtv.methanogens['deltaG0Cat'],dhtv.methanogens['deltaH0Cat'],corr=True)
    # Computing the biological steady-state
    cstar,success = uhtv.RunBioHTV_Qstar(Conc0,T,dg0vec,em,dgdiss,qcat,dth,Stoi)
    
    bmass = (jc/qcat)*(Conc0['H2']-cstar['H2'])
    bmasstot = np.sum(bmass*2*np.pi*r)*dr
    
    inhabited_plume = phc.Fluxes(jc,cstar,r,dr)
    uninhabited_plume = phc.Fluxes(jc,Conc0,r,dr)
    totalconvF = phc.Total_buoyflux(jc,r,dr)
    return(inhabited_plume,uninhabited_plume,totalconvF,bmasstot,success)
