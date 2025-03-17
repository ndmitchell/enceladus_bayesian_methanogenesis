#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computation of physical parameters inside the htv chimney




This code is under MIT license. See the License.txt file.
This module contains the functions useful to numerically solve the model

Antonin Affholder
antonin.affholder@bio.ens.psl.eu
"""

import numpy as np


glo_rho   = 1000
glo_Cp    = 4200
glo_alpha = 3e-4

enceladus = {'g':0.12,}

def t_grad_par(F,Tf,To,Jmax=None,c=None):
    """
    Computes the parameters to the hydrothermal flux density function
    With energy conservation

    Arguments
    ----------------------------------------------------------------------------
    F    : float
         The hot spot total power (W)
    Tf   : float
         Hydrothermal fluid temperature (K)
    To   : float
         Ocean temperature (K)
    Jmax : float optional
         Maximum hydrothermal flux density allowed
    c    : float optional
         Standard deviation of hydrothermal flux density

    Returns
    ----------------------------------------------------------------------------
    Jmax : float
         Maximal flux density allowed
    c    : float
         Flux density standard deviation
    """
    Jf = F/(glo_Cp*(Tf-To))
    if Jmax==None and c==None:
        return('error specify a parameter to constrain')
    elif Jmax == None:
        Jmax = Jf/(np.pi*c**2)
        return(Jmax)
    elif c==None:
        c = np.sqrt(Jf/(np.pi*Jmax))
        return(c)

def buoyflux(epsilon,g,T,To):
    """
    Computes the flux density out of the mixing layer (Goodman 2004)

    Arguments
    ----------------------------------------------------------------------------
    epsilon : float
            thickness of mixing layer (m)
    g       : float
            Gravity acceleration (m.s-2)
    T       : float or array
            temperature (K)
    To      : float
            Ocean temperature (K)

    Returns
    ----------------------------------------------------------------------------
    Jc : float or array
       Convective flux density
    """
    return(glo_rho*np.sqrt(g*2*epsilon*glo_alpha*(T-To)))

def thermflux(c,Jmax,r):
    """
    Computes the local hydrothermal flux density

    Arguments
    ----------------------------------------------------------------------------
    c    : float
         Standard deviation
    Jmax : float
         Maximal flux allowed
    r    : float or array
         r coordinate

    Returns
    ----------------------------------------------------------------------------
    jf : float or array
       Local hydrothermal flux density
    """
    return(Jmax*np.exp(-(r**2)/c**2))

def localtempss(Jf,Tf,To,epsilon,g):
    """
    Computes the steady-state of the temperatures in the mixing layer

    Arguments
    ----------------------------------------------------------------------------
    Jf      : float or array
            Hydrothermal flux
    Tf      : float
            Hydrothermal fluid temperature
    To      : float
            Ocean temperature
    epsilon : float
            Mixing layer thickness
    g       : float
            Acceleration of gravity

    Returns
    ----------------------------------------------------------------------------
    T : float or array
        Local steady state temperature
    """
    return(((Jf*(Tf-To))/(glo_rho*np.sqrt(g*2*epsilon*glo_alpha)))**(2/3) + To)

def abiotic_conc(oconc,fconc,jf,jc):
    """
    Computes the abiotic concentrations

    Arguments
    ----------------------------------------------------------------------------
    oconc : dict
          Dictionnary containig the concentrations in the ocean waters
    fconc : dict
          Dictionnary containig the concentrations in the hydrothermal waters
    jf    : array
          The hydrothermal flux density
    jc    : array
          The convective flux density

    Returns
    ----------------------------------------------------------------------------
    abconc : dict
           Dictionnary containig the arrays of concentrations at abiotic steady
           -state
    """
    keys = list(oconc.keys())
    abconc = dict()
    for k in keys:
        c = np.zeros_like(jc)
        c[jc > 0] = (jf[jc > 0]/jc[jc > 0])*(fconc[k]-oconc[k])+oconc[k]
        c[jc == 0] = oconc[k]
        abconc[k] = c
    return(abconc)

def Total_buoyflux(jc,r,dr):
    """
    Computes the total flux out of the ML (kg.s-1)
    """
    return(np.sum((jc)*2*np.pi*r)*dr)

def Fluxes(jc,conc,r,dr):
    """
    Returns the fluxes in mol per second of every elements
    """
    return({k: np.sum(jc*conc[k]*2*np.pi*r)*dr for k in conc.keys()})

def PlumeMixing(jc,conc,r,dr):
    """
    Simple model of mixing in the plume

    Arguments
    ----------------------------------------------------------------------------
    jc   : array
         Convective flux density
    conc : dict
         Dictionnary containing the concentration arrays
    r    : array
         r coordinate
    dr   : float
         space step

    Returns
    ----------------------------------------------------------------------------
    pconc : dict
          Concentrations in the plume
    """
    Fc = Fluxes(jc,conc,r,dr)
    return({k:Fc[k]/Total_buoyflux(jc,r,dr) for k in conc.keys()})
