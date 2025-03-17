#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal computation tools for the thermodynamical model a
dapted to the specificity of hydrothermal vents environment




This code is under MIT license. See the License.txt file.
This module contains the functions useful to numerically solve the model

Antonin Affholder
antonin.affholder@ens.fr
"""

import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp
from pyEnceladus.data_htv import aH,aC,aG,methanogens

### Thermodynamical constants (global variables)
glo_R = 8.314             # Perfect gas constant J/K/mol
glo_TS = 298              # Standard temperature of 298K
glo_kb = 1.38064852*1e-23 # Boltzman constant J/K
glo_h = 6.62607004*1e-34  # Planck constant J s

## Thermodynamical functions

def ReactionQuotient(C,S):
    """
    Computes the reaction quotient associated to
    ractants in concentrations C for a reaction
    of stoichiometry S

    Arguments
    ------------
    C : array
        Array of the concentrations
    S : array
        Array of the stoichiometric (signed)
        coefficients (corresponding to C)
    """
    return(np.product(C**S))

def Slim(Concentrations, Stoichiometry,d=0):
    """
    Finds the limiting substrate concentration

    Arguments
    ------------------------------------------
    Concentrations : array
                   The concentrations  in the
                   same orders as Stoichiometry
    Stoichiometry  : array
                   The stoichiometry
    d              : int
                   Dimensions on which evaluate
                   max is 2 (I guess), default
                   is 1

    """
    if d==0:
        to_min = Concentrations/Stoichiometry
        return(Concentrations[np.argmin(to_min)],Stoichiometry[np.argmin(to_min)])
    elif d==1:
        return(np.array([Slim(Concentrations[:,j],Stoichiometry)
                         for j in range(len(Concentrations[0]))]))
    else:
        return(np.array([[Slim(Concentrations[:,i,j],Stoichiometry)
                          for i in range(len(Concentrations[0,:,0]))]
                         for j in range(len(Concentrations[0,0,:]))]))

def Dgdiss(NoC,gamma,**kwargs):
    """
    Returns the dissipaed energy during metabolism **REFERENCE**

    Arguments
    ------------------------------------
    NoC : int
        length of carbon source chain
    gamma : int
        oxydation level of carbon in carbon source
    """
    return((200 + 18*(6 - NoC)**1.8 + np.exp((((-0.2 - gamma)**2)**(0.16))*(3.6 + 0.4*NoC)))*1000)


def SolutionCorrection(T,Y,Kh):
    """
    Kh is the vector containing the Henry parameters Ci/Pi
    """
    return(-glo_R*T*np.sum(Y*np.log(Kh)))

def DeltaG0(T,DG0S,DH0S,corr=True,**kwargs):
    """
    Computes Gibbs Free Energy of a reaction with reaction quotient Q
    and standard gibbs free energy DG0S and standard enthalpy DH0S

    Arguments
    ---------------
    Q    : float
        reaction quotient of the reaction
    T    : float
        temperature (K)
    DG0S : float
        Standard gibbs free energy at 298K
    DH0S : float
        Standard enthalpy at 298K
    **kwargs
    """    
    DG0 = DG0S*(T/glo_TS) + DH0S*((glo_TS-T)/glo_TS)
    if corr:
        if isinstance(T,(np.ndarray)):
            correction = np.array([SolutionCorrection(T[i],Y=methanogens['St_cata'],Kh=np.array([aH,aC(T[i]),aG])) for i in range(len(T))])
        else :
            Kh = np.array([aH,aC(T),aG])
            correction = SolutionCorrection(T,Y=methanogens['St_cata'],Kh=Kh)
        DG0 += correction
    return(DG0)

def DeltaG(Q,T,DG0S,DH0S,corr=True,**kwargs):
    """
    Computes Gibbs Free Energy of a reaction with reaction quotient Q
    and standard gibbs free energy DG0S and standard enthalpy DH0S

    Arguments
    ---------------
    Q    : float
        reaction quotient of the reaction
    T    : float
        temperature (K)
    DG0S : float
        Standard gibbs free energy at 298K
    DH0S : float
        Standard enthalpy at 298K
    **kwargs
    """
    return(DeltaG0(T,DG0S,DH0S,corr=corr)+ glo_R*T*np.log(Q))
       

## Traits functions

def Bstruct(V):
    """
    Computes the structural biomass
    V is cell volume

    Arguments
    -----------
    V : float
        cell volume in mum
    """
    return(18*(1e-15)*V**(0.94))

def Compute_d(Qm,Qcat):
    """
    returns the decay rate

    Arguments
    -------------
    Qm   : float
        maintenance requirement rate (d-1)
    Qcat : float
        catabolic rate
    """
    if len(np.shape(Qm)) >= 1:
        d_mat = np.ones(np.shape(Qm))*glo_m
        indexes = np.where(Qcat <= Qm)
        d_mat[indexes] = glo_m + glo_kd*(Qm[indexes] - Qcat[indexes])
        return(d_mat)
    else:
        if Qcat > Qm:
            return(glo_m)
        else:
            return(glo_m + glo_kd*(Qm-Qcat))

def MaintenanceE(T):
    """
    Returns the maintenance energy rate per unit of biomass
    it is analog to an Arrhenius equation, representing
    the kinetics of biomass destabilizing
    From Tijhuis

    Arguments
    ---------------------------
    T : float
        Temperature (K)

    Returns
    ----------------------------
    Em : float
        Energy loss rate (J/d)
    """
    return(24*3.5*np.exp((69400/8.314)*(1/298-1/T))*1000)

def Bstar(qana,bstruct,rmax,theta):
    """
    Returns the internal biomass equilibrium

    Arguments
    ----------------------
    qana : float
        anabolic rate per unit of biomass (molC/molC.day)
    bstruct : float
        internal biomass quantity (molC)
    rmax : float
        maximal division rate (d-1)
    theta : integer
        steepness of division function parameter

    Returns
    -----------------------
    bstar : float
        internal biomass at equilibrium (molC)
    """
    if len(np.shape(qana))>0:
        err = False
        if np.any(qana >= rmax-1):
            err = True
            #print('Error! qana > rmax, internal biomass cannot reach an equilibrium, qana is artificially set at rmax-1')
            eridx = qana >= rmax-1
            qana[eridx] = rmax-1

        bstar = bstruct*((rmax/qana-1)**(-(1/theta))+2)
    else:
        if qana <= rmax-1:
            bstar = bstruct*((np.divide(rmax,qana)-1)**(-(1/theta))+2)
        else:
            bstar = bstruct*((rmax/(rmax-1))**(-(1/theta))+2)
    #if err:
        #print(bstar[eridx][0])
    return(bstar)

def Vmax(T_eq,dgacat,dhaeq,**kwargs):
    """
    Returns a function of temperature allowing to compute
    the maximal catalyzed reaction rate per unit of biomass
    which is named qmax/B in this model

    based on the equilibrium model from Daniel et al 2010

    Arguments
    --------------------------
    T_eq     : float
        Temperature midpoint of the quilibrium Eact/Einact (K)
    dgacat   : float
        activation energy of the catalysed reaction (J/mol)
    dhaeq    : float
        change in enthalpy associated with the Eact/Einact equilibrium (J/mol)

    Global variables
    ---------------------------
    glo_h  : float
        Planck constant
    glo_R  : float
        Perfect gas constant
    glo_kb : float
        Boltzmann constant
    """
    def vmaxfun(T):
        kcat = (glo_kb/glo_h)*T*np.exp(-(dgacat/(glo_R*T)))
        keq = np.exp((dhaeq/glo_R)*(1/T_eq-1/T))
        return((kcat/(1+keq))*60*60*24)
    return(vmaxfun)


## Solvers
# ------------------------------------------------------------------------------
# Solving the chemostat-type model using the reaction quotient method

def QstarVal(T,DG0,em,dgdiss,qcat,dth):
    """
    Computes the value of the reaction quotient at equilibrium

    Arguments
    ----------------------------------------------------------------------------
    T      : float or float array
             Temperature of the medium (K)
    DG0    : float or float array
              Standard Gibbs energy as a function of Temperature (J.mol^-1)
    em     : float or float array
             Maintenance energy rate (J.molC^-1.d^-1)
    dgdiss : float
             Dissipated energy associated with the catabolism
    qcat   : float or float array
             Catabolic rate (moleD.molCx^-1.d^-1)
    dth    : float
             basal death rate (d^-1)
    Returns
    ----------------------------------------------------------------------------
    Qstar  : float or float array
             Value of the reaction quotient at steady-state of the system
             ONLY IN CASE OF BIOLOGICAL ACTIVITY
    """
    return(np.exp(-(1/(glo_R*T))*(DG0+(dth+(em/dgdiss))*(dgdiss/qcat))))

def QstarToSol(Qstar,Conc0,Stoi):
    """
    Generates the function which root is the solution for the electron donor

    Arguments
    ----------------------------------------------------------------------------
    T      : float
           temperature (K)
    DG0    : float
           Standard gibbs free energy of catabolic reaction at temperature T
    em     : float
           Maintenance energy rate at temperature T
    dgdiss : float
           Dissipation energy
    qcat   : float
           Catabolic rate
    dth    : float
           Basal mortality rate
    Conc0  : dict
           Dictionnary, each key has only one float element,
           the initial concentration
           Convention is that the electron donor is the first key
    Stoi   : array
           Stoichiometry. eD first

    Returns
    ----------------------------------------------------------------------------
    eq : function
       Function which root is the eD at steady-state
    """
    keys = list(Conc0.keys())
    eD0 = Conc0[keys[0]]
    def eq(eD):
        return(Qstar -
               (1/eD)*np.product(np.array(
                   [(Conc0[keys[k]]-Stoi[k]*(eD-eD0))**(Stoi[k])
                    for k in range(1,len(keys))])))
    return(eq)

def RunBioHTV_Qstar(Conc0,T,dg0cat,em,dgdiss,qcat,dth,Stoi):
    """
    Solves the steady state in the mixing layer using the Qstar method

    Arguments
    ----------------------------------------------------------------------------
    Conc0  : dict
           Dictionnary containing the initial concentrations
    T      : array
           Temperature (K)
    dg0cat : array
           Standard Gibbs energy at temperature T
    em     : array
           Maintenance energy rate
    dgdiss : float
           Dissipation energy
    qcat   : array
           Catabolic rate
    dth    : float
           Basal mortality rate
    Stoi   : array
           Stoichiometry of the catabolic reaction

    Returns
    ----------------------------------------------------------------------------
    Concstar : dict
             Concentrations at steady state
    S        : bool
             Whether all went right
    """
    keys = list(Conc0.keys())
    eDvec = np.zeros_like(T)
    success = np.zeros_like(T,dtype=bool)
    eD0 = Conc0[keys[0]]
    for i in range(len(T)):
        Qstar = QstarVal(T[i],dg0cat[i],em[i],dgdiss,qcat[i],dth)
        eq = QstarToSol(Qstar,{key:Conc0[key][i] for key in Conc0.keys()},Stoi)
        infbd = np.max([eD0[i]+(1/Stoi[1])*Conc0[keys[1]][i]+1e-10,1e-10])
        Q0 = np.product(np.array([Conc0[key][i] for key in Conc0.keys()])**Stoi)
        if Q0 < Qstar:
            if eq(eD0[i])*eq(infbd) < 0 :
                eDsol = optimize.root_scalar(eq,bracket=[infbd,eD0[i]],method='brentq')
                converged = eDsol.converged
                eDstar = eDsol.root

            else:
                eDstar = infbd
                converged = True
        else:
            eDstar = eD0[i]
            converged = True
        eDvec[i] = eDstar
        success[i] = converged
    Concstar = {keys[k] : Stoi[k]*(eD0-eDvec)+Conc0[keys[k]] for k in range(len(keys))}
    S = np.all(success)
    return(Concstar,S)

def Nstar(bstar,Jf,Jc,cfed,coed,cstared,qcat):
    """
    Attempts to compute nstar
    """
    return((1/(bstar*qcat))*(Jf*(cfed-coed)+Jc*(coed-cstared))*3600*24)

#-------------------------------------------------------------------------------
# Solving the biological model ODE
def Nderpar(Q,T, qmaxfun, dgdiss, bstruct,dg0scat,dh0scat,tau,dth,rmax,theta,kd,**kwargs):
    """
    Gives growth parameters in a given situation
    """
    # Catabolic energy rate
    qcat  = qmaxfun(T)*tau
    dgcat = DeltaG(Q,T,dg0scat,dh0scat)
    em    = MaintenanceE(T)
    # When there is no energy in running the catabolic reaction
    if dgcat >= 0:
        qana = 0
        d = kd #was dth+constant
        bstar = 2*bstruct # Approximation for the internal content
        return(qana,d,bstar,qcat)
    else:
        qm = -em/dgcat
        lam = -dgcat/dgdiss
        # Subcase when there is not enough catabolic energy
        if qcat < qm :
            qana = 0
            d = dth + kd*(qm-qcat)
            bstar = 2*bstruct
        # Subcase when catabolic energy is over maintenance
        else:
            qana = lam*(qcat-qm)
            d = dth
            bstar = Bstar(qana,bstruct,rmax,theta)
        return(qana, bstar, d, qcat)

def Cderivative(qbio, c, sto, Jf, Jc, co, cf, epsilon):
    """
    Derivative of a particular concentration
    Qbio is the biological rate of electron donor consumption
    """
    return((1/(epsilon))*((Jf*(cf-co)+Jc*(co-c))*3600*24+qbio*sto))

def ODEsys(Y, t, T, Jf, Jc, oconc, fconc,qmaxfun, dgdiss, bstruct, kd, epsilon, dg0scat, dh0scat, tau, dth, rmax, theta, **kwargs):
    N, H2, CO2, CH4 = Y[0], Y[1], Y[2], Y[3]
    C = np.array([H2,CO2,CH4])
    S = np.array([-1,-0.25,0.25])
    #clim,slim = uhtv.Slim(np.array([H2,CO2]), np.array([1,0.25]))
    #Q = (CH4**0.25)/(H2*CO2**0.25)
    Q = ReactionQuotient(C,S)
    qana, bstar, d, qcat = Nderpar(Q,T, qmaxfun, dgdiss, bstruct,dg0scat,dh0scat,tau,dth,rmax,theta,kd)
    qbio = N*qcat*bstar
    par = {'Jf':Jf,'Jc':Jc,'epsilon':epsilon}
    dydt = np.array([(qana-d)*N,
                    Cderivative(qbio,H2,sto=-1, co=oconc['H2'],cf=fconc['H2'],**par),
                    Cderivative(qbio,CO2,sto=-0.25,co=oconc['CO2'],cf=fconc['CO2'],**par),
                    Cderivative(qbio,CH4,sto=0.25,co=oconc['CH4'],cf=fconc['CH4'],**par)])
    return(dydt)

def MethDynInt(Y0,t_span,T, Jf, Jc, oconc, fconc, qmaxfun, dgdiss, bstruct, kd, epsilon, dg0scat, dh0scat, tau, dth, rmax, theta,method='BDF',**kwargs):
    """
    Solves the system over the interval t_span
    """
    dydt = lambda t,Y : ODEsys(Y, t, T, Jf, Jc, oconc, fconc, qmaxfun, dgdiss, bstruct, kd, epsilon, dg0scat, dh0scat, tau, dth, rmax, theta)
    sol = solve_ivp(dydt, t_span=t_span, y0=Y0, method=method,atol=1e-12,rtol=1e-12)
    return(sol)


################################################################################
#################################  DEPRECATED  #################################
################################################################################

def RunBiology(Q,T,dg0scat,dh0scat,tau,NoC,gamma,vol,T_eq,dgacat,dhaeq,rmax,theta,dth,**kwargs):
    """
    Runs the biological model for given environmental conditions and gives the initial growth rate

    Arguments
    ----------------------------------------------
    Q       : float or array of floats
            Reaction quotient of catabolic reaction
    T       : float or array of floats
            Environment temperature (K)
    dg0scat : float
            Standard gibbs free energy change of catabolic reaction at 298K (J.mol-1)
    dh0scat : float
            Standard enthalpy change of catabolic reaction at 298K (J.mol-1)
    tau     : float
            Catabolic biomass ratio (ncatenzymes/ncellmolecules)
    NoC     : int
            Length of carbon source chain
    gamma   : int
            Oxydation number of carbon in carbon source
    vol     : float
            Cell volume (in mum3)
    T_eq    : float
            Temperature midpoint of the quilibrium Eact/Einact (K)
    dgacat  : float
            Activation energy of the catalysed reaction (J/mol)
    dhaeq   : float
            Change in enthalpy associated with the Eact/Einact equilibrium (J/mol)
    rmax    : float or int
            Maximum initial division rate
    theta   : float or int
            Division funciton slope parameter
    dth     : Death rate
            When everything is alright what is life expectancy of the cell? (d-1)
    Returns
    ------------------------------------------------
    results : dict
            Dictionnary containing model outputs
            - life  : bool or array of bool
                    Whether growth occurs or not
            - qana  : float or array of floats
                    Anabolic reaction rate per C-mol of biomass (molCx.molCx-1.d-1)
                    Which is also equal to the division rate under some assumptions
            - qcat  : float or array of floats
                    Catabolic reaction rate per C-mol biomass (molED.molCx-1.d-1)
            - bstar : float or array of floats
                    Internal biomass at internal equilibrium (molCx)
    """
    bstruct = Bstruct(vol)
    dgdiss = Dgdiss(NoC,gamma)
    dg = DeltaG(Q,T,dg0scat,dh0scat)
    lam = -dg/dgdiss
    em = MaintenanceE(T)
    qmaxfun = Vmax(T_eq,dgacat,dhaeq)
    qmax = qmaxfun(T)
    qm = -em/dg
    qcat = qmax*tau
    if len(np.shape(T)) > 0:
        lam[np.where(lam<0)] = 0
        qana = lam*(qcat - qm)
        didx = qana - dth <= 0 #index of death
        qana[didx] = 0
        divrate = qana - dth
        qcat_eff = np.empty_like(qcat)
        qcat_eff[didx] = 0
        qcat_eff[~didx] = qcat[~didx]
        bstar = np.empty_like(qana)
        bstar[didx] = 0
        bstar[~didx] = Bstar(qana[~didx],bstruct,rmax,theta)
    else :
        if lam < 0 or qcat - qm < 0:
            lam = 0
            qana = 0
            qcat_eff = 0
            divrate = 0
            bstar = 0
        else :
            qana = lam*(qcat - qm)
            divrate = qana - dth
            if divrate < 0:
                divrate = 0
                qcat_eff = 0
                bstar = 0
            else:
                qcat_eff = qcat
                bstar = Bstar(qana,bstruct,rmax,theta)
    results = {'life':~didx,'qana':qana,'qcat':qcat_eff,'bstar':bstar}
    return(results)



