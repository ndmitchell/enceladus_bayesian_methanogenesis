#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data for different metabolisms modelling




This code is under MIT license. See the License.txt file.
This module contains the functions useful to numerically solve the model

Antonin Affholder
antonin.affholder@ens.fr
"""

import numpy as np

radius = 1 #mum
vol = (4/3)*np.pi*radius**3 #mum3
Bstr = 18*(1e-15)*vol**(0.94)

T0 = 273.15


## Henry law parameters Ci/Pi
## These are actually fits on simulation from Aspen Plus
aH = 7.8e-4
aC = lambda T : np.exp(9345.17/T-167.8108+23.3585*np.log(T)+(0.023517-2.3656e-4*T+4.7036e-7*T**2)*35.0) #solubility of CO2 from Sauterey 2020
aG = 1.4e-3

## Biological things

general = {'theta':10,'rmax':1,'m':0.1,'kd':0.5,'ks':1e-9,'V':vol,'Bstr':Bstr,'tau':1.73*1e-5}

phipar = {'phi':0.2,'rpore':50e-6}

# Methanogens
#     The data for the enzymatic actyvity was taken from Daniel et al 2010 for extremophiles of growth temperature
methanogens = {'St_cata':np.array([-1,-0.25,0.25]),
               'St_ana':np.array([-2.1,-1,-0.2]),
               'NoC':1,
               'gamma':4,
               'deltaG0Cat':-32575,
               'deltaG0Ana':28250,
               'deltaH0Cat':-63175,
               'deltaH0Ana':128000,
               'T_eq':90+T0,
               'dgacat':72000,
               'dhaeq':305000}

BioRunMeth = {'dg0scat':methanogens['deltaG0Cat'],
              'dh0scat':methanogens['deltaH0Cat'],
              'tau':general['tau'],
              'NoC':methanogens['NoC'],
              'gamma':methanogens['NoC'],
              'vol':vol,
              'T_eq':methanogens['T_eq'],
              'dgacat':methanogens['dgacat'],
              'dhaeq':methanogens['dhaeq'],
              'rmax':72,
              'theta':10,
              'dth':0.1,
              'catabolism':np.array(['H2','CO2','CH4']),
              'sto_th':np.array([-1,-0.25,0.25]),
              'sto_lm':np.array([1,0.25])}

## Hydrothermal vents
# Template : {'Tmax':,'pH':,'H2':,'CH4':,'CO2'}
LostCity = {'Tmax':93+T0,
            'pH':10,
            'H2':0.3e-3,
            'CH4':0.2e-3,
            'CO2':23e-3,
            'NH4':1e-5}

Taubner2018_H2CO2 = {'Tmax':80+T0,
                     'pH':7,
                     'H2':1.52*1e-3,
                     'CH4':1e-10,
                     'CO2':7.73*1e-3,
                     'NH4':1e-5}
