#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Plotting functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy import stats
import scipy.optimize as scopt
import scipy.integrate as scint

def get_Nli_range(data,var,obs):
    """
    Get a maximum likelihood interval
    """
    kernel = stats.gaussian_kde(np.array(data[var]))
    xobsmin = np.min(obs[var])
    xobsmax = np.max(obs[var])
    likelyhood = kernel.integrate_box_1d(xobsmin,xobsmax)
    tomin = lambda x : -kernel.pdf(x)
    Y,X = np.histogram(data[var])
    firstguess = X[np.argmax(Y)]
    xmaxL = scopt.minimize(tomin,firstguess).x[0]
    e = np.abs((xobsmax-xobsmin)/2)
    NLi  = likelyhood/scint.quad(kernel.pdf,xmaxL-e,xmaxL+e)[0]
    return({'var':var,'NL':NLi,'maxL':xmaxL,'width':e,'obs':np.array([xobsmin,xobsmax]),'kernel':kernel})

def NLikelihood(outputs,observations):
    """
    Returns the so-called normalized likelihoods (an estimate)
    of the different models given observations
    """
    Normalized_Likelyhood = []
    for var in observations.keys():
        distr = np.array(outputs[var])
        Lfun = stats.gaussian_kde(distr)
        tomin = lambda x : -Lfun.pdf(x)
        Y,X = np.histogram(distr)
        firstguess = X[np.argmax(Y)]
        xmaxL = scopt.minimize(tomin,firstguess).x[0]
        xobsmin = np.min(observations[var])
        xobsmax = np.max(observations[var])
        e = (xobsmax-xobsmin)/2
        xobs = np.mean(observations[var])
        NLi  = scint.quad(Lfun,xobs-e,xobs+e)[0]/scint.quad(Lfun,xmaxL-e,xmaxL+e)[0]
        if NLi >1:
            print('warning : N Likelyhood greater than 1 ({})! This may be due to numerical integration'.format(NLi))
            NLi = 1.0
        Normalized_Likelyhood.append(NLi)
    return(Normalized_Likelyhood)

def Likelihood(outputs,observations,observables):
    """
    Returns the so-called normalized likelihoods (an estimate)
    of the different models given observations
    """
    Likelyhood = []
    for var in observables:
        distr = np.array(outputs[var])
        Lfun = stats.gaussian_kde(distr)
        tomin = lambda x : -Lfun.pdf(x)
        Y,X = np.histogram(distr)
        firstguess = X[np.argmax(Y)]
        xobsmin = np.min(observations[var])
        xobsmax = np.max(observations[var])
        e = (xobsmax-xobsmin)/2
        xobs = np.mean(observations[var])
        NLi  = scint.quad(Lfun,xobs-e,xobs+e)[0]
        if NLi >1:
            print('warning : N Likelyhood greater than 1 ({})! This may be due to numerical integration'.format(NLi))
            NLi = 1.0
        Likelyhood.append(NLi)
    return(Likelyhood)