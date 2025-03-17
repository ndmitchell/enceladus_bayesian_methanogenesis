#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Plotting functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pyEnceladus.stats_analysis as homestats

import pyEnceladus.physical as phc

# Statistical

def autolabel(rects,ax, xpos='center',orientation='vertical',bottom=0):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        if orientation=='vertical':
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}
            if rect.get_height()+bottom > 0.01 or rect.get_height()+bottom == 0.:
                height = np.round(rect.get_height()+bottom,2)
                ax.annotate('{}'.format(np.round(height,2)),
                            xy=(rect.get_x() + rect.get_width() / 3, height),
                            xytext=(3,offset[xpos]*3),  # use 3 points offset
                            textcoords="offset points",  # in both directions
                            ha=ha[xpos], va='bottom',fontsize=5.5)
            else:
                height = np.round(rect.get_height()+bottom,2)
                ax.annotate(r'$<1\%$',
                            xy=(rect.get_x() + rect.get_width() / 3, height),
                            xytext=(3,offset[xpos]*3),  # use 3 points offset
                            textcoords="offset points",  # in both directions
                            ha=ha[xpos], va='bottom',fontsize=5.5)
        elif orientation=='horizontal':
            va = {'center': 'center', 'top': 'bottom', 'bottom': 'top'}
            offset = {'center': 1, 'top': 1, 'bottom': -1}
            width = np.round(rect.get_width(),2)
            ax.annotate('{}'.format(width),
                        xy=(width,rect.get_y()+ rect.get_height() / 3),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        va=va[xpos], ha='left',fontsize=5.5) 


def plot_Likelihood(ax,data,var,obs,topval,color,**kwargs):
    Ldata = homestats.get_Nli_range(data,var,obs)
    xi = np.linspace(np.min(data[var])-1,np.max(data[var])+1,50)
    ax.fill(xi,Ldata['kernel'](xi),alpha=0.5,color=color)
    ax.hist(data[var],color=color,**kwargs)
    ax.plot([obs[var][0]]*2,[0,topval],'--',color='magenta')
    ax.plot([obs[var][1]]*2,[0,topval],'--',color='magenta')
    #ax.text(3,0.2,r'$\hat{P}(\Phi_{CH_4}|I) \approx %s $' %np.round(Ldata['NL'],2))
    ax.plot([Ldata['maxL']-Ldata['width']]*2,[0,topval],'--',color='grey')
    ax.plot([Ldata['maxL']+Ldata['width']]*2,[0,topval],'--',color='grey')

def DistriPlot(ax,data,var,palette = sns.color_palette('colorblind'),Nbins=100,**kwargs):

    N = len(data)
    bins = np.linspace(np.min(data[var]),np.max(data[var]),Nbins+1)
    countsHb,binsHb = np.histogram(data[var][data['model']==0],bins=bins)
    countsH,binsH = np.histogram(data[var][data['model']==1],bins=bins)
    countsI,binsI = np.histogram(data[var][data['model']==2],bins=bins)

    ax.hist(binsHb[:-1], binsHb, weights=countsHb/N,color=palette[0],**kwargs)
    ax.hist(binsH[:-1], binsH, weights=countsH/N,color=palette[1],**kwargs)
    ax.hist(binsI[:-1], binsI, weights=countsI/N,color=palette[2],**kwargs)

def kdemodelsplot(data,var,ax,n=100,alpha=0.3,colors=sns.color_palette()):
    mxvar= np.max(data[var])
    mivar= np.min(data[var])
    span = mxvar-mivar
    box  = [mivar-0.1*span,mxvar+0.1*span]
    X    = np.linspace(box[0],box[1],n)
    kde0 = stats.gaussian_kde(data[var][data['model']==0])
    fac0 = len(data[var][data['model']==0])/len(data[var])
    p0   = kde0(X)*fac0
    ax.fill(X,p0,color=colors[0],alpha=alpha)
    ax.plot(X,p0,color=colors[0])

    if np.sum(data['model']==1) > 10:
        kde1 = stats.gaussian_kde(data[var][data['model']==1])
        fac1 = len(data[var][data['model']==1])/len(data[var])
        p1   = kde1(X)*fac1
        ax.fill(X,p1,color=colors[1],alpha=alpha)
        ax.plot(X,p1,color=colors[1])
    if np.sum(data['model']==2) > 10:
        kde2 = stats.gaussian_kde(data[var][data['model']==2])
        fac2 = len(data[var][data['model']==2])/len(data[var])
        p2   = kde2(X)*fac2
        ax.fill(X,p2,color=colors[2],alpha=alpha)
        ax.plot(X,p2,color=colors[2])



def plot_priors(priors_file,plot_name,delimiter=';'):
    """
    priors_file refer to the file containing the values resulting from randomly drawn parameters
    """
    priors = pd.read_csv(priors_file,delimiter=delimiter)
    nsubplots = len(priors.keys()[1:])
    fig, ax = plt.subplots(ncols = 2,nrows = nsubplots//2+nsubplots%2,figsize=(8,11))
    for i in range(len(priors.keys()[1:])):
        key = priors.keys()[i+1]
        axis = ax[i//2,i%2]
        if key != 'Tf':
            sns.distplot(np.log10(priors[key]),ax=axis)
        else:
            sns.distplot(priors[key],ax=axis)
        axis.set_xlabel(key)
    if nsubplots%2:
        ax[-1,-1].set_axis_off()
    plt.suptitle('Priors')
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()

def plot_inference(sim,dmat,datapoints,pih,dth,sel1,sel2,pprob1,pprob2,figname,size=(8,8),legendfontsize=12,textfont=14,dpi=None,show=False):
    mpl.rc('axes', labelsize=12)
    mpl.rc('legend', fontsize=12)
    mpl.rc('xtick', labelsize=10)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=10)
    mpl.rc('text',usetex=True)
    font = {'family' : 'sans-serif',
            'sans-serif':'DejaVu Sans',
            'weight' : 'normal',
            'size': 14}
    mpl.rc('font', **font)
    fig, axes = plt.subplots(nrows=3,ncols=3,figsize=size)

    hbar = np.round(len(sim['R1'][sim['model']==0])/len(sim),2)


    sns.set_context('paper')
    ant = (('A','B','C'),('D','E','F'))
    keys = list(sim.keys())
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            if i==j:
                kdemodelsplot(sim,keys[i],ax)
                ax.set_ylim(0,0.26)
                ax.set_yticks([0,0.06,0.12,0.18,0.24])
                #ax.set_ylabel(keys[i]+' ($log_{10}$)')
                #ax.set_xlabel(keys[i]+' ($log_{10}$)')
                ax.plot(np.ones(10)*np.log10(datapoints[i][0]),np.linspace(0,.18,10),color='red',linestyle='--')
                ax.plot(np.ones(10)*np.log10(datapoints[i][1]),np.linspace(0,.18,10),color='red',linestyle='--')
            elif i<j:
                ax.scatter(sim[keys[j]],sim[keys[i]],c=[sns.color_palette()[sim['model'][k]] for k in range(len(sim['model']))],marker='d',s=0.1,rasterized=True)
                ax.set_ylim(-6,6)
                ax.set_yticks([-5,0,5])
                ax.plot(np.log10(datapoints[j][0]),np.log10(datapoints[i][0]),'*',color='red')
                ax.plot(np.log10(datapoints[j][1]),np.log10(datapoints[i][1]),'*',color='red')
            elif i!=2 or j!=0:
                ax.axis('off')
            ax.set_xlim(-6,10)
            ax.set_xticks([-5,0,5,10])

    xticks = [r'$\overline{H}$',r'$H$',r'$I$']
    sns.heatmap(dmat,cmap='YlOrBr',ax=axes[2,0],annot=True,cbar=False,linecolor='white',linewidths=2,square=True,xticklabels=xticks,yticklabels=xticks)
    axes[2,0].text(0.1,1.1,'(g)',transform=axes[2,0].transAxes,fontsize=10)

    axes[0,2].set_xticklabels([])
    axes[0,1].set_xticklabels([])
    axes[1,2].set_xticks([])

    unin = mpatches.Patch(color=sns.color_palette()[0])
    habi = mpatches.Patch(color=sns.color_palette()[1])
    inha = mpatches.Patch(color=sns.color_palette()[2])
    dlin = mlines.Line2D([], [], color='red',linestyle='--', marker='*', markersize=5)

    axes[2,1].legend(handles=[unin,habi,inha,dlin],labels=['Uninhabitable '+r'($\bar{H}$)','Habitable '+r'($H$)','Inhabited '+r'($I$)','Data'],loc='lower center',fontsize=legendfontsize)

    models = [r'\bar{H}',r'H',r'I']
    if sel1 != sel2:
        selmod = models[sel1]+','+model[sel2]
        sign   = '='
        minprob = '\{{0},{1}\}'.format(pprob1,pprob2)
    else:
        selmod = models[sel1]
        sign   = '>'
        minprob= np.round(np.min([pprob1,pprob2]),3)

    equations = r'\begin{{eqnarray*}} P_{{prior}}(\bar{{H}}) &=& {0} \\  P_{{prior}}(I | H) &=& {1} \\ \hat{{m}} &=& {2} \\ \pi(\hat{{m}}|S(x^0)) &{3}& {4}\\ d &=& {5} \end{{eqnarray*}}'.format(hbar,pih,selmod,sign,minprob,dth)
    axes[1,0].text(-0.1,0.4,equations,verticalalignment='center',horizontalalignment='left',transform=axes[1,0].transAxes,fontsize=textfont)

    bindx = np.arange(3)
    rows = np.repeat([0,1,2],[3,2,1])
    cols = [bindx[i:] for i in range(3)]
    cols = np.array([val for sublist in cols for val in sublist])
    labels = ['({0})'.format(letter) for letter in ['a','b','c','d','e','f']]
    for i,j,lab in zip(rows,cols,labels) :
        axes[i,j].text(0.1,0.9,lab,transform=axes[i,j].transAxes,fontsize=10)

    axes[1,1].set_yticklabels([])
    #axes[2,2].set_yticklabels([])
    axes[0,2].set_yticks([-5,0,5])
    axes[0,1].set_yticklabels([])

    #axes[0,1].set_yticklabels([])

    axes[0,1].set_yticks([])
    axes[0,1].set_xticks([])
    axes[0,2].set_xticks([])
    axes[0,2].set_ylabel('R1 (log)',rotation=-90)
    axes[0,2].yaxis.set_label_position("right")
    axes[0,0].set_xlabel('R1 (log)')
    axes[1,1].set_xlabel('R2 (log)')
    axes[1,2].set_ylabel('R2 (log)',rotation=-90)
    axes[1,2].yaxis.set_label_position("right")
    axes[2,2].set_xlabel('Qp (log)')
    axes[0,0].set_ylabel('density')


    for i in range(3):
        axes[i,2].yaxis.tick_right()


    fig.subplots_adjust(wspace=0.1,hspace=0.1)
    #fig.tight_layout()
    #plt.savefig('bayesiand1_pres.pdf',dpi=400)

    plt.savefig(figname,dpi=dpi)
    if show:
        plt.show()
    plt.close()

# UI model
def plotAdvection(ax,r,c,Jmax,Tf,To,epsilon,g,sup=40,width=0.003,scale=0.001,awidth=0.05,hlength=0.1,**kwargs):
    jf = phc.thermflux(c,Jmax,r)
    T = phc.localtempss(jf,Tf,To,epsilon,g)
    jc = phc.buoyflux(epsilon,g,T,To)
    X = np.linspace(0,sup,20)
    Yf = np.zeros_like(X)
    Yc = np.ones_like(X)
    U = np.zeros_like(X)
    Vjf = phc.thermflux(c,Jmax,X)
    Tp = phc.localtempss(Vjf,Tf,To,epsilon,g)
    Vjc = phc.buoyflux(epsilon,g,Tp,To)
    #sns.set_style('white',{'xtick.bottom':True,'ytick.left':True})
    #fig, ax = plt.subplots(figsize=(6,3))
    #sns.set_context('paper')
    jfq = ax.quiver(X,Yf,U,Vjf,width=width,color='orangered',units='width',scale=1 / scale)
    jcq = ax.quiver(X,Yc,U,Vjc,width=width,color='dodgerblue',units='width',scale=1 / scale)
    qk = ax.quiverkey(jfq, 30, 0.3, 50, '10 $\mathregular{kg~s^{-1}~m^{-2}}$', labelpos='N', coordinates='data',color='k',fontproperties={'size':5.5})
    ax.plot(X,np.ones_like(X),'--k',linewidth=0.75)
    ax.plot(X,np.zeros_like(X),'--k',linewidth=0.75)
    ax.arrow(sup-2,0,0,1,width=awidth,length_includes_head=True,head_length=hlength,head_width=5*awidth,color='k')
    ax.arrow(sup-2,1,0,-1,width=awidth,length_includes_head=True,head_length=hlength,head_width=5*awidth,color='k')
    ax.text(sup-1.5,0.5,'$\epsilon$',fontsize=5.5)
    dil, = ax.plot(r,(T-To)/(Tf-To),color='k',linewidth=0.75)
    ax.legend(handles=[jfq,jcq,dil],labels=['$J_f$','$J_c$','$x$'],**kwargs)
    #plt.plot(r,jc,'dodgerblue')
    #plt.plot(r,jf,'orangered')
    ax.set_xlim(-1,sup)
    ax.set_ylim(-0.1,1.8)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','0.5','1'])

    ax.set_xlabel('Distance from center (m)')
    ax.set_ylabel('z (m) ; $x$')
    sns.despine(ax=ax,trim=True)
    #return(fig)
