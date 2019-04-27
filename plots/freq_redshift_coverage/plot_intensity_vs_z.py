import sys
sys.path.append('../../')

import CO_data
import numpy as np 
import matplotlib.pyplot as plt

from colossus.cosmology import cosmology

from labellines import labelLine, labelLines
from colour import Color

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': [r'\ttdefault']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']


def plot_intensity_vs_z():

    c1 = Color(tb_c[7])
    c2 = Color(tb_c[0])
    c3 = Color(tb_c[7])
    
    clist1 = list(c3.range_to(c2, 10))
    clist2 = list(c2.range_to(c1, 4))
    clist2 = clist2[1:]
    
    clist = np.append(clist1, clist2)
    clist = [c.get_hex() for c in clist]
    clist.append(tb_c[-1])
    
    lslist = ['dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed',
              None, None, None, None, None]
    
    xlist = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.75, 1.25, 1.5, 1.5, 8]
    
    lines = ['13-12', '12-11', '11-10', '10-9', '9-8', '8-7', 
             '7-6', '6-5', '5-4', '4-3', '3-2', '2-1', '1-0', 'CII']
    
    zlist = np.linspace(0, 10, 100)
    sigma = 1
    stable = CO_data.smit_unlog_table
    
    avg_int_list = {}
    
    for l in lines:
        nuemit = CO_data.CO_lines[l]
        L0 = CO_data.CO_L0[l]
        avg_int_list[l] = CO_data.avg_int(L0, zlist, stable, nuemit, cosmo, smooth=True, sigma=sigma)
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    for l,c,ls in zip(lines, clist, lslist):
        ax.plot(zlist, avg_int_list[l], label=l, c=c, ls=ls)
    
    labelLines(ax.get_lines(), zorder=2.5, bbox={'pad': 0, 'facecolor': 'white', 'edgecolor': 'white'}, 
                               xvals=xlist, fontsize=7.5)
    
    ax.set_yscale('log')
    
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\left< I \right> \,[\,\text{Jy}/\text{str}\,]$')
    
    fig.savefig('intensity_vs_z.pdf')



if __name__ == '__main__':
    plot_intensity_vs_z()
