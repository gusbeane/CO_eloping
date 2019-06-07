import sys
sys.path.append('../../')

import CO_data
import numpy as np 
import matplotlib.pyplot as plt

from multifield import alpha_factors

from colossus.cosmology import cosmology

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': [r'\ttdefault']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

stable = CO_data.smit_unlog_table

Jlist = np.arange(3, 13)
Jlist_label = [str(J)+'-'+str(J-1) for J in Jlist]
freqlist = [CO_data.CO_lines_LT16[Jlabel] for Jlabel in Jlist_label] # in GHz

ICOlist = []
ztarget = 7
nuCII = CO_data.CO_lines_LT16['CII']
L0CII = CO_data.CO_L0['CII']
ICII = CO_data.avg_int(L0CII, ztarget, stable, nuCII, cosmo)

for l,nu in zip(Jlist_label, freqlist):
    zinterloper = (nu/nuCII)*(1+ztarget) - 1

    L0 = CO_data.CO_L0[l]
    inten = CO_data.avg_int(L0, zinterloper, stable, nu, cosmo)
    ICOlist.append(inten)

ICOlist = np.array(ICOlist)

fig, ax = plt.subplots(1, 1)

ax.scatter(Jlist, ICOlist/ICII)

ax.set_xlabel(r'$J$')
ax.set_ylabel(r'$\left< I_{\text{CO}}(z_i)\right> / \left<I_{\text{C~\textsc{II}}}(z_t = 7)\right>$')

ax.set_xlim(3, 13)
ax.set_ylim(-0.1, 0.5)

fig.tight_layout()
fig.savefig('LT16_Fig2.pdf')
