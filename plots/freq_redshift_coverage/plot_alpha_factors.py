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

zjlist = np.linspace(0.00001, 1, 10000)

ztarget = [7, 20]

fig, ax = plt.subplots(1, 1)
for zi in ztarget:
    apar, aperp = alpha_factors(zi, zjlist, cosmo)

    ax.plot(zjlist, 1/(apar * aperp**2), label=str(zi))
    # ax.plot(zjlist, 1/aperp**2)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel(r'$z$')
ax.legend()
# ax.set_ylabel(r'$\left< I \right> \,[\,\text{Jy}/\text{str}\,]$')

plt.show()
# fig.savefig('intensity_vs_z.pdf')
