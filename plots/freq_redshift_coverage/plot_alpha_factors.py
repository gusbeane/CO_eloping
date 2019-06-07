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

zjlist = np.linspace(0, 8, 10000)

ztarget = [6, 7, 8]

fig, ax = plt.subplots(1, 1)
for zi,c in zip(ztarget, tb_c):
    apar, aperp = alpha_factors(zi, zjlist, cosmo)

    keys = np.where(zjlist < zi)[0]

    ax.plot(zjlist[keys], apar[keys], label=str(zi), c=c)
    ax.plot(zjlist[keys], aperp[keys], ls='dashed', c=c)
    # ax.plot(zjlist, 1/aperp**2)

ax.set_xlabel(r'$z$')
ax.legend()
ax.set_ylabel(r'$\alpha_{\parallel}(z)$, $\alpha_{\perp}(z)$')

ax.set_xlim(0, 8)
ax.set_ylim(0, 2)

fig.tight_layout()
fig.savefig('apar_aperp.pdf')
