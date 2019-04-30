import sys
sys.path.append('../../')

import CO_data
import numpy as np 
import matplotlib.pyplot as plt

from colossus.cosmology import cosmology
import multifield as mf

from labellines import labelLine, labelLines
from colour import Color

from tqdm import tqdm

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': [r'\ttdefault']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

def plot_CII_ps(z=7, name='CIIps_z7.pdf'):

    kmin = 0.01
    kmax = 10

    l = 'CII'
    lint = ['3-2', '4-3', '5-4', '6-5']

    nue = CO_data.CO_lines[l]
    L0 = CO_data.CO_L0[l]
    Il = CO_data.avg_int(L0, z, CO_data.smit_unlog_table, nue, cosmo, smooth=False)
    bl = 3

    k, Pi = mf.intensity_power_spectrum(z, bl, Il, cosmo, kmin=kmin, kmax=kmax, returnk=True, angle_averaged=True)

    del2 = k**3 * Pi / (2. * np.pi**2)

    fig, ax = plt.subplots(1, 1)

    ax.plot(k, del2, c=tb_c[0], label='CII')

    del2int = np.zeros(np.shape(del2))
    for li in lint:
        nue_i = CO_data.CO_lines[li]
        L0 = CO_data.CO_L0[li]
        zint = (nue_i/nue) * (1+z)
        zint -= 1

        if zint > 0:

            Ii = CO_data.avg_int(L0, zint, CO_data.smit_unlog_table, nue_i, cosmo, smooth=False)
            bi = 2
            k, Pi = mf.intensity_power_spectrum(zint, bi, Ii, cosmo, kmin=kmin, kmax=kmax, returnk=True, angle_averaged=True,
                                                                     distort=True, ztarget=z)

            del2 = k**3 * Pi / (2.*np.pi**2)

            del2int += del2

    ax.plot(k, del2int, c=tb_c[2], label='CO interlopers')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$k\,[\,h/\text{Mpc}\,]$')
    ax.set_ylabel(r'$\Delta^2 (k)\,[\,\text{Jy}^2/\text{str}^2\,]$')

    ax.set_xlim([0.01, 10])
    ax.set_ylim([10, 1E8])

    ax.legend()
    ax.set_title('z='+str(z))

    fig.tight_layout()
    fig.savefig(name)

if __name__ == '__main__':
    plot_CII_ps()
    # plot_CII_ps(8, name='CIIps_z8.pdf')

