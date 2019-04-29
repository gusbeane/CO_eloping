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


def plot_intensity_vs_nu(kplot=0.7, sigma=1):

    nulist = np.linspace(170, 270, 100)

    ltarget = 'CII'
    linterlopers = ['13-12', '12-11', '11-10', '10-9', '9-8', '8-7', 
                    '7-6', '6-5', '5-4', '4-3', '3-2', '2-1', '1-0']

    nuemit_int = [CO_data.CO_lines[l] for l in linterlopers]
    L0l_list = [CO_data.CO_L0[l] for l in linterlopers]

    nuemit_target = CO_data.CO_lines[ltarget]
    zlist = (nuemit_target - nulist)/nulist

    L0target = CO_data.CO_L0[ltarget]
    Itarget = np.array([CO_data.avg_int(L0target, z, CO_data.smit_unlog_table, nuemit_target, cosmo, smooth=False, sigma=sigma) for z in zlist])

    Patk_target = []
    Patk_int = {l: [] for l in linterlopers}

    for z,nu,It in zip(tqdm(zlist), nulist, Itarget):
        k, Ptarget = mf.intensity_power_spectrum(z, 3, It, cosmo, returnk=True, angle_averaged=True)

        Pk = np.interp(kplot, k, Ptarget)
        Patk_target.append(Pk)

        for l, nue, L0l in zip(linterlopers, nuemit_int, L0l_list):
            zl = (nue/nuemit_target)*(1+z) - 1
            if zl >= 0.0:
                Itarget = CO_data.avg_int(L0l, np.array([zl]), CO_data.smit_unlog_table, nue, cosmo, smooth=True, sigma=sigma)[0]
                Pl = mf.intensity_power_spectrum(zl, 2, It, cosmo, angle_averaged=True)

                Pk = np.interp(kplot, k, Pl)
                Patk_int[l].append(Pk)

            else:
                Patk_int[l].append(0)

    Patk_target = np.array(Patk_target)
    for l in linterlopers:
        Patk_int[l] = np.array(Patk_int[l])

    fig, ax = plt.subplots(1, 1)

    ax.plot(zlist, kplot**3 * Patk_target/(2.*np.pi**2), c=tb_c[0], label=ltarget)
    
    Ptot = np.zeros(np.shape(Patk_target))
    # for l in linterlopers:
    for l in ['2-1', '3-2', '4-3', '5-4', '6-5', '7-6']:
        ax.plot(zlist, kplot**3 * Patk_int[l]/(2.*np.pi**2), label=l)
        Ptot += Patk_int[l]

    print(Ptot)
    ax.plot(zlist, kplot**3 * Ptot/(2.*np.pi**2), c='k', label='int tot')
    ax.legend()

    ax.set_yscale('log')

    ax.set_xlabel(r'$z_{\text{C~\textsc{ii}}}$')
    ax.set_ylabel(r'$\Delta^2 (k)$')

    plt.show()

if __name__ == '__main__':
    # plot_intensity_vs_z()
    P = plot_intensity_vs_nu()

