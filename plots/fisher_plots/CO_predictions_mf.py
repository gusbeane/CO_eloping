import numpy as np 

import sys
sys.path.append('../../')

from multifield import multifield, threefield, corner_plot
import CO_data
from colossus.cosmology import cosmology

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

def _Vk_(klist, Pklist, Vsurv):
    integrand = np.divide(np.square(klist*Pklist), 2.*np.pi**2)
    Vk = np.trapz(integrand, klist)
    return np.sqrt(Vsurv*Vk)

def three_fields_corner_plot():
    survey = 'CCAT-p'
    lines = ['6-5', '5-4', '4-3']
    
    b = 3
    bandwidth = 60
    kmax = 1
    
    zobs = 0.88
    nuemit = np.array([CO_data.CO_lines[l] for l in lines])
    nuobs = nuemit / (1.+zobs)

    Vsurv = CO_data.calc_Vsurv(nuobs[0], nuemit[0], bandwidth, CO_data.survey_area[survey], cosmo)
    
    Blist, Nlist = CO_data.gen_Blist_Nlist(3, bandwidth, kmax, zobs, lines, survey, cosmo)

    tf = threefield(zobs, Blist, Nlist, kmax, Vsurv, cosmo)
    labels = [r'$B_{6-5}\,[\,\text{Jy}/\text{str}\,]$', r'$B_{5-4}\,[\,\text{Jy}/\text{str}\,]$', 
              r'$B_{4-3}\,[\,\text{Jy}/\text{str}\,]$']
    fig, ax, _ = corner_plot(zobs, Blist, Nlist, cosmo, 1.75, tf=tf, labels=labels, printtext=False)
    fig.tight_layout()
    fig.savefig('CO_tf.pdf')

def plot_all_SN(survey, lines, zbounds=[0, 10], nz=1000):

    freq_range = CO_data.survey_freq_range[survey]

    b = 3
    bandwidth = 60
    kmax = 1

    # Vsurv = CO_data.calc_Vsurv(nuobs[0], nuemit[0], bandwidth, CO_data.survey_area[survey], cosmo)
    
    # Blist, Nlist = CO_data.gen_Blist_Nlist(3, bandwidth, kmax, zobs, lines, survey, cosmo)

    zextent_list = []
    for l in lines:
        nuemit = CO_data.CO_lines[l]
        zextent = (nuemit - freq_range)/freq_range
        zextent_list.append(np.flip(zextent))
    zextent_list = np.array(zextent_list)
    print(zextent_list)

    zlist = np.linspace(zbounds[0], zbounds[1], nz)
    lines_obs_list = []
    tf_bool_list = []

    for z in zlist:
        bool1 = z > zextent_list[:,0]
        bool2 = z < zextent_list[:,1]

        keys = np.where(np.logical_and(bool1, bool2))[0]

        lines_obs = [ lines[k] for k in keys ]

        lines_obs_list.append(lines_obs)

        if len(lines_obs) >= 3:
            tf_bool_list.append(True)
        else:
            tf_bool_list.append(False)

    mf_keys = np.where(tf_bool_list)[0]

    # first generate the S/N Cramer-Rao bound for the auto-spectrum
    # this is done at every redshift for each line that is in band

    SN_auto_list = []
    SN_mf_list = []
    for z, lines_at_z, mfbool in zip(zlist, lines_obs_list, tf_bool_list):
        SN = np.zeros(np.shape(lines))
        SN_mf = np.zeros(np.shape(lines))
        
        nuemit = CO_data.CO_lines[l]
        nuobs = nuemit / (1.+z)
        
        Vsurv = CO_data.calc_Vsurv(nuobs, nuemit, bandwidth, CO_data.survey_area[survey], cosmo)
        
        Blist, Nlist = CO_data.gen_Blist_Nlist(3, bandwidth, kmax, z, lines_at_z, survey, cosmo)

        mf = multifield(z, Blist, Nlist, kmax, Vsurv, cosmo)
        Vk = _Vk_(mf.klist, mf.Pklist, Vsurv)

        for i,l in enumerate(lines_at_z):
            for key, lp in enumerate(lines):
                if l == lp:
                    SN[key] = Blist[i]**2 * Vk / Nlist[i]

        if mfbool:
            cov = np.linalg.inv(mf.fmat)
            noise = np.sqrt(np.diag(cov))
            sn = Blist / noise

            for i,l in enumerate(lines_at_z):
                for key, lp in enumerate(lines):
                    if l == lp:
                        SN_mf[key] = sn[i]

        SN_auto_list.append(SN)
        SN_mf_list.append(SN_mf)

    SN_auto_list = np.array(SN_auto_list)
    SN_mf_list = np.array(SN_mf_list)

    for i,l in enumerate(lines):
        plt.plot(zlist, SN_auto_list[:,i], ls='dashed', c=tb_c[i])
        plt.plot(zlist, SN_mf_list[:,i], label=l, c=tb_c[i])

    # now generate the S/N Cramer-Rao bound for the auto-spectrum
    # done at redshifts where we have >= 3 fields

    plt.xlabel(r'$z$')
    plt.ylabel(r'$\text{S}/\text{N}(B)$')
    plt.yscale('log')

    plt.xlim(zbounds)

    plt.legend(frameon=False, title=r'$B$')
    plt.show()

    return zlist, SN_auto_list
            

if __name__ == '__main__':
    # three_fields_corner_plot()

    lines = ['12-11', '11-10', '10-9', '9-8', '8-7', '7-6', '6-5', '5-4', '4-3', '3-2']
    # for s in ['TIME', 'CONCERTO', 'CCAT-p']:
    for s in ['CCAT-p']:
        zlist, SN_auto_list = plot_all_SN(s, lines, zbounds=[0, 6])
