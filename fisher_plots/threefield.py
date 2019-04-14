import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

class threefield(object):
    def __init__(self, z, Blist, Nlist, kmax, Vsurv, cosmo, kmin=1E-3, nint=1000,
                 noisedominated=True, whitenoise=True, Vk=None):
        
        if not noisedominated or not whitenoise:
            raise NotImplementedError()

        # store some parameters
        self.kmin, self.kmax = kmin, kmax
        self.z, self.nint, self.Vk = z, nint, Vk
        self.Vsurv, self.cosmo = Vsurv, cosmo
        self.noisedominated, self.whitenoise = noisedominated, whitenoise

        # construct linear matter power spectrum
        self.klist, self.Pklist = self._gen_lin_ps_(self.z, self.kmin, self.kmax, 
                                                    self.nint, self.cosmo)

        # check Blist, Nlist, make sure lengths match
        self.Blist, self.Nlist, self.nparam = self._check_BN_param_(Blist, Nlist)

        self.fmat = self._gen_fmat_(self.Blist, self.Nlist, self.klist, self.Pklist, 
                                    self.Vsurv, self.nparam, self.nint,
                                    self.noisedominated, self.whitenoise, self.Vk)

    def _gen_lin_ps_(self, z, kmin, kmax, nint, cosmo):
        klist = np.logspace(np.log10(kmin), np.log10(kmax), nint)
        Pklist = cosmo.matterPowerSpectrum(klist/cosmo.h, z)
        Pklist /= cosmo.h**3

        return klist, Pklist

    def _check_BN_param_(self, Blist, Nlist):
        # check Blist, Nlist, make sure lengths match
        Blist = np.asarray(Blist)
        Nlist = np.asarray(Nlist)
        assert len(Blist) == len(Nlist), "Bias and noise lists are different length!"
                                                    
        nparam = len(Blist)
        return Blist, Nlist, nparam

    def _compute_Vk_noisedom_whitenoise_(self, klist, Pklist, Vsurv):
        integrand = np.divide(np.square(klist*Pklist), 2.*np.pi**2)
        Vk = np.trapz(integrand, klist)
        return Vsurv*Vk

    def _fmat_offdiag_noisedom_whitenoise_(self, i, j, Blist, Nlist):
        return (Blist[i]*Blist[j])/(Nlist[i]*Nlist[j])

    def _fmat_diag_noisedom_whitenoise_(self, i, Blist, Nlist):
        tosum = np.square(Blist)/(Nlist[i]*Nlist)
        tosum[i] = 0
        return np.sum(tosum)

    def _gen_fmat_(self, Blist, Nlist, klist, Pklist, Vsurv, nparam, nint,
                   noisedominated, whitenoise, Vk=None):
        if noisedominated and whitenoise:
            fisher = np.zeros((nparam, nparam))
            if Vk is None:
                Vk = self._compute_Vk_noisedom_whitenoise_(klist, Pklist, Vsurv)
            for i in range(nparam):
                for j in range(nparam):
                    if i==j:
                        fisher[i][j] = self._fmat_diag_noisedom_whitenoise_(i, Blist, Nlist)
                    else:
                        fisher[i][j] = self._fmat_offdiag_noisedom_whitenoise_(i, j, Blist, Nlist)
            return Vk*fisher

def _gaussian_(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(np.sqrt(2.*np.pi*np.square(sig)))

def corner_plot(z, Blist, Nlist, cosmo, fac=1, tf=None, kmax=None, Vk=None):
    if tf is not None:
        pass
    elif kmax is not None and Vk is not None:
        tf = threefield(z, Blist, Nlist, kmax, 1, cosmo, Vk=Vk)
    else:
        raise Exception('tf is required or kmax and Vk are required')

    cov_mat = np.linalg.inv(tf.fmat)

    B1 = Blist[0]
    B2 = Blist[1]
    B3 = Blist[2]
    N1 = Nlist[0]
    N2 = Nlist[1]
    N3 = Nlist[2]

    dB1 = B1*fac
    dB2 = B2*fac
    dB3 = B3*fac
    b1list = np.linspace(B1-dB1, B1+dB1, 1000)
    b2list = np.linspace(B2-dB2, B2+dB2, 1000)
    b3list = np.linspace(B3-dB3, B3+dB3, 1000)

    b12_1, b12_2 = np.meshgrid(b1list, b2list)
    b23_2, b23_3 = np.meshgrid(b2list, b3list)
    b31_3, b31_1 = np.meshgrid(b3list, b1list)

    rho12 = cov_mat[0][1]/np.sqrt(cov_mat[0][0]*cov_mat[1][1])
    chisq_12 = (b12_1-B1)**2/cov_mat[0][0] + (b12_2-B2)**2/cov_mat[1][1] - 2*rho12*(b12_1-B1)*(b12_2-B2)/np.sqrt(cov_mat[0][0]*cov_mat[1][1])
    chisq_12 /= 1 - rho12**2

    rho23 = cov_mat[1][2]/np.sqrt(cov_mat[1][1]*cov_mat[2][2])
    chisq_23 = (b23_2-B2)**2/cov_mat[1][1] + (b23_3-B3)**2/cov_mat[2][2] - 2*rho23*(b23_2-B2)*(b23_3-B3)/np.sqrt(cov_mat[1][1]*cov_mat[2][2])
    chisq_23 /= 1 - rho23**2

    rho31 = cov_mat[2][0]/np.sqrt(cov_mat[2][2]*cov_mat[0][0])
    chisq_31 = (b31_3-B3)**2/cov_mat[2][2] + (b31_1-B1)**2/cov_mat[0][0] - 2*rho31*(b31_3-B3)*(b31_1-B1)/np.sqrt(cov_mat[2][2]*cov_mat[0][0])
    chisq_31 /= 1 - rho31**2

    p1sigma = 0.68269   # probability inside 1 sigma
    p2sigma = 0.95449   # probability inside 2 sigma
    df = len(Blist)

    delta1chisq = chi2.isf(1.-p1sigma, df=df)
    delta2chisq = chi2.isf(1.-p2sigma, df=df)
    contour_list_12 = [delta1chisq+np.min(chisq_12), delta2chisq+np.min(chisq_12)]
    contour_list_23 = [delta1chisq+np.min(chisq_23), delta2chisq+np.min(chisq_23)]
    contour_list_31 = [delta1chisq+np.min(chisq_31), delta2chisq+np.min(chisq_31)]

    fig = plt.figure(figsize=(5,6))
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 5)
    ax3 = plt.subplot(3, 3, 9)

    pb1list = _gaussian_(b1list, B1, np.sqrt(cov_mat[0][0]))
    pb2list = _gaussian_(b2list, B2, np.sqrt(cov_mat[1][1]))
    pb3list = _gaussian_(b3list, B3, np.sqrt(cov_mat[2][2]))

    ax1.plot(b1list, pb1list, c=tb_c[0])
    ax2.plot(b2list, pb2list, c=tb_c[0])
    ax3.plot(b3list, pb3list, c=tb_c[0])

    ax12 = plt.subplot(3, 3, 4, sharex=ax1)
    ax23 = plt.subplot(3, 3, 8, sharex=ax2)
    ax31 = plt.subplot(3, 3, 7, sharex=ax1, sharey=ax23)

    for bl,x in zip([b1list, b2list, b3list], [ax1, ax2, ax3]):
        x.set_xlim([np.min(bl), np.max(bl)])
        x.set_yticklabels([])

    B1str = "{:.2E}".format(Blist[0])
    B2str = "{:.2E}".format(Blist[1])
    B3str = "{:.2E}".format(Blist[2])
    N1str = "{:.2E}".format(N1)
    N2str = "{:.2E}".format(N2)
    N3str = "{:.2E}".format(N3)

    ax1.text(0.1, 1.4, r'$B_1^2='+B1str+r'$' , transform=ax1.transAxes)
    ax1.text(0.1, 1.2, r'$N_1='+N1str+r'$' , transform=ax1.transAxes)
    ax2.text(0.1, 1.4, r'$B_2^2='+B2str+r'$' , transform=ax2.transAxes)
    ax2.text(0.1, 1.2, r'$N_2='+N2str+r'$' , transform=ax2.transAxes)
    ax3.text(0.1, 1.4, r'$B_3^2='+B3str+r'$' , transform=ax3.transAxes)
    ax3.text(0.1, 1.2, r'$N_3='+N3str+r'$' , transform=ax3.transAxes)

    ax31.set_xlim([np.min(b1list), np.max(b1list)])
    ax31.set_ylim([np.min(b3list), np.max(b3list)])
    ax12.set_xlim([np.min(b1list), np.max(b1list)])
    ax12.set_ylim([np.min(b2list), np.max(b2list)])
    ax23.set_xlim([np.min(b2list), np.max(b2list)])
    ax23.set_ylim([np.min(b3list), np.max(b3list)])

    ax12.contour(b12_1, b12_2, chisq_12, levels=contour_list_12, colors=tb_c[0], linestyles=('solid', 'dashed'))
    ax23.contour(b23_2, b23_3, chisq_23, levels=contour_list_23, colors=tb_c[0], linestyles=('solid', 'dashed'))
    ax31.contour(b31_1, b31_3, chisq_31, levels=contour_list_31, colors=tb_c[0], linestyles=('solid', 'dashed'))

    ax3.set_xlabel(r'$B_3$')
    ax12.set_ylabel(r'$B_2$')
    ax31.set_xlabel(r'$B_1$')
    ax31.set_ylabel(r'$B_3$')
    ax23.set_xlabel(r'$B_2$')

    return fig, (ax12, ax23, ax31), chisq_12

if __name__ == '__main__':
    from colossus.cosmology import cosmology
    from CO_data import gen_Blist_Nlist

    params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
    cosmo = cosmology.setCosmology('myCosmo', params)

    # survey = 'CCAT-p'
    # lines = ['6-5', '5-4', '4-3']
    z = 0.88
    kmax = 1
    # bandwidth = 28
    # b = 3

    # Blist, Nlist = gen_Blist_Nlist(3, 28, kmax, z, lines, survey, cosmo)
    Blist = [1, 2, 3]
    Nlist = [1, 2, 3]
    Vsurv = 1

    tf = threefield(z, Blist, Nlist, kmax, Vsurv, cosmo)
