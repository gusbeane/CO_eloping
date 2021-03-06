import numpy as np
import itertools

from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.special import gamma
import CO_data

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

class multifield(object):
    def __init__(self, z, Blist, Nlist, kmax, Vsurv, cosmo, kmin=1E-3, nint=1000,
                 whitenoise=True, Vk=None, klist=None):
        
        if not whitenoise:
            raise NotImplementedError()

        # store some parameters
        self.kmin, self.kmax = kmin, kmax
        self.z, self.nint, self.Vk = z, nint, Vk
        self.Vsurv, self.cosmo = Vsurv, cosmo

        # ignore nint if a klist is provided
        if klist is not None:
            self.nint = len(klist)

        # construct linear matter power spectrum
        self.klist, self.Pklist = self._gen_lin_ps_(self.z, self.kmin, self.kmax, 
                                                    self.nint, self.cosmo, klist)

        # check Blist, Nlist, make sure lengths match
        self.Blist, self.Nlist, self.nparam = self._check_BN_param_(Blist, Nlist)

        # generate pairs of indices
        self.pairlist, self.npair = self._gen_pairlist_(self.nparam)

        # construct derivative vector
        self.dPldBi = self._gen_dPdB_(self.Blist, self.Pklist, self.pairlist, self.npair, self.nparam, self.nint)

        # construct cov, invcov matrices
        self.cov, self.invcov = self._gen_cov_(self.Blist, self.Nlist, self.Pklist,
                                               self.pairlist, self.nparam, self.npair, self.nint)

        # construct fisher matrix
        self.fmat = self._gen_fmat_(self.dPldBi, self.invcov, self.klist,
                                    self.Vsurv, self.npair, self.nparam, self.nint)

    def _gen_lin_ps_(self, z, kmin, kmax, nint, cosmo, klist=None):
        """Generates a the linear power spectrum at redshift z.

        Uses the input colossus cosmology to generate the linear matter
        power spectrum at redshift z. Note that kmin and kmax are ignored if
        klist is provided, but are still required arguments. If klist is not
        given, then a klist will be generated with logarithmic spacing from kmin
        to kmax, with a length of nint.
    
        Args:
            z (float): The redshift at which to compute the power spectrum.
            kmin (float): The minimum k at which to compute the power spectrum.
                Ignored if klist is given.
            kmax (float): The maximum k at which to compute the power spectrum.
                Ignored if klist is given.
            nint (int): The number of spacings for the klist. Ignored if klist is
                given.
            cosmo: A colossus cosmology object. Used to compute the matter power
                spectrum.
            klist (array, optional): A list of k values at which to compute the
                matter power spectrum. Recommended to be in ascending order. Overrides
                the kmin, kmax, and nint arguments.
    
        Returns:
            klist (array): The k values [h/Mpc] at which the matter power spectrum is 
                computed.
            Pklist (array): The power spectrum values [(Mpc/h)^3].
        """
        if klist is None:
            klist = np.logspace(np.log10(kmin), np.log10(kmax), nint)

        Pklist = cosmo.matterPowerSpectrum(klist, z)

        return klist, Pklist

    def _check_BN_param_(self, Blist, Nlist):
        # check Blist, Nlist, make sure lengths match
        Blist = np.asarray(Blist)
        Nlist = np.asarray(Nlist)
        assert len(Blist) == len(Nlist), "Bias and noise lists are different length!"
                                                    
        nparam = len(Blist)
        return Blist, Nlist, nparam

    def _gen_pairlist_(self, nparam):
        l = list(range(nparam))
        c = itertools.combinations(l, 2)
        pairlist = list(c)
        npair = len(pairlist)
        return pairlist, npair

    def _gen_dPdB_(self, Blist, Pklist, pairlist, npair, nparam, nint):
        dPldBi = np.zeros((nparam, npair, nint))
        for i, (p1, p2) in enumerate(pairlist):
            for j in range(nparam):
                if j == p1:
                    dPldBi[j][i] = Blist[p2] * Pklist
                elif j == p2:
                    dPldBi[j][i] = Blist[p1] * Pklist
                else:
                    dPldBi[j][i] = 0
        return dPldBi

    def _gen_cov_(self, Blist, Nlist, Pklist, pairlist, nparam, npair, nint):
        cov = np.zeros((npair, npair, nint))

        for i, (l1, l2) in enumerate(pairlist):
            for j, (m1, m2) in enumerate(pairlist):
                if (l1 == m1 and l2 == m2) or (l1 == m2 and l2 == m1):
                    cov[i][j] = (Blist[l1]*Blist[l2]*Pklist)**2
                    cov[i][j] += (Blist[l1]**2*Pklist + Nlist[l1]) * (Blist[l2]**2*Pklist + Nlist[l2])
                elif l1 == m1 and l2 != m2:
                    cov[i][j] = (Blist[l1]**2*Pklist + Nlist[l1])*Blist[l2]*Blist[m2]*Pklist
                    cov[i][j] += Blist[l1]**2 * Blist[l2]*Blist[m2] * Pklist**2
                elif l1 == m2 and l2 != m1:
                    cov[i][j] = (Blist[l1]**2*Pklist + Nlist[l1])*Blist[l2]*Blist[m1]*Pklist
                    cov[i][j] += Blist[l1]**2 * Blist[l2]*Blist[m1] * Pklist**2
                elif l2 == m1 and l1 != m2:
                    cov[i][j] = (Blist[l2]**2*Pklist + Nlist[l2])*Blist[l1]*Blist[m2]*Pklist
                    cov[i][j] += Blist[l2]**2 * Blist[l1]*Blist[m2] * Pklist**2
                elif l2 == m2 and l1 != m1:
                    cov[i][j] = (Blist[l2]**2*Pklist + Nlist[l2])*Blist[l1]*Blist[m1]*Pklist
                    cov[i][j] += Blist[l2]**2 * Blist[l1]*Blist[m1] * Pklist**2
                else:
                    cov[i][j] = 0.0

        invcov = np.copy(cov)
        for i in range(nint):
            invcov[:,:,i] = np.linalg.inv(cov[:,:,i])

        return cov, invcov

    def _gen_fmat_(self, dPldBi, invcov, klist, Vsurv, npair, nparam, nint):
        t1 = np.transpose(np.swapaxes(dPldBi, 0, 1))
        t2 = np.transpose(invcov)
        t3 = np.transpose(dPldBi)

        fmat = np.matmul(np.matmul(t1, t2), t3)
        fmat = np.transpose(fmat)
        fmat = self._integrate_fmat_(klist, Vsurv, fmat, nparam, nint)

        return fmat

    def _integrate_fmat_(self, klist, Vsurv, fmat, nparam, nint):
        k = np.reshape(klist, (1, 1, nint))
        k = np.repeat(k, nparam, axis=0)
        k = np.repeat(k, nparam, axis=1)

        factor = np.square(k) * Vsurv / (2. * np.pi**2)

        return np.trapz(np.multiply(factor, fmat), klist, axis=2)

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
        Pklist = cosmo.matterPowerSpectrum(klist, z)

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

def corner_plot(z, Blist, Nlist, cosmo, fac=1, tf=None, kmax=None, Vk=None, norm=False, labels=None, intstr=False, 
                printtext=True):
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
    b1list = np.linspace(B1-dB1, B1+dB1, 5000)
    b2list = np.linspace(B2-dB2, B2+dB2, 5000)
    b3list = np.linspace(B3-dB3, B3+dB3, 5000)

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

    fig = plt.figure(figsize=(5,5.2))
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 5)
    ax3 = plt.subplot(3, 3, 9)

    sig1 = np.sqrt(cov_mat[0][0])
    sig2 = np.sqrt(cov_mat[1][1])
    sig3 = np.sqrt(cov_mat[2][2])

    pb1list = _gaussian_(b1list, B1, sig1)
    pb2list = _gaussian_(b2list, B2, sig2)
    pb3list = _gaussian_(b3list, B3, sig3)

    if norm:
        fac1, fac2, fac3 = B1, B2, B3
    else:
        fac1, fac2, fac3 = 1, 1, 1

    ax1.plot(b1list/fac1, pb1list, c=tb_c[0])
    ax2.plot(b2list/fac2, pb2list, c=tb_c[0])
    ax3.plot(b3list/fac3, pb3list, c=tb_c[0])

    ax1.axvline(B1/fac1, c=tb_c[0])
    ax2.axvline(B2/fac2, c=tb_c[0])
    ax3.axvline(B3/fac3, c=tb_c[0])
    ax1.axvline((B1+sig1)/fac1, c=tb_c[0], ls='dashed', alpha=0.5)
    ax2.axvline((B2+sig2)/fac2, c=tb_c[0], ls='dashed', alpha=0.5)
    ax3.axvline((B3+sig3)/fac3, c=tb_c[0], ls='dashed', alpha=0.5)
    ax1.axvline((B1-sig1)/fac1, c=tb_c[0], ls='dashed', alpha=0.5)
    ax2.axvline((B2-sig2)/fac2, c=tb_c[0], ls='dashed', alpha=0.5)
    ax3.axvline((B3-sig3)/fac3, c=tb_c[0], ls='dashed', alpha=0.5)

    ax12 = plt.subplot(3, 3, 4, sharex=ax1)
    ax23 = plt.subplot(3, 3, 8, sharex=ax2)
    ax31 = plt.subplot(3, 3, 7, sharex=ax1, sharey=ax23)

    for x in [ax12, ax23, ax31]:
        x.locator_params('x', nbins=5)

    for bl,f,x in zip([b1list, b2list, b3list], [fac1,fac2,fac3],[ax1, ax2, ax3]):
        x.set_xlim([np.min(bl)/f, np.max(bl)/f])
        x.locator_params('x', nbins=5)
        x.set_yticklabels([])
        x.set_yticks([])
        x.set_ylim(bottom=0)

    if intstr:
        B1str = str(round(Blist[0]))
        B2str = str(round(Blist[1]))
        B3str = str(round(Blist[2]))
        N1str = str(round(Nlist[0]))
        N2str = str(round(Nlist[1]))
        N3str = str(round(Nlist[2]))
    else:
        B1str = "{:.2E}".format(Blist[0])
        B2str = "{:.2E}".format(Blist[1])
        B3str = "{:.2E}".format(Blist[2])
        N1str = "{:.2E}".format(N1)
        N2str = "{:.2E}".format(N2)
        N3str = "{:.2E}".format(N3)

    if printtext:
        ax1.text(0.05, 1.2, r'$B_1^2='+B1str+r'$' , transform=ax1.transAxes, in_layout=True)
        ax1.text(0.05, 1.05, r'$N_1='+N1str+r'$' ,  transform=ax1.transAxes, in_layout=True)
        ax2.text(0.05, 1.2, r'$B_2^2='+B2str+r'$' , transform=ax2.transAxes, in_layout=False)
        ax2.text(0.05, 1.05, r'$N_2='+N2str+r'$' ,  transform=ax2.transAxes, in_layout=False)
        ax3.text(0.05, 1.2, r'$B_3^2='+B3str+r'$' , transform=ax3.transAxes, in_layout=False)
        ax3.text(0.05, 1.05, r'$N_3='+N3str+r'$' ,  transform=ax3.transAxes, in_layout=False)

    ax31.set_xlim([np.min(b1list)/fac1, np.max(b1list)/fac1])
    ax31.set_ylim([np.min(b3list)/fac3, np.max(b3list)/fac3])
    ax12.set_xlim([np.min(b1list)/fac1, np.max(b1list)/fac1])
    ax12.set_ylim([np.min(b2list)/fac2, np.max(b2list)/fac2])
    ax23.set_xlim([np.min(b2list)/fac2, np.max(b2list)/fac2])
    ax23.set_ylim([np.min(b3list)/fac3, np.max(b3list)/fac3])

    ax12.contour(b12_1/fac1, b12_2/fac2, chisq_12, levels=(contour_list_12[0],), colors=tb_c[0], linestyles=('solid',))
    ax23.contour(b23_2/fac2, b23_3/fac3, chisq_23, levels=(contour_list_23[0],), colors=tb_c[0], linestyles=('solid',))
    ax31.contour(b31_1/fac1, b31_3/fac3, chisq_31, levels=(contour_list_31[0],), colors=tb_c[0], linestyles=('solid',))

    ax12.contour(b12_1/fac1, b12_2/fac2, chisq_12, levels=(contour_list_12[1],), colors=tb_c[0], alpha=0.5, linestyles=('solid',))
    ax23.contour(b23_2/fac2, b23_3/fac3, chisq_23, levels=(contour_list_23[1],), colors=tb_c[0], alpha=0.5, linestyles=('solid',))
    ax31.contour(b31_1/fac1, b31_3/fac3, chisq_31, levels=(contour_list_31[1],), colors=tb_c[0], alpha=0.5, linestyles=('solid',))
    
    if norm:
        ax3.set_xlabel(r'$B_3/B_{3,\text{true}}$')
        ax12.set_ylabel(r'$B_2/B_{2,\text{true}}$')
        ax31.set_xlabel(r'$B_1/B_{1,\text{true}}$')
        ax31.set_ylabel(r'$B_3/B_{3,\text{true}}$')
        ax23.set_xlabel(r'$B_2/B_{2,\text{true}}$')
    elif labels is not None:
        ax3.set_xlabel(labels[2])
        ax12.set_ylabel(labels[1])
        ax31.set_xlabel(labels[0])
        ax31.set_ylabel(labels[2])
        ax23.set_xlabel(labels[1])
    else:
        ax3.set_xlabel(r'$B_3$')
        ax12.set_ylabel(r'$B_2$')
        ax31.set_xlabel(r'$B_1$')
        ax31.set_ylabel(r'$B_3$')
        ax23.set_xlabel(r'$B_2$')

    return fig, (ax12, ax23, ax31), chisq_12

def alpha_factors(ztarget, ziloper, cosmo):
    # if zj >=0 and zi >= 0:
    if True:
        alphapar = cosmo.Hz(ztarget)/cosmo.Hz(ziloper)
        alphapar *= (1. + ziloper)/(1. + ztarget)

        # TODO: check that this is the right conversion in non-flat universes
        alphaperp = cosmo.angularDiameterDistance(ziloper)/cosmo.angularDiameterDistance(ztarget)
        alphaperp *= (1. + ziloper)/(1. + ztarget)
    else:
        alphaperp, alphapar = np.nan, np.nan

    return alphapar, alphaperp

def gen_k_meshgrid(klist, mulist, distort=False, apar=None, aperp=None):

    k, mu = np.meshgrid(klist, mulist, indexing='ij')

    if distort:
        assert apar is not None and aperp is not None, "Must specify apar, aperp to distort!"

        # convert from (k, mu) to (kpar, kperp)        
        kpar = np.multiply(k, mu)
        kpar2 = np.square(kpar)
        k2 = np.square(k)
        kperp2 = np.subtract(k2, kpar2)
        kperp = np.sqrt(kperp2)

        # distort in (kpar, kperp) space
        kpar = np.divide(kpar, apar)
        kperp = np.divide(kperp, aperp)

        # convert back to (k, mu) space
        k2 = np.add(np.square(kpar), np.square(kperp))
        k = np.sqrt(k2)
        mu = np.divide(kpar, k)
    
    return k, mu

def fomega(z, cosmo):
    D = cosmo.growthFactor(z)
    dDdz = cosmo.growthFactor(z, derivative=1)

    fomega = np.divide(np.add(1., z), D)
    fomega = np.multiply(fomega, dDdz)

    return np.negative(fomega)

def sigmap2(z, b, cosmo, kmin=1E-4, kmax=1E4, nk=1000):
    klist = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    Pklist = cosmo.matterPowerSpectrum(klist, z)
    
    Pint = np.trapz(Pklist, klist)

    f = fomega(z, cosmo)
    sigmav2 = f**2 * Pint / (3. * 2. * np.pi**2)
    sigmap2 = sigmav2/2

    return sigmap2

def intensity_power_spectrum(z, b, I, cosmo, kmin=1E-3, kmax=1, nk=256, nmu=256,
                             distort=False, ztarget=None, returnk=False, angle_averaged=False):
    klist = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    mulist = np.linspace(-1, 1, nmu)

    if distort:
        assert ztarget is not None, "Must specify ztarget to distort!"
        apar, aperp = alpha_factors(ztarget, z, cosmo)
    else:
        apar, aperp = None, None

    # generate distorted k, mu - if applicable
    # otherwise kdist, mudist = k, mu
    k, mu = gen_k_meshgrid(klist, mulist)
    kdist, mudist = gen_k_meshgrid(klist, mulist, distort=distort, apar=apar, aperp=aperp)

    Pden = cosmo.matterPowerSpectrum(kdist, z)

    B = b * I

    # compute prefactor from kaiser effect
    beta_z = fomega(z, cosmo)/b
    kaiser = np.add(1., np.multiply(beta_z, np.square(mudist)))
    kaiser = np.square(kaiser)

    # compute prefactor from fingerofgod effect
    sp2 = sigmap2(z, b, cosmo)
    x2 = sp2 * (kdist*cosmo.h)**2 * mu**2
    fingerofgod = 1./(1. + x2)

    # compute shot noise
    SFR, phi, alpha = CO_data._find_nearest_smit_(z, CO_data.smit_unlog_table)
    shot = I**2 * (2. + alpha) / (phi * gamma(2.+alpha))
    shot *= CO_data.smit_h3

    # put all the pieces together
    Pintensity = np.multiply(np.multiply(np.multiply(B**2, kaiser), fingerofgod), Pden)
    Pintensity = np.add(Pintensity, shot)

    # add in prefactor if distorted
    if distort:
        Pintensity = np.divide(Pintensity, apar * aperp**2)

    # angle average, if necessary
    if angle_averaged:
        k, Pintensity = _angle_average_ps_(k, mu, Pintensity)

    # return
    if returnk:
        if angle_averaged:
            return k, Pintensity
        else:
            return k, mu, Pintensity
    else:
        return Pintensity

def _angle_average_ps_(k, mu, Pkmu):
    klist = k[:,0]

    Pi = np.trapz(Pkmu, mu, axis=1)
    Pi = np.divide(Pi, 2.)

    return klist, Pi


if __name__ == '__main__':
    from colossus.cosmology import cosmology

    params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
    cosmo = cosmology.setCosmology('myCosmo', params)

    z = 0.88
    kmax = 1

    nint = 100
    klist = np.logspace(-3, 0, nint)

    Blist = [1, 2, 3]
    Vsurv = 1E8

    # do tf with constant N
    Nlist = [1E8, 2E8, 3E8]
    tf = threefield(z, Blist, Nlist, kmax, Vsurv, cosmo)

    # do mf with full N
    Nlist = np.array([np.full(nint, 1E8), np.full(nint, 2E8), np.full(nint, 3E8)])
    mf = multifield(z, Blist, Nlist, kmax, Vsurv, cosmo, klist=klist)