import numpy as np
from scipy.stats import chi2

class threefield(object):
    def __init__(self, z, Blist, Nlist, kmax, Vsurv, cosmo, kmin=1E-3, nint=1000,
                 noisedominated=True, whitenoise=True):
        
        if not noisedominated or not whitenoise:
            raise NotImplementedError()

        # store some parameters
        self.kmin, self.kmax = kmin, kmax
        self.z, self.nint = z, nint
        self.Vsurv, self.cosmo = Vsurv, cosmo
        self.noisedominated, self.whitenoise = noisedominated, whitenoise

        # construct linear matter power spectrum
        self.klist, self.Pklist = self._gen_lin_ps_(self.z, self.kmin, self.kmax, 
                                                    self.nint, self.cosmo)

        # check Blist, Nlist, make sure lengths match
        self.Blist, self.Nlist, self.nparam = self._check_BN_param_(Blist, Nlist)

        self.fmat = self._gen_fmat_(self.Blist, self.Nlist, self.klist, self.Pklist, 
                                    self.Vsurv, self.nparam, self.nint,
                                    self.noisedominated, self.whitenoise)

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
                   noisedominated, whitenoise):
        if noisedominated and whitenoise:
            fisher = np.zeros((nparam, nparam))
            Vk = self._compute_Vk_noisedom_whitenoise_(klist, Pklist, Vsurv)
            for i in range(nparam):
                for j in range(nparam):
                    if i==j:
                        fisher[i][j] = self._fmat_diag_noisedom_whitenoise_(i, Blist, Nlist)
                    else:
                        fisher[i][j] = self._fmat_offdiag_noisedom_whitenoise_(i, j, Blist, Nlist)
            return Vk*fisher

class chisq(object):
    def __init__(self, B1, B2, B3, cov_mat):
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

        self.cov_mat = cov_mat

        # self.sigma1 = np.sqrt(cov_mat[0][0])
        # self.sigma2 = np.sqrt(cov_mat[1][1])
        # self.sigma3 = np.sqrt(cov_mat[2][2])

        # self.sigma12 = cov_mat[0][1]
        # self.sigma23 = cov_mat[1][2]
        # self.sigma31 = cov_mat[2][0]

        # self.rho12 = self.sigma12/(self.sigma1*self.sigma2)
        # self.rho23 = self.sigma23/(self.sigma2*self.sigma3)
        # self.rho31 = self.sigma31/(self.sigma3*self.sigma1)

    def __call__(self, b1, b2, b3, chisq):
        d1 = b1 - self.B1
        d2 = b2 - self.B2
        d3 = b3 - self.B3
        d = np.array([d1, d2, d3])

        chisq = np.matmul(np.matmul(np.transpose(d), np.linalg.inv(self.cov_mat)), d)
        
        return chisq

def corner_plot(B1, B2, B3, N1, N2, N3, kmax, Vsurv):
    fisher_mat = gen_fisher(B1, B2, B3, N1, N2, N3, kmax, Vsurv)

    cov_mat = np.inverse(fisher_mat)

    b1list = np.linspace(0.7*B1, 1.3*B1, 100)
    b2list = np.linspace(0.7*B2, 1.3*B2, 100)
    b3list = np.linspace(0.7*B3, 1.3*B3, 100)

    b12_1, b12_2 = np.meshgrid(b1list, b2list)
    b23_2, b23_3 = np.meshgrid(b2list, b3list)
    b31_3, b31_1 = np.meshgrid(b3list, b1list)

    chisq_b12 = chisq()

    p1sigma = 0.68269   # probability inside 1 sigma
    p2sigma = 0.95449   # probability inside 2 sigma
    df = 3

    delta1chisq = chi2.isf(1.-p1sigma, df=df)
    delta2chisq = chi2.isf(1.-p2sigma, df=df)
    contour_list = [delta1chisq, delta2chisq]

    fig = plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 5)
    ax3 = plt.subplot(3, 3, 9)

    ax12 = plt.subplot(3, 3, 4, sharex=ax1)
    ax23 = plt.subplot(3, 3, 8, sharex=ax2)
    ax13 = plt.subplot(3, 3, 7, sharex=ax1, sharey=ax23)

    ax12.contour()

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
