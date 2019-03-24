import numpy as np
from scipy.stats import chi2

class threefield(object):
    def __init__(self, z, Blist, Nlist, kmax, cosmo, kmin=1E-3, nint=1000):
        
        # store some parameters
        self.kmin, self.kmax = kmin, kmax
        self.z, self.nint = z, nint
        self.cosmo = cosmo

        # construct linear matter power spectrum
        self.klist, self.Pklist = self._gen_lin_ps_(self.kmin, self.kmax, 
                                                    self.nint, self.cosmo)

        # check Blist, Nlist, make sure lengths match
        self.Blist, self.Nlist, self.nparam = self._check_BN_param_(Blist, Nlist)

        self.fmat = self._gen_fmat_(self.Blist, self.Nlist, self.klist, self.Pklist, 
                                    self.nparam, self.nint)

        self.explicit_fmat = self._explicit_fmat_(self.Blist, self.Nlist, self.klist, 
                                                  self.Pklist, self.nparam, self.nint)

    def _gen_lin_ps_(self, kmin, kmax, nint, cosmo):
        klist = np.logspace(np.log10(kmin), np.log10(kmax), nint)
        Pklist = cosmo.matterPowerSpectrum(klist*cosmo.h, z)
        Pklist /= cosmo.h**3

        return klist, Pklist

    def _check_BN_param_(self, Blist, Nlist):
        # check Blist, Nlist, make sure lengths match
        Blist = np.asarray(Blist)
        Nlist = np.asarray(Nlist)
        assert len(Blist) == len(Nlist), "Bias and noise lists are different length!"
                                                    
        nparam = len(Blist)
        return Blist, Nlist, nparam

    def _gen_fmat_(self, Blist, Nlist, klist, Pklist, nparam, nint):
        fmat = np.zeros((nparam, nparam, nint))
        for i in range(nparam):
            for j in range(nparam):
                fmat[i][j] = self._gen_fmat_int_(i, j, Blist, Nlist, klist, Pklist)
        
        fmat = np.trapz(fmat, klist, axis=2)

        return fmat

    def _gen_fmat_int_(self, i, j, Blist, Nlist, klist, Pklist):
        Pbiilist = np.outer(np.square(Blist), Pklist)
        Ptotlist = np.add(Pbiilist, Nlist[:,np.newaxis])

        Pbijlist = np.multiply.outer(np.outer(Blist, Blist), Pklist)

        Pbideltalist = np.outer(Blist, Pklist)

        if i==j:
            term = self._gen_fmat_diag_term_(i, Blist, Ptotlist, Pbijlist, Pbideltalist, Pklist)
        else:
            term = self._gen_fmat_offdiag_term_(i, j, Blist, Ptotlist, Pbijlist, Pbideltalist)

        factor = np.divide(np.square(klist), 2.*np.pi**2)
        factor = np.multiply(factor, np.square(Pklist))
        term = np.multiply(term, factor)

        return term

    def _gen_fmat_diag_term_(self, i, Blist, Ptot, Pbij, Pbidelta, Pklist):
        t1 = np.square(Pbij[i])
        t1 = np.add(t1, np.multiply(Ptot[i], Ptot))
        t1 = np.divide(np.square(Blist)[:,np.newaxis], t1)
        t1[i] = 0
        t1 = np.sum(t1, axis=0)

        t2 = np.multiply(Ptot[i], Pklist)
        t2 = np.add(t2, np.square(Pbidelta[i]))
        t2 = np.divide(1, t2)

        return np.add(t1, t2)

    def _gen_fmat_offdiag_term_(self, i, j, Blist, Ptot, Pbij, Pbidelta):
        t1 = np.multiply(Ptot, Pbij[i][j])
        t1 = np.add(t1, np.multiply(Pbij[i], Pbij[j]))
        t1 = np.divide(np.square(Blist)[:,np.newaxis], t1)
        t1[i] = 0
        t1[j] = 0
        t1 = np.sum(t1, axis=0)

        t2 = np.multiply(Ptot[i], Pbidelta[j])
        t2 = np.add(t2, np.multiply(Pbidelta[i], Pbij[i][j]))
        t2 = np.divide(Blist[i], t2)

        t3 = np.multiply(Ptot[j], Pbidelta[i])
        t3 = np.add(t3, np.multiply(Pbidelta[j], Pbij[j][i]))
        t3 = np.divide(Blist[j], t3)

        t4 = np.square(Pbij[i][j])
        t4 = np.add(t4, np.multiply(Ptot[i], Ptot[j]))
        t4 = np.divide(np.multiply(Blist[i], Blist[j]), t4)

        return np.add(np.add(np.add(t1, t2), t3), t4)

    def _explicit_fmat_(self, Blist, Nlist, klist, Pklist, nparam, nint):
        fmat = np.zeros((nparam, nparam))
        for i in range(nparam):
            for j in range(nparam):
                fmat[i][j] = self._explicit_fmat_term_(i, j, Blist, Nlist, klist, Pklist, nparam, nint)
        return fmat

    def _explicit_fmat_term_(self, i, j, Blist, Nlist, klist, Pklist, nparam, nint):
        Pbiilist = np.outer(np.square(Blist), Pklist)
        Ptot = np.add(Pbiilist, Nlist[:,np.newaxis])

        tot = 0
        for l in range(nparam):
            for lp in range(nparam):
                if l <= lp:
                    continue
                if l==i:
                    dPl = Blist[lp]*Pklist
                elif lp==i:
                    dPl = Blist[l]*Pklist
                else:
                    dPl = 0

                for m in range(nparam):
                    for mp in range(nparam):
                        if m <= mp:
                            continue
                        if m==j:
                            dPm = Blist[mp]*Pklist
                        elif mp==j:
                            dPm = Blist[m]*Pklist
                        else:
                            dPm = 0

                        if l!=m and l!=mp and lp!=m and lp!=mp:
                            continue
                        elif l==m and lp==mp or l==mp and lp==m:
                            cov = (Blist[i]*Blist[j]*Pklist)**2 + Ptot[i]*Ptot[j]
                        elif l==m and lp!=mp:
                            cov = (Ptot[l]*Blist[lp]*Blist[mp]*Pklist) + Blist[l]*Blist[lp]*Blist[m]*Blist[mp]*Pklist**2
                        elif l==mp and lp!=m:
                            cov = (Ptot[l]*Blist[lp]*Blist[m]*Pklist) + Blist[l]*Blist[lp]*Blist[m]*Blist[mp]*Pklist**2
                        elif lp==m and l!=mp:
                            cov = (Ptot[lp]*Blist[l]*Blist[mp]*Pklist) + Blist[l]*Blist[lp]*Blist[m]*Blist[mp]*Pklist**2
                        elif lp==mp and l!=m:
                            cov = (Ptot[lp]*Blist[l]*Blist[m]*Pklist) + Blist[l]*Blist[lp]*Blist[m]*Blist[mp]*Pklist**2
                        else:
                            continue

                        tot += dPl*dPm/cov

        tot *= np.square(klist)/(2.*np.pi**2)

        return np.trapz(tot, klist)



            
            # off-diagonal elements
            inext2 = np.mod(i+2, nparam)
            cov[i][inext] = (Blist[inext]**2*Pklist + Nlist[inext])*\
                            (Blist[i]*Blist[inext2]*Pklist)
            cov[i][inext] += Blist[i]*Blist[inext]**2*Blist[inext2]*\
                             Pklist**2
            cov[inext][i] = cov[i][inext].copy()

        return cov

def gen_Vk(kmax, Vsurv):
    ans = kmax**3/(6.*np.pi**2)
    return Vsurv*ans

def diagonal_component(B1, B2, B3, N1, N2, N3, Vk):
    B1tot = np.sqrt(B1**2 + N1**2)
    B2tot = np.sqrt(B2**2 + N2**2)
    B3tot = np.sqrt(B3**2 + N3**2)

    ans = B2**2/( (B1*B2)**2 + B1tot**2 * B2tot**2 )
    ans += B3**2/( (B1*B3)**2 + B1tot**2 * B3tot**2 )
    ans += 2*B2*B3/( B1tot**2*B2*B3 + B1**2*B2*B3 )

    return Vk*ans

def offdiagonal_component(B1, B2, B3, N1, N2, N3, Vk):
    B1tot = np.sqrt(B1**2 + N1**2)
    B2tot = np.sqrt(B2**2 + N2**2)
    B3tot = np.sqrt(B3**2 + N3**2)

    ans = B1*B2/( (B1*B2)**2 + B1tot**2*B2tot**2 )
    ans += B1*B3/( B1tot**2*B2*B3 + B1*(B1*B2*B3) )
    ans += B2*B3/( B2tot**2*B3*B1 + B2*(B1*B2*B3) )
    ans += B3*B3/( B3tot**2*B1*B2 + B3*(B1*B2*B3) )

    return Vk*ans

def gen_fisher_matrix(B1, B2, B3, N1, N2, N3, kmax, Vsurv):
    Vk = gen_Vk(kmax, Vsurv)

    F11 = diagonal_component(B1, B2, B3, N1, N2, N3, Vk)
    F22 = diagonal_component(B2, B3, B1, N2, N3, N1, Vk)
    F33 = diagonal_component(B3, B1, B2, N3, N1, N2, Vk)

    F12 = offdiagonal_component(B1, B2, B3, N1, N2, N3, Vk)
    F23 = offdiagonal_component(B2, B3, B1, N2, N3, N1, Vk)
    F31 = offdiagonal_component(B3, B1, B2, N3, N1, N2, Vk)

    F21 = F12
    F32 = F23
    F13 = F31

    mat = Vk * np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

    return mat

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

    tf = threefield(z, Blist, Nlist, kmax, cosmo)
