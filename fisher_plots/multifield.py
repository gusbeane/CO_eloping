import numpy as np
import itertools

class multifield(object):
    def __init__(self, z, Blist, Nlist, kmax, Vsurv, cosmo, kmin=1E-3, nint=1000,
                 whitenoise=True, Vk=None):
        
        if not whitenoise:
            raise NotImplementedError()

        # store some parameters
        self.kmin, self.kmax = kmin, kmax
        self.z, self.nint, self.Vk = z, nint, Vk
        self.Vsurv, self.cosmo = Vsurv, cosmo

        # construct linear matter power spectrum
        self.klist, self.Pklist = self._gen_lin_ps_(self.z, self.kmin, self.kmax, 
                                                    self.nint, self.cosmo)

        # check Blist, Nlist, make sure lengths match
        self.Blist, self.Nlist, self.nparam = self._check_BN_param_(Blist, Nlist)

        self.pairlist, self.npair = self._gen_pairlist_(self.nparam)

        self.dPldBi = self._gen_dPdB_(self.Blist, self.Pklist, self.pairlist, self.npair, self.nparam, self.nint)

        self.cov, self.invcov = self._gen_cov_(self.Blist, self.Nlist, self.Pklist,
                                               self.pairlist, self.nparam, self.npair, self.nint)

        self.fmat = self._gen_fmat_(self.dPijdBk, self.varij, self.covijk, self.klist,
                                    self.Vsurv, self.nparam, self.nint)

    def _gen_lin_ps_(self, z, kmin, kmax, nint, cosmo):
        # generate linear power spectrum from colossus
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


    def _integrate_fmat_(self, klist, Vsurv, fmat, nparam, nint):
        k = np.reshape(klist, (1, 1, nint))
        k = np.repeat(k, nparam, axis=0)
        k = np.repeat(k, nparam, axis=1)

        factor = np.square(k) * Vsurv / (2. * np.pi**2)

        return np.trapz(np.multiply(factor, fmat), klist, axis=2)


if __name__ == '__main__':
    from colossus.cosmology import cosmology

    params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
    cosmo = cosmology.setCosmology('myCosmo', params)

    z = 0.88
    kmax = 1

    Blist = [1, 2, 3]
    Nlist = [1, 2, 3]
    Vsurv = 1

    mf = multifield(z, Blist, Nlist, kmax, Vsurv, cosmo)
