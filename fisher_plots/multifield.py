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

        self.dPijdBk = self._gen_dPdB_(self.Blist, self.Pklist, self.nparam, self.nint)

        self.varij, self.covijk = self._gen_var_cov_(self.Blist, self.Nlist, self.Pklist,
                                                     self.nparam, self.nint)

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
        return list(c), len(list(c))

    def _gen_Pij_(self, Blist, Pklist, nparam, nint):
        Bouter = np.reshape(np.outer(Blist, Blist), (nparam, nparam, 1))
        Pkrepeat = np.reshape(Pklist, (1, 1, nint))
        Pkrepeat = np.repeat(Pkrepeat, nparam, axis=0)
        Pkrepeat = np.repeat(Pkrepeat, nparam, axis=1)
        Pij = np.multiply(Bouter, Pkrepeat)
        return Pij

    def _gen_dPdB_(self, Blist, Pklist, nparam, nint):
        Pij = self._gen_Pij_(Blist, Pklist, nparam, nint)

        PijoverBk = np.divide.outer(Pij, Blist)
        PijoverBk = np.swapaxes(PijoverBk, axis1=2, axis2=3)

        # will fix this ugly for loop later
        for i in range(nparam):
            for j in range(nparam):
                for k in range(nparam):
                    if i != k and j != k:
                        PijoverBk[i][j][k] = 0.0

        return PijoverBk

    def _gen_var_cov_(self, Blist, Nlist, Pklist, nparam, nint):
        # generate Pi
        BPi = np.reshape(Blist, (nparam, 1))
        PkPi = np.reshape(Pklist, (1, nint))
        PkPi = np.repeat(PkPi, nparam, axis=0)
        Pi = np.multiply(np.square(BPi), PkPi)

        # generate Pij, gets its own function because its fancy
        Pij = self._gen_Pij_(Blist, Pklist, nparam, nint)

        # generate Pitot - still only white noise
        NPitot = np.reshape(Nlist, (nparam, 1))
        NPitot = np.repeat(NPitot, nint, axis=1)
        Pitot = np.add(Pi, NPitot)

        # multiply the tots together
        t = np.multiply.outer(Pitot, Pitot)
        PitotPjtot = np.diagonal(t, axis1=1, axis2=3)

        # compute varij
        varij = np.add(np.square(Pij), PitotPjtot)

        # # # # # # # # # # # # # # # # # # # 
        # on to cov
        # # # # # # # # # # # # # # # # # # #

        #generate Pitot*Pjk
        t = np.multiply.outer(Pitot, Pij)
        PitotPjk = np.diagonal(t, axis1=1, axis2=4)

        # generate PijPik
        t = np.multiply.outer(Pij, Pij)
        t2 = np.diagonal(t, axis1=0, axis2=3)
        t3 = np.diagonal(t2, axis1=1, axis2=3)
        PijPik = np.swapaxes(t3, 0, 2)

        covijk = np.add(PitotPjk, PijPik)

        return varij, covijk

    def _check_fmat_indices_(self, i, j, l, lp, m, mp):
        if l != i and lp != i:
            return False
        
        if m != j and mp != j:
            return False

        if l == m or l == mp:
            return True
        elif lp == m or lp == mp:
            return True
        else:
            return False

    def _fmat_term_(self, dPijdBk, varij, covijk, i, j, l, lp, m, mp):
        if l != i and lp != i:
            return 0.0
        
        if m != j and mp != j:
            return 0.0

        if l != m and l != mp:
            return 0.0
        elif lp != mp and l != m:
            return 0.0

        t1 = dPijdBk[l][lp][i]
        t3 = dPijdBk[m][mp][j]

        if l == m and lp == mp:
            t2 = varij[l][lp]
        elif l == mp and lp == m:
            t2 = varij[l][lp]
        else:
            return 0.0
        # if l == m and lp != mp:
        #     t2 = covijk[l][lp][mp]
        # elif l == mp and lp != m:
        #     t2 = covijk[l][lp][m]
        # elif lp == m and l != mp:
        #     t2 = covijk[lp][l][mp]
        # elif lp == mp and l != m:
        #     t2 = covijk[lp][l][m]
        # else:
        #     raise Exception("Something went wrong with the index gymnastics in _fmat_term_")

        return t1*t3/t2

    def _integrate_fmat_(self, klist, Vsurv, fmat, nparam, nint):
        k = np.reshape(klist, (1, 1, nint))
        k = np.repeat(k, nparam, axis=0)
        k = np.repeat(k, nparam, axis=1)

        factor = np.square(k) * Vsurv / (2. * np.pi**2)

        return np.trapz(np.multiply(factor, fmat), klist, axis=2)

    def _gen_fmat_(self, dPijdBk, varij, covijk, klist, Vsurv, nparam, nint):
        fmat = np.zeros((nparam, nparam, nint))

        for i in range(nparam):
            for j in range(nparam):
                for l in range(nparam):
                    for lp in range(nparam):
                        for m in range(nparam):
                            for mp in range(nparam):
                                fmat[i][j] += self._fmat_term_(dPijdBk, varij, covijk, 
                                                               i, j, l, lp, m, mp)

        fmat = self._integrate_fmat_(klist, Vsurv, fmat, nparam, nint)

        return fmat


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
