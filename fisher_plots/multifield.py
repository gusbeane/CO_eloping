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

        cov = np.zeros((npair, npair, nint))

        print(np.shape(cov))

        for i,l in enumerate(pairlist):
            for j,m in enumerate(pairlist):
                # check for diagonal term
                t = np.append(l, m)
                t2, counts = np.unique(t, return_counts=True)
                k = np.flip( np.argsort(counts) )

                if len(k)==4:
                    cov[i][j] = 0
                elif len(k)==3:
                    cov[i][j] = covijk[t2[k][0], t2[k][1], t2[k][2]]
                elif len(k)==2:
                    cov[i][j] = varij[t2[k][0], t2[k][1]]
                else:
                    raise Exception("Same pair in the list, or auto-spectrum in list")

        # also invert
        invcov = np.zeros(np.shape(cov))
        for i in range(np.shape(cov)[2]):
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
