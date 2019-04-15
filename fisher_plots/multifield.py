import numpy as np
import itertools

class threefield(object):
    def __init__(self, z, Blist, Nlist, kmax, Vsurv, cosmo, kmin=1E-3, nint=1000,
                 noisedominated=True, whitenoise=True, Vk=None):
        
        if not noisedominated or not whitenoise:
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

        self.varij, self.covijk = self._gen_var_cov_(self.Blist, self.Nlist, self.Pklist,
                                                     self.nparam, self.nint)

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

    def _gen_varij_covijk_(Blist, Nlist, Pklist, nparam, nint):
        BPi = np.reshape(Blist, (nparam, 1))
        PkPi = np.reshape(Pklist, (1, nint))
        PkPi = np.repeat(PkPi, nparam, axis=0)
        Pi = np.multiply(np.square(BPi), PkPi)

        Bouter = np.reshape(np.outer(Blist, Blist), (nparam, nparam, 1))
        Pkrepeat = np.reshape(Pklist, (1, 1, nint))
        Pkrepeat = np.repeat(Pkrepeat, nparam, axis=0)
        Pkrepeat = np.repeat(Pkrepeat, nparam, axis=1)
        Pij = np.multiply(Bouter, Pkrepeat)

        NPitot = np.reshape(Nlist, (nparam, 1))
        NPitot = np.repeat(NPitot, nint, axis=1)
        Pitot = np.add(Pi, NPitot)

        t = np.multiply.outer(Pitot, Pitot)
        PitotPjtot = np.diagonal(t, axis1=1, axis2=3)

        varij = np.add(np.square(Pij), PitotPjtot)


        t = np.multiply.outer(Pitot, Pij)
        PitotPjk = np.diagonal(t, axis1=1, axis2=4)

        t = np.multiply.outer(Pij, Pij)
        t2 = np.diagonal(t, axis1=0, axis2=3)
        t3 = np.diagonal(t2, axis1=1, axis2=3)
        PijPik = np.swapaxes(t3, 0, 2)

        covijk = np.add(PitotPjk, PijPik)

        return varij, covijk, np.square(Pij), PitotPjtot, PitotPjk, PijPik


    def _gen_fmat_(self, Blist, Nlist, klist, Pklist, Vsurv)

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
