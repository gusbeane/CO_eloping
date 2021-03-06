import numpy as np
import itertools

from scipy.special import gamma
import CO_data

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

def fmat_term(z, i, j, ipairs, blist, Ilist, cov, nlines, cosmo):
    term = 0.0
    for l, (l1, l2) in enumerate(ipairs):
        for m, (m1, m2) in enumerate(ipairs):
            c = cov[l][m]
            
            if l1 == i:
                p1 = intensity_cross_power_spectrum(z, blist[l1], blist[l2], Ilist[l1], Ilist[l2],
                                                    cosmo, bderivative=True)
            elif l1 + nlines == i:
                p1 = intensity_cross_power_spectrum(z, blist[l1], blist[l2], Ilist[l1], Ilist[l2],
                                                    cosmo, Iderivative=True)
            elif l2 == i:
                p1 = intensity_cross_power_spectrum(z, blist[l2], blist[l1], Ilist[l2], Ilist[l1],
                                                    cosmo, bderivative=True)
            elif l2 + nlines == i:
                p1 = intensity_cross_power_spectrum(z, blist[l2], blist[l1], Ilist[l2], Ilist[l1],
                                                    cosmo, Iderivative=True)
            else:
                p1 = 0

            if m1 == j:
                p2 = intensity_cross_power_spectrum(z, blist[m1], blist[m2], Ilist[m1], Ilist[m2],
                                                    cosmo, bderivative=True)
            elif m1 + nlines == j:
                p2 = intensity_cross_power_spectrum(z, blist[m1], blist[m2], Ilist[m1], Ilist[m2],
                                                    cosmo, Iderivative=True)
            elif m2 == j:
                p2 = intensity_cross_power_spectrum(z, blist[m2], blist[m1], Ilist[m2], Ilist[m1],
                                                    cosmo, bderivative=True)
            elif m2 + nlines == j:
                p2 = intensity_cross_power_spectrum(z, blist[m2], blist[m1], Ilist[m2], Ilist[m1],
                                                    cosmo, Iderivative=True)
            else:
                p2 = 0

            integrand = np.multiply(np.multiply(p1, c), p2)
            integrand = np.multiply(integrand, np.square(k))
            print(l, l1, l2, m, m1, m2, integrand[0][0])
            term = np.add(term, integrand)
    term = np.trapz(term, mu[0,:], axis=1)
    term = np.trapz(term, k[:,0], axis=0)
    return term

def fisher_multifield(z, blist, Ilist, Vsurv, cosmo, Nfunclist=None, 
                      kmin=1E-3, kmax=1, nk=256, nmu=256):

    assert len(blist) == len(Ilist), "blist and Ilist must be same length"
    nlines = len(blist)

    k, mu, ipairs, cov = covariance(z, blist, Ilist, cosmo, Nfunclist=Nfunclist, 
                     kmin=kmin, kmax=kmax, nk=nk, nmu=nmu, returnk_and_pairs=True)
    
    nderiv = 2*nlines
    fmat = np.zeros((nderiv, nderiv))
    for i in range(nderiv):
        for j in range(nderiv):
            fmat[i][j] = fmat_term(z, i, j, ipairs, blist, Ilist, cov, nlines, cosmo)

    fmat *= Vsurv/(2.*np.pi)**2
    return fmat
                    
def convert_fisher(fmat, blist, Ilist):
    nlines = len(blist)
    nderiv = 2 * nlines
    conv_mat = np.zeros((nderiv, nlines))
    for i in range(nderiv):
        for j in range(nlines):
            if i==j:
                conv_mat[i][j] = 1/Ilist[j]
            elif i==j+nlines:
                conv_mat[i][j] = 1/blist[j]
            else:
                conv_mat[i][j] = 0
    fmatp = np.matmul(np.matmul(np.transpose(conv_mat), fmat), conv_mat)
    return fmatp

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

def sigmap2(z, cosmo, kmin=1E-4, kmax=1E4, nk=1000):
    klist = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    Pklist = cosmo.matterPowerSpectrum(klist, z)
    
    Pint = np.trapz(Pklist, klist)

    f = fomega(z, cosmo)
    sigmav2 = f**2 * Pint / (3. * 2. * np.pi**2)
    sigmap2 = sigmav2/2

    return sigmap2

def kaiser(b, mu, z, cosmo):
    beta_z = fomega(z, cosmo)/b
    kai = np.add(1., np.multiply(beta_z, np.square(mu)))
    return kai

def intensity_cross_power_spectrum(z, b1, b2, I1, I2, cosmo, kmin=1E-3, kmax=1, nk=256, nmu=256,
                             returnk=False, angle_averaged=False,
                             Iderivative=False, bderivative=False):
    # note that b derivative is always taken wrt b1, and same with I derivative

    assert not (Iderivative and bderivative), "I and b second derivative not supported"

    def _angle_average_ps_(k, mu, Pkmu):
        klist = k[:,0]

        Pi = np.trapz(Pkmu, mu, axis=1)
        Pi = np.divide(Pi, 2.)

        return klist, Pi
    
    klist = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    mulist = np.linspace(-1, 1, nmu)

    # generate distorted k, mu - if applicable
    # otherwise kdist, mudist = k, mu
    k, mu = gen_k_meshgrid(klist, mulist)

    Pden = cosmo.matterPowerSpectrum(k, z)

    B1 = b1 * I1
    B2 = b2 * I2

    # compute prefactor from kaiser effect
    kai1 = kaiser(b1, mu, z, cosmo)
    kai2 = kaiser(b2, mu, z, cosmo)

    # compute prefactor from fingerofgod effect
    sp2 = sigmap2(z, cosmo)
    x2 = sp2 * (k*cosmo.h)**2 * mu**2
    fingerofgod = 1./(1. + x2)

    prefactor = np.multiply(B1, B2)
    prefactor = np.multiply(prefactor, np.multiply(kai1, kai2))
    prefactor = np.multiply(prefactor, fingerofgod)
    
    if (not bderivative) and (not Iderivative):
        Pintensity = np.multiply(prefactor, Pden)
    elif bderivative:
        prefactor1 = np.divide(prefactor, b1)
        prefactor2 = np.divide(prefactor, kai1)
        prefactor2 = np.multiply(prefactor, (1.-np.square(kai1))/b1)
        Pintensity = np.multiply(prefactor1, Pden) + np.multiply(prefactor2, Pden)
    elif Iderivative:
        Pintensity = np.multiply(prefactor/I1, Pden)

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

def intensity_power_spectrum(z, b, I, cosmo, kmin=1E-3, kmax=1, nk=256, nmu=256,
                             distort=False, ztarget=None, returnk=False, angle_averaged=False,
                             Iderivative=False, bderivative=False):
    
    assert not (Iderivative and bderivative), "I and b second derivative not supported"

    def _angle_average_ps_(k, mu, Pkmu):
        klist = k[:,0]

        Pi = np.trapz(Pkmu, mu, axis=1)
        Pi = np.divide(Pi, 2.)

        return klist, Pi
    
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
    kaiser1 = np.add(1., np.multiply(beta_z, np.square(mudist)))
    kaiser = np.square(kaiser1)

    if bderivative:
        kd = np.multiply(2., kaiser1)
        kaiser_derivative = np.multiply(kd, np.multiply(-beta_z/b, np.square(mudist)))

    # compute prefactor from fingerofgod effect
    sp2 = sigmap2(z, cosmo)
    x2 = sp2 * (kdist*cosmo.h)**2 * mu**2
    fingerofgod = 1./(1. + x2)

    # compute shot noise
    # TODO: implement in some way, probably in CO_data
    # SFR, phi, alpha = CO_data._find_nearest_smit_(z, CO_data.smit_unlog_table)
    # shot = I**2 * (2. + alpha) / (phi * gamma(2.+alpha))
    # shot *= CO_data.smit_h3
    shot = 0

    if bderivative:
        Pintensity1 = np.multiply(np.multiply(np.multiply(B**2, kaiser_derivative), fingerofgod), Pden)
        Pintensity2 = np.multiply(np.multiply(np.multiply(2*B**2/b, kaiser), fingerofgod), Pden)
        Pintensity = np.add(Pintensity1, Pintensity2)
    else:
        # put all the pieces together
        Pintensity = np.multiply(np.multiply(np.multiply(B**2, kaiser), fingerofgod), Pden)    
        Pintensity = np.add(Pintensity, shot)

    if Iderivative:
        # this assumes that shot propto I^2, which I think is true generally
        Pintensity = np.divide(Pintensity, I)
        Pintensity = np.multiply(Pintensity, 2)

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

def covariance(z, blist, Ilist, cosmo, Nfunclist=None, kmin=1E-3, kmax=1, nk=256, nmu=256,
                returnk_and_pairs=False, angle_averaged=False):

    assert len(blist) == len(Ilist), "blist, and Ilist must be the same length"
    
    nparam = len(blist)
    ilist = np.array(list(range(nparam)))
    c = itertools.combinations(ilist, 2)
    ipairs = np.array(list(c))
    npairs = len(ipairs)

    # just to get k, mu... TODO: pull this out of PS func later
    if angle_averaged:
        k, _ = intensity_power_spectrum(z, blist[0], Ilist[0], cosmo, kmin=kmin, kmax=kmax,
                                        nk=nk, nmu=nmu, returnk=True, angle_averaged=True)
        xPSlist = np.zeros((nparam, nparam, nk))
        cov = np.zeros((npairs, npairs, nk))


    else:   
        k, mu, _ = intensity_power_spectrum(z, blist[0], Ilist[0], cosmo, kmin=kmin, kmax=kmax,
                                            nk=nk, nmu=nmu, returnk=True)
        xPSlist = np.zeros((nparam, nparam, nk, nmu))
        cov = np.zeros((npairs, npairs, nk, nmu))


    PSlist = np.array([intensity_power_spectrum(z, b, I, cosmo, kmin=kmin, kmax=kmax, 
                                             nk=nk, nmu=nmu, returnk=False, angle_averaged=angle_averaged) 
                                                for b,I in zip(blist, Ilist)])

    for i in range(nparam):
        for j in range(nparam):
            xPSlist[i][j] = intensity_cross_power_spectrum(z, blist[i], blist[j], Ilist[i], Ilist[j], 
                                                           cosmo, kmin=kmin, kmax=kmax, nk=nk, nmu=nmu,
                                                           returnk=False, angle_averaged=angle_averaged)

    if Nfunclist is not None:
        if angle_averaged:
            Nlist = np.array([ Nfunc(k) for Nfunc in Nfunclist ])
        else:
            Nlist = np.array([ Nfunc(k, mu) for Nfunc in Nfunclist ])
        Nlist = np.sum(Nlist, axis=0)
    else:
        Nlist = np.zeros(np.shape(k))

    PStotlist = np.add(PSlist, Nlist)

    for l, (l1, l2) in enumerate(ipairs):
        for m, (m1, m2) in enumerate(ipairs):
            if l == m:
                cov[l][m] = np.square(xPSlist[l1][l2])
                cov[l][m] = np.add(cov[l][m], np.multiply(PStotlist[l1], PStotlist[l2]))
            else:
                if l1 == m1 and l2 != m2:
                    cov[l][m] = np.multiply(PStotlist[l1], xPSlist[l2][m2])
                    cov[l][m] = np.add(cov[l][m], np.multiply(xPSlist[l1][l2], xPSlist[l1][m2]))
                elif l1 == m2 and l2 != m1:
                    cov[l][m] = np.multiply(PStotlist[l1], xPSlist[l2][m1])
                    cov[l][m] = np.add(cov[l][m], np.multiply(xPSlist[l1][l2], xPSlist[l1][m1]))
                elif l2 == m1 and l1 != m2:
                    cov[l][m] = np.multiply(PStotlist[l2], xPSlist[l1][m2])
                    cov[l][m] = np.add(cov[l][m], np.multiply(xPSlist[l2][l1], xPSlist[l2][m2]))
                elif l2 == m2 and l1 != m1:
                    cov[l][m] = np.multiply(PStotlist[l2], xPSlist[l1][m1])
                    cov[l][m] = np.add(cov[l][m], np.multiply(xPSlist[l2][l1], xPSlist[l2][m1]))
                else:
                    cov[l][m] = 0.0

    if returnk_and_pairs:
        if angle_averaged:
            return k, ipairs, cov
        else:
            return k, mu, ipairs, cov
    else:
        return cov

class constant_N_mu(object):
    def __init__(self, N):
        self.N = N
    def __call__(self, k, mu):
        return np.full(np.shape(k), self.N)

class constant_N(object):
    def __init__(self, N):
        self.N = N
    def __call__(self, k):
        return np.full(np.shape(k), self.N)

if __name__ == '__main__':
    cosmo = CO_data.LT16_cosmo

    N_const1 = constant_N_mu(10)
    N_const2 = constant_N_mu(20)
    Nlist = [N_const1]

    k, mu, ipairs, cov = covariance(2, np.array([1, 2, 3, 4]), np.array([100, 100, 100, 100]), cosmo, 
                     Nfunclist=Nlist, returnk_and_pairs=True)

    # reproduce the ellipses from the paper
    Vsurv = 508
    blist = 100*np.array([1, np.sqrt(2), np.sqrt(3)])
    Ilist = np.array([1, 1, 1])/100
    Nlist = [constant_N_mu(2E5)]
    k, mu, ipairs, cov = covariance(0.88, blist, Ilist, cosmo, Nfunclist=Nlist, returnk_and_pairs=True)
    fmat = fisher_multifield(0.88, blist, Ilist, Vsurv, cosmo, Nfunclist=Nlist)
    fmatp = convert_fisher(fmat, blist, Ilist)
