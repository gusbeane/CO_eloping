import numpy as np
import itertools

from scipy.special import gamma
import CO_data

class multifield(object):
    def __init__(self, z, line_list):

        self._check_assigned_bias_(line_list)
    
    def _check_assigned_bias_(self, line_list):
        for line in line_list:
            assert line.bias_assigned, "Bias not assigned for line "+line.key


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

