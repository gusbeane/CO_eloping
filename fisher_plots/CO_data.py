import numpy as np 
import astropy.units as u
from astropy.constants import c
from scipy.special import gamma

# basic info about the lines

# CO line wavelengths in microns
CO_lines_wave = {'1-0': 2610, '2-1': 1300, '3-2': 866, '4-3': 651,
                 '5-4': 521, '6-5': 434, '7-6': 372, '8-7': 325,
                 '9-8': 289, '10-9': 260, '11-10': 237,
                 '12-11': 217, '13-12': 200}

# Convert CO wavelengths to GHz
speed_of_light = c.to_value(u.micron*u.GHz)

CO_lines = { key: speed_of_light/wave for key, wave in CO_lines_wave.items() }

# ---------------------------------------------------------------- #
#                                                                  #
#                         line modelling                           #
#                                                                  #
# ---------------------------------------------------------------- #

# CO luminosity constant in Lsolar
CO_L0 = {'1-0': 3.7E3, '2-1': 2.8E4, '3-2': 7E4, '4-3': 9.7E4,
         '5-4': 9.6E4, '6-5': 9.5E4, '7-6': 8.9E4, '8-7': 7.7E4,
         '9-8': 6.9E4, '10-9': 5.3E4, '11-10': 3.8E4,
         '12-11': 2.6E4, '13-12': 1.4E4}

def _find_nearest_(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def epsilon_l(L0, z, smit_table, log=False):
    key = _find_nearest_(smit_table[:,0], z)
    smit = smit_table[key]
    
    SFR = smit[1]
    phi = smit[2]
    alpha = smit[3]

    if log:
        SFR = 10.**SFR
        phi = 10.**phi

    eps = phi * L0 * SFR * gamma(2.+alpha)
    return eps # Lsolar / Mpc^3

def avg_int(L0, z, smit_table, nurest_l, cosmo, log=False):
    eps = epsilon_l(L0, z, smit_table, log=log)

    avg = (eps/(4.*np.pi*nurest_l))
    avg *= c.to_value(u.km/u.s)/cosmo.Hz(z)

    r = 1. * u.Lsun/ (u.Mpc**2 * u.GHz) 
    r = r.to_value(u.Jansky)
    avg *= r
    
    return avg

smit_table = np.array( [[0.0, 0.91, -3.80, -1.51], 
                        [0.2, 0.88, -3.01, -1.45], 
                        [0.4, 0.97, -2.97, -1.45], 
                        [0.6, 1.06, -2.68, -1.45], 
                        [0.8, 1.19, -2.77, -1.45], 
                        [0.8, 1.10, -2.47, -1.56], 
                        [1.5, 1.41, -2.61, -1.62], 
                        [2.2, 1.71, -2.73, -1.57], 
                        [1.5, 2.28, -3.44, -1.60], 
                        [2.0, 2.27, -3.41, -1.60], 
                        [2.3, 2.35, -3.49, -1.71], 
                        [3.8, 1.54, -2.97, -1.60], 
                        [5.0, 1.36, -3.12, -1.50], 
                        [5.9, 1.07, -2.97, -1.57], 
                        [6.8, 1.00, -3.20, -1.96]] )

smit_unlog_table = smit_table.copy()
smit_unlog_table[:,1] = 10.**smit_unlog_table[:,1]
smit_unlog_table[:,2] = 10.**smit_unlog_table[:,2]

# ---------------------------------------------------------------- #
#                                                                  #
#                      survey information                          #
#                                                                  #
# ---------------------------------------------------------------- #

survey_freq_range = { 'TIME': np.array([183, 326]),
                      'CONCERTO': np.array([200, 360]),
                      'CCAT-p': np.array([210, 420]) }

def TIME_res(nu):
    if nu < 230:
        return 1.5
    else:
        return 1.9

def CONCERTO_res(nu):
    return 1.5

def CCATp_res(nu):
    return nu/100

survey_res_func = { 'TIME': TIME_res,
                    'CONCERTO': CONCERTO_res,
                    'CCAT-p': CCATp_res }

survey_area = { 'TIME': 1.3*0.0083,
                'CONCERTO': 1.4,
                'CCAT-p': 2 }

def spixtpix(freq, survey):
    assert survey in ['TIME', 'CONCERTO', 'CCAT-p'], "Survey not recognized!"

    if survey=='TIME':
        if freq > 245 and freq <= 307.5:
            return 1.6E4
        elif freq > 212 and freq <= 245:
            return 5.7E3
        else:
            return np.nan

    if survey=='CONCERTO':
        if freq > 307.5 and freq <= 376.5:
            return 4.7E4
        elif freq > 245 and freq <= 307.5:
            return 1.8E4
        elif freq > 212 and freq <= 245:
            return 8E3
        else:
            return np.nan

    if survey=='CCAT-p':
        if freq > 376.5 and freq <= 428:
            return 2.2E4
        elif freq > 307.5 and freq <= 376.5:
            return 1.2E4
        elif freq > 245 and freq <= 307.5:
            return 6.2E3
        elif freq > 212 and freq <= 245:
            return 3.9E3
        else:
            return np.nan

survey_z_spixtpix = { 'TIME': np.array([[6.0, 1.6E4], [7.4, 5.7E3]]),
                      'CONCERTO': np.array([[4.5, 4.7E4], [6.0, 1.8E4], [7.4, 8.0E3]]),
                      'CCAT-p': np.array([[3.7, 2.2E4], [4.5, 1.2E4], 
                                          [6.0, 6.2E3], [7.4, 3.9E3]]) }

def comoving_distance_at_freq(freq_obs, freq_emit, cosmo):
    z = (freq_emit - freq_obs) / freq_obs
    return cosmo.comovingDistance(0, z) / cosmo.h

def calc_Vpix(nu_obs, nu_emit, pixel_nu, survey, cosmo):
    assert survey in ['TIME', 'CONCERTO', 'CCAT-p'], "Survey not recognized!"

    totaldeg = 4.*np.pi*(180./np.pi)**2
    z = (nu_emit - nu_obs) / nu_obs

    if survey=='CCAT-p':
        thetabeam = 53./3600.
        omegabeam = 2.*np.pi*(thetabeam/2.355)**2
    else:
        D = 12 # m
        line = c.to_value(u.m*u.GHz)/nu_obs
        thetabeam = 1.22 * line/D
        thetabeam *= (180./np.pi)
        omegabeam = 2.*np.pi*(thetabeam/2.355)**2

    dup_pix = comoving_distance_at_freq(nu_obs - pixel_nu/2, nu_emit, cosmo)
    dlow_pix = comoving_distance_at_freq(nu_obs + pixel_nu/2, nu_emit, cosmo)
    Vpix = (omegabeam/totaldeg) * (4.*np.pi/3.) * (dup_pix**3 - dlow_pix**3)
    return Vpix

def calc_Vsurv(nu_obs, nu_emit, bandwidth, survey_area, cosmo):
    totaldeg = 4.*np.pi*(180./np.pi)**2

    dup = comoving_distance_at_freq(nu_obs - bandwidth/2, nu_emit, cosmo)
    dlow = comoving_distance_at_freq(nu_obs + bandwidth/2, nu_emit, cosmo)
    Vsurv = (survey_area/totaldeg) * (4.*np.pi/3.) * (dup**3 - dlow**3)
    return Vsurv

def gen_Blist_Nlist(b, bandwidth, kmax, zobs, lines, survey, cosmo):

    nuemit = np.array([CO_lines[l] for l in lines])
    nuobs = nuemit / (1.+zobs)

    resolution = np.array([survey_res_func[survey](n) for n in nuobs])

    avg_int_list = []
    for l,n in zip(lines, nuemit):
        L0 = CO_L0[l]
        stable = smit_unlog_table

        I = avg_int(L0, zobs, stable, n, cosmo)
        avg_int_list.append(I)
    avg_int_list = np.array(avg_int_list)

    Blist = avg_int_list * b

    Nlist = []
    for r,no,ne in zip(resolution,nuobs,nuemit):
        Vpix = calc_Vpix(no, ne, r, survey, cosmo)
        N = spixtpix(no, survey)**2 * Vpix
        Nlist.append(N)
    Nlist = np.array(Nlist)

    return Blist, Nlist
