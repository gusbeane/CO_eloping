import numpy as np 
import astropy.units as u
from astropy.constants import c
from scipy.special import gamma
from colossus.cosmology import cosmology

LT16_params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
LT16_cosmo = cosmology.setCosmology('myCosmo', LT16_params)

smit_h = 0.7
smit_h3 = smit_h**3

smit_table = np.array( [[0.0, 0.91, -3.80, -1.51], 
                        [0.2, 0.88, -3.01, -1.45], 
                        [0.4, 0.97, -2.97, -1.45], 
                        [0.6, 1.06, -2.68, -1.45], 
                        # [0.8, 1.19, -2.77, -1.45], 
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
    return cosmo.comovingDistance(0, z)

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

class line(object):
    def __init__(self, key, cosmo, redshift=None, bias=None):
        self.key = key
        
        if bias is not None:
            self.bias = bias
            self.bias_assigned = True
        else:
            self.bias_assigned = False

        self.freq_emit = CO_lines[key]
        self.wave_emit = CO_lines_wave[key]

        self.cosmo = cosmo

        if redshift is not None:
            self.assign_redshift(redshift)
            self.intensity(redshift, assign=True)
        else:
            self.assign_redshift(redshift)

        if bias is not None:
            self.assign_bias(bias)

    def assign_redshift(self, z):
        self.redshift = z

    def assign_bias(self, b):
        self.bias = b

    def freq_obs(self, z):
        return self.freq_emit / (1. + z)

    def wave_obs(self, z):
        return self.wave_emit * (1. + z)

    def redshift_emit_interloping(self, ztarget, freq_target):
        zinterlop = (1.+ztarget)*(self.freq_emit/freq_target) - 1
        return zinterlop

    def freq_obs_interloping(self, ztarget, freq_target):
        zinterlop = self.redshift_interloping(ztarget, freq_target)
        return self.freq_emit / (1. + zinterlop)

    def wave_obs_interloping(self, ztarget, freq_target):
        zinterlop = self.redshift_emit_interloping(ztarget, freq_target)
        return self.wave_obs(zinterlop)

class gaussian_smooth(object):
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
    
    def __call__(self, x_eval):
        delta_x = x_eval[:, None] - self.x 
        weights = np.exp(-delta_x*delta_x / (2*self.sigma*self.sigma)) / (np.sqrt(2*np.pi) * self.sigma) 
        weights /= np.sum(weights, axis=1, keepdims=True) 
        y_eval = np.dot(weights, self.y) 
        return y_eval

class LT16_COmodel(object):
    def __init__(self, use_LT16_freq=False, cosmo=LT16_cosmo, kind='smooth', smooth_sigma=1):
        self._speed_of_light_in_kms_ = 299792.458
        self.available_keys = ['1-0', '2-1', '3-2', '4-3', '5-4', '6-5', '7-6', '8-7', '9-8',
                               '10-9', '11-10', '12-11', '13-12', 'CII']
        self.use_LT16_freq = use_LT16_freq
        self.cosmo = cosmo
        self.kind = kind
        self.smooth_sigma = smooth_sigma

    def calc_intensity(self, z, assign=False):
        L0 = CO_L0[self.key]
        intensity = avg_int(L0, z, smit_unlog_table, 
                            self.freq_emit, self.cosmo, smooth=False)

        if assign:
            self.intensity = intensity
            self.intensity_assigned = True

        return intensity

    def B(self, z):
        assert self.bias is not None, "You must assign the line bias first!"
        
        if self.intensity_assigned:
            return self.intensity * self.bias
        else:
            CO_lines_wave = {'1-0': 2610, '2-1': 1300, '3-2': 866, '4-3': 651,
                             '5-4': 521, '6-5': 434, '7-6': 372, '8-7': 325,
                             '9-8': 289, '10-9': 260, '11-10': 237,
                             '12-11': 217, '13-12': 200, 'CII': 157.7}
            
            self.lines = { key: line(key, wave_emit=w) 
                           for key, w in CO_lines_wave.items() }

    def _compute_intensity_grid_(self):
        intensity_grid = {}

        z = self.smit_table[:,0]
        SFR = self.smit_table[:,1]
        phi = self.smit_table[:,2]
        alpha = self.smit_table[:,3]

        for key in self.available_keys:
            eps = phi * self.CO_L0[key] * SFR * gamma(2.+alpha)
            avg = (eps/(4.*np.pi*self.lines[key].freq_emit))
            avg = np.multiply(avg, self._speed_of_light_in_kms_/self.cosmo.Hz(z) )
            avg *= 0.04020414574289873 # unit handling, will incorporate astropy units later
            intensity_grid[key] = np.transpose([z, avg])
        
        self._intensity_grid_ = intensity_grid
    
    def _compute_interpolating_functions_(self):
        interpolating_fns = {}

        for key in self.available_keys:
            grid = self._intensity_grid_[key]
            fns = {}
            for kind in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
                fns[kind] = interp1d(np.array(grid[:,0]), np.array(grid[:,1]), kind=kind, 
                                     bounds_error=False, fill_value="extrapolate")
            fns['smooth'] = gaussian_smooth(np.array(grid[:,0]), np.array(grid[:,1]), self.smooth_sigma)
            interpolating_fns[key] = fns

        self._interpolating_fns_ = interpolating_fns
    
    def __call__(self, key, z, kind=None):
        if kind is None:
            return self._interpolating_fns_[key][self.kind](z)
        else:
            return self._interpolating_fns_[key][kind](z)

if __name__ == '__main__':
    l_freq = line('CII', 1901.03)
    l_wave = line('CII', wave_emit=157.7)

    CO1_0 = line('1-0', LT16_cosmo)
    CO2_1 = line('2-1', LT16_cosmo)
    CO3_2 = line('3-2', LT16_cosmo)
    CO4_3 = line('4-3', LT16_cosmo)
    CO5_4 = line('5-4', LT16_cosmo)
    CO6_5 = line('6-5', LT16_cosmo)
    CO7_6 = line('7-6', LT16_cosmo)
    CO8_7 = line('8-7', LT16_cosmo)
    CO9_8 = line('9-8', LT16_cosmo)
    CO10_9 = line('10-9', LT16_cosmo)
    CO11_10 = line('11-10', LT16_cosmo)
    CO12_11 = line('12-11', LT16_cosmo)
    CO13_12 = line('13-12', LT16_cosmo)
    CII = line('CII', LT16_cosmo)
