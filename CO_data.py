import numpy as np
import warnings

class line(object):
    def __init__(self, key, freq_emit=None, wave_emit=None):
        
        self._speed_of_light_ = 299792.458 # micron * GHz
        self._assign_freq_and_wave_(freq_emit, wave_emit)

        self.key = key

    def _assign_freq_and_wave_(self, freq_emit, wave_emit):        
        assert freq_emit is not None or wave_emit is not None, "Must specify emitted frequency\
                                                                or wavelength"
        
        if freq_emit is not None and wave_emit is not None:
            warnings.warn("Both emitted frequency and wavelength supplied, wavelength is ignored",
                            RuntimeWarning)

        if freq_emit is not None:
            self.freq_emit = freq_emit
            self.wave_emit = self._speed_of_light_/self.freq_emit
        else:
            self.wave_emit = wave_emit
            self.freq_emit = self._speed_of_light_/self.wave_emit

    def freq_obs(self, z):
        return self.freq_emit / (1. + z)

    def wave_obs(self, z):
        return self.wave_emit * (1. + z)

    def redshift_emit_interloping(self, ztarget, freq_target):
        zinterlop = (1. + ztarget) * (self.freq_emit/freq_target) - 1.
        return zinterlop

    def freq_obs_interloping(self, ztarget, freq_target):
        zinterlop = self.redshift_emit_interloping(ztarget, freq_target)
        return self.freq_obs(zinterlop)

    def wave_obs_interloping(self, ztarget, freq_target):
        zinterlop = self.redshift_emit_interloping(ztarget, freq_target)
        return self.wave_obs(zinterlop)
    
class LT16_COmodel(object):
    def __init__(self, use_LT16_freq=False):
        self.available_keys = ['1-0', '2-1', '3-2', '4-3', '5-4', '6-5', '7-6', '8-7', '9-8',
                               '10-9', '11-10', '12-11', '13-12', 'CII']
        self.use_LT16_freq = use_LT16_freq

        self._assign_lines_()
        self._assign_CO_L0_()        
        self._assign_smit_table_()
        

    def _assign_CO_L0_(self):
        self.CO_L0 = {'1-0': 3.7E3, '2-1': 2.8E4, '3-2': 7E4, '4-3': 9.7E4,
                      '5-4': 9.6E4, '6-5': 9.5E4, '7-6': 8.9E4, '8-7': 7.7E4,
                      '9-8': 6.9E4, '10-9': 5.3E4, '11-10': 3.8E4,
                      '12-11': 2.6E4, '13-12': 1.4E4, 'CII': 6E6}

    def _assign_smit_table_(self):
        smit_h = 0.7

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
        
        smit_table[:,1] = 10.**smit_table[:,1]
        smit_table[:,2] = 10.**smit_table[:,2]

        self.smit_table = smit_table

    def _assign_lines_(self):
        if self.use_LT16_freq:
            n = 111.52
            CO_lines_LT16 = {'1-0': n, '2-1': 2*n, '3-2': 3*n, '4-3': 4*n,
                 '5-4': 5*n, '6-5': 6*n, '7-6': 7*n, '8-7': 8*n,
                 '9-8': 9*n, '10-9': 10*n, '11-10': 11*n,
                 '12-11': 12*n, '13-12': 13*n, 'CII': 1901.0}

            self.lines = { key: line(key, freq_emit=f) for key, f in CO_lines_LT16.items() }
        
        else:
            CO_lines_wave = {'1-0': 2610, '2-1': 1300, '3-2': 866, '4-3': 651,
                             '5-4': 521, '6-5': 434, '7-6': 372, '8-7': 325,
                             '9-8': 289, '10-9': 260, '11-10': 237,
                             '12-11': 217, '13-12': 200, 'CII': 157.7}
            
            self.lines = { key: line(key, wave_emit=w) for key, w in CO_lines_wave.items() }

if __name__ == '__main__':
    l_freq = line('CII', 1901.03)
    l_wave = line('CII', wave_emit=157.7)

    COmodel = LT16_COmodel()
