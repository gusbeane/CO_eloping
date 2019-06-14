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
