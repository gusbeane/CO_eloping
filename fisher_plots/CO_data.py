import numpy as np 
import astropy.units as u
from astropy.constants import c

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

survey_z_spixtpix = { 'TIME': np.array([[6.0, 1.6E4], [7.4, 5.7E3]]),
                      'CONCERTO': np.array([[4.5, 4.7E4], [6.0, 1.8E4], [7.4, 8.0E3]]),
                      'CCAT-p': np.array([[3.7, 2.2E4], [4.5, 1.2E4], [6.0, 6.2E3]]) }


