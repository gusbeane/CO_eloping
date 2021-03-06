import numpy as np 

import sys
sys.path.append('../../')

from multifield import threefield, corner_plot
import CO_data
from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

survey = 'CCAT-p'
lines = ['6-5', '5-4', '4-3']

b = 3
bandwidth = 60
kmax = 1

zobs = 0.88
nuemit = np.array([CO_data.CO_lines[l] for l in lines])
nuobs = nuemit / (1.+zobs)

Blist, Nlist = CO_data.gen_Blist_Nlist(3, bandwidth, kmax, zobs, lines, survey, cosmo)

Vsurv = CO_data.calc_Vsurv(nuobs[0], nuemit[0], bandwidth, CO_data.survey_area[survey], cosmo)

fig, ax, _ = corner_plot(1, np.array([1, 2, 3]), np.array([100, 100, 100]), cosmo, 0.25, kmax=1, Vk=200000, norm=True, intstr=True)
fig.tight_layout()
fig.savefig('corner_Nconstant.pdf')

fig, ax, _ = corner_plot(1, np.array([1, 4, 9]), np.array([100, 100, 100]), cosmo, 0.25, kmax=1, Vk=200000, norm=True, intstr=True)
fig.tight_layout()
fig.savefig('corner_Nconstant_Bhigher.pdf')

fig, ax, _ = corner_plot(1, np.array([2, 2, 2]), np.array([100, 200, 300]), cosmo, 0.4, kmax=1, Vk=200000, norm=True, intstr=True)
fig.tight_layout()
fig.savefig('corner_Bconstant.pdf')
