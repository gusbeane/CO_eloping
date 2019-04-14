import numpy as np 

from threefield import threefield, corner_plot
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

resolution = np.array([CO_data.survey_res_func[survey](n) for n in nuobs])

avg_int = []
for l,n in zip(lines, nuemit):
    L0 = CO_data.CO_L0[l]
    stable = CO_data.smit_unlog_table

    I = CO_data.avg_int(L0, zobs, stable, n, cosmo)
    avg_int.append(I)
avg_int = np.array(avg_int)

Blist = avg_int * b

Nlist = []
for r,no,ne in zip(resolution,nuobs,nuemit):
    Vpix = CO_data.calc_Vpix(no, ne, r, survey, cosmo)
    N = CO_data.spixtpix(no, survey)**2 * Vpix
    Nlist.append(N)
Nlist = np.array(Nlist)

Vsurv = CO_data.calc_Vsurv(nuobs[0], nuemit[0], bandwidth, CO_data.survey_area[survey], cosmo)

fig, ax, _ = corner_plot(1, np.array([1, 2, 3]), np.array([1, 1, 1]), cosmo, 0.5, kmax=1, Vk=20)
fig.tight_layout()
fig.savefig('corner_Nconstant.pdf')

fig, ax, _ = corner_plot(1, np.array([2, 2, 2]), np.array([1, 2, 3]), cosmo, 0.5, kmax=1, Vk=20)
fig.tight_layout()
fig.savefig('corner_Bconstant.pdf')

tf = threefield(zobs, Blist, Nlist, kmax, Vsurv, cosmo)
fig, ax, _ = corner_plot(zobs, Blist, Nlist, cosmo, 1.75, tf=tf)
fig.tight_layout()
fig.savefig('CO.pdf')
