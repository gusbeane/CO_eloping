import numpy as np 

import sys
sys.path.append('../')

from multifield import threefield, corner_plot
import CO_data
from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

def three_fields_corner_plot():
    survey = 'CCAT-p'
    lines = ['6-5', '5-4', '4-3']
    
    b = 3
    bandwidth = 60
    kmax = 1
    
    zobs = 0.88
    nuemit = np.array([CO_data.CO_lines[l] for l in lines])
    nuobs = nuemit / (1.+zobs)

    Vsurv = CO_data.calc_Vsurv(nuobs[0], nuemit[0], bandwidth, CO_data.survey_area[survey], cosmo)
    
    Blist, Nlist = CO_data.gen_Blist_Nlist(3, bandwidth, kmax, zobs, lines, survey, cosmo)

    tf = threefield(zobs, Blist, Nlist, kmax, Vsurv, cosmo)
    labels = [r'$B_{6-5}\,[\,\text{Jy}/\text{str}\,]$', r'$B_{5-4}\,[\,\text{Jy}/\text{str}\,]$', 
              r'$B_{4-3}\,[\,\text{Jy}/\text{str}\,]$']
    fig, ax, _ = corner_plot(zobs, Blist, Nlist, cosmo, 1.75, tf=tf, labels=labels, printtext=False)
    fig.tight_layout()
    fig.savefig('CO_tf.pdf')


if __name__ == '__main__':
    three_fields_corner_plot()
