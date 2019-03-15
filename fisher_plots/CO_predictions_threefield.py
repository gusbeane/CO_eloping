import numpy as np 

from threefield import gen_fisher_matrix, gen_Vk
import CO_data
from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 70, 'Om0': 0.27, 'Ob0': 0.046, 'sigma8': 0.8, 'ns': 1.0}
cosmo = cosmology.setCosmology('myCosmo', params)

