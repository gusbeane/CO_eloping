import sys
sys.path.append('../../fisher_plots')

import CO_data
import numpy as np 
import matplotlib.pyplot as plt

from labellines import labelLine, labelLines

import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': [r'\ttdefault']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

surveys = ['TIME', 'CONCERTO', 'CCAT-p']
surveys_tex = [r'\textbf{TIME}', r'\textbf{CONCERTO}', r'\textbf{CCAT-p}']

xrnge = [150, 525]
yrnge = [0, 10]

nuobs_list = np.linspace(xrnge[0], xrnge[1], 1000)

fig, ax = plt.subplots(1, 1, figsize=(4, 6))

for l in CO_data.CO_lines_names:
    nuemit = CO_data.CO_lines[l]
    zlist = (nuemit - nuobs_list)/nuobs_list
    if l=='CII':
        ax.plot(nuobs_list, zlist, label=r'$\text{C \textsc{ii}}$', c='k', lw=0.5)
    else:
        ax.plot(nuobs_list, zlist, label=l, c='k', lw=0.5)

xlist = np.linspace(450, 460, len(CO_data.CO_lines_names))
xlist[-1] = 490
xlist[1] = 170
xlist[2] = 260
xlist[3] = 390
labelLines(ax.get_lines(), zorder=2.5, bbox={'pad': 0.2, 'facecolor': 'white', 'edgecolor': 'white'}, 
                           xvals=xlist, fontsize=7)

for s,st,c in zip(surveys, surveys_tex, tb_c):
    freq_range = CO_data.survey_freq_range[s]
    ax.axvline(freq_range[0], color=c, zorder=4)
    ax.axvline(freq_range[1], color=c, zorder=4)
    ax.axvspan(freq_range[0], freq_range[1], color=c, alpha=0.25, zorder=4)
    ax.text(freq_range[1]-20, 8, st, color=c, zorder=5, rotation=90)

ax.set_xlim(xrnge)
ax.set_ylim(yrnge)

# ax.set_xlabel(r'$\text{\texttt{observed frequency [GHz]}}$')
# ax.set_ylabel(r'$\text{\texttt{emitted redshift}}$')
ax.set_xlabel(r'$\text{observed frequency}\,[\,\text{GHz}\,]$')
ax.set_ylabel(r'$\text{emitted redshift}$')

fig.tight_layout()
fig.savefig('freq_redshift_coverage.pdf')

# for s in surveys:
#     freq_range = CO_data.survey_freq_range[s]


