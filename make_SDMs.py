# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 13:32:13 2022

@author: z.rao
"""


import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.colors import BoundaryNorm
from fast_histogram import histogram2d as fast_histogram2d 
from matplotlib.ticker import MaxNLocator

folder_1 = 'AlMg_simulated_data_3/data_noise_fcc_5.csv'
points = pd.read_csv(folder_1)
points=points.drop(columns=['Unnamed: 0'])
points.columns=['a','b','c','d']


element_1 = points.loc[(points['d'] == 27 ) , ['a','b','c']] #Al atoms
element_1 = element_1.values

element_2 = points.loc[(points['d'] == 24 ) , ['a','b','c']] # Mg/Li atoms
element_2 = element_2.values

SDM_bins = 200
tree = []
tree = spatial.cKDTree(element_1)
# SDM = np.zeros([SDM_bins,SDM_bins])

x_tot = []
y_tot = []
num_in_SDM = 0
max_cand =0
cand = tree.query_ball_point(element_1, 1.5, return_sorted=False, n_jobs = -1)

for list in cand:
    num_in_SDM += len(list);
    if (len(list) > max_cand):
        max_cand = len(list);
x_tot = np.zeros([num_in_SDM,], dtype = np.float32)
y_tot = np.zeros([num_in_SDM,], dtype = np.float32)
x = np.zeros([max_cand,], dtype = np.float32)   
y = np.zeros([max_cand,], dtype = np.float32)  

start = 0;
i = 0;
for list in cand:
    length = len(list)
    x_tot[start:(start+length)] = np.ndarray.__sub__(element_1[list,0],element_1[i,0]);
    y_tot[start:(start+length)] = np.ndarray.__sub__(element_1[list,2],element_1[i,2]);
    i += 1
    start = start+length;
notzero = (x_tot!=0)*(y_tot!=0);
SDM = fast_histogram2d(y_tot[notzero],x_tot[notzero], range = [[-1.5,1.5],[-1.5,1.5]],  bins=SDM_bins)   

xedges = np.linspace(-1.5, 1.5, num=SDM_bins+1, endpoint=True, retstep=False, dtype=np.float32)
yedges = np.linspace(-1.5, 1.5, num=SDM_bins+1, endpoint=True, retstep=False, dtype=np.float32)
fig2D = plt.figure(figsize=(8,6))
ax2D = fig2D.add_subplot(111)
levels = MaxNLocator(nbins=8).tick_values(1, SDM.max())
cmap = plt.get_cmap('jet') #BuGn


norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)   
im=ax2D.pcolormesh(yedges, xedges, SDM, cmap=cmap, norm=norm)  
ax2D.set_xlabel('$\Delta$x, nm', fontsize=20)
ax2D.set_ylabel('$\Delta$z, nm', fontsize=20)
print('element_number=', 27)
ax2D.set_aspect(1)
plt.xlim((-0.7,0.7))
plt.ylim((-0.7,0.7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb = fig2D.colorbar(im,ax=ax2D)
cb.ax.tick_params(labelsize=18)
plt.show()  

plt.savefig('Figures/fcc_SDM_data_3.png', format='png', dpi=300)

