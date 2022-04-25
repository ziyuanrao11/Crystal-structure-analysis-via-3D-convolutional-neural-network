# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:00:44 2021

@author: z.rao
"""

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import math

#%%function for probability calculation for each points
def Probability_calculation(atoms,center,sigma):
    p_single= ((atoms.iloc[:,1:4]-center)**2).sum(axis=1)
    p=np.exp(-p_single/(2*(sigma**2))).sum(axis=0)  
    return p
#%% voxelization function
def voxelize(points):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.
    Args:
    `points`: pointcloud in 3D numpy.ndarray
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters
    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    n = 16
    sigma=0.15
    padding_size=(n, n, n, 2)
    voxels = np.zeros(padding_size)
    points = points.values
    origin = (np.min(points[:, 1]), np.min(points[:, 2]), np.min(points[:, 3]))
    # set the nearest point as (0,0,0)
    points[:, 1] -= origin[0]
    points[:, 2] -= origin[1]
    points[:, 3] -= origin[2]
    
    points=pd.DataFrame(points)
    
    # Atom_Al = pd.DataFrame(columns=['number','x','y','z','Da'])
    Atom_Mg = points[points.iloc[:, 4] == 24 ]
    
    # Atom_Fe = pd.DataFrame(columns=['number','x','y','z','Da'])
    Atom_Al = points[points.iloc[:, 4] == 27 ]
    
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                min_x = i*(8/n) #8 is the length of the cubic
                max_x = (i+1)*(8/n)
                coord_x= ((2*i+1)*(8/n))/2
                min_y = j*(8/n)
                max_y = (j+1)*(8/n)
                coord_y= ((2*j+1)*(8/n))/2
                min_z = k*(8/n)
                max_z = (k+1)*(8/n)
                coord_z= ((2*k+1)*(8/n))/2
                center=[coord_x,coord_y,coord_z]
                p_Atom_Mg= Probability_calculation(Atom_Mg,center,sigma)
                # print('Mg_finish_{}_{}_{}_{}'.format(i,j,k,p_Atom_Mg))  
                p_Atom_Al= Probability_calculation(Atom_Al,center,sigma)
                # print('Al_finish_{}_{}_{}_{}'.format(i,j,k,p_Atom_Al))  
                voxels[i,j,k, 0]=p_Atom_Mg
                voxels[i,j,k, 1]=p_Atom_Al
    return voxels


#%%calculate the probability for each atom and save    
all_data = np.empty([4000,16,16,16,2])

for i in tqdm(range(2000)):
    folder_1 = 'Data_remove_0.03_x_0.2_0.8_z_0.03_0.04/AlMg_simulated_data_3/data_noise_fcc_{}.csv'.format(i)
    points = pd.read_csv(folder_1)
    voxels=voxelize(points)
    all_data[i] = voxels
    folder_2 = 'Data_remove_0.03_x_0.2_0.8_z_0.03_0.04/AlMg_simulated_data_3/data_noise_l12_{}.csv'.format(i)
    points = pd.read_csv(folder_2)
    voxels=voxelize(points)
    all_data[i+2000] = voxels
    
all_y = np.empty([4000,1])

for i in tqdm(range(2000)):
    all_y[i] = 0
    all_y[2000+i] = 1
    
a = all_data[2000,:,:,:,0]
b = all_data[2000,:,:,:,1]

#%%train and test split    
bins     = [1000,2000,3000]
bin_y =  pd.DataFrame(all_y[:,])
y_binned = np.digitize(bin_y.index, bins, right=True)
 
X_train, X_test, y_train, y_test = train_test_split(all_data, all_y, test_size=0.20, random_state=42, stratify=y_binned)
    


#%%save the data
hdf5_data = h5py.File('Data_remove_0.03_x_0.2_0.8_z_0.03_0.04/data_16_probabilisty.h5','w')
hdf5_data.create_dataset('X_train_data_3',data= X_train)
hdf5_data.create_dataset('X_test_data_3',data= X_test)
hdf5_data.create_dataset('y_train_data_3',data= y_train)
hdf5_data.create_dataset('y_test_data_3',data= y_test)
hdf5_data.close()

# with h5py.File('data_16.h5',  "a") as f:
#     del f['X_train_data_3']
#     del f['X_test_data_3']
#     del f['y_train_data_3']
#     del f['y_test_data_3']
# hdf5_data.close()
    
# with h5py.File("data.h5", "r") as hf:    

#     # Split the data into training/test features/targets
#     X_train_fcc = hf["data_fcc_0"][0:2000]  
#     X_train_fcc = np.round(X_train_fcc)
#     X_train_l12 = hf["data_l12_0"][0:2000] 
#     X_train_l12 = np.round(X_train_l12)
#     print(hf.keys())
