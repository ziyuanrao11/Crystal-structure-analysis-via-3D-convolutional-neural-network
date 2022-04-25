# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:55:26 2021

@author: z.rao
"""


# import gridToVTK
from pyevtk.hl import pointsToVTK 
import cv2
import glob
import h5py
import numpy as np
import pandas as pd

#%% wirte points data
folder_1 = 'AlMg_simulated_data_0/data_noise_fcc_0.csv'
points_fcc = pd.read_csv(folder_1).values
x = points_fcc[:,1]  
# x = x+4.4
# x=2*x
y = points_fcc[:,2] 
# y=y+1
# y=2*y
z = points_fcc[:,3]
# z=z+0.2
# z=2*z
# x, y, z = x + 4, y + 4, z + 4
Da = points_fcc[:,4] 
pointsToVTK("simulated_data_0_fcc_0", x, y, z, data = {"Da" : Da})

folder_2 = 'AlMg_simulated_data_0/data_noise_l12_0.csv'
points_l12 = pd.read_csv(folder_2).values
x = points_l12[:,1]  
# x = x+4.4
# x=2*x
y = points_l12[:,2] 
# y=y+1
# y=2*y
z = points_l12[:,3]
# z=z+0.2
# z=2*z
# x, y, z = x + 4, y + 4, z + 4
Da = points_l12[:,4] 
pointsToVTK("simulated_data_0_l12_0", x, y, z, data = {"Da" : Da})

#%%
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
                min_x = i*(8/n)
                max_x = (i+1)*(8/n)
                min_y = j*(8/n)
                max_y = (j+1)*(8/n)
                min_z = k*(8/n)
                max_z = (k+1)*(8/n)
                Atom_in_Mg = Atom_Mg[Atom_Mg.iloc[:,1].between(min_x, max_x) & Atom_Mg.iloc[:,2].between(min_y, max_y) & Atom_Mg.iloc[:,3].between(min_z, max_z)]
                Atom_in_Al = Atom_Al[Atom_Al.iloc[:,1].between(min_x, max_x) & Atom_Al.iloc[:,2].between(min_y, max_y) & Atom_Al.iloc[:,3].between(min_z, max_z)]
                voxels[i,j,k, 0]=len(Atom_in_Mg)
                voxels[i,j,k, 1]=len(Atom_in_Al)
    return voxels

#%% normalization
def normalization(data):
    range=np.max(data)-np.min(data)
    return (data-np.min(data))/range
#%%

folder_1 = 'L12AlMg/simulated_data/data_noise_fcc_0.csv'
points_fcc = pd.read_csv(folder_1)
voxels_fcc=voxelize(points_fcc)
    

folder_2 = 'L12AlMg/simulated_data/data_noise_l12_0.csv'
points_l12 = pd.read_csv(folder_2)
voxels_l12=voxelize(points_l12)

from pyevtk.hl import imageToVTK 
import numpy as np  
nx, ny, nz = 16, 16, 16 
ncells = nx * ny * nz 
npoints = (nx + 1) * (ny + 1) * (nz + 1) 
# Variables <br>
pressure = np.random.rand(ncells).reshape( (nx, ny, nz)) 
temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))
Mg_fcc = voxels_fcc[:,:,:,0] 
Mg_fcc = normalization(Mg_fcc)
Al_fcc = voxels_fcc[:,:,:,1] 
Al_fcc = normalization(Al_fcc)
MgAl_fcc = voxels_fcc[:,:,:,0] + voxels_fcc[:,:,:,1] 
MgAl_fcc = normalization(MgAl_fcc)
imageToVTK("./voxels_fcc_16", cellData = {"Mg" : Mg_fcc ,  "Al" : Al_fcc, "MgAl" : MgAl_fcc  }, pointData = {"temp" : temp} )
# imageToVTK("./voxels_fcc", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )

Mg_l12 = voxels_l12[:,:,:,0] 
Mg_l12 = normalization(Mg_l12)
Al_l12 = voxels_l12[:,:,:,1] 
Al_l12 = normalization(Al_l12)
MgAl_l12 = voxels_l12[:,:,:,0] + voxels_l12[:,:,:,1] 
MgAl_l12 = normalization(MgAl_l12)
imageToVTK("./voxels_l12_16", cellData = {"Mg" : Mg_l12 ,  "Al" : Al_l12, "MgAl" : MgAl_l12  }, pointData = {"temp" : temp} )
#%%full connected layer
folder_1 = 'L12AlMg/simulated_data/data_noise_fcc_0.csv'
points_fcc = pd.read_csv(folder_1)
voxels_fcc=voxelize(points_fcc)
    

folder_2 = 'L12AlMg/simulated_data/data_noise_l12_0.csv'
points_l12 = pd.read_csv(folder_2)
voxels_l12=voxelize(points_l12)

from pyevtk.hl import imageToVTK 
import numpy as np  
nx, ny, nz = 128, 1, 1 
ncells = nx * ny * nz 
npoints = (nx + 1) * (ny + 1) * (nz + 1) 
# Variables <br>
pressure = np.random.rand(ncells).reshape((nx, ny, nz)) 
temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))
Mg_fcc = voxels_fcc[:,:,:,0] 
Mg_fcc=np.ravel(Mg_fcc)
Mg_fcc=Mg_fcc[0:128].reshape((nx, ny, nz)) 
Mg_fcc = normalization(Mg_fcc)
Al_fcc = voxels_fcc[:,:,:,1] 
Al_fcc=np.ravel(Mg_fcc)
Al_fcc=Al_fcc[0:128].reshape((nx, ny, nz)) 
Al_fcc = normalization(Al_fcc)
# MgAl_fcc = voxels_fcc[:,:,:,0] + voxels_fcc[:,:,:,1] 
# MgAl_fcc = normalization(MgAl_fcc)
imageToVTK("./voxels_fcc_fullC", cellData = {"Mg" : Mg_fcc ,  "Al" : Al_fcc}, pointData = {"temp" : temp} )
# imageToVTK("./voxels_fcc", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )

Mg_l12 = voxels_l12[:,:,:,0] 
Mg_l12=np.ravel(Mg_l12)
Mg_l12=Mg_l12[0:128].reshape((nx, ny, nz)) 
Mg_l12 = normalization(Mg_l12)
Al_l12 = voxels_l12[:,:,:,1] 
Al_l12=np.ravel(Al_l12)
Al_l12=Al_l12[0:128].reshape((nx, ny, nz)) 
Al_l12 = normalization(Al_l12)
imageToVTK("./voxels_l12_fullC", cellData = {"Mg" : Mg_l12 ,  "Al" : Al_l12}, pointData = {"temp" : temp} )




