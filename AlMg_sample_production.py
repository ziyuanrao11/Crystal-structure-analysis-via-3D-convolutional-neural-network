# -*- coding: utf-8 -*-
"""
@author: z.rao
"""

import numpy as np
# import os
# import shutil
import matplotlib.pyplot as plt
from Euler_transformation import Euler_transformation_100, Euler_transformation_110, Euler_transformation_111
# from generator_single_SDMs import single_SDMs
from tqdm import tqdm
import pandas as pd
from numpy import random
import seaborn as sns
import os
#%%build the files you want to save
for n in range(4):
    folder_dir = 'Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_{}'.format(n)
    if not os.path.isdir(folder_dir):
      os.mkdir(folder_dir)
#%%for adding some noise
def add_data_noise(data_reconstruction, sigma_xy, sigma_z, plot_noise):
    row = data_reconstruction.shape[0]
    noise_xy = np.random.normal(mu, sigma_xy, [row ,2]) 
    noise_z = np.random.normal(mu, sigma_z, [row ,1])
    zeros = np.zeros((row, 1))
    noise = np.hstack((noise_xy, noise_z, zeros))
    data_noise = data_reconstruction + noise
    # print(data_noise.type)
    # Plot 3D crystal structure with noise
    if plot_noise == True:
        ax = plt.subplot(111, projection='3d')  # build a project
        ax.scatter(data_noise [:, 0], data_noise [:, 1], data_noise [:, 2], c=data_noise [:, 3], s=8)  # 绘制数据点
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_zlabel('Z', fontsize=16)  # axis
        ax.set_ylabel('Y', fontsize=16)
        ax.set_xlabel('X', fontsize=16)
        plt.show()
    return data_noise
#%% Input file and parameters  fcc data
for i in range(2000):
    data_o = np.loadtxt('ggoutputFile_fcc_5nm_a_0.405.txt')
    data_o=np.unique(data_o, axis=0)
    lattice_para = 0.405
    mu  = 0
    data = pd.DataFrame(data_o)
    # replace some atoms if it is needed
    replace_n = 2744 #the number of 24 in the l12 phase
    replace_indices = np.random.choice(data.index, replace_n, replace = False)   
    data.iloc[replace_indices,3] = data.iloc[replace_indices,3].replace(27,24)
    a=data[data.iloc[:,3]==24]
    plot_noise = False #default 
    data.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_0/data_noise_fcc_{}.csv'.format(i))
    #%% data with Euler transfromation
    
    #Euler_transformation
    data=data.to_numpy()
    data_1 = Euler_transformation_110(data, False)
    data_1 = pd.DataFrame(data_1)
    data_1.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_1/data_noise_fcc_{}.csv'.format(i))
    #%%with incomplete data
    #remove some data
    rand = 3
    remove_n = round(data_1.shape[0]*0.01*rand)
    drop_indices = np.random.choice(data_1.index, remove_n, replace = False)
    data_2=data_1.drop(drop_indices)
    data_2=data_2.reset_index()
    data_2=data_2.drop(columns=['index'])
    data_2.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_2/data_noise_fcc_{}.csv'.format(i))
    
    # # replace some atoms if it is needed
    # rand = random.randint(8,13)
    # replace_n = round(data_1.shape[0]*0.01*rand)
    # replace_indices = np.random.choice(data_1.index, replace_n, replace = False)   
    # data_1.iloc[replace_indices,3] = data_1.iloc[replace_indices,3].replace(27,24).to_numpy()
    # data_1=data_1.to_numpy()
    # plot_noise = False #default  
#%%add some noise
    #Generating data with noise
    sigma_xy = (np.random.randint(2,8))/10
    sigma_z = (np.random.randint(0,10))/1000
    # sigma_z = 0.03
    plot_noise = False #default  
    data_2 = add_data_noise(data_2, sigma_xy, sigma_z, plot_noise)
    data_noise=pd.DataFrame(data_2)
    data_noise.columns=['x','y','z','Da']
    # data_noise=data_noise[(data_noise['x']<2) & (data_noise['y']<2) & (data_noise['z']<2)
    #                       & (data_noise['x']>-2) & (data_noise['x']>-2) & (data_noise['x']>-2)]
    data_noise.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_3/data_noise_fcc_{}.csv'.format(i))
    print(i)

#%% Input file and parameters  l12 data
for i in range(2000):
    data_o = np.loadtxt('ggoutputFile_L12_5nm_a_0.405.txt')
    data_o=np.unique(data_o, axis=0)
    lattice_para = 0.405
    mu  = 0
    data = pd.DataFrame(data_o)
    data.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_0/data_noise_l12_{}.csv'.format(i))
    
    #Euler_transformation
    data=data.to_numpy()
    data_1 = Euler_transformation_110(data, False)
    data_1 = pd.DataFrame(data_1)
    data_1.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_1/data_noise_l12_{}.csv'.format(i))
    
    #remove some data
    rand = 23
    remove_n = round(data_1.shape[0]*0.01*rand)
    drop_indices = np.random.choice(data_1.index, remove_n, replace = False)
    data_2=data_1.drop(drop_indices)
    data_2=data_2.reset_index()
    data_2=data_2.drop(columns=['index'])
    data_2.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_2/data_noise_l12_{}.csv'.format(i))
    
    # # replace some atoms if it is needed
    # rand = random.randint(8,13)
    # replace_n = round(data_1.shape[0]*0.01*rand)
    # replace_indices = np.random.choice(data_1.index, replace_n, replace = False)   
    # data_1.iloc[replace_indices,3] = data_1.iloc[replace_indices,3].replace(27,24).to_numpy()
    # data_1=data_1.to_numpy()
    # plot_noise = False #default  

    #Generating data with noise
    sigma_xy = (np.random.randint(2,8))/10
    sigma_z = (np.random.randint(0,10))/1000
    # sigma_z = 0.03
    plot_noise = False #default  
    data_2 = add_data_noise(data_2, sigma_xy, sigma_z, plot_noise)
    data_noise=pd.DataFrame(data_2)
    data_noise.columns=['x','y','z','Da']
    # data_noise=data_noise[(data_noise['x']<2) & (data_noise['y']<2) & (data_noise['z']<2)
    #                       & (data_noise['x']>-2) & (data_noise['x']>-2) & (data_noise['x']>-2)]
    data_noise.to_csv('Data_remove_0.43_x_0.2_0.8_z_0.00_0.01/AlMg_simulated_data_3/data_noise_l12_{}.csv'.format(i))
    print(i)