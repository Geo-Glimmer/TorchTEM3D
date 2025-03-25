from sys import path
path.append("./FDTD_func/")
from torchTEM3D import *

import torch
import numpy as np
from discretize import TensorMesh

import time


#%%

# Define the mesh dimensions using discretize's TensorMesh
# Define the mesh, model, and mapping
dh = 50  # Width of the core region of the grid / initial width
XI = 24  # Number of cells along the x-axis
YJ = 24  # Number of cells along the y-axis
ZK = 24  # Number of cells along the z-axis
pad_num = 6  # Number of padding cells on each side
core_num_x = XI - 2 * pad_num  # Core computational region along the x-axis
core_num_y = YJ - 2 * pad_num  # Core computational region along the y-axis
core_num_z = ZK - 2 * pad_num  # Core computational region along the z-axis

padd_value = 2  # Expansion factor for padding
# Define the grid size
# dx X dy X dz, where length = dh
hx = [(dh, pad_num, -padd_value), (dh, core_num_x), (dh, pad_num, padd_value)]  # Grid spacing in the x-direction
hy = [(dh, pad_num, -padd_value), (dh, core_num_y), (dh, pad_num, padd_value)]  # Grid spacing in the y-direction
hz = [(dh, pad_num, -padd_value), (dh, core_num_z), (dh, pad_num, padd_value)]  # Grid spacing in the z-direction

mesh = TensorMesh([hx, hy, hz])  # Create the tensor mesh
# Calculate the grid widths in the x, y, and z directions
x = mesh.h[0]  # Grid widths in the x-direction
y = mesh.h[1]  # Grid widths in the y-direction
z = mesh.h[2]  # Grid widths in the z-direction


#%%
device = 'cuda'  # the device of computing

#%%
#% single layer
model_EC = torch.ones((XI, YJ, ZK))  # Conductivity model (XI x YJ x ZK)
model_EC[:, :, 0:7] = model_EC[:, :, 0:7] * 3e-4  # Air layer, typically set to a conductivity of 3e-4
model_EC[:, :, 7:-1] = model_EC[:, :, 7:-1] * 1e-2  # Underground layer

#%%
# three layers

# model_EC[:,:,0:7] = model_EC[:,:,0:7]*3e-4
# model_EC[:,:,7:13] = model_EC[:,:,7:13]*2e-3
# model_EC[:,:,13:19] = model_EC[:,:,13:19]*1e-2
# model_EC[:,:,19:] = model_EC[:,:,19:]*2e-2

#%%
# # complex model
# model_EC[:,:,0:7] = model_EC[:,:,0:7]*3e-4
# model_EC[:,:,7:9] = model_EC[:,:,7:9]/10
# model_EC[0:12,:,9:17] = model_EC[0:12,:,9:17]*1e-2
# model_EC[12:,:,9:17] = model_EC[12:,:,9:17]/500

# model_EC[10:12,10:12,11:12] = model_EC[10:12,10:12,11:12]/250
# model_EC[15:18,10:12,14:18] = model_EC[15:18,10:12,14:18]/150.0

# model_EC[:,:,17:] = model_EC[:,:,17:]/200

#%%
# if you need calculate Sensitivity Matrix, please set "requires_grad" = True
model_EC = torch.tensor(model_EC, device=device, requires_grad=False)

#%%
# Observation system settings
L_loop = 300  # Length of the loop
n_subloop = 64  # Number of sub-loops

Alpha = 0.7  # Alpha parameter

Rx_st = 11  # Starting point (center of the grid)
Rx_end = 11  # End point (inclusive, center of the grid)
i0, j0, k0 = 10, 10, 7  # Center position of the transmitting loop
iter_n_max = 2000  # Maximum number of iterations


#%%
# compute    
strat_time = time.time()    
    
t_H,dBz = forward_3DTEM(L_loop,n_subloop,i0, j0, k0,
                  XI, YJ, ZK,x, y, z, Alpha,
                  model_EC, 
                  Rx_st, Rx_end,
                  iter_n_max,
                  device=device)


end_time = time.time()  
print(dBz)  
print(f"time:{end_time-strat_time}")

# #%%
# # Calculate Sensitivity Matrix
# start_time = time.time()    
# dBz.backward(torch.ones_like(dBz))  # Gradient weight tensor with the same shape as dBz

# # Get the gradient of model_EC
# grad_model_EC = model_EC.grad
# end_time = time.time()    
# print(f"Calculate Sensitivity Matrix time: {end_time - start_time}")

#%%
# # Save data
# combined_data = np.column_stack((t_H, dBz))  # Combine t_H and dBz into a single array
# # Save as a CSV file (Note: Change the path and filename as needed)
# np.savetxt('./result/1-layers/TorchTEM3D_dBz.csv', combined_data, delimiter=',', header='t_H,dBz', comments='')