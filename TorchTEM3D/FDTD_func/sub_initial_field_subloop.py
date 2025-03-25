import torch
from calculate_coord import *

def sub_initial_field_subloop_torch(i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                                    subloop_x, subloop_y, L_subloop, Alpha, device='cuda'):
    """PyTorch version: GPU-accelerated initial field calculation"""

    # Ensure input tensors are on the GPU
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    permeability_vac = 4 * torch.pi * 1e-7  # Vacuum magnetic permeability

    # Pre-allocate field tensors (created directly on the GPU)
    EX_subloop = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY_subloop = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ_subloop = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX_subloop = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY_subloop = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ_subloop = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # [1] Initial conductivity
    initialfield_EC = model_EC[i0 - 1, j0 - 1, k0]  
    Txmm = L_subloop ** 2

    # [2] Calculate the initial time
    t1_E = 1.13 * permeability_vac * model_EC[i0-1, j0-1, k0-1] * dz[k0 - 1] ** 2
    dt1 = Alpha * dz[k0 - 1] * torch.sqrt(permeability_vac * model_EC[i0-1, j0-1, k0-1] * t1_E / 6)
    t1_H = t1_E + dt1 / 2

    # [4] Calculate the electric field component EX
    i = torch.arange(XI, device=device)
    j = torch.arange(1, YJ, device=device)
    k = torch.arange(1, ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

    # Calculate coord_x, coord_y, coord_z
    coord_x = calculate_coord_x_torch(ii + 1, 1, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 1, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 1, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_E))
    numerator = torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * Txmm * coord_y * (u / coord_r) ** 5
    denominator = 4 * torch.pi * initialfield_EC * torch.exp(u ** 2 / 2)
    EX_subloop[ii, jj, kk, 0] = numerator / denominator

    # [5] Calculate the electric field component EY
    i = torch.arange(1, XI, device=device)
    j = torch.arange(YJ, device=device)
    k = torch.arange(1, ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')
    # Calculate coord_x, coord_y, coord_z
    coord_x = calculate_coord_x_torch(ii + 1, 2, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 2, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 2, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_E))
    numerator = torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * Txmm * coord_x * (u / coord_r) ** 5
    denominator = 4 * torch.pi * initialfield_EC * torch.exp(u ** 2 / 2)
    EY_subloop[ii, jj, kk, 0] = numerator / denominator

    # Set Ez component to zero
    EZ_subloop[1:XI, 1:YJ, :ZK, 0] = 0

    # [6] Calculate the magnetic field component HX
    i = torch.arange(XI + 1, device=device)
    j = torch.arange(YJ, device=device)
    k = torch.arange(ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

    coord_x = calculate_coord_x_torch(ii + 1, 4, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 4, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 4, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_H))
    term = 3 * torch.erf(u) - torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * u * (3 + u ** 2) / torch.exp(
        u ** 2 / 2)
    HX_subloop[ii, jj, kk, 0] = (Txmm * coord_z * coord_x / (4 * torch.pi * coord_r ** 5)) * term

    # [7] Calculate the magnetic field component HY
    i = torch.arange(XI, device=device)
    j = torch.arange(YJ + 1, device=device)
    k = torch.arange(ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

    coord_x = calculate_coord_x_torch(ii + 1, 5, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 5, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 5, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_H))
    term = 3 * torch.erf(u) - torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * u * (3 + u ** 2) / torch.exp(
        u ** 2 / 2)
    HY_subloop[ii, jj, kk, 0] = (Txmm * coord_z * coord_y / (4 * torch.pi * coord_r ** 5)) * term

    # [8] Calculate the magnetic field component HZ
    i = torch.arange(XI, device=device)
    j = torch.arange(YJ, device=device)
    k = torch.arange(ZK + 1, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

    coord_x = calculate_coord_x_torch(ii + 1, 6, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 6, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 6, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)
    coord_r = torch.where(coord_r < 0.001 * dz[k0 - 1], 0.001 * dz[k0 - 1], coord_r)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_H))
    term1 = (2 * coord_z ** 2 - torch.sqrt(coord_x ** 2 + coord_y ** 2)) * torch.erf(u)
    term2 = (2 * coord_z ** 2 - torch.sqrt(coord_x ** 2 + coord_y ** 2) * (1 + u ** 2)) * torch.sqrt(
        torch.tensor(2.0, device=device) / torch.pi) / torch.exp(u ** 2 / 2)
    HZ_subloop[ii, jj, kk, 0] = Txmm / (4 * torch.pi * coord_r ** 5) * (term1 - term2)

    return EX_subloop, EY_subloop, EZ_subloop, HX_subloop, HY_subloop, HZ_subloop, t1_E, t1_H