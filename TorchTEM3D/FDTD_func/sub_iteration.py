import torch

def sub_iteration_torch(
        i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
        iter_n_max, Alpha, model_EC_2, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,
        device="cuda"
):

    # ============================= Parameter Preprocessing =============================
    i0 -= 1
    j0 -= 1
    k0 -= 1
    Rx_st -= 1
    Rx_end -= 1

    # Number of receivers
    Rx_size = Rx_end - Rx_st + 1

    # Convert to PyTorch tensors and move to GPU
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    # Adjust field component tensor dimensions [time, x, y, z]
    EX = EX.permute(3, 0, 1, 2)
    EY = EY.permute(3, 0, 1, 2)
    EZ = EZ.permute(3, 0, 1, 2)
    HX = HX.permute(3, 0, 1, 2)
    HY = HY.permute(3, 0, 1, 2)
    HZ = HZ.permute(3, 0, 1, 2)


    # ============================= Conductivity Node Calculation =============================
    EC_x = torch.zeros(XI, YJ, ZK, device=device)
    EC_y = torch.zeros(XI, YJ, ZK, device=device)
    EC_z = torch.zeros(XI, YJ, ZK, device=device)
    
    # EC_x 
    i, j, k = torch.meshgrid(
        torch.arange(XI, device=device),
        torch.arange(1, YJ, device=device),  
        torch.arange(1, ZK, device=device),  
        indexing='ij'
    )

    numerator = (
            dy[j-1] * dz[k-1] * model_EC_2[i, j-1, k-1] +
            dy[j ] * dz[k-1] * model_EC_2[i, j , k-1] +
            dy[j-1] * dz[k] * model_EC_2[i, j-1, k] +
            dy[j ] * dz[k ] * model_EC_2[i, j, k]
    )
    denominator = (dy[j-1] + dy[j]) * (dz[k-1] + dz[k])
    EC_x[i,j,k] = numerator / denominator

    # EC_y 
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  
        torch.arange(YJ, device=device),
        torch.arange(1, ZK, device=device),  
        indexing='ij'
    )
    numerator = (
            dx[i - 1] * dz[k - 1] * model_EC_2[i - 1, j, k - 1] +
            dx[i] * dz[k - 1] * model_EC_2[i, j, k - 1] +
            dx[i - 1] * dz[k] * model_EC_2[i - 1, j, k] +
            dx[i] * dz[k] * model_EC_2[i, j, k]
    )
    denominator = (dx[i - 1] + dx[i]) * (dz[k - 1] + dz[k])
    EC_y[i,j,k] = numerator / denominator

    # EC_z 
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  
        torch.arange(1, YJ, device=device),  
        torch.arange(ZK, device=device),
        indexing='ij'
    )
    numerator = (
            dx[i - 1] * dy[j - 1] * model_EC_2[i - 1, j - 1, k] +
            dx[i] * dy[j - 1] * model_EC_2[i, j - 1, k] +
            dx[i - 1] * dy[j] * model_EC_2[i - 1, j, k] +
            dx[i] * dy[j] * model_EC_2[i, j, k]
    )
    denominator = (dx[i - 1] + dx[i]) * (dy[j - 1] + dy[j])
    EC_z[i,j,k] = numerator / denominator

    # ============================= Time Iteration Initialization =============================
    permeability_vac = 4 * torch.pi * 1e-7
    t_iteration_E = torch.zeros(iter_n_max, device=device)
    t_iteration_H = torch.zeros(iter_n_max, device=device)
    DBZ_Rx = torch.zeros(iter_n_max, Rx_size, device=device)

    model_t = model_EC_2[i0, j0, k0].detach()
    t_iteration_E[0] = t1_E
    t_iteration_H[0] = t1_H

    dt0 = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_t * t1_E / 6)
    
    # Initial receiver data
    rx_slice = slice(Rx_st, Rx_end + 1)  # Receiver position slice
    rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

    dEx_dy = (EX[0, i0, rx_slice_plus_1, k0 + 1] - EX[0, i0, rx_slice, k0 + 1]) / dy[rx_slice]
    dEy_dx = (EY[0, i0 + 1, rx_slice, k0 + 1] - EY[0, i0, rx_slice, k0 + 1]) / dx[i0]
    DBZ_Rx[0] = dEx_dy - dEy_dx


    # ============================= Main Iteration Loop =============================
    iter_n = 1  
    while iter_n < iter_n_max:

            # Update time step
            t_iteration_E[iter_n] = t_iteration_E[iter_n - 1] + dt0
            dt1 = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_t * t_iteration_E[iter_n] / 6)
            t_iteration_H[iter_n] = t_iteration_H[iter_n - 1] + (dt0 + dt1) / 2
            gamma = (4 / permeability_vac) * (dt0 / dz[k0]) ** 2

            # --------------------------- Electric Field Update ---------------------------
            # Ex update 
            i, j, k = torch.meshgrid(
                torch.arange(XI, device=device),  
                torch.arange(1, YJ, device=device),  
                torch.arange(1, ZK, device=device),
                indexing='ij'
            )

            term1 = (2 * gamma - EC_x[i, j, k] * dt0) / (2 * gamma + EC_x[i, j, k] * dt0)
            term2 = 4 * dt0 / (2 * gamma + EC_x[i, j, k] * dt0)

            # Calculate gradient terms (strictly align dimensions)
            dHZ_dy = (HZ[0, i, j, k] - HZ[0, i, j-1, k]) / (dy[j-1] + dy[j])
            dHY_dz = (HY[0, i, j, k] - HY[0, i, j, k-1]) / (dz[k-1] + dz[k])
            EX[1, i, j, k] = term1 * EX[0,i, j, k] + term2 * (dHZ_dy - dHY_dz)
            
            # Ey update 
            i, j, k = torch.meshgrid(
                torch.arange(1, XI, device=device),  
                torch.arange(YJ, device=device),  
                torch.arange(1, ZK, device=device),
                indexing='ij'
            )
            
            term1 = (2 * gamma - EC_y[i, j, k] * dt0) / (2 * gamma + EC_y[i, j, k] * dt0)
            term2 = 4 * dt0 / (2 * gamma + EC_y[i, j, k] * dt0)
            dHX_dz = (HX[0, i, j, k] - HX[0, i, j, k-1]) / (dz[k-1] + dz[k])
            dHZ_dx = (HZ[0, i, j, k] - HZ[0, i-1, j, k]) / (dx[i-1] + dx[i])
            EY[1, i, j, k] = term1 * EY[0,i, j, k] +  term2 * (dHX_dz - dHZ_dx)

            # Ez update 
            i, j, k = torch.meshgrid(
                torch.arange(1, XI, device=device),  
                torch.arange(1, YJ, device=device),  
                torch.arange(ZK, device=device),
                indexing='ij'
            )
            term1 = (2 * gamma - EC_z[i, j, k] * dt0) / (2 * gamma + EC_z[i, j, k] * dt0)
            term2 = 4 * dt0 / (2 * gamma + EC_z[i, j, k] * dt0)
            dHY_dx = (HY[0, i, j, k] - HY[0, i-1, j, k]) / (dx[i-1] + dx[i])
            dHX_dy = (HX[0, i, j, k] - HX[0, i, j-1, k]) / (dy[j-1] + dy[j])
            EZ[1, i, j, k] = term1  * EZ[0,i, j, k] + term2 * (dHY_dx - dHX_dy)

            # ------------------------ Dirichlet Boundary Conditions ------------------------
            # EX boundary
            EX[1, :, [0, -1], :] = 0
            EX[1, :, :, [0, -1]] = 0

            # EY boundary
            EY[1, [0, -1], :, :] = 0
            EY[1, :, :, [0, -1]] = 0

            # EZ boundary
            EZ[1, [0, -1], :, :] = 0
            EZ[1, :, [0, -1], :] = 0

            # --------------------------- Magnetic Field Update ---------------------------
            i, j, k = torch.meshgrid(
                torch.arange(XI+1, device=device),  
                torch.arange(YJ, device=device),  
                torch.arange(ZK, device=device),
                indexing='ij'
            )
            # Hx update 
            dEY_dz = (EY[1,i, j, k + 1] - EY[1,i, j, k]) / dz[k]
            dEZ_dy = (EZ[1,i, j + 1, k] - EZ[1,i, j, k]) / dy[j]
            HX[1,i, j, k] = HX[0,i, j, k] + (dt0 + dt1) / (2 * permeability_vac) * (dEY_dz   -dEZ_dy)

            i, j, k = torch.meshgrid(
                torch.arange(XI, device=device),  
                torch.arange(YJ+1, device=device),  
                torch.arange(ZK, device=device),
                indexing='ij'
            )
            
            # Hy update 
            dEZ_dx = (EZ[1,i + 1, j, k] - EZ[1,i, j, k]) / dx[i]
            dEX_dz = (EX[1,i, j, k + 1] - EX[1,i, j, k]) / dz[k]
            HY[1,i, j, k] = HY[0,i, j, k] + (dt0 + dt1) / (2 * permeability_vac) * (dEZ_dx - dEX_dz)

            # Hz update 
            HZ[:, :, 0, 0] = 0
            HZ[:, :, -1, 0] = 0

            i,j = torch.meshgrid(
                torch.arange(XI, device=device),  
                torch.arange(YJ, device=device),  
                indexing='ij'
            )
            # Update from bottom to top
            for k in range(ZK, k0, -1):
                dHX_dx = (HX[1, i+1, j, k-1] - HX[1, i, j, k-1]) / dx[i]
                dHY_dy = (HY[1, i, j+1, k-1] - HY[1, i, j, k-1]) / dy[j]
                
                HZ[1, i, j, k-1] = HZ[1, i, j, k] + dz[k-1] * (dHX_dx + dHY_dy)

            m,n = torch.meshgrid(
                torch.arange(YJ, device=device),  
                torch.arange(XI, device=device),  
                indexing='ij'
            )
            # Update from top to bottom
            for k in range(1, k0):
                dHX_dx = (HX[1, n+1, m, k-1] - HX[1, n, m, k-1]) / dx[n]
                dHY_dy = (HY[1, n, m+1, k-1] - HY[1, n, m, k-1]) / dy[m]
                
                HZ[1, n, m, k] = HZ[1, n, m, k-1].detach() - dz[k-1] * (dHX_dx + dHY_dy)

            # ------------------------ Receiver Data Recording ------------------------
            rx_slice = slice(Rx_st, Rx_end + 1)  # Receiver position slice
            rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

            dEx_dy = (EX[1, i0, rx_slice_plus_1, k0+1] - EX[1, i0, rx_slice, k0+1]) / dy[rx_slice]
            dEy_dx = (EY[1, i0 + 1, rx_slice, k0+1] - EY[1, i0, rx_slice, k0+1]) / dx[i0]
            DBZ_Rx[iter_n] = dEx_dy - dEy_dx

            # [3.4] Roll time steps ---------------------------------------------
            EX[0], EY[0], EZ[0] = EX[1], EY[1], EZ[1]
            HX[0], HY[0], HZ[0] = HX[1], HY[1], HZ[1]
            dt0 = dt1
            
            iter_n += 1

    # ============================= Output Results =============================
    print(f'Computation finished.')
    return t_iteration_H.detach(), DBZ_Rx