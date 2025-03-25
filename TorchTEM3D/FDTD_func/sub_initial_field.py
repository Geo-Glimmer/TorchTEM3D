import torch
from sub_initial_field_subloop import sub_initial_field_subloop_torch

def sub_initial_field_torch(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Alpha, model_EC, device='cuda'):
    """PyTorch version: Calculate the initial field by superimposing sub-loops (GPU acceleration)"""

    print('Initial field is calculating...')

    # Ensure input parameters are tensors and move them to GPU
    L_loop = torch.tensor(L_loop, dtype=torch.float32, device=device)
    n_subloop = torch.tensor(n_subloop, dtype=torch.int32, device=device)
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)
    Alpha = torch.tensor(Alpha, dtype=torch.float32, device=device)

    # Initialize field tensors (allocated directly on GPU)
    EX = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # Calculate sub-loop parameters (ensure all operations are tensor-based)
    L_subloop = L_loop / torch.sqrt(n_subloop.float())
    n_side = int(torch.sqrt(n_subloop.float()).item())  # Convert to Python int

    # Generate sub-loop coordinate grid
    subgrid = torch.linspace(-(L_loop - L_subloop) / 2, (L_loop - L_subloop) / 2, n_side, device=device)
    subloop_x, subloop_y = torch.meshgrid(subgrid, subgrid, indexing='ij')

    # Iterate over all sub-loops
    for i in range(n_side):
        for j in range(n_side):
            # Call the sub-loop calculation function (ensure returned values are tensors)
            EX_sub, EY_sub, EZ_sub, HX_sub, HY_sub, HZ_sub, t1_E, t1_H = sub_initial_field_subloop_torch(
                i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                subloop_x[i, j], subloop_y[i, j], L_subloop, Alpha, device
            )

            # Accumulate field components
            EX += EX_sub
            EY += EY_sub
            EZ += EZ_sub
            HX += HX_sub
            HY += HY_sub    
            HZ += HZ_sub            

    print('Initial field calculation finished.')

    return EX, EY, EZ, HX, HY, HZ, t1_E, t1_H