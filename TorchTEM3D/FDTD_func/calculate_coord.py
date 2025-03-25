import torch

def calculate_coord_x_torch(x_matlab_tensor, m, i0, dx_tensor, device):
    """Calculate x-coordinate"""
    # Ensure inputs are PyTorch tensors
    if not isinstance(x_matlab_tensor, torch.Tensor):
        x_matlab_tensor = torch.tensor(x_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dx_tensor, torch.Tensor):
        dx_tensor = torch.tensor(dx_tensor, dtype=torch.float32, device=device)

    # Initialize result tensor
    coord_x_tensor = torch.zeros_like(x_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dx = torch.cumsum(dx_tensor, dim=0)

    # Handle cases where m ∈ [2, 3, 4]
    if m in [2, 3]:
        mask = x_matlab_tensor <= i0
        # Calculate partial sums for all indices less than or equal to i0
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # Calculate partial sums for all indices greater than i0
        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1] - dx_tensor[indices_greater_i0]

    elif m in [4]:
        mask = x_matlab_tensor <= i0
        # Calculate partial sums for all indices less than or equal to i0
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # Calculate partial sums for all indices greater than i0
        indices_greater_i0 = (x_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1]

    else:
        mask = x_matlab_tensor <= i0
        # Calculate partial sums for all indices less than or equal to i0
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # Calculate partial sums for all indices greater than i0
        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * (dx_tensor[indices_less_i0] + dx_tensor[i0 - 1]) + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 - 0.5 * dx_tensor[indices_greater_i0] + 0.5 * dx_tensor[i0 - 1]

    return coord_x_tensor


def calculate_coord_y_torch(y_matlab_tensor, m, j0, dy, device):
    """Calculate y-coordinate"""
    # Ensure inputs are PyTorch tensors
    if not isinstance(y_matlab_tensor, torch.Tensor):
        y_matlab_tensor = torch.tensor(y_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dy, torch.Tensor):
        dy = torch.tensor(dy, dtype=torch.float32, device=device)

    # Initialize result tensor
    coord_y_tensor = torch.zeros_like(y_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dy = torch.cumsum(dy, dim=0)

    # Handle cases where m ∈ [1, 3, 5]
    if m in [1, 3]:
        mask = y_matlab_tensor <= j0
        # Calculate partial sums for all indices less than or equal to j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # Calculate partial sums for all indices greater than j0
        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1] - dy[indices_greater_j0]
    elif m in [5]:
        mask = y_matlab_tensor <= j0
        # Calculate partial sums for all indices less than or equal to j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # Calculate partial sums for all indices greater than j0
        indices_greater_j0 = (y_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1]

    else:
        mask = y_matlab_tensor <= j0
        # Calculate partial sums for all indices less than or equal to j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # Calculate partial sums for all indices greater than j0
        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * (dy[indices_less_j0] + dy[j0 - 1]) + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 - 0.5 * dy[indices_greater_j0] + 0.5 * dy[j0 - 1]

    return coord_y_tensor


def calculate_coord_z_torch(z_matlab_tensor, m, k0, dz_tensor, device):
    """Calculate z-coordinate"""
    # Ensure inputs are PyTorch tensors
    if not isinstance(z_matlab_tensor, torch.Tensor):
        z_matlab_tensor = torch.tensor(z_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dz_tensor, torch.Tensor):
        dz_tensor = torch.tensor(dz_tensor, dtype=torch.float32, device=device)

    # Initialize result tensor
    coord_z_tensor = torch.zeros_like(z_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dz = torch.cumsum(dz_tensor, dim=0)

    # Handle cases where m ∈ [3, 4, 5]
    if m in [3, 4, 5]:
        mask = z_matlab_tensor <= k0
        # Calculate partial sums for all indices less than or equal to k0
        indices_less_k0 = (z_matlab_tensor[mask] - 1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]
        # Calculate partial sums for all indices greater than k0
        indices_greater_k0 = (z_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0 - 1]

        coord_z_tensor[mask] = -(sum_dz_less_k0 + 0.5 * dz_tensor[indices_less_k0])
        coord_z_tensor[~mask] = sum_dz_greater_k0 - 0.5 * dz_tensor[indices_greater_k0]

    else:
        mask = z_matlab_tensor <= k0
        # Calculate partial sums for all indices less than or equal to k0
        indices_less_k0 = (z_matlab_tensor[mask] - 1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]
        # Calculate partial sums for all indices greater than k0
        indices_greater_k0 = (z_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0 - 1]

        coord_z_tensor[mask] = -(sum_dz_less_k0 + dz_tensor[indices_less_k0])
        coord_z_tensor[~mask] = sum_dz_greater_k0

    return coord_z_tensor