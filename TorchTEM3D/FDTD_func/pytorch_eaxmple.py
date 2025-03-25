import torch

def sub_initial_field_torch(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Alpha, model_EC, device='cuda'):
    """PyTorch 版：通过子环叠加计算初始场 (GPU 加速)"""

    print('Initial field is calculating...')

    # 确保输入参数为张量并移至 GPU
    L_loop = torch.tensor(L_loop, dtype=torch.float32, device=device)
    n_subloop = torch.tensor(n_subloop, dtype=torch.int32, device=device)
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)
    Alpha = torch.tensor(Alpha, dtype=torch.float32, device=device)

    # 初始化场张量 (直接分配在 GPU 上)
    EX = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # 计算子环参数（确保所有操作为张量操作）
    L_subloop = L_loop / torch.sqrt(n_subloop.float())
    n_side = int(torch.sqrt(n_subloop.float()).item())  # 转换为 Python int

    # 生成子环坐标网格
    subgrid = torch.linspace(-(L_loop - L_subloop) / 2, (L_loop - L_subloop) / 2, n_side, device=device)
    subloop_x, subloop_y = torch.meshgrid(subgrid, subgrid, indexing='ij')

    # 遍历所有子环
    for i in range(n_side):
        for j in range(n_side):
            # 调用子环计算函数（确保返回张量）
            EX_sub, EY_sub, EZ_sub, HX_sub, HY_sub, HZ_sub, t1_E, t1_H = sub_initial_field_subloop_torch(
                i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                subloop_x[i, j], subloop_y[i, j], L_subloop, Alpha, device
            )


            # 累加场分量
            EX += EX_sub
            EY += EY_sub
            EZ += EZ_sub
            HX += HX_sub
            HY += HY_sub    
            HZ += HZ_sub
            
            # del EX_sub, EY_sub, EZ_sub, HX_sub, HY_sub, HZ_sub
            # torch.cuda.empty_cache()
    print('Initial field calculation finished.')

    return EX, EY, EZ, HX, HY, HZ, t1_E, t1_H


def sub_initial_field_subloop_torch(i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                                    subloop_x, subloop_y, L_subloop, Alpha, device='cuda'):
    """PyTorch 版：GPU 加速的初始场计算"""

    # 确保输入张量在 GPU 上
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    permeability_vac = 4 * torch.pi * 1e-7  # 真空磁导率

    # 预分配场张量（直接创建在 GPU 上）
    EX_subloop = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY_subloop = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ_subloop = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX_subloop = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY_subloop = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ_subloop = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # [1] 初始电导率
    initialfield_EC = model_EC[i0 - 1, j0 - 1, k0]  # 注意索引转换
    Txmm = L_subloop ** 2

    # [2] 计算初始时间
    t1_E = 1.13 * permeability_vac * model_EC[i0-1, j0-1, k0-1] * dz[k0 - 1] ** 2
    dt1 = Alpha * dz[k0 - 1] * torch.sqrt(permeability_vac * model_EC[i0-1, j0-1, k0-1] * t1_E / 6)
    t1_H = t1_E + dt1 / 2

    # [4] 计算电场分量 EX
    i = torch.arange(XI, device=device)
    j = torch.arange(1, YJ, device=device)
    k = torch.arange(1, ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

    # 计算 coord_x, coord_y, coord_z
    coord_x = calculate_coord_x_torch(ii + 1, 1, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 1, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 1, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_E))
    numerator = torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * Txmm * coord_y * (u / coord_r) ** 5
    denominator = 4 * torch.pi * initialfield_EC * torch.exp(u ** 2 / 2)
    EX_subloop[ii, jj, kk, 0] = numerator / denominator
    
    
    # [5] 计算电场分量 EY
    i = torch.arange(1, XI, device=device)
    j = torch.arange(YJ, device=device)
    k = torch.arange(1, ZK, device=device)
    ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')
    # 计算 coord_x, coord_y, coord_z
    coord_x = calculate_coord_x_torch(ii + 1, 2, i0, dx, device) + subloop_x
    coord_y = calculate_coord_y_torch(jj + 1, 2, j0, dy, device) + subloop_y
    coord_z = calculate_coord_z_torch(kk + 1, 2, k0, dz, device)
    coord_r = torch.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)

    u = coord_r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * initialfield_EC / (1e7 * t1_E))
    numerator = torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * Txmm * coord_x * (u / coord_r) ** 5
    denominator = 4 * torch.pi * initialfield_EC * torch.exp(u ** 2 / 2)
    EY_subloop[ii, jj, kk, 0] = numerator / denominator

    # Ez分量置零
    EZ_subloop[1:XI, 1:YJ, :ZK, 0] = 0

    # [6] 计算磁场分量 HX
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

    # [7] 计算磁场分量 HY
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

    # [8] 计算磁场分量 HZ
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




def calculate_coord_x_torch( x_matlab_tensor,m, i0, dx_tensor, device):
    """x坐标计算 (输入为MATLAB索引)"""
    # 确保输入是 PyTorch 张量
    if not isinstance(x_matlab_tensor, torch.Tensor):
        x_matlab_tensor = torch.tensor(x_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dx_tensor, torch.Tensor):
        dx_tensor = torch.tensor(dx_tensor, dtype=torch.float32, device=device)

    # 初始化结果张量
    coord_x_tensor = torch.zeros_like(x_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dx = torch.cumsum(dx_tensor, dim=0)

    # 处理 m ∈ [2, 3, 4] 的情况
    if m in [2,  3]:
        mask = x_matlab_tensor <= i0
        # 计算所有小于等于 i0 的部分和
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # 计算所有大于 i0 的部分和
        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1] - dx_tensor[indices_greater_i0]

    elif m in [4]:
        mask = x_matlab_tensor <= i0
        # 计算所有小于等于 i0 的部分和
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # 计算所有大于 i0 的部分和
        indices_greater_i0 = (x_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1]

    else:
        mask = x_matlab_tensor <= i0
        # 计算所有小于等于 i0 的部分和
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # 计算所有大于 i0 的部分和
        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * (dx_tensor[indices_less_i0] + dx_tensor[i0 - 1]) + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 - 0.5 * dx_tensor[indices_greater_i0] + 0.5 * dx_tensor[i0 - 1]

    return coord_x_tensor


def calculate_coord_y_torch(y_matlab_tensor, m, j0, dy, device):
    """y坐标计算 (输入为MATLAB索引)"""
    # 确保输入是 PyTorch 张量
    if not isinstance(y_matlab_tensor, torch.Tensor):
        y_matlab_tensor = torch.tensor(y_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dy, torch.Tensor):
        dy = torch.tensor(dy, dtype=torch.float32, device=device)

    # 初始化结果张量
    coord_y_tensor = torch.zeros_like(y_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dy = torch.cumsum(dy, dim=0)

    # 处理 m ∈ [1, 3, 5] 的情况
    if m in [1, 3]:
        mask = y_matlab_tensor <= j0
        # 计算所有小于等于 j0 的部分和
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # 计算所有大于 j0 的部分和
        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1] - dy[indices_greater_j0]
    elif m in [5]:
        mask = y_matlab_tensor <= j0
        # 计算所有小于等于 j0 的部分和
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # 计算所有大于 j0 的部分和
        indices_greater_j0 = (y_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1]


    else:
        mask = y_matlab_tensor <= j0
        # 计算所有小于等于 j0 的部分和
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]
        # 计算所有大于 j0 的部分和
        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * (dy[indices_less_j0] + dy[j0 - 1]) + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 - 0.5 * dy[indices_greater_j0] + 0.5* dy[j0 - 1]

    return coord_y_tensor

def calculate_coord_z_torch(z_matlab_tensor, m, k0, dz_tensor,device):
    """z坐标计算 (输入为MATLAB索引)"""
    if not isinstance(z_matlab_tensor, torch.Tensor):
        z_matlab_tensor = torch.tensor(z_matlab_tensor, dtype=torch.float32,device=device)
    if not isinstance(dz_tensor, torch.Tensor):
        dz_tensor = torch.tensor(dz_tensor, dtype=torch.float32,device=device)

    # Initialize the result tensor
    coord_z_tensor = torch.zeros_like(z_matlab_tensor, dtype=torch.float32,device=device)
    cumsum_dz = torch.cumsum(dz_tensor, dim=0)

    # Handle the case where m is in [3, 4, 5]
    if m in [3, 4, 5]:
        mask = z_matlab_tensor <= k0
        # 计算所有小于等于k0的部分和
        indices_less_k0 = (z_matlab_tensor[mask]-1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]
        # 计算所有大于k0的部分和
        indices_greater_k0 = (z_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0-1]

        coord_z_tensor[mask] = -(sum_dz_less_k0 + 0.5 * dz_tensor[indices_less_k0])
        coord_z_tensor[~mask] = sum_dz_greater_k0 - 0.5 * dz_tensor[indices_greater_k0]

    else:
        mask = z_matlab_tensor <= k0
        # 计算所有小于等于k0的部分和
        indices_less_k0 = (z_matlab_tensor[mask] - 1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]
        # 计算所有大于k0的部分和
        indices_greater_k0 = (z_matlab_tensor[~mask]-2).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0 - 1]

        coord_z_tensor[mask] = -(sum_dz_less_k0 + dz_tensor[indices_less_k0])
        coord_z_tensor[~mask] = sum_dz_greater_k0
    return coord_z_tensor


def sub_iteration_torch(
        i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
        iter_n_max, Alpha, model_EC_2, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,
        device="cuda"
):
    """PyTorch GPU 加速版瞬态电磁场迭代计算 (完整维度修正版)"""

    # ============================= 参数预处理 =============================
    # 转换 MATLAB 1-based 索引到 Python 0-based
    i0 -= 1
    j0 -= 1
    k0 -= 1
    Rx_st -= 1
    Rx_end -= 1

    # 接收器数量
    Rx_size = Rx_end - Rx_st + 1

    # 转换为 PyTorch 张量并移至 GPU
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    # 调整场分量张量维度 [time, x, y, z]
    EX = EX.permute(3, 0, 1, 2)
    EY = EY.permute(3, 0, 1, 2)
    EZ = EZ.permute(3, 0, 1, 2)
    HX = HX.permute(3, 0, 1, 2)
    HY = HY.permute(3, 0, 1, 2)
    HZ = HZ.permute(3, 0, 1, 2)


    # ============================= 电导率节点计算 =============================
    EC_x = torch.zeros(XI, YJ, ZK, device=device)
    EC_y = torch.zeros(XI, YJ, ZK, device=device)
    EC_z = torch.zeros(XI, YJ, ZK, device=device)
    
    # model_EC_2 = model_EC.clone()
    
    # EC_x (公式13)
    i, j, k = torch.meshgrid(
        torch.arange(XI, device=device),
        torch.arange(1, YJ, device=device),  # 对应 y_valid
        torch.arange(1, ZK, device=device),  # 对应 z_valid
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
    


    # EC_y (公式15)
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  # x_valid
        torch.arange(YJ, device=device),
        torch.arange(1, ZK, device=device),  # z_valid
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
    



    # EC_z (公式17)
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  # x_valid
        torch.arange(1, YJ, device=device),  # y_valid
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

    # ============================= 时间迭代初始化 =============================
    permeability_vac = 4 * torch.pi * 1e-7
    t_iteration_E = torch.zeros(iter_n_max, device=device)
    t_iteration_H = torch.zeros(iter_n_max, device=device)
    DBZ_Rx = torch.zeros(iter_n_max, Rx_size, device=device)


    t_iteration_E[0] = t1_E
    t_iteration_H[0] = t1_H
    dt = torch.zeros(2, device=device)  # dt[0]=当前, dt[1]=下一步
    dt[0] = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_EC_2[i0, j0, k0] * t1_E / 6)
    # 初始时刻的接收器数据
    rx_slice = slice(Rx_st, Rx_end + 1)  # 接收器位置切片
    rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

    dEx_dy = (EX[0, i0, rx_slice_plus_1, k0 + 1] - EX[0, i0, rx_slice, k0 + 1]) / dy[rx_slice]
    dEy_dx = (EY[0, i0 + 1, rx_slice, k0 + 1] - EY[0, i0, rx_slice, k0 + 1]) / dx[i0]
    DBZ_Rx[0] = dEx_dy - dEy_dx


    # ============================= 主迭代循环 =============================
    iter_n = 1  # Python 0-based 索引
    while iter_n < iter_n_max:

  
        # with torch.no_grad():

        # 更新时间步
        # t_iteration_E[iter_n] = t_iteration_E[iter_n - 1] + dt[0]
        dt[1] = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_EC_2[i0, j0, k0] * (t_iteration_E[iter_n - 1] + dt[0]) / 6)
        t_iteration_H[iter_n] = t_iteration_H[iter_n - 1] + (dt[0] + dt[1]) / 2
        gamma = (4 / permeability_vac) * (dt[0] / dz[k0]) ** 2

        # --------------------------- 电场更新 ---------------------------
        # Ex 更新 (公式12)
        i, j, k = torch.meshgrid(
            torch.arange(XI, device=device),  # x_valid
            torch.arange(1, YJ, device=device),  # y_valid
            torch.arange(1, ZK, device=device),
            indexing='ij'
        )
        term1 = (2 * gamma - EC_x[i, j, k] * dt[0]) / (2 * gamma + EC_x[i, j, k] * dt[0])
        term2 = 4 * dt[0] / (2 * gamma + EC_x[i, j, k] * dt[0])

        # 计算梯度项 (严格对齐维度)
        dHZ_dy = (HZ[0, i, j, k] - HZ[0, i, j-1, k]) / (dy[j-1] + dy[j])
        dHY_dz = (HY[0, i, j, k] - HY[0, i, j, k-1]) / (dz[k-1] + dz[k])
        EX[1, i, j, k] =term1 * EX[0,i, j, k] + term2 * (dHZ_dy - dHY_dz)

        # Ey 更新 (公式14)
        i, j, k = torch.meshgrid(
            torch.arange(1, XI, device=device),  # x_valid
            torch.arange(YJ, device=device),  # y_valid
            torch.arange(1, ZK, device=device),
            indexing='ij'
        )
        term1 = (2 * gamma - EC_y[i, j, k] * dt[0]) / (2 * gamma + EC_y[i, j, k] * dt[0])
        term2 = 4 * dt[0] / (2 * gamma + EC_y[i, j, k] * dt[0])
        dHX_dz = (HX[0, i, j, k] - HX[0, i, j, k-1]) / (dz[k-1] + dz[k])
        dHZ_dx = (HZ[0, i, j, k] - HZ[0, i-1, j, k]) / (dx[i-1] + dx[i])
        EY[1, i, j, k] = term1 * EY[0,i, j, k] +  term2 * (dHX_dz - dHZ_dx)


        # Ez 更新 (公式16)
        i, j, k = torch.meshgrid(
            torch.arange(1, XI, device=device),  # x_valid
            torch.arange(1, YJ, device=device),  # y_valid
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        term1 = (2 * gamma - EC_z[i, j, k] * dt[0]) / (2 * gamma + EC_z[i, j, k] * dt[0])
        term2 = 4 * dt[0] / (2 * gamma + EC_z[i, j, k] * dt[0])
        dHY_dx = (HY[0, i, j, k] - HY[0, i-1, j, k]) / (dx[i-1] + dx[i])
        dHX_dy = (HX[0, i, j, k] - HX[0, i, j-1, k]) / (dy[j-1] + dy[j])
        EZ[1, i, j, k] = term1  * EZ[0,i, j, k] + term2 * (dHY_dx - dHX_dy)
    
        # ------------------------ Dirichlet 边界条件 ------------------------
        # EX 边界
        EX[1, :, [0, -1], :] = 0
        EX[1, :, :, [0, -1]] = 0

        # EY 边界
        EY[1, [0, -1], :, :] = 0
        EY[1, :, :, [0, -1]] = 0

        # EZ 边界
        EZ[1, [0, -1], :, :] = 0
        EZ[1, :, [0, -1], :] = 0

        # --------------------------- 磁场更新 ---------------------------
        i, j, k = torch.meshgrid(
            torch.arange(XI+1, device=device),  # x_valid
            torch.arange(YJ, device=device),  # y_valid
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        # Hx 更新 (公式18)
        dEY_dz = (EY[1,i, j, k + 1] - EY[1,i, j, k]) / dz[k]
        dEZ_dy = (EZ[1,i, j + 1, k] - EZ[1,i, j, k]) / dy[j]
        HX[1,i, j, k] = HX[0,i, j, k] + (dt[0] + dt[1]) / (2 * permeability_vac) * (dEY_dz   -dEZ_dy)

        i, j, k = torch.meshgrid(
            torch.arange(XI, device=device),  # x_valid
            torch.arange(YJ+1, device=device),  # y_valid
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        # Hy 更新 (公式19)
        dEZ_dx = (EZ[1,i + 1, j, k] - EZ[1,i, j, k]) / dx[i]
        dEX_dz = (EX[1,i, j, k + 1] - EX[1,i, j, k]) / dz[k]
        HY[1,i, j, k] = HY[0,i, j, k] + (dt[0] + dt[1]) / (2 * permeability_vac) * (dEZ_dx - dEX_dz) 

        # Hz 更新 (公式20)
        HZ[:, :, 0, 0] = 0
        HZ[:, :, -1, 0] = 0


        i,j = torch.meshgrid(
            torch.arange(XI, device=device),  # y_valid
            torch.arange(YJ, device=device),  # x_valid
            indexing='ij'
        )
        # 从底部向上更新
        for k in range(ZK, k0, -1):  # MATLAB ZK+1 → Python ZK (0-based)
            dHX_dx = (HX[1, i + 1, j, k - 1] - HX[1, i, j, k - 1]) / dx[i]
            dHY_dy = (HY[1, i, j + 1, k - 1] - HY[1, i, j, k - 1]) / dy[j]
            HZ[1,i, j, k-1] =  HZ[1,i, j, k] + dz[k - 1]  * (dHX_dx + dHY_dy)


        m,n = torch.meshgrid(
            torch.arange(YJ, device=device),  # x_valid
            torch.arange(XI, device=device),  # y_valid
            indexing='ij'
        )
        # 从顶部向下更新
        for k in range(1, k0):
            dHX_dx = (HX[1, n + 1, m, k - 1] - HX[1, n, m, k - 1]) / dx[n]
            dHY_dy = (HY[1, n, m + 1, k - 1] - HY[1, n, m, k - 1]) / dy[m]
            HZ[1,n, m, k] = HZ[1,n, m, k - 1] - dz[k - 1] * (dHX_dx + dHY_dy)


        # ------------------------ 接收器数据记录 ------------------------
        rx_slice = slice(Rx_st, Rx_end + 1)  # 接收器位置切片
        rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

        dEx_dy = (EX[1, i0, rx_slice_plus_1, k0+1] - EX[1, i0, rx_slice, k0+1]) / dy[rx_slice]
        dEy_dx = (EY[1, i0 + 1, rx_slice, k0+1] - EY[1, i0, rx_slice, k0+1]) / dx[i0]
        DBZ_Rx[iter_n] = dEx_dy - dEy_dx

        # [3.4] 滚动时间步 ---------------------------------------------
        EX[0], EY[0], EZ[0] = EX[1], EY[1], EZ[1]
        HX[0], HY[0], HZ[0] = HX[1], HY[1], HZ[1]
        dt[0] = dt[1]

        iter_n += 1
    # ============================= 结果输出 =============================
    print(f'Computation finished. Final dt: {dt[0].item():.3e} sec')
    return t_iteration_H, DBZ_Rx


# 示例参数
import numpy as np
from discretize import TensorMesh

# Define the mesh, model and mapping
dh = 50
XI = 24
YJ = 24
ZK = 24
pad_num = 6
core_num_x = XI - 2* pad_num
core_num_y = YJ - 2* pad_num
core_num_z = ZK - 2* pad_num

padd_layer = 2
# 定义网格大小
# dx X dy X dz   length = dh
hx = [(dh, pad_num,-padd_layer),(dh,core_num_x),(dh,pad_num,padd_layer)]
hy = [(dh, pad_num,-padd_layer),(dh,core_num_y),(dh,pad_num,padd_layer)]
hz = [(dh, pad_num,-padd_layer),(dh,core_num_x),(dh,pad_num,padd_layer)]
# mesh = TensorMesh([hx, hy, hz], origin="CCC")
mesh = TensorMesh([hx, hy, hz])
x = mesh.h[0]
y = mesh.h[1]
z = mesh.h[2]
# mesh.nodes_z 网格的宽度

# x = [1280,640,320,160,80,40,20,20,20,20,20,40,80,160,320,640,1280]
# y = [1280,640,320,160,80,40,20,10,10,15,20,20,40,80,160,320,640,1280]
# z = [1920,960,480,240,120,60,30,15,15,15,15,15,15,30,90,270,810,2430]

device = 'cuda'
model_EC = torch.ones((XI, YJ, ZK))  # 电导率模型 (XI x YJ x ZK)
model_EC_2 = torch.ones((XI, YJ, ZK))  # 电导率模型 (XI x YJ x ZK)

model_EC[:,:,0:7] = model_EC[:,:,0:7]*3e-4
model_EC[:,:,7:-1] = model_EC[:,:,7:-1]*1e-2

model_EC_2[:,:,0:7] = model_EC[:,:,0:7]*3e-4
model_EC_2[:,:,7:-1] = model_EC[:,:,7:-1]*1e-2

model_EC = torch.tensor(model_EC, device='cuda', requires_grad=False)
model_EC_2 = torch.tensor(model_EC_2, device='cuda', requires_grad=False)

L_loop=300
n_subloop=64


Alpha=0.7

Rx_st=11
Rx_end=11
i0,j0,k0 = 11, 11, 7
iter_n_max = 1000

def forward_3DTEM(L_loop=100,n_subloop=64,i0=12, j0=12, k0=8,
                  XI=XI, YJ=YJ, ZK=ZK,dx=x, dy=y, dz=z,Alpha=0.7,
                  model_EC=model_EC,model_EC_2 = model_EC_2,Rx_st=Rx_st, Rx_end=Rx_end,iter_n_max=iter_n_max,device=device):
    # 调用主函数
    EX, EY, EZ, HX, HY, HZ, t1_E, t1_H = sub_initial_field_torch(
        L_loop,
        n_subloop,
        i0, j0, k0,
        XI, YJ, ZK,
        dx, dy, dz,
        Alpha,
        model_EC_2,
        device = device
    )
    t_H, dBz = sub_iteration_torch(i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
                             iter_n_max, Alpha, model_EC, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,device = device)
    
    return t_H, dBz

    
import time
strat_time = time.time()    
    
t_H,dBz = forward_3DTEM(L_loop,n_subloop,i0, j0, k0,
                  XI, YJ, ZK,x, y, z, Alpha,
                  model_EC, model_EC_2, 
                  Rx_st, Rx_end,
                  iter_n_max)

end_time = time.time()    
print(f"time:{end_time-strat_time}")

# # print(t_H)
# # print(dBz)
# combined_data = np.column_stack((t_H.cpu().numpy(), dBz.cpu().numpy()))
# # 保存为 CSV 文件
# np.savetxt('./测试/t_H_dBz.csv', combined_data, delimiter=',', header='t_H,dBz', comments='')


# grad = torch.autograd.grad(outputs=dBz.sum(), inputs=model_EC)
# print(grad)

# dBz_real = torch.randn(t_H.shape).cuda()
# # loss = abs(dBz_real-dBz).sum()  # 示例：以 t_H 的第一个元素作为目标损失
# dBz.backward(torch.ones_like(dBz)  # 权重张量与 y 形状相同
# )  # 计算梯度

# # 获取 model_EC 的梯度
# grad_model_EC = model_EC.grad.cpu().numpy()
# print(np.sum(grad_model_EC))

