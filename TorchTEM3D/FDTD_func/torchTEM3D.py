from sub_initial_field import *
from sub_iteration import *

#%%
# calculate dbz/dt
def forward_3DTEM(L_loop,n_subloop,i0, j0, k0,
                  XI, YJ, ZK,dx, dy, dz,Alpha,
                  model_EC,Rx_st, Rx_end,iter_n_max,device='cuda'):
    '''
    Parameters
    ----------
    L_loop : Size of the transmitting loop (side length, in meters).
        TYPE, optional
        DESCRIPTION. The default is 100.

    n_subloop : Number of magnetic dipoles in the loop, typically fixed at 64.
        TYPE, optional
        DESCRIPTION. The default is 64.

    (i0, j0, k0) : Position of the transmitting loop (grid indices, located at the center of the grid).
    i0 : TYPE, optional
        DESCRIPTION. The default is 12.
    j0 : TYPE, optional
        DESCRIPTION. The default is 12.
    k0 : TYPE, optional
        DESCRIPTION. The default is 8.

    (XI, YJ, ZK) : Number of tensor staggered grids in the x, y, and z directions.
    XI : TYPE, optional
        DESCRIPTION. The default is 17.
    YJ : TYPE, optional
        DESCRIPTION. The default is 17.
    ZK : TYPE, optional
        DESCRIPTION. The default is 18.

    (dx, dy, dz) : Length of the grid edges in the x, y, and z directions.
    dx : TYPE, optional
        DESCRIPTION. The default is x.
    dy : TYPE, optional
        DESCRIPTION. The default is y.
    dz : TYPE, optional
        DESCRIPTION. The default is z.

    Alpha : Coefficient in the time step size calculation equation.
        TYPE, optional
        DESCRIPTION. The default is 0.7.

    model_EC : Conductivity value of the anomalous body.
        TYPE, optional
        DESCRIPTION. The default is model_EC.

    Rx_st : Initial position of the receiver.
        TYPE, optional
        DESCRIPTION. The default is Rx_st.

    Rx_end : Final position of the receiver (inclusive).
        TYPE, optional
        DESCRIPTION. The default is Rx_end.

    iter_n_max : Number of iterations (affects observation time).
        TYPE, optional
        DESCRIPTION. The default is iter_n_max.

    device : Computational device; "cuda" for GPU computation, "cpu" for CPU computation.
        TYPE, optional
        DESCRIPTION. The default is device.

    Returns
    -------
    TYPE
        torch.
    TYPE
        torch.
'''
    # Initial field calculation
    EX, EY, EZ, HX, HY, HZ, t1_E, t1_H = sub_initial_field_torch(
        L_loop,
        n_subloop,
        i0, j0, k0,
        XI, YJ, ZK,
        dx, dy, dz,
        Alpha,
        model_EC.detach(),
        device = device
    )
    
    # Iterative computation
    t_H, dBz = sub_iteration_torch(i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
                             iter_n_max, Alpha, model_EC, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,device = device)
    
    return t_H, dBz