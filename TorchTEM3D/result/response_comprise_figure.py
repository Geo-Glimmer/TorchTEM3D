import numpy as np
import matplotlib.pyplot as plt
import os

#%%
# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'

#%%
# 3-layers
data_1D = np.loadtxt("./3-layers/Analytic_dBz_test_simpeg_3layers.csv")

data_simpeg = np.load("./3-layers/simpeg_dBz_simpeg_3layers.npy")

data_custem = np.load("./3-layers/custem_dBz_simpeg_3layers.npy")
t_custem = data_custem[:, 0]
dBz_custem = data_custem[:, 1]

data = np.loadtxt('./3-layers/torchTEM3D_dBz_simpeg_3layers.csv', delimiter=',', skiprows=1)
t_H_loaded = data[:, 0]
dBz_loaded = data[:, 1]
name = '3-layers'

#%%
# 1-layers
data_1D = np.loadtxt("./1-layers/Analytic_dBz_100.csv")

data_simpeg = np.load("./1-layers/simpeg_dBz_100.npy")

data_custem = np.load("./1-layers/custem_dBz_100.npy")
t_custem = data_custem[:, 0]
dBz_custem = data_custem[:, 1]

data = np.loadtxt('./1-layers/torchTEM3D_dBz_100.csv', delimiter=',', skiprows=1)
t_H_loaded = data[:, 0]
dBz_loaded = data[:, 1]
name = '1-layers'

#%%
time_channels = np.logspace(-5, -2, 41)

plt.figure(figsize=(8, 8))  # Adjust to a larger canvas size
# Custom colors, styles, and sizes
colors = ['#1f77b4', '#d62728', '#2ca02c', 'black']
markers = ['o', 's', 'o', '+']
marker_sizes = [10, 10, 8, 12]
linewidths = [3, 4, 2, 4]
linestyles = ['-', ':', '--', '']
markevery_vals = [1, 2, 2, 2]

# Plot curves
simpeg1d_line, = plt.loglog(time_channels, -data_1D, 
                            color=colors[1], linewidth=linewidths[1], linestyle=linestyles[1],
                            marker=markers[1], markersize=marker_sizes[1],
                            markevery=markevery_vals[1], markerfacecolor='white', 
                            markeredgewidth=1.5, label='Analytic')

pytorch_line, = plt.loglog(t_H_loaded, -dBz_loaded, 
                           color=colors[0], linewidth=linewidths[0], linestyle=linestyles[0],
                           label='TorchTEM3D')  # Remove marker parameter

simpeg3d_line, = plt.loglog(time_channels, -data_simpeg, 
                            color=colors[2], linewidth=linewidths[2], linestyle=linestyles[2],
                            marker=markers[2], markersize=marker_sizes[2],
                            markevery=markevery_vals[2], alpha=0.9, label='SimPEG-3D')

custem_line, = plt.loglog(t_custem, -dBz_custem, 
                          color=colors[3], linewidth=linewidths[3], linestyle=linestyles[3],
                          marker=markers[3], markersize=marker_sizes[3],
                          markevery=markevery_vals[3], alpha=0.9, label='custEM-3D')

# Axis settings
plt.xlim(9e-5, 1e-2)

# Labels and title
plt.xlabel('Time (s)', fontsize=26, labelpad=3)
plt.ylabel('dBz/dt (V/m²)', fontsize=26, labelpad=3)

# Tick settings
ax = plt.gca()
ax.tick_params(axis='both', which='both', labelsize=24, length=1, width=1.5)  # Set font size for both major and minor ticks to 20

# Use a loop to set the line width for all outer borders
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Set the line width of the outer border to 2

# Legend settings
legend = plt.legend(
    loc='upper right',
    fontsize=24,
    frameon=False,
    fancybox=False,
    framealpha=0.9,
    edgecolor='#2A2A2A',
    borderpad=0.2,
    handlelength=1.8,
    handletextpad=0.5
)

# Adjust layout
plt.tight_layout(pad=3.0)

# Save as high-resolution image and vector graphic
os.chdir(name)
plt.savefig(f"{name}.png", dpi=400, bbox_inches='tight')

plt.show()