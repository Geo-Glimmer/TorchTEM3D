import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# Globally set the font to Arial
plt.rcParams['font.family'] = 'Arial'
# Globally set font and figure parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 40,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30
})

#%%
# Double-body
data_simpeg = np.load("./double-body/grad_simpeg_dBz_double.npy")

grad_simpeg = data_simpeg.reshape(18, 24, 24)
grad_simpeg = np.transpose(grad_simpeg, (2, 1, 0)) 
grad_simpeg = grad_simpeg[:, :, ::-1]
grad_simpeg = grad_simpeg[6:18, 6:18, 0:12]

# Load t_H and dBz data
data = np.load('./double-body/model_double.npy')

num = 7
grad_model = data[6:18, 6:18, num:num+12]

data = np.load('./double-body/grad_torchTEM3D_dBz_double.npy')
grad_pytorch = data[6:18, 6:18, num:num+12]

name = 'double-body'

#%%
# Visualization parameter settings
slice_idz = 4  # Z = -200 m
figure_name = 'Sensitivity-Matrix-Z-200m '
slice_idz = 8  # Z = -400 m
figure_name = 'Sensitivity-Matrix-Z-800m '

#%%
# Create a 12x12 grid
X = np.linspace(-300, 300, 12)  
Y = np.linspace(-300, 300, 12)
Z = np.linspace(-600, 0, 12)

# Create canvas
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

# Encapsulate the plotting function
def plot_slice(ax, x, y, data, title, y_label=0, imshow_figure=0):
    
    if imshow_figure == 1:
        norm = Normalize(vmin=data.min(), vmax=data.max())
        cmap = plt.cm.RdYlBu
        im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, rasterized=True)
    else:
        im = ax.contourf(x, y, data, cmap='plasma', aspect='auto', 
                         origin='lower')
    
    # Set X-axis position
    ax.xaxis.tick_top()            # Move ticks to the top
    ax.xaxis.set_label_position('top')  # Move labels to the top
    ax.tick_params(axis='x', which='both', bottom=False)  # Hide bottom ticks
    
    # ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_xticks([-300, -150, 0, 150, 300])  # Set X-axis ticks

    if y_label == 1:
        ax.set_ylabel('Y (m)')
    
    # Use a loop to set the line width for all outer borders
    for spine in ax.spines.values():
        spine.set_linewidth(4)  # Set the line width of the outer border to 4
        
    return im

#%%  
# Plot XZ slices

x, y = np.meshgrid(X, Y)

im = plot_slice(axes[0], x, y, grad_model[:, :, slice_idz], 
                'Custom Gradient', y_label=1, imshow_figure=1)

plot_slice(axes[1], x, y, grad_simpeg[:, :, slice_idz], 
           'TorchTEM3D Gradient')
plot_slice(axes[2], x, y, grad_pytorch[:, :, slice_idz], 
           'PyTorch Gradient')

#%%
# Add a color bar; a shared color bar can be achieved through the same figure
mappable = plt.cm.ScalarMappable(cmap='plasma')
mappable.set_array(grad_simpeg[:, :, slice_idz])
cbar = fig.colorbar(mappable, ax=axes, fraction=0.04, pad=-0.11)  # Add a color bar
cbar.ax.tick_params(labelsize=28)      # Set the size of the color bar tick labels

#%%
plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for the color bar

# Save as a high-resolution image and vector graphic
os.chdir(name)
plt.savefig(f"grad_{figure_name}.png", dpi=400, bbox_inches='tight')

plt.show()