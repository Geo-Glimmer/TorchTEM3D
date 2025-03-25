# TorchTEM3D
TorchTEM3D, a time-domain finite difference forward simulation platform for 3D transient electromagnetics. Can quickly calculate dBz/dt, and can quickly calculate sensitivity matrix based on automatic differentiation

Install
	• Create a new environment
conda create -n TorchTEM3D python=3.12
	• Install PyTorch
 pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
