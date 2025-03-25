# TorchTEM3D
TorchTEM3D, a time-domain finite difference forward simulation platform for 3D transient electromagnetics. Can quickly calculate dBz/dt, and can quickly calculate sensitivity matrix based on automatic differentiation

Install
	• Install Anaconda and create a new environment
wget  https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
conda create -n TorchTEM3D python=3.12
	• Install PyTorch
	• pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
