# Use the official CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    ca-certificates \
    libopenblas-dev \
    cmake \
    build-essential \
    libffi-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.10.9 conda && \
    /opt/conda/bin/conda clean -ya

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH
RUN pip install --upgrade "pip<24.1"
COPY environment.yml .
COPY requirements.txt .
RUN conda env create -f environment.yml
RUN pip install -r requirements.txt
RUN pip3 install --no-build-isolation pycocotools
RUN pip3 install --no-build-isolation pyyaml

# # Copy the project files into the container
# COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Copy all necessary folders to /workspace in the container
COPY . .

# Change directory to install projectaria_tools
WORKDIR /workspace/projectaria_tools
RUN python3 -m pip install projectaria-tools'[all]'

# set working dir
WORKDIR /workspace
# Install PyTorch and torchvision with the specified CUDA version
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

# # # Clone and install MinkowskiEngine
WORKDIR /workspace/third_party
RUN git clone "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas


WORKDIR /workspace/third_party
# Clone and build ScanNet Segmentator
RUN git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make

# # # Install pointnet2
WORKDIR /workspace/third_party/pointnet2
RUN python setup.py install

# # Go back to the root directory and install pytorch-lightning
WORKDIR /workspace
RUN pip3 install fvcore
RUN pip3 install pytorch-lightning==1.7.2
RUN pip3 install open3d
RUN pip3 install hydra-core
RUN pip3 install torch_geometric
RUN pip3 install openai==0.28
RUN pip3 install git+https://github.com/openai/CLIP.git
# # Install the Mask3D package
RUN pip3 install .

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]
