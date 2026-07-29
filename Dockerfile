# Swapped to a modern, verified CUDA ecosystem base image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Setup NRSC Network Proxy Arguments
ENV http_proxy=http://rrscnorth:NRSC%40User@192.168.0.9:8080
ENV https_proxy=http://rrscnorth:NRSC%40User@192.168.0.9:8080
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and the core binary dependencies for cfgrib/eccodes
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    g++ \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

# Upgrade pip and install tensorflow[and-cuda], core dependencies, and missing packages
RUN pip install --no-cache-dir --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org && \
    pip install --no-cache-dir \
    tensorflow[and-cuda]==2.16.1 \
    tf_keras \
    tqdm \
    xarray \
    cfgrib \
    netcdf4 \
    pyshp \
    matplotlib \
    pandas \
    scipy \
    requests \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org

# Fallback requirement file processing if you keep extra packages there
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org; fi

COPY . .

# Use standard execute entrypoint mapping to accept custom workspace paths smoothly
ENTRYPOINT ["python3"]