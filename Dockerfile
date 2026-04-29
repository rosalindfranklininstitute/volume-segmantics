FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository and install runtime libraries only
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y \
    cuda-libraries-12-6 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python

RUN git clone https://github.com/rosalindfranklininstitute/volume-segmantics.git /opt/volume-segmantics
WORKDIR /opt/volume-segmantics

RUN pip install --no-cache-dir poetry

RUN pip install --no-cache-dir \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

RUN poetry config virtualenvs.create false \
    && poetry install --no-root \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir "opencv-python-headless==4.11.0.86"

WORKDIR /root

CMD ["/bin/bash"]
