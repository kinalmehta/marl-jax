
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG="C.UTF-8" LC_ALL="C.UTF-8"

RUN apt-get update -y && apt-get install -q -y \
    wget \
    curl \
    unzip \
    git \
    ffmpeg \
    build-essential \
    rsync
RUN apt-get install -q -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# ========== Create dirs for libraries ==========
RUN mkdir -p "/root/libs"

# ========== Installing Requirements ==========
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ========== Acme Installation ==========
WORKDIR "/root/libs"
RUN \
  mkdir acme && \
  cd acme && \
  git init . && \
  git remote add origin https://github.com/deepmind/acme && \
  git fetch --depth 1 origin 4525ade7015c46f33556e18d76a8d542b916f264 && \
  git checkout FETCH_HEAD
RUN cd acme && pip install .[jax]

# ========== MeltingPot Installation ==========
WORKDIR "/root/libs"
RUN \
  mkdir meltingpot && \
  cd meltingpot && \
  git init . && \
  git remote add origin https://github.com/deepmind/meltingpot && \
  git fetch --depth 1 origin 9d3c74e68f9b506571706dd8be89e0809a8b4744 && \
  git checkout FETCH_HEAD
WORKDIR "/root/libs/meltingpot"

# Install lab2d (appropriate version for architecture)
RUN if [ "$(uname -m)" != 'x86_64' ]; then \
    echo "No Lab2d wheel available for $(uname -m) machines." >&2 \
    exit 1; \
  elif [ "$(uname -s)" = 'Linux' ]; then \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl ;\
  else \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-macosx_10_15_x86_64.whl ;\
  fi
RUN curl -SL https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-2.1.0.tar.gz \
    | tar -xz --directory=meltingpot
RUN pip install -e .

# ========== Important Exports ==========
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# ========== Setup working directory ==========
RUN mkdir -p "/root/code"
# COPY . "/root/code"
WORKDIR "/root/code"

# ========== Start Running code ==========
ENV CUDA_VISIBLE_DEVICES=-1
ENV WANDB_API_KEY=""
ENV WANDB_ACCOUNT=""
RUN git config --global --add safe.directory /root/code
ENTRYPOINT ["python"]
