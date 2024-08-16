# NOTE(llogan): This dockerfile assumes that
# hermes github is the current working directory

# Install ubuntu 22.04
FROM ubuntu:22.04
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="0.0"
LABEL description="MegaMmap Docker image"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update ubuntu
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install

# Install some basic packages
RUN apt install -y \
    openssh-server \
    sudo \
    git \
    gcc g++ gfortran make binutils gpg \
    tar zip xz-utils bzip2 \
    perl m4 libncurses5-dev libxml2-dev diffutils \
    pkg-config cmake pkg-config \
    python3 python3-pip python3 python3-distutils python3-venv\
    doxygen lcov zlib1g-dev hdf5-tools \
    build-essential ca-certificates \
    coreutils curl environment-modules \
    gfortran git gpg lsb-release \
    unzip zip \
    bash jq gdbserver gdb

# Setup basic environment
ENV USER="root"
ENV HOME="/root"
ENV SPACK_DIR="${HOME}/spack"
ENV SPACK_VERSION="v0.20.2"
COPY ci/module_load.sh module_load.sh

# Install Spack
RUN . /module_load.sh && \
    git clone -b ${SPACK_VERSION} https://github.com/spack/spack ${SPACK_DIR} && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack external find

# Git clone mega_mmap \
RUN git clone https://github.com/lukemartinlogan/mega_mmap && \
    cd mega_mmap && \
    . /module_load.sh && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    . deps.sh
