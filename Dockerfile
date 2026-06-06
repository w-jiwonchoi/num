# ─── Base image matching Vessel AI environment ───────────────────────────────
# The PPTX specifies: quay.io/vessl-ai/cuda:12.8 with Kokkos 4.7.02
# and the pre-built image: chaeyeunpark/veesl-ai-ubuntu2404-cuda1208:kokkos-ampere80-v0.1
#
# This Dockerfile reproduces that environment from scratch.
# For Vessel AI usage, prefer the pre-built image (see README).
FROM quay.io/vessl-ai/cuda:12.8

LABEL maintainer="kokkos-spmv"
LABEL description="Multi-GPU SpMV + CG library (Kokkos 4.7.02 + MPI + nanobind)"

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
WORKDIR /root

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ninja-build \
        libopenblas-openmp-dev \
        # Python 3.12 is pre-installed in the Vessel AI image; add dev headers
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ─── Kokkos 4.7.02 (matches Vessel AI pre-built image) ───────────────────────
# Use nvcc_wrapper as CXX so CUDA lambdas work with host-side C++17 headers.
RUN wget https://github.com/kokkos/kokkos/releases/download/4.7.02/kokkos-4.7.02.tar.gz \
 && tar -xf kokkos-4.7.02.tar.gz \
 && pushd kokkos-4.7.02 \
 && cmake -Bbuild \
        -DCMAKE_CXX_COMPILER="$PWD/bin/nvcc_wrapper" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DBUILD_SHARED_LIBS=ON \
        . \
 && cmake --build ./build --parallel $(nproc) \
 && cmake --install ./build \
 && popd \
 && rm -rf kokkos-4.7.02.tar.gz kokkos-4.7.02

# ─── KokkosKernels 4.7.02 ────────────────────────────────────────────────────
RUN wget https://github.com/kokkos/kokkos-kernels/releases/download/4.7.02/kokkos-kernels-4.7.02.tar.gz \
 && tar -xf kokkos-kernels-4.7.02.tar.gz \
 && pushd kokkos-kernels-4.7.02 \
 && cmake -Bbuild \
        -DCMAKE_CXX_COMPILER=/usr/local/bin/nvcc_wrapper \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DKokkos_ROOT=/usr/local \
        -DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
        -DBUILD_SHARED_LIBS=ON \
        . \
 && cmake --build ./build --parallel $(nproc) \
 && cmake --install ./build \
 && popd \
 && rm -rf kokkos-kernels-4.7.02.tar.gz kokkos-kernels-4.7.02

# ─── OpenMPI 5.0.10 with CUDA support ────────────────────────────────────────
# Matches the Vessel AI Dockerfile shown in the PPTX.
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz \
 && tar -xf openmpi-5.0.10.tar.gz \
 && pushd openmpi-5.0.10 \
 && ./configure --with-cuda=/usr/local/cuda --prefix=/usr/local \
 && make -j$(nproc) all install \
 && popd \
 && rm -rf openmpi-5.0.10.tar.gz openmpi-5.0.10 \
 && ldconfig

# ─── Python packages ──────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir nanobind \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ─── Environment ──────────────────────────────────────────────────────────────
ENV PATH="/usr/local/bin:${PATH}"
# Allow MPI to run as root (required in Vessel AI containers)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Optional: enable CUDA-aware MPI if supported
ENV OMPI_MCA_opal_cuda_support=true

# ─── Build the library ────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

RUN mkdir build && \
    cd build && \
    CXX=nvcc_wrapper cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DKOKKOS_ENABLE_CUDA=ON \
        -DKOKKOS_ARCH=AMPERE80 \
        -DBUILD_TESTS=ON \
 && make -j$(nproc)

# Symlink .so for Python import (filename has Python version embedded)
RUN ln -s /workspace/build/kokkos_spmv*.so /workspace/python/ 2>/dev/null || true

WORKDIR /workspace/python
ENV PYTHONPATH=/workspace/python

CMD ["bash"]
