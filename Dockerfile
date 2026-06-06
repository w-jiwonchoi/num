# ─── Base image: CUDA 12.4 + Ubuntu 22.04 ────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="kokkos-spmv"
LABEL description="Multi-GPU SpMV + CG library (Kokkos + MPI + nanobind)"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ninja-build \
        # MPI
        libopenmpi-dev \
        openmpi-bin \
        # Python
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        # Profiling
        # nsys is provided by the CUDA toolkit image
    && rm -rf /var/lib/apt/lists/*

# ─── Default python ───────────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ─── Python packages ──────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ─── nanobind (Python package for headers + cmake config) ────────────────────
RUN pip install --no-cache-dir nanobind

# ─── Kokkos 4.3.01 ───────────────────────────────────────────────────────────
ARG KOKKOS_VERSION=4.3.01
ARG CUDA_ARCH=80   # A100 = 80, V100 = 70, H100 = 90

RUN git clone --depth 1 --branch ${KOKKOS_VERSION} \
        https://github.com/kokkos/kokkos.git /opt/kokkos-src \
 && cmake -S /opt/kokkos-src -B /opt/kokkos-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/kokkos \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ARCH_AMPERE${CUDA_ARCH}=ON \
        -DCMAKE_CXX_EXTENSIONS=OFF \
 && ninja -C /opt/kokkos-build install \
 && rm -rf /opt/kokkos-src /opt/kokkos-build

# ─── KokkosKernels 4.3.01 ────────────────────────────────────────────────────
ARG KK_VERSION=4.3.01
RUN git clone --depth 1 --branch ${KK_VERSION} \
        https://github.com/kokkos/kokkos-kernels.git /opt/kk-src \
 && cmake -S /opt/kk-src -B /opt/kk-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/kokkos-kernels \
        -DKokkos_ROOT=/opt/kokkos \
        -DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
 && ninja -C /opt/kk-build install \
 && rm -rf /opt/kk-src /opt/kk-build

# ─── Environment ──────────────────────────────────────────────────────────────
ENV Kokkos_ROOT=/opt/kokkos
ENV KokkosKernels_ROOT=/opt/kokkos-kernels
ENV PATH="/opt/kokkos/bin:${PATH}"
# CUDA-aware MPI requires this on some systems
ENV OMPI_MCA_opal_cuda_support=true

# ─── Build the library ────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ROOT=/opt/kokkos \
        -DKokkosKernels_ROOT=/opt/kokkos-kernels \
        -DKOKKOS_ENABLE_CUDA=ON \
        -DKOKKOS_ARCH=AMPERE80 \
        -DBUILD_TESTS=ON \
 && ninja -C build

# Create symlink so Python can import the module
RUN ln -s /workspace/build/kokkos_spmv*.so /workspace/python/

WORKDIR /workspace/python
ENV PYTHONPATH=/workspace/python

CMD ["bash"]
