FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=off \
    PYTHONDONTWRITEBYTECODE=1

ARG CUDA_ARCH="70;80;86;89;90"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /opt/build

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel setuptools

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Build/install raysim from source.
COPY i4h-sensor-simulation/ultrasound-raytracing /opt/build/ultrasound-raytracing
# The source tree may contain a host-generated scikit-build/CMake cache
# (`build-release`) with absolute paths from the developer machine. If copied
# into the image, CMake fails with a cache path mismatch during pip install.
RUN rm -rf /opt/build/ultrasound-raytracing/build-release
RUN CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
    pip install --no-cache-dir "/opt/build/ultrasound-raytracing[all]"

# Install runtime dependencies for the demo API and pullback geometry helpers.
COPY IVUS_demo/requirements.txt /opt/build/IVUS_demo/requirements.txt
RUN pip install --no-cache-dir -r /opt/build/IVUS_demo/requirements.txt

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=build /opt/venv /opt/venv
# Copy the in-tree raysim package produced during the source build. The compiled
# `ray_sim_python` extension is emitted into the source tree's `raysim/`
# directory, not reliably into site-packages for wheel installs.
COPY --from=build /opt/build/ultrasound-raytracing/raysim /opt/ivus_demo/raysim

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH="/opt/ivus_demo"

WORKDIR /opt/ivus_demo

# Application code.
COPY IVUS_demo/app.py /opt/ivus_demo/app.py
COPY simulated_pullback_from_CT/src /opt/ivus_demo/pullback_src

# Static assets loaded at startup.
COPY patient_23 /opt/ivus_demo/patient_23
COPY instrument-calibration/p035_visions/volcano_s5i.yaml /opt/ivus_demo/instrument-calibration/p035_visions/volcano_s5i.yaml
COPY ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_off_g50_d60.npy /opt/ivus_demo/ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_off_g50_d60.npy

EXPOSE 8000

CMD ["python", "/opt/ivus_demo/app.py"]
