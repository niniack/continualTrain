# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit

# Start from the NVIDIA PyTorch image
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install required tools and libraries, and clean up to reduce layer size
RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y software-properties-common &&\
    add-apt-repository -y ppa:ubuntu-toolchain-r/test &&\
    apt-get update &&\
    # Combined package list from both commands
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libgtk2.0-dev \
        libstdc++6 \
        libgl1 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install btop
RUN wget "https://github.com/aristocratos/btop/releases/latest/download/btop-x86_64-linux-musl.tbz" \
    && tar xvjf btop-x86_64-linux-musl.tbz -C /usr/local/bin \
    && rm btop-x86_64-linux-musl.tbz \
    && apt-get remove --purge -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build openCV from source
ARG OPENCV_VERSION=4.7.0
RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
	-DOPENCV_GENERATE_PKGCONFIG=YES \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

RUN apt-get update -y &&\
    apt-get install -y libturbojpeg0-dev 

# Set Poetry variables
# The system site packages are important because we are using docker to give 
# us torch!
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=1
    # POETRY_CACHE_DIR=~/.cache/pypoetry/cache

WORKDIR /app

# Install poetry with pinned version
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:$POETRY_HOME/bin"
# Set site packages to be available
# The env vars should have handled this 
# But that is currently broken for OPTIONS
RUN poetry config virtualenvs.options.system-site-packages true
RUN poetry config installer.max-workers 10

# Copy poetry files
COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml

# RUN poetry lock --no-update

# Install the heavy dependencies for continualTrain
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --only torch
    # && rm -rf ~/.cache/pypoetry/cache \
    # && rm -rf ~/.cache/pypoetry/artifacts

# Install the remaining dependencies
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --with docker
    # && rm -rf ~/.cache/pypoetry/cache \
    # && rm -rf ~/.cache/pypoetry/artifacts

# Install additional dependencies from project
ARG ADDITIONAL_DEPS
ARG GIT_DEPS

# Add non-git repositories
RUN --mount=type=cache,target=/root/.cache \
    if [ -n "$ADDITIONAL_DEPS" ] ; then \
    poetry add $ADDITIONAL_DEPS ; \
    fi

# Cache busting mechanism based on the latest commit from a GitHub repo
ARG CACHEBUSTER=undefined

# Add git repositories
RUN --mount=type=cache,target=/root/.cache \
    if [ -n "$GIT_DEPS" ] ; then \
    poetry add $GIT_DEPS ; \
    fi

FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04 as runtime

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
    # Install curl and Python
    # && apt-get install -y curl python3 python3-pip python3-venv

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /usr/local /usr/local
COPY --from=builder ${POETRY_HOME} ${POETRY_HOME}
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Symlink python 
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

COPY ../continualUtils /app/continualUtils
ENV PYTHONPATH "${PYTHONPATH}:/app"

# USER $USERNAME
