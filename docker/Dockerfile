# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set up environment variables to ensure conda is prioritized
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive

# Install required tools and libraries, and clean up to reduce layer size
RUN apt-get update && apt-get install -y wget software-properties-common \
    && wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" \
    && bash Mambaforge-$(uname)-$(uname -m).sh -b -p /opt/conda \
    && rm Mambaforge-$(uname)-$(uname -m).sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && /opt/conda/bin/conda clean -tipy \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y libstdc++6 \
    && wget "https://github.com/aristocratos/btop/releases/latest/download/btop-x86_64-linux-musl.tbz" \
    && tar xvjf btop-x86_64-linux-musl.tbz -C /usr/local/bin \
    && rm btop-x86_64-linux-musl.tbz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Get environment file from avalanche and your local environment file
RUN wget -O avalanche-environment.yml https://raw.githubusercontent.com/ContinualAI/avalanche/master/environment.yml
COPY local-environment.yml .

# Merge and install from the environment files
RUN mamba install -y -c conda-forge conda-merge \
    && conda-merge avalanche-environment.yml local-environment.yml > merged.yml \
    && mamba env update --name base --file merged.yml \
    && mamba clean -y --all
