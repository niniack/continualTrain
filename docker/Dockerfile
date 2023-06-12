# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Install Mambaforge
RUN apt-get update && apt-get install -y wget && \
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p /opt/conda && \
    rm Mambaforge-$(uname)-$(uname -m).sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -tipy

# Update libstdc++.so.6 for this image
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libstdc++6

# Install btop
RUN wget "https://github.com/aristocratos/btop/releases/latest/download/btop-x86_64-linux-musl.tbz" && \
    tar xvjf btop-x86_64-linux-musl.tbz -C /usr/local/bin && \
    rm btop-x86_64-linux-musl.tbz

# Make sure the environment is activated
ENV PATH /opt/conda/bin:$PATH

# Activate base environment when starting bash
RUN echo "conda activate base" >> ~/.bashrc

# Prioritize Conda libraries over system ones
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH

# Copy environment file from avalanche into the Docker image
RUN wget -O avalanche-environment.yml https://raw.githubusercontent.com/ContinualAI/avalanche/master/environment.yml

# Copy local environment file into the Docker image
COPY local-environment.yml .

# Install conda-merge in the base environment
RUN mamba install -y -c conda-forge conda-merge

# Merge the environment files and create the environment
RUN conda-merge avalanche-environment.yml local-environment.yml > merged.yml && \
    mamba env update --name base --file merged.yml