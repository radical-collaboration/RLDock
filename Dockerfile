from continuumio/miniconda
# is based on debian:latest

ARG conda_env=RLDock

# $ docker build . -t aclyde11/rldock

# gcc and mpi for mdanalysis libdcd, mpi4py, and vmstat for ray
RUN apt-get update && apt-get install -y git \
    g++ \
    libopenmpi-dev \
    sysstat \
    procps

RUN git clone https://github.com/radical-collaboration/RLDock.git $HOME/$conda_env && \
    cd $HOME/RLDock && git checkout lstm && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

ENTRYPOINT [ "python", "$HOME/$conda_env/runner.py" ]
