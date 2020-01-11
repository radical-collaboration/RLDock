from continuumio/miniconda
# is based on debian:latest

ARG conda_env=RLDock
ENV NCCS_PRJ_PATH=/gpfs/alpine/proj-shared/lrn005/$conda_env

# $ docker build . -t aclyde11/rldock

# gcc and mpi for mdanalysis libdcd, mpi4py, and vmstat for ray
RUN apt-get update && apt-get install -y git \
    g++ \
    libopenmpi-dev \
    sysstat \
    procps

RUN mkdir -p /PycharmProjects/$conda_env
RUN git clone https://github.com/radical-collaboration/RLDock.git $HOME/$conda_env && \
    cd $HOME/$conda_env && git checkout lstm && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

COPY oe_license.txt /PycharmProjects/$conda_env/oe_license.txt
ENV OE_LICENSE /PycharmProjects/$conda_env/oe_license.txt

RUN echo "#!/bin/bash\nconda run -n $CONDA_DEFAULT_ENV python $HOME/$CONDA_DEFAULT_ENV/runner.py $@" > /docker-entrypoint.sh
RUN chmod 700 /docker-entrypoint.sh

ENTRYPOINT [ "/docker-entrypoint.sh" ]
