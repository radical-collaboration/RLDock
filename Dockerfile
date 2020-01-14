from continuumio/miniconda
# is based on debian:latest

ARG prj_name=PycharmProjects
ARG conda_env=RLDock
ENV NCCS_PRJ_PATH=/gpfs/alpine/proj-shared/lrn005/$conda_env

# $ docker build . -t aclyde11/rldock

# gcc and mpi for mdanalysis libdcd, mpi4py, and vmstat for ray
RUN apt-get update && apt-get install -y git \
    g++ \
    libopenmpi-dev \
    sysstat \
    procps

RUN mkdir -p /$prj_name/$conda_env
RUN git clone https://github.com/radical-collaboration/RLDock.git $HOME/$conda_env && \
    cd $HOME/$conda_env && git checkout lstm && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

COPY oe_license.txt /$prj_name/$conda_env/oe_license.txt
ENV OE_LICENSE /$prj_name/$conda_env/oe_license.txt

RUN echo "#!/bin/bash\nconda run -n $CONDA_DEFAULT_ENV python $HOME/$CONDA_DEFAULT_ENV/runner.py $@" > /docker-entrypoint.sh
RUN chmod 700 /docker-entrypoint.sh

COPY watch.py /watch.py
RUN mkdir -p /$prj_name/$conda_env/gpcr/cache
RUN echo "#!/bin/bash\n. /opt/conda/etc/profile.d/conda.sh\nconda activate $CONDA_DEFAULT_ENV\ncd $prj_name\npython /watch.py" > /docker-entrypoint.sh
RUN chmod 755 /docker-entrypoint.sh

RUN git clone https://github.com/radical-collaboration/RLDock.git /$prj_name/$conda_env/src/ && cd /$prj_name/$conda_env/src && git checkout lstm
RUN chgrp -R 0 /$prj_name && \
    chmod -R g=u /$prj_name

RUN chown -R 15798:27061 /opt/conda/envs/$conda_env
RUN chown -R 15798:27061 /$prj_name
ENV local_prj_path=/$prj_name/$conda_env/src

ENTRYPOINT [ "/docker-entrypoint.sh" ]
