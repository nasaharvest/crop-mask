# syntax = docker/dockerfile:experimental
FROM gpuci/miniconda-cuda:10.2-runtime-ubuntu18.04 as base

FROM base as reqs
COPY environment.gpu.yml environment.gpu.yml
RUN conda env create -f environment.gpu.yml
ENV PATH /opt/conda/envs/landcover-mapping/bin:$PATH
RUN /bin/bash -c "source activate landcover-mapping"

FROM reqs as get-data-stage
COPY .dvc /crop-mask/.dvc
COPY .git /crop-mask/.git
COPY data /crop-mask/data
WORKDIR /crop-mask
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    dvc pull data/features && dvc pull data/models

RUN mkdir -p /vol /vol/input /vol/predict

FROM get-data-stage as final-stage
COPY scripts /crop-mask/scripts
COPY src /crop-mask/src
WORKDIR /crop-mask/scripts
