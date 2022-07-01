# syntax = docker/dockerfile:experimental
FROM pytorch/torchserve:0.4.2-cpu as base

USER root

RUN pip3 install --disable-pip-version-check --no-compile pip -U torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && pip install --disable-pip-version-check --no-compile cropharvest==0.3.0 google-cloud-storage netCDF4 pandas rasterio xarray

FROM base as build-torchserve
COPY src/torchserve_handler.py /home/model-server/handler.py

# Ensures that everytime models.dvc is updated 
# This following docker steps are rerun
COPY data/models.dvc /home/model-server
COPY data/models/*.pt /home/model-server/

WORKDIR /home/model-server

ARG MODELS
RUN for m in $MODELS; \
    do torch-model-archiver \
    --model-name $m \
    --version 1.0 \
    --serialized-file $m.pt \
    --handler handler.py \
    --export-path=model-store; \
    done

ADD scripts/torchserve_start.sh /usr/local/bin/start.sh
RUN chmod 777 /usr/local/bin/start.sh
ENV MODELS ${MODELS}
CMD ["/usr/local/bin/start.sh", "\"${MODELS}\""]