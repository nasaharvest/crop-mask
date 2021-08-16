FROM python:3.8

RUN apt-get update && apt-get install -y tzdata \
    libgdal-dev

RUN pip install --upgrade pip
RUN pip install gdal==2.4.4

CMD ["python"]