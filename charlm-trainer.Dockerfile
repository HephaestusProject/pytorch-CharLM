FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /charlm

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    vim

# Install python
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
    /bin/bash Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    numpy==1.16.5 \
    torch==1.6.0 \
    pytest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV LANG C.UTF-8
