FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    cmake \
    make \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    openmpi-bin \
    libopenmpi-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    pocl-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-mpi4py \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./sequential /app/page-rank/sequential
COPY ./parallel /app/page-rank/parallel
COPY ./distributed /app/page-rank/distributed
COPY ./accelerated /app/page-rank/accelerated
COPY ./test-runner /app/test-runner

RUN g++ -O2 -std=c++17 /app/page-rank/sequential/sequential.cpp -o /app/page-rank/pageRankSequential
RUN g++ -O2 -std=c++17 -fopenmp /app/page-rank/parallel/parallel.cpp -o /app/page-rank/pageRankParallel
RUN mpic++ -O2 -std=c++17 /app/page-rank/distributed/distributed.cpp -o /app/page-rank/pageRankDistributed
RUN g++ -O2 -std=c++17 /app/page-rank/accelerated/accelerated.cpp -o /app/page-rank/pageRankAccelerated -lOpenCL

CMD ["python3", "test-runner/test_runner.py"]

CMD ["python3", "test-runner/test_runner.py"]
