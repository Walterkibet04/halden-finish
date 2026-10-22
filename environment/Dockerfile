FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.4.4 pillow==12.1.1 scipy==1.17.1

COPY data/pairs /data/pairs
WORKDIR /app
