FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir pytest==9.1.1 pytest-json-ctrf==0.5.2 numpy==2.4.4 pillow==12.1.1 scipy==1.17.1

# Unprivileged account the submitted program runs as
RUN useradd -m -u 1500 runner

# Bake tests into the image; harbor does not upload tests/ at verify time.
# Root-only, so the submitted program cannot read the sealed references.
COPY . /tests/
RUN chown -R root:root /tests && chmod -R go-rwx /tests

# Parent dir of the declared artifact
RUN mkdir -p /app
