FROM us-docker.pkg.dev/vertex-ai/training/pytorch-tpu.2-0:latest
COPY . /src
RUN /src/deployment/tpu_install.sh
ENTRYPOINT ["python", "/src/scripts/main.py"]