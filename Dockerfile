FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310
COPY . /src
RUN pip install -r src/requirements.txt
ENTRYPOINT ["python", "/src/scripts/test.py"]
