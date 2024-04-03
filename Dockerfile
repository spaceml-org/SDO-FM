FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

RUN pip install -r requirements.txt

COPY SDO-FM /src

ENTRYPOINT ["python", "/src/scripts/test.py"]
