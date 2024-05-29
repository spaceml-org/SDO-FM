FROM us-docker.pkg.dev/vertex-ai/training/pytorch-tpu.2-1.cp310:latest 

COPY . /src

RUN pip install -r /src/requirements.txt
ENTRYPOINT ["python3", "/src/scripts/main.py"]