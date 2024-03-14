FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY requirements.txt /tmp/requiremnts.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir /container_app

COPY src/main.py /container_app
COPY src/network.py /container_app
COPY src/utils.py /container_app

WORKDIR /container_app