FROM tensorflow/tensorflow:1.12.0-gpu-py3

ENV PIP_VERSION_TO_INSTALL="18.1"

MAINTAINER Piyush

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc iproute wondershaper python3-pip

RUN pip install -U pip==${PIP_VERSION_TO_INSTALL}

WORKDIR /app
COPY requirements.txt /app

RUN pip install -r requirements.txt -i https://pypi.org/simple

ADD . /app
#COPY main.py /app/main.py

# allow other containers/PCs to connect; maybe not necessary
EXPOSE 5551

# when using docker-compose, this command can be overwritten
CMD ["python", "/app/main.py", "--ip", "172.17.0.1"]
