#pub
#FROM python:3.7.1-slim

#FROM jjanzic/docker-python3-opencv
FROM borda/docker_python-opencv-ffmpeg

MAINTAINER Piyush Yadav

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc iperf3

WORKDIR /app
COPY requirements.txt /app

#COPY requirements.txt /app
#COPY dnnmodel.py /app
#COPY window.py /app
#COPY mobilenet_model.h5 /app

RUN pip install -r requirements.txt -i https://pypi.org/simple
ADD . /app

#COPY main.py /app/main.py

# allow other containers/PCs to connect; maybe not necessary
EXPOSE 5551

# when using docker-compose, this command can be overwritten
CMD ["python", "/app/main.py", "--ip", "172.17.0.1"]
