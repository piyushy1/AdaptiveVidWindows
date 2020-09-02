#pub
#FROM python:3.7.1-slim

FROM jjanzic/docker-python3-opencv

MAINTAINER Piyush Yadav

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc

#RUN apt-get install libgtk2.0-dev

WORKDIR /app
COPY requirements.txt  /app
COPY window.py /app
COPY videostreamer.py /app
COPY probe.py /app
RUN pip install -r requirements.txt

COPY main.py /app/main.py

# when using docker-compose, this command can be overwritten
# Change "yo.ur.i.p" to your machine IP (something similar to: "192.168.99.1") when using `docker run `
CMD ["python", "main.py", "--ip", "yo.ur.i.p"]
#CMD ["python", "main.py", "--ip", "172.17.255.255"]
