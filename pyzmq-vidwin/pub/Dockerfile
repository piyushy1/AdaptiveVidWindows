FROM dsalwala/vid-cep:latest

MAINTAINER Piyush Yadav

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  gcc

WORKDIR /app
COPY requirements.txt /app

#COPY requirements.txt /app
#COPY dnnmodel.py /app
#COPY window.py /app
#COPY mobilenet_model.h5 /app

RUN pip3 install -r requirements.txt -i https://pypi.org/simple

COPY --from=nvidia/cuda:10.1-devel-ubuntu18.04 /usr/local/cuda-10.1/ /usr/local/cuda


ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib/stubs:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64
ENV LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}

ADD . /app

#COPY main.py /app/main.py

# allow other containers/PCs to connect; maybe not necessary
EXPOSE 5551

# when using docker-compose, this command can be overwritten
CMD ["python", "/app/main.py", "--ip", "172.17.0.1"]
