FROM nvidia/cuda:11.6.2-base-ubuntu20.04
RUN apt update
RUN apt install --no-install-recommends -y python3.8 python3-pip git
RUN apt clean
ADD ssl-pablo /ssl-pablo
WORKDIR /ssl-pablo
RUN pip3 install -r requirements.txt