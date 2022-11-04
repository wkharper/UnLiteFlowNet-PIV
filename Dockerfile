FROM nvidia/cuda:10.1-devel-ubuntu18.04 as base
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install python3-pip python3-tk -y
RUN pip3 install --upgrade pip
RUN pip3 install livelossplot
RUN pip3 install flowiz -U
RUN pip3 install GPUtil
RUN pip3 install torch==1.5.0
RUN pip3 install scikit-learn==0.22.2
RUN pip3 install cupy-cuda101
RUN pip3 install imageio
RUN pip3 install progress