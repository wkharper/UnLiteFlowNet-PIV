FROM nvidia/cuda:11.2.2-devel-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install python3-pip python3-tk -y
RUN pip3 install --upgrade pip
RUN pip3 install livelossplot
RUN pip3 install flowiz -U
RUN pip3 install GPUtil
RUN pip3 install torch==1.13.0
RUN pip3 install scikit-learn==0.22.2
RUN pip3 install cupy-cuda112
RUN pip3 install imageio
RUN pip3 install progress
RUN pip3 install IPython
