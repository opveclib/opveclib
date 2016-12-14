# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# Pick up ubuntu 14.04 with Cuda and CuDNN - need developer version to get nvcc
FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Karen Brems <karen.brems@hpe.com>

# pick up proxy from command line arguement to docker build command
ARG proxy
ENV http_proxy ${proxy}
ENV https_proxy ${proxy}

# install dependencies
# install nose2 so we can run OVL tests
# and protobuf compiler to get header files for test operator
RUN apt-get update && apt-get install -y \
        curl \
        g++ \
        python-numpy \
        python-nose2 \
        wget \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# get pip2 and pip3
# note, this generates a warning due to missing ca cert. 
# need to figure out how to get ssl certs inside docker or just ignore with curl -k option?
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# install nose2 for python 3
RUN pip3 install nose2

# Install TensorFlow CPU version for both python2.7 and python 3.4.
ENV TENSORFLOW_VERSION 0.11.0
RUN pip2 --no-cache-dir install --upgrade \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
RUN pip3 --no-cache-dir install --upgrade \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp34-cp34m-linux_x86_64.whl

# set default workdir
WORKDIR /usr/opveclib

# install sphinx so we can run doctest
# note we use the python2 version here and the python3 version for cpu to get better coverage
RUN pip2 install -U sphinx

# install protoc 3.0.0 and its C++ libraries
#RUN wget --no-check-certificate  https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.tar.gz
#RUN tar xvf protobuf-cpp-3.0.0-beta-2.tar.gz
#RUN cd protobuf-3.0.0-beta-2/
#RUN /usr/opveclib/protobuf-3.0.0-beta-2/configure
#RUN make
#RUN make install
#RUN ldconfig

# copy opveclib source that was already cloned from github via jenkins
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"


COPY opveclib /usr/opveclib/opveclib
COPY documentation /usr/opveclib/documentation
COPY README.rst /usr/opveclib/
COPY setup.py /usr/opveclib/
COPY runBAT.sh /usr/opveclib/
COPY runALL.sh /usr/opveclib/
COPY nose2.cfg /usr/opveclib/

CMD /bin/bash
