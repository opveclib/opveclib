# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# Start from ubuntu 14.04 
FROM ubuntu:14.04

MAINTAINER Karen Brems <karen.brems@hpe.com>

# pick up proxy from command line arguement to docker build command
ARG proxy
ENV http_proxy ${proxy}
ENV https_proxy ${proxy}

# install dependencies
# install nose2 for python 2 so we can run OVL tests
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
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
RUN pip3 --no-cache-dir install --upgrade \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp34-cp34m-linux_x86_64.whl

# set default workdir
WORKDIR /usr/opveclib

# install protoc 3.0.0 and its C++ libraries
#RUN wget --no-check-certificate  https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.tar.gz
#RUN tar xvf protobuf-cpp-3.0.0-beta-2.tar.gz
#RUN cd protobuf-3.0.0-beta-2/
#RUN /usr/opveclib/protobuf-3.0.0-beta-2/configure
#RUN make
#RUN make install
#RUN ldconfig

# install sphinx so we can run doctest
# note we use the python3 version here and the python2 version for gpu to get better coverage
RUN pip3 install -U sphinx

# copy opveclib source that was already cloned from github via jenkins
COPY opveclib /usr/opveclib/opveclib
COPY documentation /usr/opveclib/documentation
COPY README.rst /usr/opveclib/
COPY setup.py /usr/opveclib/
COPY runBAT.sh /usr/opveclib/
COPY runALL.sh /usr/opveclib/
COPY nose2.cfg /usr/opveclib/


CMD /bin/bash
