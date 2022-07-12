FROM ros:noetic

ENV PYTHONPATH="${PYTHONPATH}:/work"

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
     python3 \
     ros-noetic-urdfdom-py \
     liburdfdom-tools \
     git \
     vim \
     python3-pip \
     python3-dev \
     python3-pykdl \
     libopenexr-dev

RUN git clone https://github.com/herzig/hrl-kdl.git

RUN cd /hrl-kdl/pykdl_utils/ && python3 setup.py install
RUN cd /hrl-kdl/hrl_geom/ && python3 setup.py install

RUN mkdir -p /work

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow pytest lxml transformations tensorflow-graphics progressbar2 pandas 

RUN apt-get install 
WORKDIR /work
