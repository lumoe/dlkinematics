FROM ubuntu

# ENV PYTHONPATH="${PYTHONPATH}:/work"

ENV TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
     python3 \
     git \
     vim \
     build-essential \
     cmake \
     python3-pip \
     python3-dev \
     libopenexr-dev \ 
     # Install dependencies for orcos_kdl
     libeigen3-dev\
     libcppunit-dev \
     # Install dependencies for python_orocos_kdl
     python3-psutil \ 
     python3-future

# https://github.com/orocos/orocos_kinematics_dynamics/blob/release-1.5/orocos_kdl/INSTALL.md#without-catkin
RUN mkdir /install && \
     cd /install && \
     git clone https://github.com/orocos/orocos_kinematics_dynamics && \
     cd /install/orocos_kinematics_dynamics/ && \
     git submodule update --init

# Compile and install orcos_kdl
# https://github.com/orocos/orocos_kinematics_dynamics/blob/master/python_orocos_kdl/INSTALL.md#without-catkin
RUN cd /install/orocos_kinematics_dynamics/orocos_kdl && \
     mkdir build && \ 
     cd build && \ 
     cmake .. && \
     make && \ 
     make install 

# Compile and install python_orcos_kdl
RUN cd /install/orocos_kinematics_dynamics/python_orocos_kdl && \
     mkdir build && \ 
     cd build && \ 
     cmake .. && \
     make && \ 
     make install && \
     ldconfig

ADD requirements_dev.txt /requirements_dev.txt
RUN pip3 install -r /requirements_dev.txt

WORKDIR /work