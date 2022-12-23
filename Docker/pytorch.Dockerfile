# docker build -t pytorch_1.12.1 -f pytorch.Dockerfile .

#FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# for old nvidia containers using the old signing key
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt -y update

RUN apt-get install -y \
    software-properties-common \
    # build-essential \
    # checkinstall \
    # cmake \
    # pkg-config \
    # yasm \
    git 
    # vim \
    # curl \
    # wget \
    # gfortran \
    # libjpeg8-dev \
    # libpng-dev \
    # libtiff5-dev \
    # libtiff-dev \
    # libavcodec-dev \
    # libavformat-dev \
    # libswscale-dev \
    # libdc1394-22-dev \
    # libxine2-dev \
    # sudo \
    # apt-transport-https \
    # libcanberra-gtk-module \
    # libcanberra-gtk3-module \
    # dbus-x11 \
    # vlc \
    # iputils-ping \
    # python3-dev \
    # python3-pip

# RUN apt-get install -y ffmpeg

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

# upgrade python to version 3.8 (IMPT: remove python3-dev and python3-pip if already installed)
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.8-dev python3.8-venv python3-pip
RUN apt -y update
# Set python3.8 as the default python
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
# RUN update-alternatives --set python3 /usr/bin/python3.8
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

# INSTALL SUBLIME TEXT
# RUN apt install -y ca-certificates
# RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - && add-apt-repository "deb https://download.sublimetext.com/ apt/stable/"
# RUN apt update && apt install -y sublime-text

RUN rm -rf /var/cache/apt/archives/

### APT END ###

# RUN /usr/bin/python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade pip setuptools
RUN pip install opencv-python

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install timm matplotlib tqdm easyfsl sklearn jupyter tensorboard plotly
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt