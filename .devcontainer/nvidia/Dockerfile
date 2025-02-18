ARG CUDA_VERSION=12.6.0
FROM nvidia/cuda:$CUDA_VERSION-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_NAME=jaxgm

# Set the timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install apt packages
RUN apt-get -q update \
    && apt-get -q -y upgrade \
    && apt-get -q install --no-install-recommends -y \
    sudo \
    python3-pip \
    python3-dev \
    python3-venv \
    apt-utils \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Configure the ubuntu non-root user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -a -G dialout $USERNAME

# Switch to the ubuntu user
USER $USERNAME
ENV USER=$USERNAME

ENV USER_WORKSPACE=/home/$USERNAME/
WORKDIR $USER_WORKSPACE

# Create a new virtual environment for Python
ENV VIRTUAL_ENV=$USER_WORKSPACE/.venv/$PROJECT_NAME
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV \
    && echo "source ${VIRTUAL_ENV}/bin/activate" >> /home/$USERNAME/.bashrc \
    && touch .venv/COLCON_IGNORE
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python requirements
COPY --chown=$USER_UID:$USER_GID requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install NVIDIA software
RUN sudo apt-get update \
    && sudo apt-get -q -y upgrade \
    && sudo apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV QT_X11_NO_MITSHM=1
