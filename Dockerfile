# Refer to Dockerfile.base for the image below.
FROM jyoo332/flint:latest

# Flint specific Pytorch, Torchtitan, Chakra have been installed

# User specific setting. Feel free to edit.
ARG USER_ID=3029572
ARG GROUP_ID=2626
ARG USER_NAME=jyoo332
ARG GROUP_NAME=gtperson

RUN groupadd -g $GROUP_ID $GROUP_NAME
RUN useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USER_NAME

RUN chown -R $USER_ID:$GROUP_ID /usr/local/lib/python3.12/dist-packages/torch
RUN chown -R $USER_ID:$GROUP_ID /workspace/chakra

RUN echo 'export PATH=/home/$USER_NAME/.local/bin:$PATH' >> /home/$USER_NAME/.bashrc
