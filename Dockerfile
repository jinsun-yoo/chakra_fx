FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG USER_ID=3029572
ARG GROUP_ID=2626
ARG USER_NAME=jyoo332
ARG GROUP_NAME=gtperson

RUN pip uninstall torch torchvision torchaudio -y

# Delete torch from dep
RUN sed -i '/torch==/d' /etc/pip/constraint.txt
RUN sed -i '/torchvision==/d' /etc/pip/constraint.txt

# Install specific version of torch.
RUN pip install --no-input --pre torch==2.9.0.dev20250704+cu129 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu129

# Edit the files we need to edit
RUN chown -R $USER_ID:$GROUP_ID /usr/local/lib/python3.12/dist-packages/torch

# Apply patch
COPY changes.patch /tmp/changes.patch
RUN patch /usr/local/lib/python3.12/dist-packages/torch/distributed/tensor/placement_types.py /tmp/changes.patch

# Install other dependencies
RUN git clone https://github.com/pytorch/torchtitan && cd torchtitan && git checkout 183f6fce1a586ce66027deee2c4fdb823616cd75
RUN cd torchtitan && pip install -r requirements.txt && pip install .

RUN git clone https://github.com/mlcommons/chakra.git
RUN cd chakra && pip install .
RUN pip install --upgrade protobuf

RUN groupadd -g $GROUP_ID $GROUP_NAME
RUN useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USER_NAME

RUN chown -R $USER_ID:$GROUP_ID /usr/local/lib/python3.12/dist-packages/torch
RUN chown -R $USER_ID:$GROUP_ID /workspace/chakra

RUN echo 'export PATH=/home/$USER_NAME/.local/bin:$PATH' >> /home/$USER_NAME/.bashrc
