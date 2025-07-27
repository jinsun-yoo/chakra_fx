FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN pip uninstall torch torchvision torchaudio -y

# Delete torch from dep
RUN sed -i '/torch==/d' /etc/pip/constraint.txt
RUN sed -i '/torchvision==/d' /etc/pip/constraint.txt

# Install specific version of torch.
RUN pip install --no-input --pre torch==2.9.0.dev20250704+cu129 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu129

# Edit the files we need to edit
RUN chown -R 3029572:2626 /usr/local/lib/python3.12/dist-packages/torch

# Apply patch
COPY changes.patch /tmp/changes.patch
RUN patch /usr/local/lib/python3.12/dist-packages/torch/distributed/tensor/placement_types.py /tmp/changes.patch

RUN groupadd -g 2626 gtperson
RUN useradd -u 3029572 -g 2626 -m -s /bin/bash jyoo332
