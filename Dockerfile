FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN groupadd -g 2626 gtperson
RUN useradd -u 3029572 -g 2626 -m -s /bin/bash jyoo332
