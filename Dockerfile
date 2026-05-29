FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    git \
    ffmpeg \
    libglib2.0-0 \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/geraf

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python setup.py build_ext --inplace
RUN pip install -e . --no-build-isolation

RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /workspace/geraf

USER ${LDAP_USERNAME}

CMD ["python", "train.py", "--help"]
