# LLVM 18 toolchain for compile-run collection
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=18

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg software-properties-common ca-certificates \
    python3.11 python3.11-venv python3-pip \
    build-essential git curl \
    linux-tools-generic msr-tools \
    && rm -rf /var/lib/apt/lists/*

# LLVM 18 from apt.llvm.org
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
    && apt-get update && apt-get install -y --no-install-recommends \
    clang-${LLVM_VERSION} llvm-${LLVM_VERSION} llvm-${LLVM_VERSION}-dev \
    mlir-${LLVM_VERSION}-tools libmlir-${LLVM_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/opt opt /usr/bin/opt-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/llvm-diff llvm-diff /usr/bin/llvm-diff-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/mlir-opt mlir-opt /usr/bin/mlir-opt-${LLVM_VERSION} 100 \
    && ln -sf /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang++ 

ENV PATH="/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LLVM_CONFIG=llvm-config-${LLVM_VERSION}

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY rl_uco ./rl_uco

RUN python3.11 -m pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]
