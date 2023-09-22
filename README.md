<div align="center">
  <img src="https://cdn.openai.com/triton/assets/triton-logo.png" alt="Triton logo" width="88" height="100">
</div>

[![Wheels](https://github.com/openai/triton/actions/workflows/wheels.yml/badge.svg?branch=release/2.0.x)](https://github.com/openai/triton/actions/workflows/wheels.yml)

We're hiring! If you are interested in working on Triton at OpenAI, we have roles open for [Compiler Engineers](https://openai.com/careers/software-engineer-triton-compiler) and [Kernel Engineers](https://openai.com/careers/kernel-engineer).

**`Documentation`** |
------------------- |
[![Documentation](https://github.com/openai/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/)

# Triton Developer Conference Registration Now Closed
The Triton Developer Conference will be held in a hybrid mode at the Microsoft Silicon Valley Campus in Mountain View, California. The conference will be held on September 20th from 10am to 4pm, followed by a reception till 5:30 pm.

Tentative Agenda for the conference (subject to change):

|Time    |Title  |Speaker
|--------|-------|-------|
|10:00 AM|Welcome|Kevin Scott (Microsoft)|
|10:20 AM|The Triton Compiler: Past, Present and Future|Phil Tillet (OpenAI)|
|11:00 AM|**Break**||
|11:20 AM|Hopper support in Triton|Gustav Zhu (Nvidia)|
|11:40 AM|Bringing Triton to AMD GPUs|Jason Furmanek, Lixun Zhang (AMD)|
|12:00 PM|Intel XPU Backend for Triton|Eikan Wang (Intel)|
|12:20 PM|Vectorization of Triton Kernels for Qualcomm Hexagon Backend|Javed Absar (Qualcomm)|
|12:30 PM|**Lunch**||
|1:40 PM |Triton for MTIA|Roman Levenstein et al, (Meta)|
|2:00 PM |Using Triton IR for high-performance fusions in XLA|George Karpenkov (Google)|
|2:20 PM |Triton for All: Triton as a device-independent language|Ian Bearman (Microsoft)|
|2:40 PM|**Break**||
|3:00 PM|PyTorch 2.0 and TorchInductor|Jason Ansel, Horace He (Meta)|
|3:20 PM|Pallas: A JAX Kernel Language|Sharad Vikram (Google)|
|3:40 PM|Writing Grouped GEMMs in Triton|Vinod Grover (Nvidia)|
|4:00 PM|**Reception**||


# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.

# Quick Installation

You can install the latest stable release of Triton from pip:

```bash
pip install triton
```
Binary wheels are available for CPython 3.7-3.11 and PyPy 3.8-3.9.

And the latest nightly release:

```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

# Install from source

```
git clone https://github.com/openai/triton.git;
cd triton;

pip install ninja cmake; # build-time dependencies
pip install -e python
```

Or with a virtualenv:

```
git clone https://github.com/openai/triton.git;
cd triton;

python -m venv .venv --prompt triton;
source .venv/bin/activate;

pip install ninja cmake; # build-time dependencies
pip install -e python
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.  Check `python/setup.py`
   for a line like

       version = "llvm-17.0.0-c5dede880d17"

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) c5dede880d17.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  ../llvm -DLLVM_ENABLE_PROJECTS="mlir"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables.

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>/python
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e python

# Changelog

Version 2.0 is out! New features include:
- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/openai/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).


# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs
