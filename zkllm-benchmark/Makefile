# Compilers. Default: the conda env's nvcc. Set CUDA_HOME to build with a CUDA toolkit instead (on the cluster:
#   CUDA_HOME=$(module load StdEnv/2023 cuda/12.9 && echo $EBROOTCUDA) make ...). Use CUDA 12.9: ptxas from 12.6 and
# 12.8 miscompiles the PTX carry-chain field multiply at -O2+ in some kernels (see bls12-381.cu). With CUDA_HOME the
# binaries link cudart statically, so running them needs no module.
ifdef CUDA_HOME
NVCC := $(CUDA_HOME)/bin/nvcc
INCLUDES :=
LIBS :=
else
NVCC := $(CONDA_PREFIX)/bin/nvcc

# Include and library paths
INCLUDES := -I$(CONDA_PREFIX)/include
LIBS := -L$(CONDA_PREFIX)/lib -lcudart
endif

NVCC_RELEASE := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
ifneq (,$(filter 12.6 12.8,$(NVCC_RELEASE)))
$(warning nvcc $(NVCC_RELEASE): its ptxas miscompiles the PTX field multiply in some kernels; use CUDA_HOME with CUDA 12.9 or -DZKLLM_PORTABLE_MUL)
endif

# Compute capability of the target GPU. Override per machine, e.g. `make ARCH=sm_90 ...` for H100
# (sm_86 = A6000/3090, sm_89 = 4090, sm_80 = A100, sm_90 = H100).
ARCH ?= sm_86

# NVCC compiler flags
NVCC_FLAGS := -arch=$(ARCH) -std=c++17

# Source and object files
CU_SRCS := bls12-381.cu blendy.cu witness-store.cu weight-stream.cu int-matmul.cu squash.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu llama-layer.cu opt-layer.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
TARGETS := matmul-bm hadamard-bm com-bm tlookup-bm ppgen commit-param partition
TARGET_OBJS := $(TARGETS:=.o)

# Every object depends on every header: the pattern rules alone would leave stale objects
# behind after a .cuh edit.
$(CU_OBJS) $(CPP_OBJS) $(TARGET_OBJS): $(wildcard *.cuh *.hpp *.h)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# General rule for building each target
$(TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ -dlto

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(TARGETS)

# Default rule
all: $(TARGETS)
