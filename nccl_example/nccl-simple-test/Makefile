# -------------------------------
# Makefile – MPI + NCCL example
# -------------------------------

### Chemins d’installation (modifiable par « make CUDA_HOME=/opt/cuda ... »)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
MPI_HOME  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi

# Outils
NVCC   := $(CUDA_HOME)/bin/nvcc
MPICC  := $(MPI_HOME)/bin/mpicc   # non utilisé ici mais laissé à titre d’exemple

# Cibles
TARGET := mpi_nccl_reduce
SRC    := mpi_nccl_reduce.cu

# Flags de compilation
NVCCFLAGS := -std=c++14 -O3 -lineinfo

# Inc. & lib
INCFLAGS := -I$(MPI_HOME)/include -I$(NCCL_HOME)/include
LDFLAGS  := -L$(MPI_HOME)/lib -lmpi \
            -L$(NCCL_HOME)/lib64 -lnccl

# -------------- Règles --------------

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	@echo "Exemple d’exécution (4 rangs / 4 GPU) :"
	@echo "mpiexec -n 4 ./$(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: all run clean
