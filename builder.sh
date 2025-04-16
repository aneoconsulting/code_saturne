#!/bin/bash -e

if [ -f /opt/nvidia/nvhpc.sh ]; then
  source /opt/nvidia/nvhpc.sh
fi

# NVTX configuration

: ${NVTX_FALLBACK_INCLUDE_DIR:=/opt/nvidia/nsight-compute/2024.3.0/host/target-linux-x64/nvtx/include/}

if [ -z ${NVHPC_ROOT+x} ]; then
  echo NVHPC_ROOT is unset, configuring NVTX include flag for Rocky.
  COMPILE_FLAGS+="-I$NVTX_FALLBACK_INCLUDE_DIR "
else
  echo NVHPC_ROOT is set to $NVHPC_ROOT, configuring NVTX include flag for Arch.
  COMPILE_FLAGS+="-I$NVHPC_ROOT/profilers/Nsight_Compute/host/target-linux-x64/nvtx/include/ "
fi

# nvcc configuration

if [ -z ${CUDA_PATH+x} ]; then
  echo CUDA_PATH is unset.
  exit 1
fi

export NVCC=$CUDA_PATH/bin/nvcc

# Find configuration path

if [ -z ${1+x} ]; then
  CONFIGURE_PATH=$(realpath ../code_saturne_aneo/configure)
else
  CONFIGURE_PATH=$(realpath $1)
fi

# Other compile flags for profiling

COMPILE_FLAGS+="-g "
COMPILE_FLAGS+="-DCS_ENABLE_NVTX=1 "

echo
echo ==========================================
echo COMPILE_FLAGS: "$COMPILE_FLAGS"
echo CONFIGURE_PATH: "$CONFIGURE_PATH"
echo ==========================================
echo

function build_config () {
  TOP_DIR=$(pwd)

  echo
  echo ==========================================
  echo Config build: $1
  echo Additional compile flags: $2
  echo ==========================================
  echo

  BUILD_DIR=/home/rocky/damien/build_code_saturne_aneo
  INSTALL_DIR=/home/rocky/damien/exe_code_saturne_aneo

  mkdir -p $BUILD_DIR
  cd $BUILD_DIR

  export CFLAGS="$COMPILE_FLAGS $2"
  export CXXFLAGS="$COMPILE_FLAGS $2"
  export NVCCFLAGS="$COMPILE_FLAGS $2"
  export LDFLAGS="-L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12/"
#   export LDFLAGS="-L$CUDA_PATH/lib $2"

  $CONFIGURE_PATH \
    --prefix=$INSTALL_DIR \
    --enable-cuda --with-cuda=$CUDA_PATH \
    --disable-gui \
    --enable-mpi=/usr/local/openmpi/ \
    --with-mpi-lib=/usr/local/openmpi/lib/ \
    --with-mpi-include=/usr/local/openmpi/include/

  cd $TOP_DIR
}

#build_config mempool_sorted "-DCS_MEM_POOL_SORT"
build_config nvtx ""