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
  CONFIGURE_PATH=$(realpath ./configure)
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

PROJECT_DIR=$(pwd)
SATURNE_CASES_DIR=$PROJECT_DIR/saturne-open-cases
PROFILER_OUTPUT_DIR=$PROJECT_DIR/profiler-output

function build_config () {
  echo
  echo ==========================================
  echo Config build: $1
  echo Additional compile flags: $2
  echo Branch: $3
  echo ==========================================
  echo

  cd $PROJECT_DIR
  git checkout $3
  ./sbin/bootstrap

  BUILD_DIR=$PROJECT_DIR/build/$1
  INSTALL_DIR=$PROJECT_DIR/install/$1

  mkdir -p $BUILD_DIR
  cd $BUILD_DIR

  export CFLAGS="$COMPILE_FLAGS $2"
  export CXXFLAGS="$COMPILE_FLAGS $2"
  export NVCCFLAGS="$COMPILE_FLAGS $2"

  $CONFIGURE_PATH \
    --prefix=$INSTALL_DIR \
    --enable-cuda --with-cuda=$CUDA_PATH \
    --disable-gui \
    --enable-mpi=/usr/local/openmpi/ \
    --with-mpi-lib=/usr/local/openmpi/lib/ \
    --with-mpi-include=/usr/local/openmpi/include/

  make -j16
  make install
}

# Profiling

function prepare_cases () {
  rm -rf $SATURNE_CASES_DIR
  git clone git@github.com:code-saturne/saturne-open-cases.git $SATURNE_CASES_DIR
}

function profile_for_config () {
  CS_EXEC=$PROJECT_DIR/install/$1/bin/code_saturne

  echo
  echo ==========================================
  echo Config profile: $1
  echo Benchmark case: $2
  echo code_saturne executable path: $CS_EXEC
  echo ==========================================
  echo

  PROFILER_COMMAND="nsys profile -t mpi,cuda,nvtx --force-overwrite=true -b dwarf --cudabacktrace=all --cuda-memory-usage=true -o $PROFILER_OUTPUT_DIR/$1-$2.nsys-rep"

  cd $SATURNE_CASES_DIR/BUNDLE/$2
  $CS_EXEC up
  mkdir -p $PROFILER_OUTPUT_DIR
  SHELL=/bin/bash $CS_EXEC run --tool-args="$PROFILER_COMMAND"
}

build_config default "" master
build_config mempool_sorted "-DCS_MEM_POOL_SOR=1" nvtx_mem_pool_scope
build_config mempool_unsorted "" nvtx_mem_pool_scope

prepare_cases

profile_for_config default BENCH_C016_PREPROCESS
profile_for_config default BENCH_C016_01
profile_for_config mempool_sorted BENCH_C016_01
profile_for_config mempool_unsorted BENCH_C016_01
