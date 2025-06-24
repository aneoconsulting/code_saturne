#!/bin/bash -e

source /etc/profile

SHELL=/bin/bash

# NVTX configuration

if [ -e /opt/nvidia/nsight-systems/2024.7.1/target-linux-x64/nvtx/include/ ]; then
  echo "Configuring NVTX for Rocky..."
  COMPILE_FLAGS+="-I /opt/nvidia/nsight-systems/2024.7.1/target-linux-x64/nvtx/include/ "
elif [ -e /opt/cuda/nsight_systems/target-linux-x64/nvtx/include/ ]; then
  echo "Configuring NVTX for Arch..."
  COMPILE_FLAGS+="-I /opt/cuda/nsight_systems/target-linux-x64/nvtx/include/ "
else
  echo "Warning: no NVTX configuration was detected."
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

COMPILE_FLAGS+="-std=c++17 "
COMPILE_FLAGS+="-g "
COMPILE_FLAGS+="-DCS_PROFILING=1 "

echo
echo ==========================================
echo COMPILE_FLAGS: "$COMPILE_FLAGS"
echo CONFIGURE_PATH: "$CONFIGURE_PATH"
echo ==========================================
echo

PROJECT_DIR=$(pwd)
SATURNE_CASES_DIR=$PROJECT_DIR/saturne-open-cases
PROFILER_OUTPUT_DIR=$PROJECT_DIR/profiler-output

NPROC=$(nproc --all)

function build_config () {
  BUILD_DIR=$PROJECT_DIR/build/$1
  INSTALL_DIR=$PROJECT_DIR/install/$1

  export CFLAGS="$COMPILE_FLAGS $2"
  export CXXFLAGS="$COMPILE_FLAGS $2"
  export NVCCFLAGS="$COMPILE_FLAGS -extended-lambda --expt-relaxed-constexpr $2"

  echo
  echo ==========================================
  echo Config build: $1
  echo Additional compile flags: $2
  echo Branch: $3
  echo
  echo Build directory: $BUILD_DIR
  echo Install directory: $INSTALL_DIR
  echo
  echo CFLAGS: $CFLAGS
  echo CXXFLAGS: $CXXFLAGS
  echo NVCCFLAGS: $NVCCFLAGS
  echo ==========================================
  echo

  cd $PROJECT_DIR
  git checkout $3
  ./sbin/bootstrap

  mkdir -p $BUILD_DIR
  cd $BUILD_DIR

  $CONFIGURE_PATH \
    --prefix=$INSTALL_DIR \
    --disable-gui \
    --enable-cuda --with-cuda=$CUDA_PATH

#     --enable-mpi=/usr/local/openmpi/ \
#     --with-mpi-lib=/usr/local/openmpi/lib/ \
#     --with-mpi-include=/usr/local/openmpi/include/

  if [ -e $PROJECT_DIR/compile_commands.json -o \( ! -e /usr/bin/bear \) ]; then
    make -j$NPROC
  else
    echo "Generating compile_commands using bear..."
    bear --output $PROJECT_DIR/compile_commands.json -- make -j$NPROC
  fi

  make install
}

# Profiling

function prepare_cases () {
  sudo rm -rf $SATURNE_CASES_DIR
  git clone git@github.com:code-saturne/saturne-open-cases.git $SATURNE_CASES_DIR
}

function profile_for_config () {

#   PROFILER_CMD+="nsys profile "
#   PROFILER_CMD+="-t cuda,nvtx --cuda-memory-usage=true "
#   PROFILER_CMD+="-b dwarf --cudabacktrace=all --cuda-um-gpu-page-faults=true "
#   PROFILER_CMD+="-o $PROFILER_OUTPUT_DIR/$1-$2.nsys-rep --force-overwrite=true "

  CS_EXEC=$PROJECT_DIR/install/$1/bin/code_saturne
  NCU_PATH=$(which ncu)
  SATURNE_WORKDIR=$SATURNE_CASES_DIR/BUNDLE/$2

  cd $SATURNE_WORKDIR

  # Using systemd-run to grant CAP_SYS_ADMIN capability to the process
#   PROFILER_CMD+="systemd-run --uid=jpenuchot --gid=jpenuchot -d -t "
#   PROFILER_CMD+="-p CapabilityBoundingSet=CAP_SYS_ADMIN "
#   PROFILER_CMD+="-p PassEnvironment=\"PATH CUDA_PATH LD_LIBRARY_PATH CFLAGS CXXFLAGS NVCCFLAGS\" "

  PROFILER_CMD+="$NCU_PATH -o $PROFILER_OUTPUT_DIR/$1-$2 -f --nvtx "

  echo
  echo ==========================================
  echo Config profile: $1
  echo Benchmark case: $2
  echo code_saturne executable path: $CS_EXEC
  echo Profiler command: $PROFILER_CMD
  echo ==========================================
  echo

  mkdir -p $PROFILER_OUTPUT_DIR
  sudo -E $CS_EXEC up
  sudo -E $CS_EXEC run --tool-args="$PROFILER_CMD"
}

function permission_cleanup () {
  sudo chown -hR jpenuchot:jpenuchot $PROFILER_OUTPUT_DIR $SATURNE_CASES_DIR
}

# User code

build_config dev "" jpenuchot/nccl

# prepare_cases
# profile_for_config dev BENCH_C016_PREPROCESS
# profile_for_config dev BENCH_C016_01
#
# permission_cleanup
