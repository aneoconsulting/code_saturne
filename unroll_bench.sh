#!/bin/bash

mkdir -p unroll_bench_results/

for GRID_SIZE in 1 2 4 8 16 32 64 128 256
do
  nsys profile -t nvtx \
    -o unroll_bench_results/bench-$GRID_SIZE.nsys-rep --force-overwrite=true \
    ./build/dev/tests/cs_dispatch_test $GRID_SIZE
done
