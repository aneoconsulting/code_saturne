# Kernel loop unrolling summary

Kernels `_lsq_scalar_gradient_hyd_p` and `_lsq_scalar_gradient_hyd_p_gather`
supposedly had low arithmetic complexity which raised the question of unrolling.
Upon inspection it seems the parallel for loops in these kernels are already
quite complex, therefore the low arithmetic complexity functions might be
located somewhere else.

`cs_cuda_kernel_parallel_for` was modified to take an integer as a
template parameter and executes a parallel_for on the GPU unrolled
by an arbitrary factor. It relies on
[metaprogramming techniques](https://ieeexplore.ieee.org/document/8514391)
implemented in `cs_unroll` instead of `#pragma unroll` for consistent unrolling,
and possibly better performance.

## Implementation

`cs_dispatch.h`:
  - `cs_unroll` was added with an auxiliary function in the `detail` namespace.
    It unrolls lambda functions, passing a static index for each invocation.
  - `cs_cuda_kernel_parallel_for` was modified to provide an optional
    unroll template parameter.

The unroll factor is set to 1 (ie. no unroll) by default,
however this can be changed later.

**NB:** Implementing this kind of modifications with confidence is difficult
since there are no unit tests for dispatch algorithms

## Benchmarks and results

Benchmarking is done using NVTX and Nsight Systems in 
`tests/cs_dispatch_test.cpp`.

no gain on our synthetic benchmark
(filling a vector with incremental values)

**NB:** `cs_unroll` can be used in other places where unrolling make sense
and must be carried with certainty, eg. iterating over small arrays.
It can also be used to iterate over statically indexed objects such as
`std::tuple`, or to iterate over dimensions of N-D arrays.

## Conclusion

- Unrolling parallel_for provides no performance benefits
- `cs_unroll` might be useful in other parts of the code
