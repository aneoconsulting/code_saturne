# Unrolling notes

Kernels `_lsq_scalar_gradient_hyd_p` and `_lsq_scalar_gradient_hyd_p_gather` 
supposedly had low arithmetic complexity which raised the question of unrolling.
Upon inspection it seems the parallel for loops in these kernels are already 
quite complex, therefore the low arithmetic complexity functions might be 
located somewhere else.

`parallel_for_unrolled` takes an integer as a template parameter and executes
a parallel_for on the GPU with user-defined static unrolling. It relies on 
[metaprogramming techniques](https://ieeexplore.ieee.org/document/8514391) 
and not `#pragma unroll` for more robustness and possibly more performance.
