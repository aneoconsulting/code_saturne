# Profiling code_saturne

Fichier partagé pour le profiling de code_saturne. On peut y mettre les liens 
vers les différents fichiers de profiling et détailler les kernels/fonctions à 
traîter en indiquant si possible le type d'événement problématique observé et 
la backtrace pour pouvoir retracer dans le code.

## Fichiers de profiling

- Cas C016_01, RTX 4060 + AMD Ryzen 7 7435HS, mempool désactivé, version 
2025.1.1.131-251135540420v0: 
[https://drive.proton.me/urls/N5VPDNFG6C#2jFM5ZXH6IwF](https://drive.proton.me/
urls/N5VPDNFG6C#2jFM5ZXH6IwF)
- Cas C016_01, Rocky (NVIDIA L40S x4 + AMD EPYC 7R13), mempool désactivé: 
[https://drive.proton.me/urls/YND4851HJ4#X7TytTtD2uob](https://drive.proton.me/
urls/YND4851HJ4#X7TytTtD2uob)
- Case C016_04, RTX 3080 + Intel 10900, cs version 
ceb66e479a71d29d77aae7cb9a2d4e692eb8d403: 
[https://drive.proton.me/urls/0AW9N9B1TR#Y4gK2fHfx53W](https://drive.proton.me/
urls/0AW9N9B1TR#Y4gK2fHfx53W)
- Repo aneoconsulting du 17/04, branches `master` et `nvtx_profiling`, Rocky, 
backtraces activées, cas C016: 
[https://drive.proton.me/urls/9Q3T5H4REM#0KK3nmxlfYGn](https://drive.proton.me/
urls/9Q3T5H4REM#0KK3nmxlfYGn)

## Liste des kernels et fonctions à traiter

### cs_equation_iterative_solve_scalar

- Backtrace:

```
Nsight Systems frames
libcuda.so.570.133.07!0x740f1c8ac7a6
libcudart.so.12.8.90!cudaLaunchKernel
libsaturne-9.1.so
libsaturne-9.1.so!auto cs_combined_context<...>::parallel_for<...>(le const*, 
cs_field_bc_coeffs_t const*, double const*, double const*, double const*, double 
const*, double const*, double const*, double (*) [6], double const (*) [2], 
double const*, int, int const*, double const*, double*, double*, double*, double 
const*, double*), &cs_equation_iterative_solve_scalar, 6u>, void (int), double*, 
double*, double>>(int, __nv_hdl_wrapper_t<t*, int, int const*, double const*, 
double*, double*, double*, double const*, double*), 
&cs_equation_iterative_solve_scalar, 6u>, 
libsaturne-9.1.so!cs_equation_iterative_solve_scalar
libsaturne-9.1.so!cs_turbulence_ke
libsaturne-9.1.so!cs_solve_all
libsaturne-9.1.so!cs_time_stepping
libcs_solver-9.1.so!main
libc.so.6!0x740f5aa35488
libc.so.6!__libc_start_main
cs_solver!_start
```

- Observations: succession de page faults précédés d'un creux d'utilisation du 
GPU avec des transferts DtoH successifs

NB: Ce pattern avec cette fonction spécifiquement revient souvent dans le 
profiler.

### ???

```
Nsight Systems frames
libcuda.so.570.133.07!0x740f1c8ac7a6
libcudart.so.12.8.90!cudaStreamSynchronize
libsaturne-9.1.so!_compute_coarse_quantities_msr_with_faces(...)
libsaturne-9.1.so!cs_grid_coarsen
libsaturne-9.1.so!_setup_hierarchy
libsaturne-9.1.so!cs_multigrid_setup_conv_diff
libsaturne-9.1.so!_multigrid_pc_setup
libsaturne-9.1.so!cs_sles_it_setup_priv
libsaturne-9.1.so!cs_sles_it_setup
libsaturne-9.1.so!cs_sles_it_solve
libsaturne-9.1.so!cs_sles_solve
libsaturne-9.1.so!cs_sles_solve_ccc_fv
libsaturne-9.1.so!_pressure_correction_fv
libsaturne-9.1.so!cs_solve_navier_stokes
libsaturne-9.1.so!cs_solve_all
libsaturne-9.1.so!cs_time_stepping
libcs_solver-9.1.so!main
libc.so.6!0x740f5aa35488
libc.so.6!__libc_start_main
cs_solver!_start
```

- Observations: succession de page faults


- List of kernels (Damien)
    - void _*equation_*iterative_solve_strided
        - Call function cpu : cs_matrix_compute_coeffs

### cs_boundary_conditions_set_coeffs

```
Nsight Systems frames
libcuda.so.560.28.03!0x7fb8ab311db6
libcudart.so.12.6.37!cudaFree
libsaturne-9.1.so!cs_mem_cuda_free(...)
libsaturne-9.1.so!cs_mem_free
libsaturne-9.1.so!cs_boundary_conditions_set_coeffs
libsaturne-9.1.so!cs_solve_all
libsaturne-9.1.so!cs_time_stepping
libcs_solver-9.1.so!main
libc-2.28.so!__libc_start_main
cs_solver!_start
```

Temps mort de 33ms
