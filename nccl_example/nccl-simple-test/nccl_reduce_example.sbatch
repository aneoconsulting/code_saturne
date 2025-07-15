#!/bin/bash
#SBATCH --job-name=mpi_nccl_tst
#SBATCH --partition=aws-rc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log
#SBATCH --export=ALL

# Transport intra-noeud (CUDA IPC + mémoire partagée)
export UCX_TLS=sm,cuda_copy,cuda_ipc
export NCCL_UCX_TLS=rc_x,cuda_copy
export NCCL_UCX_RNDV_THRESH=0
export UCX_MEMTYPE_CACHE=n
export NCCL_COLLNET_ENABLE=0
export NCCL_PLUGIN_P2P=ucx
export NCCL_DEBUG=info
export NCCL_DEBUG_SUBSYS=NET
#export LD_LIBRARY_PATH=$HPX_HOME/nccl_rdma_sharp_plugin/lib/:$LD_LIBRARY_PATH

mpiexec -np $SLURM_NTASKS --bind-to socket              \
        -x LD_LIBRARY_PATH                              \
        -x UCX_TLS                                      \
        -x NCCL_UCX_TLS                                 \
        -x NCCL_UCX_RNDV_THRESH                         \
        -x UCX_MEMTYPE_CACHE                            \
        -x NCCL_COLLNET_ENABLE                          \
        -x NCCL_PLUGIN_P2P                              \
        -x NCCL_DEBUG                                   \
        -x NCCL_DEBUG_SUBSYS                            \
        ./mpi_nccl_reduce

