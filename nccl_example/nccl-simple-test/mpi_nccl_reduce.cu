/*********************************************************************
 * Example: MPI + NCCL AllReduce across 4 GPUs on a single node
 * Each rank sets its GPU value = rank + 1.0
 * ncclAllReduce(sum) should give N*(N+1)/2 = 10 for 4 ranks
 *********************************************************************/

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_MPI(cmd)  do {                               \
  int e = (cmd);                                           \
  if (e != MPI_SUCCESS) {                                  \
    fprintf(stderr,"MPI error %d at %s:%d\n", e,           \
            __FILE__, __LINE__);                           \
    MPI_Abort(MPI_COMM_WORLD, e);                          \
  }                                                        \
} while(0)

#define CHECK_CUDA(cmd) do {                               \
  cudaError_t e = (cmd);                                   \
  if (e != cudaSuccess) {                                  \
    fprintf(stderr,"CUDA error %s at %s:%d\n",             \
            cudaGetErrorString(e), __FILE__, __LINE__);    \
    exit(EXIT_FAILURE);                                    \
  }                                                        \
} while(0)

#define CHECK_NCCL(cmd) do {                               \
  ncclResult_t r = (cmd);                                  \
  if (r != ncclSuccess) {                                  \
    fprintf(stderr,"NCCL error %s at %s:%d\n",             \
            ncclGetErrorString(r), __FILE__, __LINE__);    \
    exit(EXIT_FAILURE);                                    \
  }                                                        \
} while(0)

int main(int argc, char *argv[])
{
  /*----------------------------------------------------------------*/
  /* 1. Init MPI                                                    */
  /*----------------------------------------------------------------*/
  CHECK_MPI( MPI_Init(&argc, &argv) );
  int rank, size;
  CHECK_MPI( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
  CHECK_MPI( MPI_Comm_size(MPI_COMM_WORLD, &size) );
  if (size != 4) {
    if (rank == 0)
      fprintf(stderr,"Error: launch with exactly 4 ranks (-n 4)\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /*----------------------------------------------------------------*/
  /* 2. Choisir un GPU par rang                                      */
  /*----------------------------------------------------------------*/
  int numDev = 0;
  CHECK_CUDA( cudaGetDeviceCount(&numDev) );
  if (numDev < size) {
    if (rank == 0)
      fprintf(stderr,"Error: found only %d GPU(s)\n", numDev);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CHECK_CUDA( cudaSetDevice(rank) );          /* 1 GPU / rank            */
  cudaStream_t stream;
  CHECK_CUDA( cudaStreamCreate(&stream) );

  /*----------------------------------------------------------------*/
  /* 3. Initialiser NCCL (id broadcasté via MPI)                     */
  /*----------------------------------------------------------------*/
  ncclUniqueId id;
  if (rank == 0) CHECK_NCCL( ncclGetUniqueId(&id) );
  CHECK_MPI( MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) );

  ncclComm_t ncclComm;
  CHECK_NCCL( ncclCommInitRank(&ncclComm, size, id, rank) );

  /*----------------------------------------------------------------*/
  /* 4. Allouer & initialiser buffer GPU                             */
  /*----------------------------------------------------------------*/
  double hostValue = (double)(rank + 1);      /* 1.0, 2.0, 3.0, 4.0      */
  double *dBuf = nullptr;
  CHECK_CUDA( cudaMalloc(&dBuf, sizeof(double)) );
  CHECK_CUDA( cudaMemcpyAsync(dBuf, &hostValue, sizeof(double),
                              cudaMemcpyHostToDevice, stream) );

  /*----------------------------------------------------------------*/
  /* 5. ncclAllReduce : somme de tous les rangs                      */
  /*----------------------------------------------------------------*/
  CHECK_NCCL( ncclAllReduce((const void*)dBuf, (void*)dBuf,
                            1, ncclDouble, ncclSum,
                            ncclComm, stream) );

  /* Synchro : s’assurer que la réduction est terminée               */
  CHECK_CUDA( cudaStreamSynchronize(stream) );

  /*----------------------------------------------------------------*/
  /* 6. Ramener le résultat sur host et afficher                     */
  /*----------------------------------------------------------------*/
  double result = 0.0;
  CHECK_CUDA( cudaMemcpy(&result, dBuf, sizeof(double),
                         cudaMemcpyDeviceToHost) );

  if (rank == 0)
    printf("[rank %d]  AllReduce result = %f (expect 10.0)\n", rank, result);

  /*----------------------------------------------------------------*/
  /* 7. Nettoyage                                                   */
  /*----------------------------------------------------------------*/
  CHECK_NCCL( ncclCommDestroy(ncclComm) );
  CHECK_CUDA( cudaFree(dBuf) );
  CHECK_CUDA( cudaStreamDestroy(stream) );
  CHECK_MPI ( MPI_Finalize() );
  return 0;
}
