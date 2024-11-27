#ifndef CS_BASE_MEMPOOL_H
#define CS_BASE_MEMPOOL_H

#include "cs_base.h" // Inclure les en-têtes nécessaires
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

/*----------------------------------------------------------------------------*/

#include "cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Standard C++ library headers
 *----------------------------------------------------------------------------*/

#include <map>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft_error.h"
#include "bft_mem.h"

#if defined(HAVE_CUDA)
#include "cs_base_cuda.h"
#endif

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_base_accel.h"

class MemoryPool {
public:
  static MemoryPool &
  instance();

  cs_mem_block_t
  allocate(size_t          ni,
           size_t          size,
           cs_alloc_mode_t mode,
           const char     *var_name,
           const char     *file_name,
           int             line_num);

  cs_mem_block_t
  reallocate(cs_mem_block_t  &me_old,
             size_t          ni,
             size_t          size,
             cs_alloc_mode_t mode,
             const char     *var_name,
             const char     *file_name,
             int             line_num);

  void
  deallocate(cs_mem_block_t &me,
             const char     *var_name,
             const char     *file_name,
             int             line_num);

  //   void
  //   allocate_device(cs_mem_block_t &me);

private:
  MemoryPool();
  ~MemoryPool();

  cs_mem_block_t
  allocate_new_block(size_t          ni,
                     size_t          size,
                     cs_alloc_mode_t mode,
                     const char     *var_name,
                     const char     *file_name,
                     int             line_num);

  void
  free_block(cs_mem_block_t &me,
             const char     *var_name,
             const char     *file_name,
             int             line_num);

  std::mutex                                 mutex_;
  std::unordered_map<void *, cs_mem_block_t> allocated_blocks_;
  std::unordered_map<cs_alloc_mode_t, std::vector<cs_mem_block_t>> free_blocks_;
};

#endif // CS_BASE_MEMPOOL_H
