/*============================================================================
 * Definitions, global variables, and base functions for accelerators.
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2024 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*----------------------------------------------------------------------------*/

#include "cs_mem_pool.h"
#include "bft_mem.h"
#include "cs_base.h"

#if defined(HAVE_CUDA)
#include "cs_mem_cuda_priv.h"
#endif

MemoryPool &
MemoryPool::instance()
{
  static std::once_flag flag;
  static MemoryPool    *instance = nullptr;
  std::call_once(flag, []() { instance = new MemoryPool(); });
  return *instance;
}

MemoryPool::MemoryPool() = default;

MemoryPool::~MemoryPool()
{
  for (auto &mode_blocks : free_blocks_) {
    cs_alloc_mode_t mode = mode_blocks.first;
    for (auto &me : mode_blocks.second) {
      free_block(me, "me.hostptr", __FILE__, __LINE__);
    }
  }
}

cs_mem_block_t
MemoryPool::allocate(size_t          ni,
                     size_t          size,
                     cs_alloc_mode_t mode,
                     const char     *var_name,
                     const char     *file_name,
                     int             line_num)
{
  std::lock_guard<std::mutex> lock(mutex_);

  size_t adjusted_size = ni * size;

  const int TTL_MAX = 300;

  auto &free_blocks = free_blocks_[mode];

  auto it = free_blocks.begin();
  while (it != free_blocks.end()) {
    it->ttl += 1;
    if (it->ttl >= TTL_MAX) {
      free_block(*it, var_name, file_name, line_num);
      it = free_blocks.erase(it);
    }
    else {
      ++it;
    }
  }

  for (auto it2 = free_blocks.begin(); it2 != free_blocks.end(); ++it2) {
    if (it2->size == adjusted_size) {
      cs_mem_block_t me = *it2;
      free_blocks.erase(it2);
      if (me.host_ptr != nullptr)
        allocated_blocks_[me.host_ptr] = me;
      else if (me.device_ptr != nullptr) {
        allocated_blocks_[me.device_ptr] = me;
      }
      else {
        assert(me.device_ptr == nullptr && me.host_ptr == nullptr);
      }
      return me;
    }
  }

  cs_mem_block_t me =
    allocate_new_block(ni, size, mode, var_name, file_name, line_num);
  if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    if (me.host_ptr != nullptr)
      allocated_blocks_[me.host_ptr] = me;
    else if (me.device_ptr != nullptr) {
      allocated_blocks_[me.device_ptr] = me;
    }
  }
  return me;
}

cs_mem_block_t
MemoryPool::reallocate(cs_mem_block_t &me_old,
                       size_t          ni,
                       size_t          size,
                       cs_alloc_mode_t mode,
                       const char     *var_name,
                       const char     *file_name,
                       int             line_num)
{
  cs_mem_block_t me;

  if (me_old.mode <= CS_ALLOC_HOST_DEVICE && mode <= CS_ALLOC_HOST_DEVICE) {
    me = allocate(ni, size, mode, var_name, file_name, line_num);
    deallocate(me_old, var_name, file_name, line_num);
  }

  return me;
}

void
MemoryPool::deallocate(cs_mem_block_t &me,
                       const char     *var_name,
                       const char     *file_name,
                       int             line_num)
{
  if (me.host_ptr == nullptr && me.device_ptr == nullptr)
    return;

  std::lock_guard<std::mutex> lock(mutex_);

  void *ptr = nullptr;

  if (me.host_ptr == nullptr && me.device_ptr != nullptr)
    ptr = me.device_ptr;
  else if (me.host_ptr != nullptr)
    ptr = me.host_ptr;

  auto it = allocated_blocks_.find(ptr);
  if (it != allocated_blocks_.end()) {
    cs_mem_block_t me = it->second;
    allocated_blocks_.erase(it);
    me.ttl = 0;
    free_blocks_[me.mode].push_back(me);
  }
  else {
    cs_mem_block_t me = get_block_info_try(ptr);
    if (me.mode == CS_ALLOC_HOST_DEVICE_SHARED) {
      me.ttl = 0;
      free_blocks_[me.mode].push_back(me);
    }
    else {
     free_block(me, var_name, file_name, line_num);
    }
  }
}

cs_mem_block_t
MemoryPool::allocate_new_block(size_t          ni,
                               size_t          size,
                               cs_alloc_mode_t mode,
                               const char     *var_name,
                               const char     *file_name,
                               int             line_num)
{
  cs_mem_block_t me = { .host_ptr   = nullptr,
                        .device_ptr = nullptr,
                        .size       = ni * size,
                        .mode       = mode };
  me.ttl            = 0;

  if (mode < CS_ALLOC_HOST_DEVICE_PINNED) {
    me.host_ptr = cs_mem_malloc(ni, size, var_name, nullptr, 0);
  }

  // Device allocation will be postponed later thru call to
  // cs_get_device_ptr. This applies for CS_ALLOC_HOST_DEVICE
  // and CS_ALLOC_HOST_DEVICE_PINNED modes

#if defined(HAVE_CUDA)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED)
    me.host_ptr =
      cs_mem_cuda_malloc_host(me.size, var_name, file_name, line_num);

  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr =
      cs_mem_cuda_malloc_managed(me.size, var_name, file_name, line_num);
    me.device_ptr = me.host_ptr;
  }

  else if (mode == CS_ALLOC_DEVICE)
    me.device_ptr =
      cs_mem_cuda_malloc_device(me.size, var_name, file_name, line_num);

#elif defined(SYCL_LANGUAGE_VERSION)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED)
    me.host_ptr = _sycl_mem_malloc_host(me.size, var_name, file_name, line_num);

  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr =
      _sycl_mem_malloc_shared(me.size, var_name, file_name, line_num);
    me.device_ptr = me.host_ptr;
  }

  else if (mode == CS_ALLOC_DEVICE)
    me.device_ptr =
      _sycl_mem_malloc_device(me.size, var_name, file_name, line_num);

#elif defined(HAVE_OPENMP_TARGET)

  else if (mode == CS_ALLOC_HOST_DEVICE_PINNED)
    me.host_ptr =
      _omp_target_mem_malloc_host(me.size, var_name, file_name, line_num);

  else if (mode == CS_ALLOC_HOST_DEVICE_SHARED) {
    me.host_ptr =
      _omp_target_mem_malloc_managed(me.size, var_name, file_name, line_num);
    me.device_ptr = me.host_ptr;
  }

  else if (mode == CS_ALLOC_DEVICE)
    me.device_ptr =
      _omp_target_mem_malloc_device(me.size, var_name, file_name, line_num);

#endif

  return me;
}

void
MemoryPool::free_block(cs_mem_block_t &me,
                       const char     *var_name,
                       const char     *file_name,
                       int             line_num)
{
  if (me.host_ptr != nullptr) {
#if defined(HAVE_CUDA)
    if (me.mode < CS_ALLOC_HOST_DEVICE_PINNED)
    {
      free(me.host_ptr);
    }
    else if (me.mode == CS_ALLOC_HOST_DEVICE_SHARED)
      cs_mem_cuda_free(me.host_ptr, var_name, file_name, line_num);
    else
      cs_mem_cuda_free_host(me.host_ptr, var_name, file_name, line_num);

#elif defined(SYCL_LANGUAGE_VERSION)

    sycl::free(me.host_ptr, cs_glob_sycl_queue);

#elif defined(HAVE_OPENMP_TARGET)

    omp_target_free(me.host_ptr, _omp_target_device_id);

#endif
  }

  if (me.device_ptr != nullptr && me.device_ptr != me.host_ptr) {
#if defined(HAVE_CUDA)

    cs_mem_cuda_free(me.device_ptr, var_name, file_name, line_num);

#elif defined(SYCL_LANGUAGE_VERSION)

    sycl::free(me.device_ptr, cs_glob_sycl_queue);

#elif defined(HAVE_OPENMP_TARGET)

    omp_target_free(me.device_ptr, _omp_target_device_id);

#endif
  }
}
