/*
 *  This file is part of code_saturne, a general-purpose CFD tool.
 *
 *  Copyright (C) 1998-2025 EDF S.A.
 *
 *  This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option) any
 * later version.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#pragma once

#include "base/cs_log.h"
#include "base/cs_mem.h"

/*==============================================================================
 * RAII wrapper for automatic cs_mem_pool activation/deactivation.
 *============================================================================*/

struct cs_mem_pool_scope_t {
  // static std::mutex _mem_pool_scope_mutex;

  /// False by default, set to true when the first instance of cs_mem_pool_scope
  /// is instantiated, and back to false when the same instance is destroyed.
  /// This ensures only one instance of cs_mem_pool_scope is in charge of
  /// activating and deactivating the memory pool.
  static bool _mem_pool_scope_latch;

  /// Set to true if the current instance is the one in charge of handling
  /// activation and deactivation of cs_mem_device_pool.
  bool _enable_scope;

  /// Enables mem_device_pool
  cs_mem_pool_scope_t()
  {
    // std::lock_guard const guard(_mem_pool_scope_mutex);

    if (_mem_pool_scope_latch) {
      _enable_scope = false;
      cs_log_warning("An instance of cs_mem_pool_scope was instantiated while "
                     "another one is already active.");
    }
    else {
      _enable_scope = true;
      cs_mem_device_pool_set_active(true);
      cs_log_printf(CS_LOG_DEFAULT, "cs_mem_pool_scope: enabled memory pool");
    }
  }

  /// Deactivates mem_device_pool and clears it.
  ~cs_mem_pool_scope_t()
  {
    if (!_enable_scope) {
      return;
    }

    // std::lock_guard const guard(_mem_pool_scope_mutex);

    cs_mem_device_pool_clear();
    cs_mem_device_pool_set_active(false);

    _mem_pool_scope_latch = false;

    cs_log_printf(CS_LOG_DEFAULT, "cs_mem_pool_scope: disabled memory pool");
  }
};
