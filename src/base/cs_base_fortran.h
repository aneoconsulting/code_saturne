#ifndef __CS_BASE_FORTRAN_H__
#define __CS_BASE_FORTRAN_H__

/*============================================================================
 * Initializtion and handling of Fortran-related mechanisms
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2025 EDF S.A.

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

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*=============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Replace default bft_printf() mechanism with internal mechanism.
 *
 * This variant is designed to allow switching from C to Fortran output,
 * whithout disabling regular C stdout output when switched to Fortran.
 *
 * This allows redirecting or suppressing logging for different ranks.
 *
 * parameters:
 *   log_name    <-- base file name for log, or NULL for stdout
 *   rn_log_flag <-- redirection for ranks > 0 log:
 *                   false:  to "/dev/null" (suppressed)
 *                   true: redirected to <log_name>_n*.log" file;
 *----------------------------------------------------------------------------*/

void
cs_base_fortran_bft_printf_set(const char  *log_name,
                               bool         rn_log_flag);

/*----------------------------------------------------------------------------
 * Switch bft_printf() mechanism to C output.
 *
 * This function may only be called after cs_base_fortran_bft_printf_set()
 *----------------------------------------------------------------------------*/

void
cs_base_fortran_bft_printf_to_c(void);

/*----------------------------------------------------------------------------
 * Switch bft_printf() mechanism to Fortran output.
 *
 * This function may only be called after cs_base_fortran_bft_printf_set()
 *----------------------------------------------------------------------------*/

void
cs_base_fortran_bft_printf_to_f(void);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_BASE_FORTRAN_H__ */
