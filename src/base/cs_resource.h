#ifndef __CS_RESOURCE_H__
#define __CS_RESOURCE_H__

/*============================================================================
 * Resource allocation management (available time).
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

#include "base/cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Macro Definitions
 *============================================================================*/

/*=============================================================================
 * Type Definitions
 *============================================================================*/

/*============================================================================
 *  Global variables
 *============================================================================*/

/*============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get current wall-clock time limit.
 *
 * \return current wall-time limit (in seconds), or -1
 */
/*----------------------------------------------------------------------------*/

double
cs_resource_get_wt_limit(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set wall-clock time limit.
 *
 * \param[in]  wt  wall-time limit (in seconds), or -1
 */
/*----------------------------------------------------------------------------*/

void
cs_resource_set_wt_limit(double  wt);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Limit number of remaining time steps if the remaining allocated
 *        time is too small to attain the requested number of steps.
 *
 * \param[in]       ts_cur  current time step number
 * \param[in, out]  ts_max  maximum time step number
 */
/*----------------------------------------------------------------------------*/

void
cs_resource_get_max_timestep(int   ts_cur,
                             int  *ts_max);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_RESOURCE_H__ */
