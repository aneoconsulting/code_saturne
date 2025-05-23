#ifndef __CS_LAGR_RESTART_H__
#define __CS_LAGR_RESTART_H__

/*============================================================================
 * Checkpoint/restart handling for Lagrangian module.
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

#include "base/cs_defs.h"
#include "base/cs_field.h"
#include "base/cs_map.h"
#include "base/cs_restart.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*=============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Read particle data from checkpoint.
 *
 * parameters:
 *   r <->  associated restart file pointer
 *
 * returns:
 *   number of particle arrays read
 *----------------------------------------------------------------------------*/

int
cs_lagr_restart_read_particle_data(cs_restart_t  *r);

/*----------------------------------------------------------------------------
 * Write particle data to checkpoint.
 *
 * parameters:
 *   r <->  associated restart file pointer
 *
 * returns:
 *   number of particle arrays written
 *----------------------------------------------------------------------------*/

int
cs_lagr_restart_write_particle_data(cs_restart_t  *r);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_LAGR_RESTART_H__ */
