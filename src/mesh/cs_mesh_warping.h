#ifndef __CS_MESH_WARPING_H__
#define __CS_MESH_WARPING_H__

/*============================================================================
 * Cut warped faces in serial or parallel with/without periodicity.
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
#include "mesh/cs_mesh.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*=============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Cut warped faces.
 *
 * Updates border face connectivity and associated mesh quantities.
 *
 * parameters:
 *   mesh           <-> pointer to mesh structure.
 *   max_warp_angle <-- criterion to know which face to cut
 *   post_flag      <-- 1 if we have to post-process cut faces, 0 otherwise
 *----------------------------------------------------------------------------*/

void
cs_mesh_warping_cut_faces(cs_mesh_t  *mesh,
                          double      max_warp_angle,
                          bool        post_flag);

/*----------------------------------------------------------------------------
 * Set defaults for cutting of warped faces.
 *
 * parameters:
 *   max_warp_angle <-- maximum warp angle (in degrees) over which faces will
 *                      be cut; negative (-1) if faces should not be cut
 *   postprocess    <-- 1 if postprocessing should be activated when cutting
 *                      warped faces, 0 otherwise
 *----------------------------------------------------------------------------*/

void
cs_mesh_warping_set_defaults(double  max_warp_angle,
                             int     postprocess);

/*----------------------------------------------------------------------------
 * Get defaults for cutting of warped faces.
 *
 * parameters:
 *   max_warp_angle --> if non-null, returns maximum warp angle (in degrees)
 *                      over which faces will be cut, or -1 if faces should
 *                      not be cut
 *   postprocess    --> if non-null, returns 1 if postprocessing should be
 *                      activated when cutting warped faces, 0 otherwise
 *----------------------------------------------------------------------------*/

void
cs_mesh_warping_get_defaults(double  *max_warp_angle,
                             int     *postprocess);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_MESH_WARPING_H__ */
