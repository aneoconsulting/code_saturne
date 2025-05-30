/*============================================================================
 * Specific physical models selection.
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

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_mem.h"
#include "bft/bft_error.h"
#include "bft/bft_printf.h"

/*----------------------------------------------------------------------------
 * Header for the current file
 *----------------------------------------------------------------------------*/

#include "pprt/cs_physical_model.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_physical_model.cpp
        Specific physical models selection.

  \var cs_physical_model_type_t::CS_JOULE_EFFECT
       Joule model flag position. Flag values can be:
       - -1: module not activated
       -  1: use of a real potential
       -  2: use of a complex potential
       -  3: use of real potential and specific boundary conditions
       -  4: use of complex potential and specific boundary conditions

  \var cs_physical_model_type_t::CS_ELECTRIC_ARCS
       Electric arcs model flag position. Flag values can be:
       - -1: module not activated
       -  1: determination of the magnetic field by means of the Ampere’
             theorem
       -  2: determination of the magnetic field by means of the vector
             potential
*/

/*----------------------------------------------------------------------------*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Global variables
 *============================================================================*/

/*! Status of specific physical models */

int cs_glob_physical_model_flag[CS_N_PHYSICAL_MODEL_TYPES] = {
  -1,  /* global specific physics flag */
       /* Combustion models: */
  -1,  /*    3-point model */
  -1,  /*    Steady laminar flamelet model */
  -1,  /*    EBU combustion model */
  -1,  /*    Libby-Williams combustion model */
  -1,  /*    Coal combustion model */
       /*  Electro-magnetism models: */
  -1,  /*    Joule effect */
  -1,  /*    Electric arcs */
       /*  Other code_saturne's modules */
  -1,  /*    Compressible model */
  -1,  /*    Atmospheric model */
  -1,  /*    Cooling towers */
  -1,  /*    Gas mix model */
  -1,  /*    Groundwater flows */
  -1,  /*    Solidification process */
  -1,  /*    Heat transfer (in solids) */
  -1   /*  NEPTUNE_CFD solver */
};

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Prototypes for functions intended for use only by Fortran wrappers.
 * (descriptions follow, with function bodies).
 *============================================================================*/

void
cs_f_physical_model_get_pointers(int     **ippmod);

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*============================================================================
 * Fortran wrapper function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Get pointers to members of the global physical model flags.
 *
 * This function is intended for use by Fortran wrappers, and
 * enables mapping to Fortran global pointers.
 *
 * parameters:
 *   ippmod --> pointer to cs_glob_physical_model_flag
 *----------------------------------------------------------------------------*/

void
cs_f_physical_model_get_pointers(int     **ippmod)
{
  *ippmod = cs_glob_physical_model_flag;
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*=============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/

END_C_DECLS
