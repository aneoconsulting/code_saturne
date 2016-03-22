/*============================================================================
 * Definition of advanced options relative to parallelism.
 *============================================================================*/

/* VERS */

/*
  This file is part of Code_Saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2016 EDF S.A.

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

#include "cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "bft_error.h"
#include "bft_mem.h"
#include "bft_printf.h"

#include "cs_base.h"
#include "cs_file.h"
#include "cs_grid.h"
#include "cs_matrix.h"
#include "cs_matrix_default.h"
#include "cs_parall.h"
#include "cs_partition.h"
#include "cs_renumber.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_prototypes.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*----------------------------------------------------------------------------*/
/*!
 * \file cs_user_performance_tuning-numbering.c
 *
 * \brief Mesh numbering example.
 *
 * See \subpage cs_user_performance_tuning for examples.
 */
/*----------------------------------------------------------------------------*/

/*============================================================================
 * User function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define advanced mesh numbering options.
 */
/*----------------------------------------------------------------------------*/

void
cs_user_numbering(void)
{
  BEGIN_EXAMPLE_SCOPE

  /*! [performance_tuning_numbering] */

  /* Force the target number of threads for mesh renumbering
     (by default, OMP_NUM_THREADS if OpenMP is enabled, 1 otherwise) */

  cs_renumber_set_n_threads(4);

  /* Set the minimum subset sizes when renumbering for threads. */

  cs_renumber_set_min_subset_size(64,   /* min. interior_subset_size */
                                  64);  /* min. boundary subset_size */

  /* Select renumbering algorithms.

     For cells, available algorithms are:

     CS_RENUMBER_CELLS_SCOTCH_PART     (SCOTCH sub-partitioning, if available)
     CS_RENUMBER_CELLS_SCOTCH_ORDER    (SCOTCH ordering, if available)
     CS_RENUMBER_CELLS_METIS_PART      (METIS sub-partitioning, if available)
     CS_RENUMBER_CELLS_METIS_ORDER     (METIS ordering, if available)
     CS_RENUMBER_CELLS_MORTON          (Morton space filling curve)
     CS_RENUMBER_CELLS_HILBERT         (Hilbert space filling curve)
     CS_RENUMBER_CELLS_NONE            (no renumbering)

     For interior faces, available algorithms are:

     CS_RENUMBER_I_FACES_BLOCK       (no shared cell in block)
     CS_RENUMBER_I_FACES_MULTIPASS   (use multipass face numbering)
     CS_RENUMBER_I_FACES_SIMD        (renumbering for SIMD)
     CS_RENUMBER_I_FACES_NONE        (no interior face numbering)

     Before applying one of those algorithms, interior faces are pre-ordered
     by a lexicographal ordering based on adjacent cells; this ordering
     may be based on the lowest or highest adjacent id first, as defined
     by the CS_RENUMBER_ADJACENT_LOW or CS_RENUMBER_ADJACENT_HIGH value.

     For boundary faces, available algorithms are:

     CS_RENUMBER_B_FACES_THREAD      (renumber for threads)
     CS_RENUMBER_B_FACES_SIMD        (renumbering for SIMD)
     CS_RENUMBER_B_FACES_NONE        (no interior face numbering)
  */

  cs_renumber_set_algorithm
    (false,                           /* halo_adjacent_cells_last */
     false,                           /* halo_adjacent_i_faces_last */
     CS_RENUMBER_ADJACENT_LOW,        /* interior face base ordering  */
     CS_RENUMBER_CELLS_NONE,          /* cells_pre_numbering */
     CS_RENUMBER_CELLS_NONE,          /* cells_numbering */
     CS_RENUMBER_I_FACES_MULTIPASS,   /* interior faces numbering */
     CS_RENUMBER_B_FACES_THREAD);     /* boundary faces numbering */

  /*! [performance_tuning_numbering] */

  END_EXAMPLE_SCOPE
}

/*----------------------------------------------------------------------------*/

END_C_DECLS