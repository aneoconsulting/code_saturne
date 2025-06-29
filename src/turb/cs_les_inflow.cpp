/*============================================================================
 * Turbulent inflow generation
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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_error.h"
#include "bft/bft_printf.h"

#include "base/cs_array.h"
#include "base/cs_base.h"
#include "base/cs_dispatch.h"
#include "base/cs_field.h"
#include "base/cs_field_pointer.h"
#include "base/cs_log.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_parall.h"
#include "base/cs_random.h"
#include "base/cs_reducers.h"
#include "base/cs_timer.h"
#include "mesh/cs_mesh_location.h"
#include "base/cs_restart.h"
#include "base/cs_restart_default.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "turb/cs_les_inflow.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro Definitions
 *============================================================================*/

/*=============================================================================
 * Local Structure Definitions
 *============================================================================*/

/* Inlet definition */
/*------------------*/

struct _cs_inlet_t {

  /* Synthetic inflow method */

  cs_les_inflow_type_t   type;
  void                  *inflow;

  int                    initialize;         /* Indicator of initialization */
  int                    verbosity;          /* Indicator of verbosity level*/

  /* Geometric informations */

  const cs_zone_t       *zone;               /* Pointer to associated
                                                boundary zone */

  cs_real_3_t           *face_center;
  cs_real_t             *face_surface;

  /* Mean flow information */

  cs_real_t              vel_m[3];           /* Mean velocity */
  cs_real_t              k_r;                /* Level of energy */
  cs_real_t              eps_r;              /* Level of dissipation rate */

  /* Counters */

  cs_real_t             wt_tot;                /* Total wall-clock time used  */
  cs_real_t             cpu_tot;               /* Total (local) CPU used      */

};

/* Batten method */
/*---------------*/

typedef struct _cs_inflow_batten_t {

  int           n_modes;              /* Number of modes */

  cs_real_t    *frequency;            /* Frequency:   normal law N(1,1) */
  cs_real_3_t  *wave_vector;          /* Wave vector: normal law N(0,1/2) */
  cs_real_3_t  *amplitude_cos;        /* Amplitudes of the cosines */
  cs_real_3_t  *amplitude_sin;        /* Amplitudes of the sines */

} cs_inflow_batten_t;

/* Synthetic Eddy Method (SEM) */
/*-----------------------------*/

typedef struct {
  double  val;
  int     rank;
} _mpi_double_int_t;

/*============================================================================
 *  Global variables
 *============================================================================*/

/* Names for synthetic turbulence generation method */

const char *cs_inflow_type_name[] = {"Laminar",
                                     "Random",
                                     "Batten",
                                     "SEM"};

/* Structures associated to the inlets */

static int            cs_glob_inflow_n_inlets    = 0;
static cs_inlet_t   **cs_glob_inflow_inlet_array = nullptr;
static cs_restart_t  *_inflow_restart = nullptr;

static bool  _allow_restart_read = true;
static bool  _allow_restart_write = true;

/* SEM when restarting from another turbulence model */

static int   _n_sem_vol_restart_structures = 50;

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Generation of synthetic turbulence via a Gaussian random method.
 *
 * parameters:
 *   n_points       --> Local number of points where turbulence is generated
 *   fluctuations   <-- Velocity fluctuations generated
 *----------------------------------------------------------------------------*/

static void
_random_method(cs_lnum_t    n_points,
               cs_real_3_t  fluctuations[])
{
  cs_real_t random[3];

  for (cs_lnum_t point_id = 0; point_id < n_points; point_id++) {
    cs_random_normal(3, random);
    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      fluctuations[point_id][coo_id] = random[coo_id];
  }
}

/*----------------------------------------------------------------------------
 * Generation of synthetic turbulence via the Batten method.
 *
 * parameters:
 *   n_points          --> Local number of points where turbulence is generated
 *   point_coordinates --> Coordinates of the points
 *   initialize        --> Indicator of initialization
 *   inflow            --> Specific structure for Batten method
 *   time              --> Current time at the present iteration
 *   rij_l             --> Reynolds stresses at each point
 *   eps_r             --> Dissipation rate at each point
 *   fluctuations      <-- Velocity fluctuations generated at each point
 *----------------------------------------------------------------------------*/

static void
_batten_method(cs_lnum_t            n_points,
               const cs_real_3_t   *point_coordinates,
               int                  initialize,
               cs_inflow_batten_t  *inflow,
               cs_real_t            time,
               const cs_real_6_t   *rij_l,
               const cs_real_t     *eps_r,
               cs_real_3_t         *fluctuations)
{
  const cs_real_t  two_pi              = 2.*acos(-1.);
  const cs_real_t  sqrt_three_half     = sqrt(1.5);
  const cs_real_t  sqrt_two_by_n_modes = sqrt(2./inflow->n_modes);

  cs_real_t *frequency = inflow->frequency;
  cs_real_3_t *wave_vector = inflow->wave_vector;
  cs_real_3_t *amplitude_cos = (cs_real_3_t *)inflow->amplitude_cos;
  cs_real_3_t *amplitude_sin = (cs_real_3_t *)inflow->amplitude_sin;
  const cs_lnum_t n_modes = inflow->n_modes;

  cs_dispatch_context ctx;

  if (initialize == 1) {

    const int three_n_modes   = 3*n_modes;
    const cs_real_t one_by_sqrt_two = sqrt(0.5);

    if (cs_glob_rank_id <= 0) {

      /* Random generation of the n_modes frequencies following a normal
         law with a mean of 1 and a variance of 1 (i.e. N(1,1)). */

      cs_random_normal(n_modes, frequency);

      for (cs_lnum_t mode_id = 0; mode_id < n_modes; mode_id++)
        frequency[mode_id] += 1.;

      /* Random generation of the n_modes wave vectors following a normal
         law with a mean of 0 and a variance of 0.5 (i.e. N(0,1/2)). */

      cs_random_normal(three_n_modes, (cs_real_t *)wave_vector);

      for (cs_lnum_t mode_id = 0; mode_id < n_modes; mode_id++) {
        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
          wave_vector[mode_id][coo_id] *= one_by_sqrt_two;
      }

      /* Generation of the n_modes amplitude vector for both the sines and
         the cosines. */

      for (cs_lnum_t mode_id = 0; mode_id < n_modes; mode_id++) {

        cs_real_t  rcos[3];
        cs_real_t  rsin[3];

        /* Temporary random vectors following a normal law N(0,1) necessary
           to compute the random amplitudes of the sines and cosines */

        cs_random_normal(3, rcos);
        cs_random_normal(3, rsin);

        cs_math_3_cross_product(rcos,
                                wave_vector[mode_id],
                                amplitude_cos[mode_id]);

        cs_math_3_cross_product(rsin,
                                wave_vector[mode_id],
                                amplitude_sin[mode_id]);

      }

    }

#if defined(HAVE_MPI)

    if (cs_glob_rank_id >= 0) {

      MPI_Bcast(inflow->frequency,     n_modes, CS_MPI_REAL, 0,
                cs_glob_mpi_comm);
      MPI_Bcast(inflow->wave_vector,   three_n_modes, CS_MPI_REAL, 0,
                cs_glob_mpi_comm);
      MPI_Bcast(inflow->amplitude_cos, three_n_modes, CS_MPI_REAL, 0,
                cs_glob_mpi_comm);
      MPI_Bcast(inflow->amplitude_sin, three_n_modes, CS_MPI_REAL, 0,
                cs_glob_mpi_comm);

    }

#endif

  }

  ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {

    cs_real_t spectral_time;
    cs_real_t spectral_coordinates[3];

    /*
      Compute integral scales of turbulence :
      -  Tb = k / epsilon
      -  Vb = sqrt(k)
      -  Lb = Tb * Vb     ( = k^(3/2) / epsilon )
    */

    cs_real_t k_r = 0.5 * cs_math_6_trace(rij_l[point_id]);

    cs_real_t time_scale     = k_r / eps_r[point_id];
    cs_real_t velocity_scale = sqrt(k_r);
    cs_real_t lenght_scale   = time_scale * velocity_scale;

    /* Spectral position of the point in space and time */

    spectral_time = two_pi * time / time_scale;

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
      spectral_coordinates[coo_id]
        = two_pi * point_coordinates[point_id][coo_id] / lenght_scale;
    }

    /* Compute the velocity fluctuations */

    for (cs_lnum_t mode_id = 0; mode_id < n_modes; mode_id++) {

      cs_real_t mod_wave_vector[3];

      cs_real_t norm_wave_vector =
        cs_math_3_norm((const cs_real_t *)(wave_vector[mode_id]));

      cs_real_t spectral_velocity_scale
        = cs_math_3_sym_33_3_dot_product(wave_vector[mode_id],
                                         rij_l[point_id],
                                         wave_vector[mode_id]);

      spectral_velocity_scale =   sqrt_three_half*sqrt(spectral_velocity_scale)
                                / norm_wave_vector;

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
        mod_wave_vector[coo_id]
          =   wave_vector[mode_id][coo_id]
            * velocity_scale / spectral_velocity_scale;
      }

      cs_real_t dxpot
        =    cs_math_3_dot_product(mod_wave_vector, spectral_coordinates)
          +  frequency[mode_id]*spectral_time;

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
        fluctuations[point_id][coo_id]
          +=   amplitude_cos[mode_id][coo_id]*cos(dxpot)
             + amplitude_sin[mode_id][coo_id]*sin(dxpot);
      }

    }

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      fluctuations[point_id][coo_id] *= sqrt_two_by_n_modes;

  });

  ctx.wait();
}

/*----------------------------------------------------------------------------
 * Modify the normal component of the fluctuations such that the mass flowrate
 * of the fluctuating field is zero.
 *
 * parameters:
 *   n_points          --> Local number of points where turbulence is generated
 *   face_ids          --> Local id of inlet boundary faces
 *   fluctuations      <-> Velocity fluctuations
 *----------------------------------------------------------------------------*/

static void
_rescale_flowrate(cs_lnum_t         n_points,
                  const cs_lnum_t   face_ids[],
                  cs_real_3_t       fluctuations[])
{
  /* Compute the mass flow rate of the fluctuating field */
  /* and the area of the inlet */

  cs_dispatch_context ctx;

  cs_real_t *density = CS_F_(rho)->val;
  const cs_mesh_t  *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t  *mq = cs_glob_mesh_quantities;

  const cs_lnum_t *b_face_cells = mesh->b_face_cells;
  const cs_lnum_t *b_face_vtx_idx = mesh->b_face_vtx_idx;
  const cs_lnum_t *b_face_vtx_lst = mesh->b_face_vtx_lst;
  const cs_real_3_t *vtx_coord = (const cs_real_3_t *)mesh->vtx_coord;
  const cs_nreal_3_t *b_face_u_normal = mq->b_face_u_normal;
  const cs_real_t *b_face_surf = mq->b_face_surf;

  struct cs_double_n<2> rd_sum_2d;
  struct cs_reduce_sum_nr<2> reducer_sum_2d;

  ctx.parallel_for_reduce
    (n_points, rd_sum_2d, reducer_sum_2d,
     [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_double_n<2> &res) {

    cs_lnum_t face_id = face_ids[point_id];
    cs_lnum_t cell_id = b_face_cells[face_id];

    const cs_real_t *fluct = fluctuations[point_id];
    const cs_nreal_t *u_normal = b_face_u_normal[face_id];

    cs_real_t dot_product
      = b_face_surf[face_id] * cs_math_3_dot_product(fluct, u_normal);

    res.r[0] = density[cell_id]*dot_product;
    res.r[1] = b_face_surf[face_id];
  });

  ctx.wait();

  cs_parall_sum(2, CS_DOUBLE, rd_sum_2d.r);

  double mass_flow_rate = rd_sum_2d.r[0];
  double area = rd_sum_2d.r[1];

  ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {

    /* Decompose the fluctuation in a local coordinate system */
    /* (not valid for warped boundary faces) */

    cs_lnum_t face_id = face_ids[point_id];
    cs_lnum_t cell_id = b_face_cells[face_id];

    cs_lnum_t idx = b_face_vtx_idx[face_id];
    cs_lnum_t vtx_id1 = b_face_vtx_lst[idx];
    cs_lnum_t vtx_id2 = b_face_vtx_lst[idx+1];

    cs_real_t tangent_unit1[3], tangent_unit2[3];

    const cs_nreal_t *normal_unit = b_face_u_normal[face_id];

    const cs_real_t *fluct = fluctuations[point_id];

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
      tangent_unit1[coo_id] =   vtx_coord[vtx_id1][coo_id]
                              - vtx_coord[vtx_id2][coo_id];
    }

    cs_math_3_cross_product(normal_unit, tangent_unit1, tangent_unit2);
    cs_math_3_normalize(tangent_unit1, tangent_unit1);
    cs_math_3_normalize(tangent_unit2, tangent_unit2);

    cs_real_t normal_comp = cs_math_3_dot_product(fluct, normal_unit);
    cs_real_t tangent_comp1 = cs_math_3_dot_product(fluct, tangent_unit1);
    cs_real_t tangent_comp2 = cs_math_3_dot_product(fluct, tangent_unit2);

    /* Rescale the normal component and return in Cartesian coordinates*/

    normal_comp -= mass_flow_rate/(density[cell_id]*area);

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      fluctuations[point_id][coo_id] =   normal_comp*normal_unit[coo_id]
                                       + tangent_comp1*tangent_unit1[coo_id]
                                       + tangent_comp2*tangent_unit2[coo_id];

  });

  ctx.wait();
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Creation of structures for the LES inflows
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_initialize(void)
{
  /* Definition of the global parameters of the inlets */

  cs_user_les_inflow_define();

  cs_log_separator(CS_LOG_DEFAULT);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Finalize turbulent inflow generation API.
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_finalize(void)
{
  if (cs_glob_inflow_n_inlets == 0)
    return;

  /* Destruction of each inlet structure */
  /*-------------------------------------*/

  for (int inlet_id = 0; inlet_id < cs_glob_inflow_n_inlets; inlet_id++) {

    cs_inlet_t *inlet = cs_glob_inflow_inlet_array[inlet_id];

    bft_printf(_("\n"
                 "Summary of synthetic turbulence generation for inlet \"%d\""
                 " (%s) :\n\n"
                 "  Accumulated wall-clock time:      %12.3f\n"),
               inlet_id + 1, cs_inflow_type_name[inlet->type],
               inlet->wt_tot);

    if (cs_glob_rank_id < 0)
      bft_printf(_("  Accumulated CPU time:             %12.3f\n"),
                 inlet->cpu_tot);

#if defined(HAVE_MPI)

    else {

      double cpu_min, cpu_max, cpu_tot;
      double cpu_loc = inlet->cpu_tot;

      MPI_Allreduce(&cpu_loc, &cpu_min, 1, MPI_DOUBLE, MPI_MIN,
                    cs_glob_mpi_comm);
      MPI_Allreduce(&cpu_loc, &cpu_max, 1, MPI_DOUBLE, MPI_MAX,
                    cs_glob_mpi_comm);
      MPI_Allreduce(&cpu_loc, &cpu_tot, 1, MPI_DOUBLE, MPI_SUM,
                    cs_glob_mpi_comm);

      bft_printf(_("  Accumulated CPU time:\n"
                   "    local min:                      %12.3f\n"
                   "    local max:                      %12.3f\n"
                   "    mean:                           %12.3f\n"),
                 cpu_min, cpu_max, cpu_tot/cs_glob_n_ranks);

    }

#endif

    /* Mesh */

    CS_FREE(inlet->face_center);
    CS_FREE(inlet->face_surface);

    /* Turbulence level */

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      inlet->vel_m[coo_id] = 0.;

    inlet->k_r   = 0.;
    inlet->eps_r = 0.;

    /* Generation method of synthetic turbulence */

    inlet->initialize = 0;

    switch(inlet->type) {

    case CS_INFLOW_LAMINAR:

      inlet->inflow = nullptr;
      break;

    case CS_INFLOW_RANDOM:

      inlet->inflow = nullptr;
      break;

    case CS_INFLOW_BATTEN:

      {
        cs_inflow_batten_t *inflow = (cs_inflow_batten_t *) inlet->inflow;

        CS_FREE(inflow->frequency);
        CS_FREE(inflow->wave_vector);
        CS_FREE(inflow->amplitude_cos);
        CS_FREE(inflow->amplitude_sin);

        CS_FREE(inflow);

        inlet->inflow = nullptr;
      }
      break;

    case CS_INFLOW_SEM:

      {
        cs_inflow_sem_t *inflow = (cs_inflow_sem_t *) inlet->inflow;

        CS_FREE(inflow->position);
        CS_FREE(inflow->energy);

        CS_FREE(inflow);

        inlet->inflow = nullptr;
      }
      break;

    default:
      break;

    }

    inlet->wt_tot  = 0.;
    inlet->cpu_tot = 0.;

    CS_FREE(inlet);
  }

  /* Global array of inlets */
  /*------------------------*/

  cs_glob_inflow_n_inlets = 0;
  CS_FREE(cs_glob_inflow_inlet_array);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add an inlet definition for synthetic turbulence inflow generation.
 *
 *  \remark:
 *  - eps_r is used only for CS_INFLOW_BATTEN and CS_INFLOW_SEM types.
 *  - Strictly positive values are required for k_r and eps_r.
 *  - Finer definition of the statistics of the flow at the inlet
 *    can be done later using \ref cs_user_les_inflow_advanced.
 *
 * \param[out]  type         type of inflow method at the inlet
 * \param[out]  volume_mode  if true, generate turbulence over the whole domain
 *                           (only if type is CS_INFLOW_SEM)
 * \param[in]   zone         pointer to associated boundary zone
 * \param[out]  n_entities   number of structures or modes
 * \param[out]  verbosity    verbosity level
 * \param[out]  vel_r        reference mean velocity
 * \param[out]  k_r          reference turbulent kinetic energy
 * \param[out]  eps_r        reference turbulent dissipation
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_add_inlet(cs_les_inflow_type_t   type,
                        bool                   volume_mode,
                        const cs_zone_t       *zone,
                        int                    n_entities,
                        int                    verbosity,
                        const cs_real_t       *vel_r,
                        cs_real_t              k_r,
                        cs_real_t              eps_r)
{
  cs_inlet_t  *inlet = nullptr;

  bft_printf(_(" Definition of the LES inflow for zone \"%s\" \n"),
             zone->name);

  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;
  const cs_real_3_t *b_face_cog = mq->b_face_cog;
  const cs_real_t *b_face_surf = mq->b_face_surf;

  cs_dispatch_context ctx;

  /* Allocating inlet structures */
  /*-----------------------------*/

  CS_REALLOC(cs_glob_inflow_inlet_array,
             cs_glob_inflow_n_inlets + 1, cs_inlet_t *);

  CS_MALLOC(inlet, 1, cs_inlet_t);

  inlet->zone = zone;

  /* Mesh */

  cs_lnum_t n_elts = zone->n_elts;
  const cs_lnum_t *face_ids = zone->elt_ids;

  inlet->face_center = nullptr;
  inlet->face_surface = nullptr;

  if (n_elts > 0) {

    CS_MALLOC_HD(inlet->face_center, n_elts, cs_real_3_t, cs_alloc_mode);
    CS_MALLOC_HD(inlet->face_surface, n_elts, cs_real_t, cs_alloc_mode);

    cs_real_3_t *face_center = inlet->face_center;
    cs_real_t *face_surface = inlet->face_surface;

    ctx.parallel_for(n_elts, [=] CS_F_HOST_DEVICE (cs_lnum_t i) {
      const cs_lnum_t face_id = face_ids[i];
      face_surface[i] = b_face_surf[face_id];

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
        face_center[i][coo_id] = b_face_cog[face_id][coo_id];
      }
    });

    ctx.wait();
  }

  /* Turbulence level */

  if (vel_r != nullptr) {
    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      inlet->vel_m[coo_id] = vel_r[coo_id];
  }
  else {
    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      inlet->vel_m[coo_id] = 0;
  }

  inlet->k_r   = k_r;
  inlet->eps_r = eps_r;

  /* Generation method of synthetic turbulence */
  /*-------------------------------------------*/

  if (type > 3)
    bft_error
      (__FILE__, __LINE__, 0,
       _("Invalid choice of synthetic turbulence generation method (%d).\n"
         "Valid choices are:\n"
         "\t0 -> laminar\n\t1 -> random\n\t2 -> batten\n\t3 -> SEM\n"),
       type);
  else
    inlet->type = type;

  switch(inlet->type) {

  case CS_INFLOW_LAMINAR:
    inlet->inflow = nullptr;
    bft_printf(_("   \n"));
    break;

  case CS_INFLOW_RANDOM:
    inlet->inflow = nullptr;
    bft_printf(_("   \n"));
    break;

  case CS_INFLOW_BATTEN:
    {
      if (n_entities <= 0)
        bft_error(__FILE__, __LINE__, 0,
                  _("The number of modes for the Batten method "
                    "must be strictly positive. %d is given here.\n"),
                  n_entities);

      cs_inflow_batten_t *inflow;
      CS_MALLOC(inflow, 1, cs_inflow_batten_t);

      inflow->n_modes = n_entities;

      CS_MALLOC(inflow->frequency,     inflow->n_modes, cs_real_t);
      CS_MALLOC(inflow->wave_vector,   inflow->n_modes, cs_real_3_t);
      CS_MALLOC(inflow->amplitude_cos, inflow->n_modes, cs_real_3_t);
      CS_MALLOC(inflow->amplitude_sin, inflow->n_modes, cs_real_3_t);

      inlet->inflow = inflow;

      bft_printf(_("   Number of modes: %d\n\n"),n_entities);
    }
    break;

  case CS_INFLOW_SEM:
    {
      if (n_entities <= 0)
        bft_error(__FILE__, __LINE__, 0,
                  _("The number of eddies for the SEM "
                    "must be strictly positive. %d is given here.\n"),
                  n_entities);

      cs_inflow_sem_t *inflow;
      CS_MALLOC(inflow, 1, cs_inflow_sem_t);
      inflow->volume_mode = (volume_mode == true) ? 1 : 0;
      inflow->n_structures = n_entities;

      CS_MALLOC(inflow->position, inflow->n_structures, cs_real_3_t);
      CS_MALLOC(inflow->energy,   inflow->n_structures, cs_real_3_t);

      inlet->inflow = inflow;

      bft_printf(_("   Number of structures: %d\n\n"),n_entities);
    }
    break;

  }

  /* Others */
  /*--------*/

  inlet->initialize = 1;

  inlet->verbosity = verbosity;

  inlet->wt_tot  = 0.;
  inlet->cpu_tot = 0.;

  /* Global array of inlets */
  /*------------------------*/

  cs_glob_inflow_inlet_array[cs_glob_inflow_n_inlets] = inlet;
  cs_glob_inflow_n_inlets++;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief General synthetic turbulence generation
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_compute(void)
{
  if (cs_glob_inflow_n_inlets == 0)
    return;

  constexpr cs_real_t two_third = 2./3.;

  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_lnum_t  n_cells = mesh->n_cells;
  const cs_lnum_t  n_b_faces = mesh->n_b_faces;
  const cs_real_3_t *cell_cen = cs_glob_mesh_quantities->cell_cen;

  cs_dispatch_context ctx;

  for (int inlet_id = 0; inlet_id < cs_glob_inflow_n_inlets; inlet_id++) {

    cs_inlet_t *inlet = cs_glob_inflow_inlet_array[inlet_id];

    cs_user_les_inflow_update(inlet->zone,
                              inlet->vel_m,
                              &(inlet->k_r),
                              &(inlet->eps_r));

    cs_real_3_t *fluctuations = nullptr;

    cs_real_t wt_start, wt_stop, cpu_start, cpu_stop;

    wt_start  = cs_timer_wtime();
    cpu_start = cs_timer_cpu_time();

    cs_lnum_t n_elts = inlet->zone->n_elts;
    const cs_lnum_t *elt_ids = inlet->zone->elt_ids;

    /* Mean velocity profile, one-point statistics and dissipation rate */
    /*------------------------------------------------------------------*/

    cs_real_3_t *vel_m_l;
    cs_real_6_t *rij_l;
    cs_real_t   *eps_r;
    CS_MALLOC_HD(vel_m_l, n_elts, cs_real_3_t, cs_alloc_mode);
    CS_MALLOC_HD(rij_l, n_elts, cs_real_6_t, cs_alloc_mode);
    CS_MALLOC_HD(eps_r, n_elts, cs_real_t, cs_alloc_mode);

    cs_real_t *vel_m = inlet->vel_m;
    const cs_real_t two_third_k_r = two_third*inlet->k_r;
    const cs_real_t inlet_eps_r = inlet->eps_r;

    /* Initialization by the turbulence scales given by the user */

    ctx.parallel_for(n_elts, [=] CS_F_HOST_DEVICE (cs_lnum_t i) {

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
        vel_m_l[i][coo_id] = vel_m[coo_id];
        rij_l[i][coo_id] = two_third_k_r;
      }

      for (cs_lnum_t coo_id = 3; coo_id < 6; coo_id++)
        rij_l[i][coo_id] = 0.;

      eps_r[i] = inlet_eps_r;

    });

    /* Modification by the user */

    cs_user_les_inflow_advanced(inlet->zone,
                                vel_m_l,
                                rij_l,
                                eps_r);

    /* Generation of the synthetic turbulence
       --------------------------------------*/

    CS_MALLOC_HD(fluctuations, n_elts, cs_real_3_t, cs_alloc_mode);
    cs_arrays_set_value<cs_real_t, 1>(3*n_elts, 0., (cs_real_t *)fluctuations);

    const cs_time_step_t *time_step = cs_glob_time_step;

    switch(inlet->type) {

    case CS_INFLOW_LAMINAR:
      break;
    case CS_INFLOW_RANDOM:
      _random_method(n_elts, fluctuations);
      break;
    case CS_INFLOW_BATTEN:
      _batten_method(n_elts,
                     inlet->face_center,
                     inlet->initialize,
                     (cs_inflow_batten_t *) inlet->inflow,
                     time_step->t_cur,
                     rij_l,
                     eps_r,
                     fluctuations);
      break;
    case CS_INFLOW_SEM:
      {
        if (inlet->verbosity > 0)
          bft_printf(_("\n------------------------------"
                       "-------------------------------\n\n"
                       "SEM INFO, inlet \"%d\" \n\n"), inlet_id);

        cs_inflow_sem_t *inflowsem = (cs_inflow_sem_t *)inlet->inflow;

        if (inflowsem->volume_mode == 1) {

          const cs_real_t dissiprate = eps_r[0];
          cs_lnum_t n_points = cs_glob_mesh->n_cells;
          cs_real_t *point_weight = nullptr;

          CS_REALLOC_HD(rij_l, n_cells, cs_real_6_t, cs_alloc_mode);
          CS_REALLOC_HD(vel_m_l, n_cells, cs_real_3_t, cs_alloc_mode);
          CS_REALLOC_HD(eps_r, n_points, cs_real_t, cs_alloc_mode);
          CS_REALLOC_HD(fluctuations, n_points, cs_real_3_t, cs_alloc_mode);

          ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t cell_id) {

            eps_r[cell_id] = dissiprate;

            for (cs_lnum_t j = 0; j < 3; j++) {
              rij_l[cell_id][j] = two_third_k_r;
              vel_m_l[cell_id][j] = 0.;
              fluctuations[cell_id][j] = 0.;
            }

            for (cs_lnum_t j = 3; j < 6; j++)
              rij_l[cell_id][j] = 0.;
          });

          ctx.wait();

          cs_les_synthetic_eddy_method(cs_glob_mesh->n_cells,
                                       elt_ids,
                                       cell_cen,
                                       point_weight,
                                       inlet->initialize,
                                       inlet->verbosity,
                                       (cs_inflow_sem_t *)inlet->inflow,
                                       time_step->dt[0],
                                       vel_m_l,
                                       rij_l,
                                       eps_r,
                                       fluctuations);
        }
        else {
          cs_les_synthetic_eddy_method(n_elts,
                                       elt_ids,
                                       inlet->face_center,
                                       inlet->face_surface,
                                       inlet->initialize,
                                       inlet->verbosity,
                                       (cs_inflow_sem_t *)inlet->inflow,
                                       time_step->dt[0],
                                       vel_m_l,
                                       rij_l,
                                       eps_r,
                                       fluctuations);
        }

        if (inlet->verbosity > 0)
          bft_printf("------------------------------"
                     "-------------------------------\n");
      }
      break;
    }

    inlet->initialize = 0;

    CS_FREE(eps_r);

    /* Rescaling of the synthetic fluctuations by the statistics */
    /*-----------------------------------------------------------*/

    if (inlet->type == CS_INFLOW_SEM){
      cs_inflow_sem_t *inflowsem = (cs_inflow_sem_t *) inlet->inflow;
      if (inflowsem->volume_mode ==1) {  /* Rescale over the whole domain */
        cs_les_rescale_fluctuations(cs_glob_mesh->n_cells,
                                    rij_l,
                                    fluctuations);
      }
      else {
        cs_les_rescale_fluctuations(n_elts,
                                    rij_l,
                                    fluctuations);
      }
    }

    else if (   inlet->type == CS_INFLOW_RANDOM
             || inlet->type == CS_INFLOW_BATTEN) {
      cs_les_rescale_fluctuations(n_elts, rij_l, fluctuations);
    }

    CS_FREE(rij_l);

    /* Rescaling of the mass flow rate */
    /*---------------------------------*/

    if (inlet->type == CS_INFLOW_RANDOM || inlet->type == CS_INFLOW_BATTEN){
      _rescale_flowrate(n_elts,
                        elt_ids,
                        fluctuations);
    }
    else if (inlet->type == CS_INFLOW_SEM){
      cs_inflow_sem_t *inflowsem = (cs_inflow_sem_t *)inlet->inflow;
      if (inflowsem->volume_mode == 1) {
          _rescale_flowrate(n_elts,
                            elt_ids,
                            fluctuations);
      }
      inflowsem->volume_mode=-1;
    }

    /* Boundary conditions */
    /*---------------------*/

    cs_real_t *rcodcl1_u = CS_F_(vel)->bc_coeffs->rcodcl1;
    cs_real_t *rcodcl1_v = rcodcl1_u + n_b_faces;
    cs_real_t *rcodcl1_w = rcodcl1_v + n_b_faces;

    ctx.parallel_for(n_elts, [=] CS_F_HOST_DEVICE (cs_lnum_t i) {
      cs_lnum_t face_id = elt_ids[i];

      rcodcl1_u[face_id] = vel_m_l[i][0] + fluctuations[i][0];
      rcodcl1_v[face_id] = vel_m_l[i][1] + fluctuations[i][1];
      rcodcl1_w[face_id] = vel_m_l[i][2] + fluctuations[i][2];
    });

    ctx.wait();

    CS_FREE(vel_m_l);
    CS_FREE(fluctuations);

    wt_stop  = cs_timer_wtime();
    cpu_stop = cs_timer_cpu_time();

    inlet->wt_tot  += (wt_stop - wt_start);
    inlet->cpu_tot += (cpu_stop - cpu_start);
  }
}

/*----------------------------------------------------------------------------
 * Read the restart file of les inflow module.
 *----------------------------------------------------------------------------*/

void
cs_les_synthetic_eddy_restart_read(void)
{
  if (_allow_restart_read == false || cs_glob_inflow_n_inlets == 0)
    return;

  bool  corresp_cel, corresp_fac, corresp_fbr, corresp_som;
  int   indfac, ierror;

  cs_restart_t *r;

  bft_printf(_(" Reading the LES inflow module restart file...\n"));

  ierror = CS_RESTART_SUCCESS;

  /* Open the restart file */
  const char filename[] = "les_inflow.csc";

  _inflow_restart
    = cs_restart_create(filename, nullptr, CS_RESTART_MODE_READ);

  if (_inflow_restart == nullptr)
    bft_error(__FILE__, __LINE__, 0,
              _("Abort while opening the LES inflow module restart file "
                "in read mode.\n"
                "Verify the existence and the name of the restart file: %s\n"),
              filename);

  /* Pointer to the global restart structure */
  r = _inflow_restart;

  /* Verification of the associated "support" to the restart file */
  cs_restart_check_base_location(r, &corresp_cel, &corresp_fac,
                                 &corresp_fbr, &corresp_som);

  /* Only boundary faces are of interest */
  indfac = (corresp_fbr == true ? 1 : 0);
  if (indfac == 0)
    bft_error(__FILE__, __LINE__, 0,
              _("Abort while reading the LES inflow module restart file.\n"
                "The number of boundary faces has been modified\n"
                "Verify that the restart file corresponds to "
                "the present study.\n"));


  { /* Read the header */
    char   nomrub[] = "version_fichier_suite_turbulence_synthetique";
    int    tabvar[1];

    ierror = cs_restart_read_section(r,
                                     nomrub,
                                     CS_MESH_LOCATION_NONE,
                                     1,
                                     CS_TYPE_int,
                                     tabvar);

    if (ierror < CS_RESTART_SUCCESS)
      bft_error(__FILE__, __LINE__, 0,
                _("Abort while reading the LES inflow module restart file.\n"
                  "\n"
                  "The file %s does not seem to be a restart file\n"
                  "for the LES inflow module.\n"
                  "The calculation will not be run.\n"
                  "\n"
                  "Verify that the restart file corresponds to a\n"
                  "restart file for the LES inflow module."),
                filename);
  }

  { /* Read the number of inlets */
    char       nomrub[] = "nb_inlets";
    int        n_inlets = 0;

    ierror = cs_restart_read_section(r,
                                     nomrub,
                                     CS_MESH_LOCATION_NONE,
                                     1,
                                     CS_TYPE_int,
                                     &n_inlets);

    if (ierror < CS_RESTART_SUCCESS)
      bft_error(__FILE__, __LINE__, 0,
                _("Problem while reading section in the restart file\n"
                  "for the LES inflow module:\n"
                  "<%s>\n"
                  "The calculation will not be run.\n"), nomrub);

    /* Coherence check */
    if (cs_glob_inflow_n_inlets != n_inlets)
      bft_error(__FILE__, __LINE__, 0,
                _("Stop reading the LES inflow module restart file.\n"
                  "The calculation is defined with %d LES inlets "
                  "while the restart file contains %d.\n"),
                cs_glob_inflow_n_inlets, n_inlets);
  }

  { /* Read the structure of each inlet */

    for (int inlet_id = 0; inlet_id < cs_glob_inflow_n_inlets; inlet_id++) {

      cs_inlet_t *inlet = cs_glob_inflow_inlet_array[inlet_id];
      char sec_name[64];
      char postfix[32];

      if (inlet_id == 0)
        postfix[0] = '\0';
      else {
        snprintf(postfix, 31, "_%d", inlet_id);
        postfix[31] = '\0';
      }

      { /* type of inlet */
        int tabvar[1];
        cs_les_inflow_type_t type;

        snprintf(sec_name, 63, "type_inlet%s", postfix); sec_name[63] = '\0';
        ierror = cs_restart_read_section(r,
                                         sec_name,
                                         CS_MESH_LOCATION_NONE,
                                         1,
                                         CS_TYPE_int,
                                         tabvar);

        if (ierror < CS_RESTART_SUCCESS)
          bft_error(__FILE__, __LINE__, 0,
                    _("Problem while reading section in the restart file\n"
                      "for the LES inflow module:\n"
                      "<%s>\n"
                      "The calculation will not be run.\n"), sec_name);

        type = (cs_les_inflow_type_t)tabvar[0];

        /* Coherence check */
        if (inlet->type != type)
          bft_error(__FILE__, __LINE__, 0,
                    _("Stop reading the LES inflow module restart file.\n"
                      "The inlet %d uses the method %d (%s) instead of "
                      "%d (%s) in the restart file.\n"),
                    inlet_id + 1,
                    inlet->type, cs_inflow_type_name[inlet->type],
                    type, cs_inflow_type_name[type]);
      }

      switch(inlet->type) {

      case CS_INFLOW_LAMINAR:
        break;

      case CS_INFLOW_RANDOM:
        break;

      case CS_INFLOW_BATTEN:

        {
          cs_inflow_batten_t *inflow = (cs_inflow_batten_t *) inlet->inflow;

          { /* number of modes */
            int n_modes = 0;

            snprintf(sec_name, 63, "batten_number_modes%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             1,
                                             CS_TYPE_int,
                                             &n_modes);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* Coherence check */
            if (inflow->n_modes != n_modes)
              bft_error(__FILE__, __LINE__, 0,
                        _("Stop reading the LES inflow module restart file.\n"
                          "%d modes are given for the Batten method "
                          "while the restart file contains %d.\n"),
                        inflow->n_modes, n_modes);
          }

          { /* frequencies */
            snprintf(sec_name, 63, "batten_frequencies%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             inflow->n_modes,
                                             CS_TYPE_cs_real_t,
                                             inflow->frequency);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* wave vector */
            snprintf(sec_name, 63, "batten_wave_vector%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             3*inflow->n_modes,
                                             CS_TYPE_cs_real_t,
                                             inflow->wave_vector);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* amplitude cos */
            snprintf(sec_name, 63, "batten_amplitude_cos%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             3*inflow->n_modes,
                                             CS_TYPE_cs_real_t,
                                             inflow->amplitude_cos);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* amplitude sin */
            snprintf(sec_name, 63, "batten_amplitude_sin%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             3*inflow->n_modes,
                                             CS_TYPE_cs_real_t,
                                             inflow->amplitude_sin);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

          }

        }
        break;

      case CS_INFLOW_SEM:

        {
          cs_inflow_sem_t *inflow = (cs_inflow_sem_t *)inlet->inflow;

          { /* number of structures */
            int n_structures = 0;

            snprintf(sec_name, 63, "sem_number_structures%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             1,
                                             CS_TYPE_int,
                                             &n_structures);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* Coherence check */
            if (inflow->n_structures != n_structures)
              bft_error(__FILE__, __LINE__, 0,
                        _("Stop reading the LES inflow module restart file.\n"
                          "%d eddies are given for the SEM "
                          "while the restart file contains %d.\n"),
                        inflow->n_structures, n_structures);
          }

          { /* positions */

            snprintf(sec_name, 63, "sem_positions%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             3*inflow->n_structures,
                                             CS_TYPE_cs_real_t,
                                             inflow->position);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

            /* energies */
            snprintf(sec_name, 63, "sem_energies%s", postfix);
            sec_name[63] = '\0';
            ierror = cs_restart_read_section(r,
                                             sec_name,
                                             CS_MESH_LOCATION_NONE,
                                             3*inflow->n_structures,
                                             CS_TYPE_cs_real_t,
                                             inflow->energy);

            if (ierror < CS_RESTART_SUCCESS)
              bft_error(__FILE__, __LINE__, 0,
                        _("Problem while reading section in the restart file\n"
                          "for the LES inflow module:\n"
                          "<%s>\n"
                          "The calculation will not be run.\n"), sec_name);

          }

        }
        break;

      }

      inlet->initialize = 0;

    }
  }

  cs_restart_read_fields(r, CS_RESTART_LES_INFLOW);

  /* Close the restart file and free structures */
  cs_restart_destroy(&_inflow_restart);

  bft_printf(_(" ...completed\n"));
}

/*----------------------------------------------------------------------------
 * Write the restart file of les inflow module.
 *----------------------------------------------------------------------------*/

void
cs_les_synthetic_eddy_restart_write(void)
{
  if (_allow_restart_write == false || cs_glob_inflow_n_inlets == 0)
    return;

  bft_printf(_("\n Writing the LES inflow module restart file...\n"));

  /* Open the restart file */
  const char filename[] = "les_inflow.csc";

  _inflow_restart
    = cs_restart_create(filename, nullptr, CS_RESTART_MODE_WRITE);

  if (_inflow_restart == nullptr)
    bft_error(__FILE__, __LINE__, 0,
              _("Abort while opening the LES inflow module restart "
                "file in write mode.\n"
                "Verify the existence and the name of the restart file: %s\n"),
              filename);

  /* Pointer to the global restart structure */
  cs_restart_t *r = _inflow_restart;

  { /* Write the header */
    int    tabvar[1] = {120};
    cs_restart_write_section(r,
                             "version_fichier_suite_turbulence_synthetique",
                             CS_MESH_LOCATION_NONE,
                             1,
                             CS_TYPE_int,
                             tabvar);
  }

  { /* Write the number of inlets */
    cs_restart_write_section(r,
                             "nb_inlets",
                             CS_MESH_LOCATION_NONE,
                             1,
                             CS_TYPE_int,
                             &cs_glob_inflow_n_inlets);
  }

  { /* Write the structure of each inlet */

    for (int inlet_id = 0; inlet_id < cs_glob_inflow_n_inlets; inlet_id++) {

      cs_inlet_t *inlet = cs_glob_inflow_inlet_array[inlet_id];
      char sec_name[64];
      char postfix[32];

      if (inlet_id == 0)
        postfix[0] = '\0';
      else {
        snprintf(postfix, 31, "_%d", inlet_id);
        postfix[31] = '\0';
      }

      { /* type of inlet */
        int tabvar[1] = {inlet->type};

        snprintf(sec_name, 63, "type_inlet%s", postfix); sec_name[63] = '\0';
        cs_restart_write_section(r,
                                 sec_name,
                                 CS_MESH_LOCATION_NONE,
                                 1,
                                 CS_TYPE_int,
                                 tabvar);
      }

      switch(inlet->type) {

      case CS_INFLOW_LAMINAR:
        break;

      case CS_INFLOW_RANDOM:
        break;

      case CS_INFLOW_BATTEN:
        {
          cs_inflow_batten_t *inflow = (cs_inflow_batten_t *)inlet->inflow;

          { /* number of modes */
            int tabvar[1] = {inflow->n_modes};

            snprintf(sec_name, 63, "batten_number_modes%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     1,
                                     CS_TYPE_int,
                                     tabvar);

            /* frequencies */
            snprintf(sec_name, 63, "batten_frequencies%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     inflow->n_modes,
                                     CS_TYPE_cs_real_t,
                                     inflow->frequency);

            /* wave vector */
            snprintf(sec_name, 63, "batten_wave_vector%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     3*inflow->n_modes,
                                     CS_TYPE_cs_real_t,
                                     inflow->wave_vector);

            /* amplitude cos */
            snprintf(sec_name, 63, "batten_amplitude_cos%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     3*inflow->n_modes,
                                     CS_TYPE_cs_real_t,
                                     inflow->amplitude_cos);

            /* amplitude sin */
            snprintf(sec_name, 63, "batten_amplitude_sin%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     3*inflow->n_modes,
                                     CS_TYPE_cs_real_t,
                                     inflow->amplitude_sin);
          }

        }
        break;

      case CS_INFLOW_SEM:

        {
          cs_inflow_sem_t *inflow = (cs_inflow_sem_t *)inlet->inflow;

          { /* number of structures */
            int tabvar[1] = {inflow->n_structures};

            snprintf(sec_name, 63, "sem_number_structures%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     1,
                                     CS_TYPE_int,
                                     tabvar);

            /* positions */
            snprintf(sec_name, 63, "sem_positions%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     3*inflow->n_structures,
                                     CS_TYPE_cs_real_t,
                                     inflow->position);

            /* energies */
            snprintf(sec_name, 63, "sem_energies%s", postfix);
            sec_name[63] = '\0';
            cs_restart_write_section(r,
                                     sec_name,
                                     CS_MESH_LOCATION_NONE,
                                     3*inflow->n_structures,
                                     CS_TYPE_cs_real_t,
                                     inflow->energy);
          }

        }
        break;

      }

    }

  }

  cs_restart_write_fields(r, CS_RESTART_LES_INFLOW);

  /* Close the restart file and free structures */
  cs_restart_destroy(&_inflow_restart);

  bft_printf(_(" ...completed\n"));
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Generation of synthetic turbulence via the Synthetic Eddy Method (SEM).
 *
 * \param[in]   n_points            local number of points where
 *                                  turbulence is generated
 * \param[in]   elt_ids             local id of inlet boundary faces
 * \param[in]   point_coordinates   point coordinates
 * \param[in]   point_weight        point weights (surface, volume or nullptr)
 * \param[in]   initialize          initialization indicator
 * \param[in]   verbosity           verbosity level
 * \param[in]   inflow              pointer to structure for Batten method
 * \param[in]   t_cur               current time
 * \param[in]   vel_m_l             mean velocity at each point
 * \param[in]   rij_l               Reynolds stresses at each point
 * \param[in]   eps_l               dissipation rate at each point
 * \param[out]  fluctuations        velocity fluctuations at each point
 */
/*----------------------------------------------------------------------------*/

void
cs_les_synthetic_eddy_method(cs_lnum_t           n_points,
                             const cs_lnum_t     elt_ids[],
                             const cs_real_3_t   point_coordinates[],
                             const cs_real_t    *point_weight,
                             int                 initialize,
                             int                 verbosity,
                             cs_inflow_sem_t    *inflow,
                             cs_real_t           t_cur,
                             const cs_real_3_t   vel_m_l[],
                             const cs_real_6_t   rij_l[],
                             const cs_real_t     eps_l[],
                             cs_real_3_t         fluctuations[])
{
  cs_real_t alpha;
  cs_real_t random = -1.;

  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;

  const cs_lnum_t *b_face_cells = mesh->b_face_cells;
  const cs_lnum_t *b_face_vtx_idx = mesh->b_face_vtx_idx;
  const cs_lnum_t *b_face_vtx_lst = mesh->b_face_vtx_lst;
  const cs_real_3_t *vtx_coord = (const cs_real_3_t *)mesh->vtx_coord;
  const cs_real_3_t *cell_cen = mq->cell_cen;
  const cs_real_t *cell_vol = mq->cell_vol;

  cs_dispatch_context ctx;

  /* Computation of the characteristic scale of the synthetic eddies */
  /*-----------------------------------------------------------------*/

  cs_real_3_t *length_scale;
  CS_MALLOC_HD(length_scale, n_points, cs_real_3_t, cs_alloc_mode);

  struct cs_int_n<3> rd_sum_3i;
  struct cs_reduce_sum_ni<3> reducer_sum_3i;

  if (inflow->volume_mode == 1) { //Generate turbulence over the whole domain

    ctx.parallel_for_reduce
      (n_points, rd_sum_3i, reducer_sum_3i,
       [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_int_n<3> &res) {

      res.i[0] = 0, res.i[1] = 0, res.i[2] = 0;

      // TODO: add a mesh algorithm to compute better estimation of longest
      // and shortest cell lengths (would also be useful for HTLES and
      // possibly other models.

      cs_real_t length_scale_min = 2.*cbrt(cell_vol[point_id]);

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

        length_scale[point_id][coo_id]
          =    pow(1.5*rij_l[point_id][coo_id], 1.5)
             / eps_l[point_id];

        length_scale[point_id][coo_id]
          = 0.5*length_scale[point_id][coo_id];

        length_scale[point_id][coo_id]
          = fmax(length_scale[point_id][coo_id], length_scale_min);

        if (cs::abs(length_scale[point_id][coo_id]-length_scale_min)
            < cs_math_epzero) {
          res.i[coo_id] = 1;
        }
      }
    });
  }
  else {
    ctx.parallel_for_reduce
      (n_points, rd_sum_3i, reducer_sum_3i,
       [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_int_n<3> &res) {

      res.i[0] = 0, res.i[1] = 0, res.i[2] = 0;

      cs_lnum_t face_id = elt_ids[point_id];
      cs_lnum_t cell_id = b_face_cells[face_id];

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

        cs_real_t length_scale_min = -cs_math_big_r;

        for (cs_lnum_t j = b_face_vtx_idx[face_id];
             j < b_face_vtx_idx[face_id + 1];
             j++) {
          cs_lnum_t vtx_id = b_face_vtx_lst[j];
          length_scale_min
            = cs::max(length_scale_min,
                      2.*cs::abs(cell_cen[cell_id][coo_id]
                                 - vtx_coord[vtx_id][coo_id]));
        }

        length_scale[point_id][coo_id]
          = pow(1.5*rij_l[point_id][coo_id], 1.5) / eps_l[point_id];

        length_scale[point_id][coo_id]
          = 0.5*length_scale[point_id][coo_id];

        length_scale[point_id][coo_id]
          = cs::max(length_scale[point_id][coo_id], length_scale_min);

        if (cs::abs(length_scale[point_id][coo_id] - length_scale_min)
            < cs_math_epzero)
          res.i[coo_id] = 1;

      }
    });
  }

  if (verbosity > 0) {
    char direction[4] = "xyz";

    bft_printf(_("Max. size of synthetic eddies:\n"));

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

      struct cs_float_n<4> rd_4r;
      cs_reduce_max1float_bcast3float reducer_max1r_bcast3r;

      ctx.parallel_for_reduce
        (n_points, rd_4r, reducer_max1r_bcast3r,
         [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_float_n<4> &res) {

        res.r[3] = length_scale[point_id][coo_id];

        res.r[0] = point_coordinates[point_id][0];
        res.r[1] = point_coordinates[point_id][1];
        res.r[2] = point_coordinates[point_id][2];

      });

      ctx.wait();

      cs_real_t xyzmax[3] = {rd_4r.r[0], rd_4r.r[1], rd_4r.r[2]};
      cs_real_t max = rd_4r.r[3];
      cs_parall_max_loc_vals(3, &max, xyzmax);

      cs_real_t ls_max = rd_4r.r[3];

      bft_printf(_("   max(sigma_%c) = %f, at coordinates (%f,%f,%f)\n"),
                 direction[coo_id], ls_max, xyzmax[0], xyzmax[1], xyzmax[2]);

    }

    bft_printf(_("\n"));

    bft_printf(_("Number of min. clippings (eddy size equals grid size):\n"));

    cs_gnum_t count[3] = {static_cast<unsigned long>(rd_sum_3i.i[0]),
                          static_cast<unsigned long>(rd_sum_3i.i[1]),
                          static_cast<unsigned long>(rd_sum_3i.i[2])};

    cs_parall_counter(count, 3);

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      bft_printf(_("   sigma_%c clipped %ld times\n"),
                 direction[coo_id], (long)count[coo_id]);

    bft_printf(_("\n"));
  }

  /* Definition of the box on which eddies are generated */
  /*-----------------------------------------------------*/

  struct cs_data_3float_3float rd_3r_3r;
  struct cs_reduce_min3float_max3float reducer_3r_3r;

  ctx.parallel_for_reduce
    (n_points, rd_3r_3r, reducer_3r_3r,
     [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_data_3float_3float &res) {

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {
      res.r1[coo_id] =  point_coordinates[point_id][coo_id]
                      - length_scale[point_id][coo_id];

      res.r2[coo_id] =  point_coordinates[point_id][coo_id]
                      + length_scale[point_id][coo_id];
    }
  });

  ctx.wait();

  cs_parall_min(3, CS_FLOAT, rd_3r_3r.r1);
  cs_parall_max(3, CS_FLOAT, rd_3r_3r.r2);

  cs_real_t box_min_coord[3] = {rd_3r_3r.r1[0], rd_3r_3r.r1[1], rd_3r_3r.r1[2]};
  cs_real_t box_max_coord[3] = {rd_3r_3r.r2[0], rd_3r_3r.r2[1], rd_3r_3r.r2[2]};

  cs_real_t box_length[3];
  for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
    box_length[coo_id] = box_max_coord[coo_id] - box_min_coord[coo_id];

  cs_real_t box_volume = box_length[0]*box_length[1]*box_length[2];

  if (box_volume <= -cs_math_big_r) {
    bft_printf(_("%s: empty virtual box\n"), __func__);
    CS_FREE(length_scale);
    return;
  }

  if (verbosity > 0)
    bft_printf(_("LES SEM: dimensions of the virtual box: \n"
                 "   Lx = %f, coo_min : %f, coo_max : %f\n"
                 "   Ly = %f, coo_min : %f, coo_max : %f\n"
                 "   Lz = %f, coo_min : %f, coo_max : %f\n\n"),
               box_length[0], box_min_coord[0], box_max_coord[0],
               box_length[1], box_min_coord[1], box_max_coord[1],
               box_length[2], box_min_coord[2], box_max_coord[2]);

  /* Initialization of the eddy field */
  /*----------------------------------*/

  if (initialize == 1) {

    if (cs_glob_rank_id <= 0) {

      for (int struct_id = 0; struct_id < inflow->n_structures; struct_id++) {

        /* Random intensities */

        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

          cs_random_uniform(1, &random);
          inflow->energy[struct_id][coo_id] = (random < 0.5) ? -1. : 1.;

        }

        /* Position of the eddies in the box */

        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

          cs_random_uniform(1, &random);
          inflow->position[struct_id][coo_id]
            = box_min_coord[coo_id] + random*box_length[coo_id];

        }

      }

    }

#if defined(HAVE_MPI)

    if (cs_glob_rank_id >= 0) {

      MPI_Bcast(inflow->energy, 3*inflow->n_structures,
                CS_MPI_REAL, 0, cs_glob_mpi_comm);
      MPI_Bcast(inflow->position, 3*inflow->n_structures,
                CS_MPI_REAL, 0, cs_glob_mpi_comm);

    }

#endif

  }

  /* Estimation of the convection speed (with weighting by surface) */
  /*----------------------------------------------------------------*/

  cs_real_t *weight;
  CS_MALLOC_HD(weight, n_points, cs_real_t, cs_alloc_mode);

  if (point_weight == nullptr) {
    ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {
      weight[point_id] = 1.;
    });
  }
  else {
    ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {
      weight[point_id] = point_weight[point_id];
    });
  }

  ctx.wait();

  struct cs_double_n<4> rd_sum_4d;
  struct cs_reduce_sum_nr<4> reducer_sum_4d;

  ctx.parallel_for_reduce
    (n_points, rd_sum_4d, reducer_sum_4d,
     [=] CS_F_HOST_DEVICE (cs_lnum_t point_id, cs_double_n<4> &res) {

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      res.r[coo_id] = vel_m_l[point_id][coo_id]*weight[point_id];

    res.r[3] = weight[point_id];
  });

  ctx.wait();

  CS_FREE(weight);

  cs_parall_sum(4, CS_DOUBLE, rd_sum_4d.r);

  cs_real_t weight_tot = rd_sum_4d.r[3];
  cs_real_t vel_m[3] = {rd_sum_4d.r[0], rd_sum_4d.r[1], rd_sum_4d.r[2]};

  for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
    vel_m[coo_id] /= weight_tot;

  /* Time evolution of the eddies */
  /*------------------------------*/

  if (cs_glob_rank_id <= 0) {

    /* Time advancement of the eddies */

    for (int struct_id = 0; struct_id < inflow->n_structures; struct_id++) {

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
        inflow->position[struct_id][coo_id] += vel_m[coo_id]*t_cur;

    }

    /* Checking if the structures are still in the box */

    int compt_born = 0;

    for (int struct_id = 0; struct_id < inflow->n_structures; struct_id++) {

      int new_struct = 0;
      int randomize[3] = {1, 1, 1};

      /* If the eddy leaves the box by one side, one convects it */

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

        if (inflow->position[struct_id][coo_id] < box_min_coord[coo_id]) {
          new_struct = 1;
          randomize[coo_id] = 0;
          inflow->position[struct_id][coo_id] += box_length[coo_id];
        }
        else if (inflow->position[struct_id][coo_id] > box_max_coord[coo_id]) {
          new_struct = 1;
          randomize[coo_id] = 0;
          inflow->position[struct_id][coo_id] -= box_length[coo_id];
        }

      }

      if (new_struct == 1) {

        /* The other directions are randomized */

        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

          if (randomize[coo_id] == 1) {
            cs_random_uniform(1, &random);
            inflow->position[struct_id][coo_id]
              = box_min_coord[coo_id] + random*box_length[coo_id];
          }

        }

        /* New randomization of the energy */

        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++) {

          cs_random_uniform(1, &random);
          inflow->energy[struct_id][coo_id] = (random < 0.5) ? -1. : 1.;
        }

      }

      compt_born += new_struct;

    }

    if (verbosity > 0)
      bft_printf(_("Number of eddies leaving the box (regenerated): %i\n\n"),
                 compt_born);

  }

#if defined(HAVE_MPI)

  if (cs_glob_rank_id >= 0) {

    MPI_Bcast(inflow->energy,   3*inflow->n_structures,
              CS_MPI_REAL, 0, cs_glob_mpi_comm);
    MPI_Bcast(inflow->position, 3*inflow->n_structures,
              CS_MPI_REAL, 0, cs_glob_mpi_comm);

  }

#endif

  /* Computation of the eddy signal */
  /*--------------------------------*/

  alpha = sqrt(box_volume / (double)inflow->n_structures);
  const int n_structures = inflow->n_structures;
  const cs_real_3_t *position = inflow->position;
  const cs_real_3_t *energy = inflow->energy;

  ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {

    cs_real_t distance[3];

    for (int struct_id = 0; struct_id < n_structures; struct_id++) {

      for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
        distance[coo_id] =
          cs::abs(  point_coordinates[point_id][coo_id]
                  - position[struct_id][coo_id]);

      if (   distance[0] < length_scale[point_id][0]
          && distance[1] < length_scale[point_id][1]
          && distance[2] < length_scale[point_id][2]) {

        cs_real_t form_function = 1.;
        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
          form_function *=
            (1.-distance[coo_id]/length_scale[point_id][coo_id])
            /sqrt(2./3.*length_scale[point_id][coo_id]);

        for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
          fluctuations[point_id][coo_id]
            += energy[struct_id][coo_id]*form_function;

      }

    }

    for (cs_lnum_t coo_id = 0; coo_id < 3; coo_id++)
      fluctuations[point_id][coo_id] *= alpha;

  });

  ctx.wait();

  CS_FREE(length_scale);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Rescale fluctuations by statistics following the Lund method.
 *
 * One assumes that the statistics are interlaced and ordered as follows:
 *   <u'u'>  <v'v'>  <w'w'>  <u'v'>  <v'w'>  <u'w'>
 *
 * \param[in]       n_points      local number of points where
 *                                turbulence is generated
 * \param[in]       statistics    statistics (i.e. Reynolds stresses)
 * \param[in, out]  fluctuations  velocity fluctuations generated
 *----------------------------------------------------------------------------*/

void
cs_les_rescale_fluctuations(cs_lnum_t          n_points,
                            const cs_real_6_t  statistics[],
                            cs_real_3_t        fluctuations[])
{
  cs_dispatch_context ctx;

  ctx.parallel_for(n_points, [=] CS_F_HOST_DEVICE (cs_lnum_t point_id) {

    /* Reynolds stresses */

    cs_real_t r11 = statistics[point_id][0];
    cs_real_t r22 = statistics[point_id][1];
    cs_real_t r33 = statistics[point_id][2];
    cs_real_t r12 = statistics[point_id][3];
    cs_real_t r13 = statistics[point_id][4];
    cs_real_t r23 = statistics[point_id][5];

    /* Lund's coefficients */

    cs_real_t a11 = sqrt(r11);
    cs_real_t a21 = r12 / a11;
    cs_real_t a22 = sqrt(cs_math_fmax(r22 - a21*a21, 0));
    cs_real_t a31 = r13 / a11;
    cs_real_t a32 = (r23 - a21*a31) / a22;
    cs_real_t a33 = sqrt(cs_math_fmax(r33 - a31*a31 - a32*a32, 0));

    /* Rescaling of velocity fluctuations */

    cs_real_t up_corr =   a11*fluctuations[point_id][0];
    cs_real_t vp_corr =   a21*fluctuations[point_id][0]
                        + a22*fluctuations[point_id][1];
    cs_real_t wp_corr =   a31*fluctuations[point_id][0]
                        + a32*fluctuations[point_id][1]
                        + a33*fluctuations[point_id][2];

    fluctuations[point_id][0] = up_corr;
    fluctuations[point_id][1] = vp_corr;
    fluctuations[point_id][2] = wp_corr;

  });

  ctx.wait();
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set number of structures used for volume SEM when
 *        restarting from another turbulence model.
 *
 * By default, a restart file is read if present, and a checkpoint written.
 * If not read, synthetic fluctuations are re-initialized.
 *
 * \param[in]  n_structures  number of structures for initialization
 */
/*----------------------------------------------------------------------------*/

void
cs_les_synthetic_eddy_set_n_restart_structures(int  n_structures)
{
  _n_sem_vol_restart_structures = n_structures;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return number of structures used for volume SEM when
 *        restarting from another turbulence model.
 *
 * \return   number of structures for initialization
 */
/*----------------------------------------------------------------------------*/

int
cs_les_synthetic_eddy_get_n_restart_structures(void)
{
  return _n_sem_vol_restart_structures;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Query behavior of the LES inflow module in case of restart.
 *
 * See \ref cs_les_synthetic_eddy_set_restart for details.
 *
 * \param[out]  allow_read   pointer to read flag, or nullptr
 * \param[out]  allow_write  pointer to write flag, or nullptr
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_get_restart(bool  *allow_read,
                          bool  *allow_write)
{
  if (allow_read != nullptr)
    *allow_read = _allow_restart_read;
  if (allow_write != nullptr)
    *allow_write = _allow_restart_write;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define behavior of the LES inflow module in case of restart.
 *
 * By default, a specific file is read if present in the restart folder,
 * and files written in the checkpoint folder at global checkpoint intervals.
 *
 * If not read, synthetic fluctuations are re-initialized.
 *
 * \param[in]  allow_read   allow reading a relevant checkpoint if present
 * \param[in]  allow_write  allow writing a relevant checkpoint if present
 */
/*----------------------------------------------------------------------------*/

void
cs_les_inflow_set_restart(bool  allow_read,
                          bool  allow_write)
{
  _allow_restart_read = allow_read;
  _allow_restart_write = allow_write;
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
