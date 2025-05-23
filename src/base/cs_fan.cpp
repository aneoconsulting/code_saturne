/*============================================================================
 * Fan modeling through velocity source terms.
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
#include <string.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"
#include "base/cs_field.h"
#include "base/cs_log.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh_location.h"
#include "base/cs_parall.h"
#include "base/cs_post.h"

/*----------------------------------------------------------------------------
 * Header for the current file
 *----------------------------------------------------------------------------*/

#include "base/cs_fan.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_fan.cpp
        Fan modeling through velocity source terms.
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Local Type Definitions
 *============================================================================*/

/* Structure associated to a fan */

struct _cs_fan_t {

  int            id;                     /* Fan id */
  int            dim;                    /* 2D or 3D geometry */
  int            mode;                   /* Use of the fan as a wind turbine (1)
                                            or as a fan (0)(default) */
  double         inlet_axis_coords[3];   /* Axis point coordinates of the
                                            inlet face */
  double         outlet_axis_coords[3];  /* Axis point coordinates of the
                                            outlet face */
  double         axis_dir[3];            /* Unit vector of the axis
                                            (inlet to outlet) */
  double         thickness;              /* Fan thickness */
  double         surface;                /* Fan total surface */
  double         volume;                 /* Fan total volume */
  double         volume_expected;        /* Fan theoretical volume */

  double         fan_radius;             /* Fan radius */
  double         blades_radius;          /* Blades radius */
  double         hub_radius;             /* Hub radius */
  double         curve_coeffs[3];        /* Coefficients of the terms of
                                            degree 0, 1 and 2 of the
                                            pressure drop/flow rate
                                            characteristic curve */
  double         axial_torque;           /* Fan axial torque */

  cs_lnum_t      n_cells;                /* Number of cells */

  cs_lnum_t     *cell_list;              /* List of the cells belonging
                                            to the fan */

  double         in_flow;                /* Current inlet flow */
  double         out_flow;               /* Current outlet flow */
  double         delta_p;                /* Pressure drop */

};

/*============================================================================
 * Global variables
 *============================================================================*/

/* Fans array */

static cs_lnum_t    _cs_glob_n_fans_max = 0;

static cs_lnum_t    _cs_glob_n_fans = 0;
static cs_fan_t  ** _cs_glob_fans = nullptr;

/*============================================================================
 * Macro definitions
 *============================================================================*/

enum {X, Y, Z};

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Fan definition (added to the ones previously defined)
 *
 * Fans are handled as explicit momentum source terms at the given location,
 * based on the fan's axis and diameter.
 * The fan's pressure characteristic curve is defined by 3 coefficients,
 * such that:
 * \f$\delta P = C_0 + C_1.flow + C_2.flow^2\f$.
 * An axial torque may also be defined for the 3D model.
 *
 * \param[in]    fan_dim             fan dimension:
 *                                     2: pseudo-2D (extruded mesh)
 *                                     3: 3D (standard)
 * \param[in]    mode                mode:
 *                                     0: fan
 *                                     1: wind turbine
 * \param[in]    inlet_axis_coords   intersection of axis and inlet face
 * \param[in]    outlet_axis_coords  intersection of axis and outlet face
 * \param[in]    fan_radius          fan radius
 * \param[in]    blades_radius       blades radius
 * \param[in]    hub_radius          hub radius
 * \param[in]    curve_coeffs        coefficients of degre 0, 1 and 2 of
 *                                   the pressure drop/flow rate
                                     characteristic curve
 * \param[in]    axial_torque        fan axial torque
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_define(int              fan_dim,
              int              mode,
              const cs_real_t  inlet_axis_coords[3],
              const cs_real_t  outlet_axis_coords[3],
              cs_real_t        fan_radius,
              cs_real_t        blades_radius,
              cs_real_t        hub_radius,
              const cs_real_t  curve_coeffs[3],
              cs_real_t        axial_torque)
{
  cs_fan_t  *fan = nullptr;

  /* Define a new fan */

  CS_MALLOC(fan, 1, cs_fan_t);

  fan->id = _cs_glob_n_fans;

  fan->dim = fan_dim;

  fan->mode = mode;

  for (int i = 0; i < 3; i++) {
    fan->inlet_axis_coords[i] = inlet_axis_coords[i];
    fan->outlet_axis_coords[i] = outlet_axis_coords[i];
  }

  fan->fan_radius = fan_radius;
  fan->blades_radius  = blades_radius;
  fan->hub_radius  = hub_radius;

  for (int i = 0; i < 3; i++)
    fan->curve_coeffs[i] = curve_coeffs[i];
  fan->axial_torque = axial_torque;

  fan->n_cells = 0;
  fan->cell_list = nullptr;

  /* Compute the axis vector */

  fan->thickness = 0.0;

  for (int i = 0; i < 3; i++) {
    fan->axis_dir[i] = outlet_axis_coords[i] - inlet_axis_coords[i];
  }

  fan->thickness = cs_math_3_norm(fan->axis_dir);
  cs_math_3_normalize(fan->axis_dir, fan->axis_dir);

  /* Compute fan theoretical volume */

  fan->volume_expected = cs_math_pi*cs_math_pow2(fan->fan_radius)*fan->thickness;

  /* Surface/volume initialized to 0, will be set by cs_fan_build_all */

  fan->surface = 0.0;
  fan->volume = 0.0;

  /* Flows initialized to 0 */

  fan->in_flow = 0.0;
  fan->out_flow = 0.0;

  /* Increase the fans array if necessary */

  if (_cs_glob_n_fans == _cs_glob_n_fans_max) {
    _cs_glob_n_fans_max = (_cs_glob_n_fans_max + 1) * 2;
    CS_REALLOC(_cs_glob_fans, _cs_glob_n_fans_max, cs_fan_t *);
  }

  /* Adds in the fans array */

  _cs_glob_fans[_cs_glob_n_fans] = fan;
  _cs_glob_n_fans += 1;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Destroy the structures associated with fans.
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_destroy_all(void)
{
  for (int i = 0; i < _cs_glob_n_fans; i++) {
    cs_fan_t  *fan = _cs_glob_fans[i];
    CS_FREE(fan->cell_list);
    CS_FREE(fan);
  }

  _cs_glob_n_fans_max = 0;
  _cs_glob_n_fans = 0;
  CS_FREE(_cs_glob_fans);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return number of fans.
 *
 * \return  number of defined fans
 */
/*----------------------------------------------------------------------------*/

int
cs_fan_n_fans(void)
{
  return _cs_glob_n_fans;
}

/*----------------------------------------------------------------------------
 * Create fans field.
 *----------------------------------------------------------------------------*/

void
cs_fan_field_create(void)
{
  if (cs_fan_n_fans() > 0) {

    /* Get ids */
    const int k_log = cs_field_key_id("log");
    const int k_vis = cs_field_key_id("post_vis");

    cs_field_t *f_fan
      = cs_field_create("fan_id",
                        CS_FIELD_PROPERTY, CS_MESH_LOCATION_CELLS, 1, false);

    cs_field_set_key_int(f_fan, k_log, 1);
    cs_field_set_key_int(f_fan, k_vis, CS_POST_ON_LOCATION);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Log fans definition setup information.
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_log_setup(void)
{
  if (_cs_glob_n_fans < 1)
    return;

  cs_log_printf(CS_LOG_SETUP,
                _("\n"
                  "Fans\n"
                  "----\n"));

  for (int i = 0; i < _cs_glob_n_fans; i++) {
    cs_fan_t  *fan = _cs_glob_fans[i];
    cs_log_printf
      (CS_LOG_SETUP,
       _("  Fan id:  %d\n"
         "    Fan mesh dimension:  %d\n"
         "    Wind turbine:        %d\n"
         "    Axis coordinates:    [%11.4e, %11.4e, %11.4e,\n"
         "                          %11.4e, %11.4e, %11.4e]\n"
         "    Fan radius:          %11.4e\n"
         "      Blades radius:     %11.4e\n"
         "      Hub radius:        %11.4e\n"
         "    Curve coefficients:  C0: %10.3e, C1: %10.3e, C2: %10.3e\n"
         "    Axial torque:        %10.3e\n"),
       fan->id, fan->dim, fan->mode,
       fan->inlet_axis_coords[0],
       fan->inlet_axis_coords[1],
       fan->inlet_axis_coords[2],
       fan->outlet_axis_coords[0],
       fan->outlet_axis_coords[1],
       fan->outlet_axis_coords[2],
       fan->fan_radius, fan->blades_radius, fan->hub_radius,
       fan->curve_coeffs[0],
       fan->curve_coeffs[1],
       fan->curve_coeffs[2],
       fan->axial_torque);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Log fan information for a given iteration.
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_log_iteration(void)
{
  if (_cs_glob_n_fans < 1)
    return;

  cs_log_printf(CS_LOG_DEFAULT,
                _("\n"
                  "Fans\n"
                  "----\n"));

  cs_log_printf(CS_LOG_DEFAULT,
                 _("    id      surface  volume(real) volume(th.)"
                   "       flow       deltaP\n"
                   "  ----  -----------  -----------  -----------"
                   "  ---------  -----------\n" ));

  for (int i = 0; i < _cs_glob_n_fans; i++) {
    cs_fan_t  *fan = _cs_glob_fans[i];
    cs_log_printf(CS_LOG_DEFAULT,
                  " %5d  %11.4e  %11.4e  %11.4e  %11.4e  %11.4e\n",
                  fan->id, fan->surface, fan->volume, fan->volume_expected,
                  0.5*(fan->out_flow - fan->in_flow),
                  fan->delta_p);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Define the cells belonging to the different fans.
 *
 * \param[in]   mesh             associated mesh structure
 * \param[in]   mesh_quantities  mesh quantities
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_build_all(const cs_mesh_t              *mesh,
                 const cs_mesh_quantities_t   *mesh_quantities)
{
  cs_real_t  coo_axis;
  cs_real_t  d_cel_axis[3];
  cs_real_t  l_surf;

  cs_fan_t  *fan = nullptr;
  cs_lnum_t  *cpt_cel_vtl = nullptr;
  int  *cell_fan_id = nullptr;

  const cs_lnum_t  n_cells = mesh->n_cells;
  const cs_lnum_t  n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_2_t  *i_face_cells = mesh->i_face_cells;
  const cs_lnum_t  *b_face_cells = mesh->b_face_cells;
  const cs_real_3_t *restrict cell_cen = mesh_quantities->cell_cen;
  const cs_real_t *restrict i_face_surf = mesh_quantities->i_face_surf;
  const cs_real_t *restrict b_face_surf = mesh_quantities->b_face_surf;

  /* Reset fans in case already built */

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {
    fan = _cs_glob_fans[fan_id];
    fan->n_cells = 0;
    fan->surface = 0;
    fan->volume = 0;
  }

  /* Create an array for cells flaging */
  /*-----------------------------------*/

  CS_MALLOC(cell_fan_id, n_cells_ext, int);

  for (cs_lnum_t cell_id = 0; cell_id < n_cells_ext; cell_id++)
    cell_fan_id[cell_id] = -1;

  /* Main loop on cells */

  for (cs_lnum_t cell_id = 0; cell_id < n_cells; cell_id++) {

    /* Loop on fans */

    for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {

      fan = _cs_glob_fans[fan_id];

      /* Vector from the outlet face axis point to the cell center */

      for (int coo_id = 0; coo_id < 3; coo_id++) {
        d_cel_axis[coo_id] =   (cell_cen[cell_id][coo_id])
                             - fan->inlet_axis_coords[coo_id];
      }

      /* Dot product with the axis vector */

      coo_axis = cs_math_3_dot_product(d_cel_axis, fan->axis_dir);

      /* Cell potentially in the fan if its center projection on the axis
         is within the thickness */

      if (coo_axis >= 0. && coo_axis <= fan->thickness) {

        /* Projection of the vector from the outlet face axis point
           to the cell center in the fan plane */

        for (int coo_id = 0; coo_id < 3; coo_id++)
          d_cel_axis[coo_id] -= coo_axis * fan->axis_dir[coo_id];

        /* Distance to the axis */
        cs_real_t d_axis = cs_math_3_norm(d_cel_axis);

        /* If the cell is in the fan */

        if (d_axis <= fan->fan_radius) {

          cell_fan_id[cell_id] = fan_id;
          fan->n_cells += 1;
          fan->volume += mesh_quantities->cell_vol[cell_id];
        }

      }

    } /* End of loop on fans */

  } /* End of main loop on cells */

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {
    cs_parall_sum(1, CS_DOUBLE, &((_cs_glob_fans[fan_id])->volume));
  }

  /* Synchronize cell_fan_id */
  if (mesh->halo != nullptr)
    cs_halo_sync_untyped(mesh->halo,
                         CS_HALO_EXTENDED,
                         sizeof(int),
                         cell_fan_id);

  /* Create the lists of cells belonging to each fan */
  /*-------------------------------------------------*/

  CS_MALLOC(cpt_cel_vtl, _cs_glob_n_fans, cs_lnum_t);

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {

    fan = _cs_glob_fans[fan_id];

    CS_REALLOC(fan->cell_list, fan->n_cells, cs_lnum_t);

    cpt_cel_vtl[fan_id] = 0;
  }

  for (cs_lnum_t cell_id = 0; cell_id < n_cells; cell_id++) {

    if (cell_fan_id[cell_id] > -1) {
      int fan_id = cell_fan_id[cell_id];
      fan = _cs_glob_fans[fan_id];
      fan->cell_list[cpt_cel_vtl[fan_id]] = cell_id;
      cpt_cel_vtl[fan_id] += 1;
    }

  }

#if defined(DEBUG) && !defined(NDEBUG)
  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {
    fan = _cs_glob_fans[fan_id];
    assert(cpt_cel_vtl[fan_id] == fan->n_cells);
  }
#endif

  /* Compute each fan surface */
  /*--------------------------*/

  /* Contribution to the domain interior */

  for (cs_lnum_t face_id = 0; face_id < mesh->n_i_faces; face_id++) {

    cs_lnum_t cell_id_1 = i_face_cells[face_id][0];
    cs_lnum_t cell_id_2 = i_face_cells[face_id][1];

    if (   cell_id_1 < mesh->n_cells /* ensure the contrib is from one domain */
        && cell_fan_id[cell_id_1] != cell_fan_id[cell_id_2]) {

      l_surf = i_face_surf[face_id];
      if (cell_fan_id[cell_id_1] > -1) {
        int fan_id = cell_fan_id[cell_id_1];
        fan = _cs_glob_fans[fan_id];
        fan->surface += l_surf;
      }
      if (cell_fan_id[cell_id_2] > -1) {
        int fan_id = cell_fan_id[cell_id_2];
        fan = _cs_glob_fans[fan_id];
        fan->surface += l_surf;
      }
    }
  }

  /* Contribution to the domain boundary */

  for (cs_lnum_t face_id = 0; face_id < mesh->n_b_faces; face_id++) {

    if (cell_fan_id[b_face_cells[face_id]] > -1) {
      l_surf = b_face_surf[face_id];
      int fan_id = cell_fan_id[b_face_cells[face_id]];
      fan = _cs_glob_fans[fan_id];
      fan->surface += l_surf;
    }
  }

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++)
    cs_parall_sum(1, CS_DOUBLE, &((_cs_glob_fans[fan_id])->surface));

  /* Free memory */


  CS_FREE(cpt_cel_vtl);
  CS_FREE(cell_fan_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the flows through the fans.
 *
 * \param[in]  mesh             mesh structure
 * \param[in]  mesh_quantities  mesh quantities
 * \param[in]  i_mass_flux      interior faces mass flux
 * \param[in]  b_mass_flux      boundary faces mass flux
 * \param[in]  c_rho            density at cells
 * \param[in]  b_rho            density at boundary faces
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_compute_flows(const cs_mesh_t             *mesh,
                     const cs_mesh_quantities_t  *mesh_quantities,
                     const cs_real_t              i_mass_flux[],
                     const cs_real_t              b_mass_flux[],
                     const cs_real_t              c_rho[],
                     const cs_real_t              b_rho[])
{
  cs_lnum_t   direction;

  cs_real_t  flow;

  cs_fan_t  *fan = nullptr;
  int *cell_fan_id = nullptr;

  const cs_lnum_t  n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t  nbr_fac = mesh->n_i_faces;
  const cs_lnum_t  nbr_fbr = mesh->n_b_faces;
  const cs_lnum_2_t *i_face_cells = mesh->i_face_cells;
  const cs_lnum_t   *b_face_cells = mesh->b_face_cells;
  const cs_nreal_3_t *restrict i_face_u_normal
    = mesh_quantities->i_face_u_normal;
  const cs_nreal_3_t *restrict b_face_u_normal
    = mesh_quantities->b_face_u_normal;

  /* Flag the cells */

  CS_MALLOC(cell_fan_id, n_cells_ext, int);

  cs_fan_flag_cells(mesh, cell_fan_id);

  /* Set the fans flows to zero */

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {
    fan = _cs_glob_fans[fan_id];
    fan->in_flow = 0.0;
    fan->out_flow = 0.0;
  }

  /* Contribution to the domain interior */

  for (cs_lnum_t face_id = 0; face_id < nbr_fac; face_id++) {

    cs_lnum_t cell_id_1 = i_face_cells[face_id][0];
    cs_lnum_t cell_id_2 = i_face_cells[face_id][1];

    if (   cell_id_1 < mesh->n_cells /* Make sure the contrib is from one domain */
        && cell_fan_id[cell_id_1] != cell_fan_id[cell_id_2]) {

      for (int i = 0; i < 2; i++) {

        cs_lnum_t cell_id = i_face_cells[face_id][i];
        int fan_id = cell_fan_id[cell_id];

        if (fan_id > -1) {
          fan = _cs_glob_fans[fan_id];
          direction = (i == 0 ? 1 : - 1);
          flow = i_mass_flux[face_id]/c_rho[cell_id] * direction;
          if (  cs_math_3_dot_product(fan->axis_dir, i_face_u_normal[face_id])
              * direction > 0.0)
            fan->out_flow += flow;
          else
            fan->in_flow += flow;
        }
      }

    }

  }

  /* Contribution to the domain boundary */

  for (cs_lnum_t face_id = 0; face_id < nbr_fbr; face_id++) {

    int fan_id = cell_fan_id[b_face_cells[face_id]];

    if (fan_id > -1) {

      fan = _cs_glob_fans[fan_id];

      flow = b_mass_flux[face_id]/b_rho[face_id];
      if (cs_math_3_dot_product(fan->axis_dir, b_face_u_normal[face_id]) > 0.0)
        fan->out_flow += flow;
      else
        fan->in_flow += flow;

    }
  }

#if defined(HAVE_MPI)
  if (cs_glob_n_ranks > 1) {

    for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {

      fan = _cs_glob_fans[fan_id];

      cs_real_t flow_glob[2];
      cs_real_t flow_loc[2] = {fan->out_flow, fan->in_flow};

      MPI_Allreduce(flow_loc, flow_glob, 2, CS_MPI_REAL, MPI_SUM,
                    cs_glob_mpi_comm);

      fan->out_flow = flow_glob[0];
      fan->in_flow = flow_glob[1];

    }
  }
#endif

  /* In 2D, the flow is normalized by the surface */

  if (fan->dim == 2) {
    cs_real_t  surf_2d;
    surf_2d =   (0.5*fan->surface - 2*fan->fan_radius*fan->thickness)
              /                    (2*fan->fan_radius+fan->thickness);
    fan->out_flow = fan->out_flow / surf_2d;
    fan->in_flow = fan->in_flow / surf_2d;
  }

  /* Free memory */

  CS_FREE(cell_fan_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the force induced by the fans
 *        (needs a previous calculation of the flows through each fan).
 *
 * \param[in]  mesh_quantities  mesh quantities
 * \param[in]  source_t         explicit source term for the velocity
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_compute_force(const cs_mesh_quantities_t  *mesh_quantities,
                     cs_real_3_t                  source_t[])
{
  cs_real_t  f_z, f_theta;
  cs_real_t  f_rot[3];

  const cs_real_3_t *restrict cell_cen = cs_glob_mesh_quantities_g->cell_cen;
  const cs_real_t  *cell_f_vol = mesh_quantities->cell_vol;
  const cs_real_t  pi = 4.*atan(1.);

  /* Compute the force induced by fans */

  /* Loop on fans */
  /*--------------*/

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {

    cs_fan_t *fan = _cs_glob_fans[fan_id];

    const cs_real_t r_hub = fan->hub_radius;
    const cs_real_t r_blades = fan->blades_radius;
    const cs_real_t r_fan = fan->fan_radius;

    const cs_real_t mean_flow = 0.5 * (fan->out_flow - fan->in_flow);

    fan->delta_p = (fan->curve_coeffs[2] * mean_flow*mean_flow)
                 + (fan->curve_coeffs[1] * mean_flow)
                 + (fan->curve_coeffs[0]);

    /* Loop on fan cells */
    /*-------------------*/

    for (cs_lnum_t i = 0; i < fan->n_cells; i++) {

      cs_lnum_t cell_id = fan->cell_list[i];

      f_z = 0.0;
      f_theta = 0.0;
      f_rot[0] = 0.0, f_rot[1] = 0.0, f_rot[2] = 0.0;

      if (r_blades < 1.0e-12 && r_hub < 1.0e-12) {

        f_z = fan->delta_p / fan->thickness;
        f_theta = 0.0;

      }
      else if (r_hub < r_blades) {

        cs_real_t  r_1, r_2, aux_1, aux_2, coo_axis, d_axis, d_cel_axis[3];

        r_1 = 0.7  * fan->blades_radius;
        r_2 = 0.85 * fan->blades_radius;

        if (fan->dim == 2) {
          if (fan->mode == 1) {
            aux_1 = -  (fan->delta_p * 2.0 * r_fan)
                    / (fan->thickness * (1.15*r_blades - r_hub));
          }
          else{
            aux_1 =   (fan->delta_p * 2.0 * r_fan)
                    / (fan->thickness * (1.15*r_blades - r_hub));
          }
          aux_2 = 0.0;
        }
        else {
          const cs_real_t r_hub4 = r_hub * r_hub * r_hub * r_hub;
          const cs_real_t r_hub3 = r_hub * r_hub * r_hub;
          const cs_real_t r_blades4 = r_blades * r_blades * r_blades * r_blades;
          const cs_real_t r_blades3 = r_blades * r_blades * r_blades;
          const cs_real_t r_blades2 = r_blades * r_blades;
          const cs_real_t r_fan2 = r_fan * r_fan;
          cs_real_t f_base =   (0.7*r_blades - r_hub)
                             / (  1.0470*fan->thickness
                                * (  r_hub3
                                   + 1.4560*r_blades3
                                   - 2.570*r_blades2*r_hub));
          cs_real_t f_orth =   (0.7*r_blades - r_hub)
                             / (  fan->thickness
                                * (  1.042*r_blades4
                                   + 0.523*r_hub4
                                   - 1.667*r_blades3*r_hub));
          if (fan->mode == 1)
            aux_1 = - f_base * fan->delta_p * pi * r_fan2;
          else
            aux_1 = f_base * fan->delta_p * pi * r_fan2;
          aux_2 = f_orth * fan->axial_torque;
        }

        /* Vector from the outlet face axis point to the cell center */

        for (int coo_id = 0; coo_id < 3; coo_id++) {
          d_cel_axis[coo_id] =   (cell_cen[cell_id][coo_id])
                               - fan->inlet_axis_coords[coo_id];
        }

        /* Projection of the cell center on the fan axis */

        coo_axis = cs_math_3_dot_product(d_cel_axis, fan->axis_dir);

        /* Projection of the vector from the outlet face axis point
           to the cell center in the fan plane */

        for (int coo_id = 0; coo_id < 3; coo_id++)
          d_cel_axis[coo_id] -= coo_axis * fan->axis_dir[coo_id];

        d_axis = cs_math_3_norm(d_cel_axis); /* Distance to the axis */

        cs_math_3_cross_product(fan->axis_dir, d_cel_axis, f_rot);

        cs_math_3_normalize(f_rot, f_rot);

        if (d_axis < r_hub) {
          f_z     = 0.0;
          f_theta = 0.0;
        }
        else if (d_axis < r_1) {
          f_z     = aux_1 * (d_axis - r_hub) / (r_1 - r_hub);
          f_theta = aux_2 * (d_axis - r_hub) / (r_1 - r_hub);
        }
        else if (d_axis < r_2) {
          f_z     = aux_1;
          f_theta = aux_2;
        }
        else if (d_axis < r_blades) {
          f_z     = aux_1 * (r_blades - d_axis) / (r_blades - r_2);
          f_theta = aux_2 * (r_blades - d_axis) / (r_blades - r_2);
        }
        else {
          f_z     = 0.0;
          f_theta = 0.0;
        }

      }

      for (int coo_id = 0; coo_id < 3; coo_id++)
        source_t[cell_id][coo_id]
          +=    (   (f_z * fan->axis_dir[coo_id])
                 + (f_theta * f_rot[coo_id]))
             * fan->volume_expected / fan->volume /* Correction factor */
             * cell_f_vol[cell_id];

    }  /* End of loop on fan cells */

  } /* End of loop on fans */

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Flag the cells belonging to the different fans
 *        (by the fan id, -1 otherwise)
 *
 * \param[in]   mesh          assosiated mesh structure
 * \param[out]  cell_fan_id   fan id (or -1) for each cell
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_flag_cells(const cs_mesh_t  *mesh,
                  int               cell_fan_id[])
{
  cs_fan_t  *fan;

  const cs_lnum_t  n_cells_ext = mesh->n_cells_with_ghosts;

  /* Flag the cells */
  for (cs_lnum_t cell_id = 0; cell_id < n_cells_ext; cell_id++)
    cell_fan_id[cell_id] = -1;

  for (int fan_id = 0; fan_id < _cs_glob_n_fans; fan_id++) {

    fan = _cs_glob_fans[fan_id];

    for (cs_lnum_t i = 0; i < fan->n_cells; i++) {
      cs_lnum_t cell_id = fan->cell_list[i];
      cell_fan_id[cell_id] = fan_id;
    }

  }
  /* Synchronize cell_fan_id */
  if (mesh->halo != nullptr)
    cs_halo_sync_untyped(mesh->halo,
                         CS_HALO_EXTENDED,
                         sizeof(int),
                         cell_fan_id);

  /* Store the cell_fan_id in the postprocessing field */

  cs_field_t *c_fan_id = cs_field_by_name("fan_id");
  for (cs_lnum_t cell_id = 0; cell_id < n_cells_ext; cell_id++)
    c_fan_id->val[cell_id] = (cs_real_t)cell_fan_id[cell_id];
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Selection function for cells belonging to fans.
 *
 * This function may be used for the definition of postprocessing meshes.
 *
 * \param[in, out]  input    pointer to input (unused here)
 * \param[out]      n_cells  number of selected cells
 * \param[out]      cell_ids array of selected cell ids (0 to n-1 numbering)
 */
/*----------------------------------------------------------------------------*/

void
cs_fan_cells_select(void         *input,
                    cs_lnum_t    *n_cells,
                    cs_lnum_t   **cell_ids)
{
  CS_UNUSED(input);

  cs_lnum_t _n_cells;

  int *cell_fan_id = nullptr;
  cs_lnum_t *_cell_ids = nullptr;

  const cs_mesh_t *m = cs_glob_mesh;

  /* Preallocate selection list */

  CS_MALLOC(_cell_ids, m->n_cells, cs_lnum_t);

  /* Allocate working array */

  CS_MALLOC(cell_fan_id, m->n_cells_with_ghosts, int);

  /* Now flag cells and build list */

  cs_fan_build_all(cs_glob_mesh, cs_glob_mesh_quantities);
  cs_fan_flag_cells(m, cell_fan_id);

  _n_cells = 0;

  for (cs_lnum_t i = 0; i < m->n_cells; i++) {
    if (cell_fan_id[i] > -1)
      _cell_ids[_n_cells++] = i;
  }

  /* Free memory */
  CS_FREE(cell_fan_id);
  CS_REALLOC(_cell_ids, _n_cells, cs_lnum_t);

  /* Set return values */

  *n_cells = _n_cells;
  *cell_ids = _cell_ids;
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
