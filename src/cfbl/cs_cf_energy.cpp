/*============================================================================
 * Solve the convection/diffusion equation (with eventual source terms)
 * for total energy over a time step.
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_printf.h"

#include "base/cs_array.h"
#include "base/cs_assert.h"
#include "alge/cs_blas.h"
#include "cfbl/cs_cf_boundary_conditions.h"
#include "cfbl/cs_cf_model.h"
#include "cfbl/cs_cf_thermo.h"
#include "alge/cs_divergence.h"
#include "base/cs_equation_iterative_solve.h"
#include "alge/cs_face_viscosity.h"
#include "base/cs_field_default.h"
#include "base/cs_field_operator.h"
#include "base/cs_field_pointer.h"
#include "base/cs_gas_mix.h"
#include "base/cs_mass_source_terms.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "pprt/cs_physical_model.h"
#include "base/cs_prototypes.h"
#include "base/cs_scalar_clipping.h"
#include "turb/cs_turbulence_model.h"
#include "base/cs_volume_mass_injection.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cfbl/cs_cf_energy.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_cf_energy.cpp

  Solve the convection/diffusion equation (with eventual source terms)
  for total energy over a time step.
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!                                          v
 * \brief Compute DIVERG = DIVERG + DIV(SIGMA .U)
 *
 *           v               t
 * with SIGMA = MU (GRAD(U) + GRAD(U)) + (KAPPA - 2/3 MU) DIV(U) Id
 *
 * and MU = MU_LAMINAR + MU_TURBULENT
 *
 * \param[inout]  div          divergence
 */
/*----------------------------------------------------------------------------*/

static void
_cf_div(cs_real_t div[])
{

  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t *fvq = cs_glob_mesh_quantities;
  const cs_halo_t *halo = mesh->halo;
  const cs_lnum_t n_i_faces = mesh->n_i_faces;
  const cs_lnum_t n_b_faces = mesh->n_b_faces;
  const cs_lnum_t n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t n_cells = mesh->n_cells;

  const cs_lnum_t *b_face_cells = mesh->b_face_cells;
  const cs_lnum_2_t *i_face_cells = (const cs_lnum_2_t *)mesh->i_face_cells;
  const cs_real_3_t *i_f_face_normal
    = (const cs_real_3_t *)fvq->i_face_normal;
  const cs_real_3_t *b_f_face_normal
    = (const cs_real_3_t *)fvq->b_face_normal;

  const int itytur = cs_glob_turb_model->itytur;

  /* Initialization
     -------------- */

  cs_field_t *f_vel = CS_F_(vel);
  cs_real_3_t *vel = (cs_real_3_t *)f_vel->val;

  /* Allocate temporary arrays */

  cs_real_t *vistot;
  cs_real_33_t *gradv;
  cs_real_3_t *tempv;
  CS_MALLOC(vistot, n_cells_ext, cs_real_t);
  CS_MALLOC(gradv, n_cells_ext, cs_real_33_t);
  CS_MALLOC(tempv, n_cells_ext, cs_real_3_t);

  const cs_real_t *viscl = CS_F_(mu)->val;
  const cs_real_t *visct = CS_F_(mu_t)->val;

  cs_field_t *f_viscv = cs_field_by_name_try("volume_viscosity");
  cs_real_t *cpro_kappa;

  if (f_viscv != nullptr)
    cpro_kappa = f_viscv->val;

  /* Compute total viscosity */

  if (itytur == 3) {
#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++)
      vistot[c_id] = viscl[c_id];
  }
  else {
#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      vistot[c_id] = viscl[c_id] + visct[c_id];
    }
  }

  /* Periodicity and parallelism process */

  if (cs_glob_rank_id > -1 || mesh->periodicity != nullptr) {

    cs_halo_sync_var(halo, CS_HALO_STANDARD, vistot);

    if (f_viscv != nullptr)
      cs_halo_sync_var(halo, CS_HALO_STANDARD, cpro_kappa);
  }

  /* Compute the divegence of (sigma.u)
     ---------------------------------- */

  cs_field_gradient_vector(f_vel,
                           true, /* iprev = 1 */
                           1, /* inc */
                           gradv);

  /* Compute the vector \tens{\sigma}.\vect{v}
     i.e. sigma_ij v_j e_i */

  /* Variable kappa in space */
  const cs_real_t viscv0 = cs_glob_fluid_properties->viscv0;

# pragma omp parallel for if (n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {

    const cs_real_t kappa = (f_viscv != nullptr) ? cpro_kappa[c_id] : viscv0;
    const cs_real_t mu = vistot[c_id];

    const cs_real_t trgdru = cs_math_33_trace(gradv[c_id]);

    /* Symmetric matrix sigma */
    cs_real_t sigma[6] = {0., 0., 0., 0., 0., 0.};

    sigma[0] =   mu * 2.0 * gradv[c_id][0][0]
      + (kappa - 2.0 / 3.0 * mu) * trgdru;

    sigma[1] =   mu * 2.0 * gradv[c_id][1][1]
      + (kappa - 2.0 / 3.0 * mu) * trgdru;

    sigma[2] =   mu * 2.0 * gradv[c_id][2][2]
      + (kappa - 2.0 / 3.0 * mu) * trgdru;

    sigma[3] = mu * (gradv[c_id][0][1] + gradv[c_id][1][0]);

    sigma[4] = mu * (gradv[c_id][1][2] + gradv[c_id][2][1]);

    sigma[5] = mu * (gradv[c_id][0][2] + gradv[c_id][2][0]);

    cs_math_sym_33_3_product(sigma, vel[c_id], tempv[c_id]);

  }

  /* Periodicity and parallelism process */

  if (cs_glob_rank_id > -1 || mesh->periodicity != nullptr) {
    cs_halo_sync_var_strided(halo,
                             CS_HALO_STANDARD,
                             (cs_real_t *)tempv,
                             3);

    if (mesh->n_init_perio > 0)
      cs_halo_perio_sync_var_vect(halo,
                                  CS_HALO_STANDARD,
                                  (cs_real_t *)tempv,
                                  3);

  }

  /* Initialize diverg(ncel+1, ncelet)
     (unused value, but need to be initialized to avoid Nan values) */

  if (n_cells_ext > n_cells) {
#   pragma omp parallel for if ((n_cells_ext-n_cells) > CS_THR_MIN)
    for (cs_lnum_t c_id = n_cells; c_id < n_cells_ext; c_id++)
      div[c_id] = 0.0;
  }

  /* Interior faces contribution */

  for (cs_lnum_t f_id = 0; f_id < n_i_faces; f_id++) {
    const cs_lnum_t c_id0 = i_face_cells[f_id][0];
    const cs_lnum_t c_id1 = i_face_cells[f_id][1];

    const cs_real_t vecfac
      =   0.5*i_f_face_normal[f_id][0]*(tempv[c_id0][0]+tempv[c_id1][0])
        + 0.5*i_f_face_normal[f_id][1]*(tempv[c_id0][1]+tempv[c_id1][1])
        + 0.5*i_f_face_normal[f_id][2]*(tempv[c_id0][2]+tempv[c_id1][2]);

    div[c_id0] = div[c_id0] + vecfac;
    div[c_id1] = div[c_id1] - vecfac;
  }

  /* Boundary faces contribution */

  for (cs_lnum_t f_id = 0; f_id < n_b_faces; f_id++) {
    const cs_lnum_t c_id = b_face_cells[f_id];

    const cs_real_t vecfac =   b_f_face_normal[f_id][0] * tempv[c_id][0]
                             + b_f_face_normal[f_id][1] * tempv[c_id][1]
                             + b_f_face_normal[f_id][2] * tempv[c_id][1];

    div[c_id] = div[c_id] + vecfac;
  }

  /* Free memory */
  CS_FREE(gradv);
  CS_FREE(tempv);
  CS_FREE(vistot);
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Perform the solving of the convection/diffusion equation (with
 * eventual source terms) for total energy over a time step. It is the third
 * step of the compressible algorithm at each time iteration.
 *
 * Please refer to the <a href="../../theory.pdf#cfener"><b>cfener</b></a> section
 * of the theory guide for more informations.
 *
 * \param[in]     f_sc_id       scalar field id
 */
/*----------------------------------------------------------------------------*/

void
cs_cf_energy(int f_sc_id)
{
  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t *fvq = cs_glob_mesh_quantities;
  const cs_halo_t *halo = mesh->halo;
  const cs_lnum_t n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t n_cells = mesh->n_cells;
  const cs_lnum_t n_i_faces = mesh->n_i_faces;
  const cs_lnum_t n_b_faces = mesh->n_b_faces;
  const cs_real_t *restrict weight = fvq->weight;
  const cs_lnum_2_t *i_face_cells = (const cs_lnum_2_t *)mesh->i_face_cells;
  const cs_lnum_t *b_face_cells = mesh->b_face_cells;
  const cs_real_t *b_dist = fvq->b_dist;
  const cs_real_3_t *cell_cen = (const cs_real_3_t *)fvq->cell_cen;
  const cs_real_3_t *i_face_cog = (const cs_real_3_t *)fvq->i_face_cog;
  const cs_real_t *cell_f_vol = fvq->cell_vol;
  const cs_rreal_3_t *restrict diipb = fvq->diipb;
  const cs_real_3_t *restrict dijpf = fvq->dijpf;

  const int kivisl  = cs_field_key_id("diffusivity_id");
  const int ksigmas = cs_field_key_id("turbulent_schmidt");

  const cs_fluid_properties_t *fluid_props = cs_glob_fluid_properties;
  const cs_real_t cp0 = fluid_props->cp0;
  const cs_real_t cv0 = fluid_props->cv0;
  const int icp = fluid_props->icp;
  const int icv = fluid_props->icv;

  const int n_species_solved = cs_glob_gas_mix->n_species_solved;
  const int k_id = cs_gas_mix_get_field_key();
  const int idtvar = cs_glob_time_step_options->idtvar;

  const cs_real_t *gxyz = cs_get_glob_physical_constants()->gravity;
  cs_real_t *dt = CS_F_(dt)->val;

  /* Computation number and post-treatment number of the
     scalar total energy */

  /* Map field arrays */

  cs_field_t *f_sc = cs_field_by_id(f_sc_id);
  cs_field_t *f_vel = CS_F_(vel);
  cs_field_t *f_pr = CS_F_(p);
  cs_field_t *f_tempk = CS_F_(t);

  cs_real_t *energy_pre = f_sc->val_pre;
  cs_real_t *energy = f_sc->val;
  cs_real_t *tempk = f_tempk->val;
  cs_real_3_t *vel = (cs_real_3_t *)f_vel->val;
  cs_real_t *pr = f_pr->val;

  cs_equation_param_t *eqp_vel = cs_field_get_equation_param(f_vel);
  cs_equation_param_t *eqp_p = cs_field_get_equation_param(f_pr);
  cs_equation_param_t *eqp_e = cs_field_get_equation_param(f_sc);

  if (eqp_e->verbosity >= 1) {
    cs_log_printf(CS_LOG_DEFAULT,
                  _("\n"
                    "   ** RESOLUTION FOR THE VARIABLE %s\n"
                    "      ---------------------------\n"),
                  f_sc->name);
  }

  /* Barotropic version */
  if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 1) {
    cs_array_real_set_scalar(n_cells, fluid_props->eint0, energy);

    if (halo != nullptr) {
      cs_halo_sync_var(halo, CS_HALO_STANDARD, pr);
      cs_halo_sync_var(halo, CS_HALO_STANDARD, energy);
      cs_halo_sync_var(halo, CS_HALO_STANDARD, tempk);
    }

    return;
  }

  cs_real_t *cpro_cp = nullptr, *cpro_cv = nullptr;
  if (icp >= 0)
    cpro_cp = CS_F_(cp)->val;

  if (icv >= 0)
    cpro_cv = cs_field_by_id(icv)->val;

  /* Initialization */

  /* Allocate a temporary array */
  cs_real_t *wb, *rhs, *rovsdt;
  CS_MALLOC(wb, n_b_faces, cs_real_t);
  CS_MALLOC(rhs, n_cells_ext, cs_real_t);
  CS_MALLOC(rovsdt, n_cells_ext, cs_real_t);

  /* Allocate work arrays */
  cs_real_t *w7, *w9, *dpvar;
  cs_real_3_t *grad;
  CS_MALLOC(grad, n_cells_ext, cs_real_3_t);
  CS_MALLOC(w7, n_cells_ext, cs_real_t);
  CS_MALLOC(w9, n_cells_ext, cs_real_t);
  CS_MALLOC(dpvar, n_cells_ext, cs_real_t);

  /* Physical property numbers */
  cs_real_t *crom = CS_F_(rho)->val;
  cs_real_t *crom_pre = CS_F_(rho)->val_pre;
  cs_real_t *brom = CS_F_(rho_b)->val;

  const cs_real_t *visct = CS_F_(mu_t)->val;

  cs_real_t *frace = nullptr, *fracv = nullptr, *fracm = nullptr;
  if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 2) {
    fracv = (cs_real_t *)CS_F_(volume_f)->val;
    fracm = (cs_real_t *)CS_F_(mass_f)->val;
    frace = (cs_real_t *)CS_F_(energy_f)->val;
  }

  int iflmas = cs_field_get_key_int(f_sc, cs_field_key_id("inner_mass_flux_id"));
  const cs_real_t *i_mass_flux = cs_field_by_id(iflmas)->val;

  int iflmab = cs_field_get_key_int(f_sc, cs_field_key_id("boundary_mass_flux_id"));
  const cs_real_t *b_mass_flux = cs_field_by_id(iflmab)->val;

  const int ifcvsl = cs_field_get_key_int(f_sc, kivisl);
  cs_real_t *viscls = nullptr;
  if (ifcvsl > -1)
    viscls = cs_field_by_id(ifcvsl)->val;

  /* Source terms */

  /* Theta-scheme:
     for now, theta=1 is assumed and the theta-scheme is not implemented */

  /* Initialization */

# pragma omp parallel for if (n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    rhs[c_id] = 0.0;
    rovsdt[c_id] = 0.0;
  }

  /* Heat volume source term: rho * phi * volume
     -------------------------------------------- */

  cs_user_source_terms(cs_glob_domain, f_sc->id, rhs, rovsdt);

# pragma omp parallel for if (n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    rhs[c_id] += rovsdt[c_id] * energy[c_id];
    rovsdt[c_id] = cs::max(-rovsdt[c_id], 0.0);
  }

  /* Mass source terms
     ----------------- */

  /* gamma[c_id] = smcel_pl[c_id]

     Implicit term : gamma*volume

     Explicit term : gamma*volume*e   - gamma*volume*e
                                         inj
  */

  if (eqp_e->n_volume_mass_injections > 0) {
    cs_lnum_t ncesmp = 0;
    const cs_lnum_t *icetsm = nullptr;
    int *itpsm = nullptr;
    cs_real_t *smcel_p = nullptr, *smcel_sc = nullptr;

    cs_volume_mass_injection_get_arrays(f_sc,
                                        &ncesmp,
                                        &icetsm,
                                        &itpsm,
                                        &smcel_sc,
                                        &smcel_p);

    cs_mass_source_terms(1, /* iterns */
                         1, /* dim */
                         ncesmp,
                         icetsm,
                         itpsm,
                         cell_f_vol,
                         energy,
                         smcel_sc,
                         smcel_p,
                         rhs,
                         rovsdt,
                         nullptr);
  }

  /*                         rho*volume
    Unsteady implicit term : ----------
    ----------------------       dt
  */

  if (eqp_e->istat > 0) {
#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++)
      rovsdt[c_id] += (crom_pre[c_id] / dt[c_id]) * cell_f_vol[c_id];
  }

  /*                                       __        v
     Viscous dissipation term        : >  ((SIGMA *U).n)  *S
     -----------------------          --               ij  ij
  */

  if (eqp_vel->idiff >= 1)
    _cf_div(rhs);

  /*                              __   P        n+1
     Pressure transport term  : - >  (---)  *(Q    .n)  *S
     -----------------------      --  RHO ij   pr     ij  ij
  */

  cs_real_t *iprtfl, *bprtfl;
  CS_MALLOC(iprtfl, n_i_faces, cs_real_t);
  CS_MALLOC(bprtfl, n_b_faces, cs_real_t);

  /* No reconstruction yet */

  /* Internal faces */
# pragma omp parallel for if (n_i_faces > CS_THR_MIN)
  for (cs_lnum_t f_id = 0; f_id < n_i_faces; f_id++) {
    const cs_lnum_t c_id0 = i_face_cells[f_id][0];
    const cs_lnum_t c_id1 = i_face_cells[f_id][1];
    const cs_real_t f_mass_flux = i_mass_flux[f_id];
    const cs_real_t f_abs_mass_flux = fabs(f_mass_flux);

    iprtfl[f_id] = - pr[c_id0]/crom[c_id0]*0.5*(f_mass_flux + f_abs_mass_flux)
                   - pr[c_id1]/crom[c_id1]*0.5*(f_mass_flux - f_abs_mass_flux);
  }

  /* Boundary faces: for the faces where a flux (Rusanov or analytical) has been
     computed, the standard contribution is replaced by this flux in bilsc2. */

  const cs_real_t *coefa_p = f_pr->bc_coeffs->a;
  const cs_real_t *coefb_p = f_pr->bc_coeffs->b;
  int *icvfli = cs_cf_boundary_conditions_get_icvfli();

# pragma omp parallel for if (n_b_faces > CS_THR_MIN)
  for (cs_lnum_t f_id = 0; f_id < n_b_faces; f_id++) {

    if (icvfli[f_id] == 0) {
      const cs_lnum_t c_id = b_face_cells[f_id];

      bprtfl[f_id] =  - b_mass_flux[f_id]
                      * (coefa_p[f_id] + coefb_p[f_id]*pr[c_id])
                      / brom[f_id];
    }
    else
      bprtfl[f_id] = 0.0;
  }

  /* Divergence */
  cs_divergence(mesh,
                0, /* init */
                iprtfl,
                bprtfl,
                rhs);

  CS_FREE(iprtfl);
  CS_FREE(bprtfl);

  /* Gravitation force term: rho*g.u *cvolume
     ---------------------- */

# pragma omp parallel for if (n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    rhs[c_id] +=   crom[c_id]*cell_f_vol[c_id]
                   * cs_math_3_dot_product(gxyz, vel[c_id]);
  }

  /*                              Kij*Sij           Lambda   Cp      mut
     Face diffusion "Velocity" : --------- with K = ------ + -- .------------
     ------------------------      IJ.nij             Cv     Cv  turb_schmidt
  */

  /* Only SGDH available */

  cs_real_t *c_viscs_t, *i_visc, *b_visc;
  CS_MALLOC(c_viscs_t, n_cells_ext, cs_real_t);
  CS_MALLOC(i_visc, n_i_faces, cs_real_t);
  CS_MALLOC(b_visc, n_b_faces, cs_real_t);

  if (eqp_e->idiff >= 1) {

    const cs_real_t turb_schmidt = cs_field_get_key_double(f_sc, ksigmas);
    const int kvisl0 = cs_field_key_id("diffusivity_ref");

    const int n_i_groups = mesh->i_face_numbering->n_groups;
    const int n_i_threads = mesh->i_face_numbering->n_threads;
    const int n_b_threads = mesh->b_face_numbering->n_threads;
    const cs_lnum_t *restrict i_group_index = mesh->i_face_numbering->group_index;
    const cs_lnum_t *restrict b_group_index = mesh->b_face_numbering->group_index;

#   pragma omp parallel if (n_cells > CS_THR_MIN)
    {

      cs_lnum_t s_id, e_id;
      cs_parall_thread_range(n_cells, sizeof(cs_real_t), &s_id, &e_id);

      /* mu_t/turb_schmidt */

      for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
        c_viscs_t[c_id] = visct[c_id] / turb_schmidt;

      /* cp*mu_t/turb_schmidt */
      if (icp >= 0) {
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] *= cpro_cp[c_id];
      }
      else {
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] *= cp0;
      }

      /* (cp/cv)*mu_t/turb_schmidt */
      if (icv >= 0) {
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] /= cpro_cv[c_id];
      }
      else {
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] /= cv0;
      }

      /* (cp/cv)*mu_t/turb_schmidt+lambda/cv */
      if (ifcvsl < 0) {
        cs_real_t visls_0 = cs_field_get_key_double(f_sc, kvisl0);
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] += visls_0;
      }
      else {
        for (cs_lnum_t c_id = s_id; c_id < e_id; c_id++)
          c_viscs_t[c_id] += viscls[c_id];
      }

    } /* End of OpenMP section */

    cs_face_viscosity(mesh,
                      fvq,
                      eqp_vel->imvisf,
                      c_viscs_t,
                      i_visc,
                      b_visc);

    /* Complementary diffusive term: - div( K grad ( epsilon - Cv.T ) )
       ----------------------------                  1  2
                                     - div( K grad ( -.u  ) )
                                                     2
    */

    /* Complementary term at cell centers */

    /* Compute e - CvT */

    /* At cell centers */
    cs_cf_thermo_eps_sup(crom, w9, n_cells);

    /* At boundary faces centers */
    cs_cf_thermo_eps_sup(brom, wb, n_b_faces);

    /* Divergence computation with reconstruction */

    /* Computation of the gradient of (0.5*u*u+EPSILONsup) */
#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      w7[c_id] = 0.5 * cs_math_3_dot_product(vel[c_id], vel[c_id]) + w9[c_id];
    }

    /* Note: by default, since the parameters are unknown, the
       velocity parameters are taken */

    int imrgrp = eqp_p->imrgra;
    cs_halo_type_t halo_type = CS_HALO_STANDARD;
    cs_gradient_type_t gradient_type = CS_GRADIENT_GREEN_ITER;
    cs_gradient_type_by_imrgra(imrgrp,
                               &gradient_type,
                               &halo_type);

    cs_gradient_scalar("Work array",
                       gradient_type,
                       halo_type,
                       1, /* inc */
                       eqp_vel->nswrgr,
                       0,             /* iphydp */
                       1,             /* w_stride */
                       eqp_vel->verbosity,
                       (cs_gradient_limit_t) eqp_vel->imligr,
                       eqp_vel->epsrgr,
                       eqp_vel->climgr,
                       nullptr,          /* f_ext */
                       nullptr,          /* bc_coeffs */
                       w7,
                       nullptr,          /* c_weight */
                       nullptr,          /* cpl */
                       grad);

    /* Internal faces */

    for (int g_id = 0; g_id < n_i_groups; g_id++) {

#     pragma omp parallel for
      for (int t_id = 0; t_id < n_i_threads; t_id++) {

        for (cs_lnum_t f_id = i_group_index[(t_id*n_i_groups + g_id)*2];
             f_id < i_group_index[(t_id*n_i_groups + g_id)*2 + 1];
             f_id++) {

          const cs_lnum_t c_id0 = i_face_cells[f_id][0];
          const cs_lnum_t c_id1 = i_face_cells[f_id][1];

          const cs_real_t *_dijpf = dijpf[f_id];

          const cs_real_t pnd = weight[f_id];

          /* Computation of II' and JJ' */

          const cs_real_t diipf[3]
            = {i_face_cog[f_id][0] - (cell_cen[c_id0][0] + (1.0-pnd)*_dijpf[0]),
               i_face_cog[f_id][1] - (cell_cen[c_id0][1] + (1.0-pnd)*_dijpf[1]),
               i_face_cog[f_id][2] - (cell_cen[c_id0][2] + (1.0-pnd)*_dijpf[2])};

          const cs_real_t djjpf[3]
            = {i_face_cog[f_id][0] -  cell_cen[c_id1][0] +  pnd*_dijpf[0],
               i_face_cog[f_id][1] -  cell_cen[c_id1][1] +  pnd*_dijpf[1],
               i_face_cog[f_id][2] -  cell_cen[c_id1][2] +  pnd*_dijpf[2]};

          const cs_real_t pip =   w7[c_id0]
                                + cs_math_3_dot_product(grad[c_id0], diipf);
          const cs_real_t pjp =   w7[c_id1]
                                + cs_math_3_dot_product(grad[c_id1], djjpf);

          const cs_real_t flux = i_visc[f_id] * (pip - pjp);

          rhs[c_id0] = rhs[c_id0] + flux;
          rhs[c_id1] = rhs[c_id1] - flux;

        }

      }

    }

    cs_real_t *i_visck = nullptr, *b_visck = nullptr;

    if (cs_glob_physical_model_flag[CS_GAS_MIX] > 0) {

      /* Diffusion flux for the species at internal faces */

      cs_real_t *kspe;
      CS_MALLOC(kspe, n_cells_ext, cs_real_t);
      CS_MALLOC(i_visck, n_i_faces, cs_real_t);
      CS_MALLOC(b_visck, n_b_faces, cs_real_t);

      /* Diffusion coefficient  T*lambda*Cvk/Cv */
#     pragma omp parallel for if (n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        kspe[c_id] = c_viscs_t[c_id] * tempk[c_id];
      }

      cs_face_viscosity(mesh,
                        fvq,
                        eqp_e->imvisf,
                        kspe,
                        i_visck,
                        b_visck);

      CS_FREE(kspe);

      cs_real_t *grad_dd;
      CS_MALLOC(grad_dd, n_i_faces, cs_real_t);
      cs_array_real_fill_zero(n_i_faces, grad_dd);

      for (cs_lnum_t spe_id = 0; spe_id < n_species_solved; spe_id++) {

        const int f_spe_id = cs_glob_gas_mix->species_to_field_id[spe_id];
        cs_field_t *f_spe = cs_field_by_id(f_spe_id);
        cs_real_t *yk = f_spe->val;

        cs_gas_mix_species_prop_t s_k;
        cs_field_get_key_struct(f_spe, k_id, &s_k);

        cs_real_t mol_mas = s_k.mol_mas;
        cs_real_t cp = s_k.cp;
        cs_real_t cv = cp - cs_physical_constants_r / mol_mas;

        cs_field_gradient_scalar(f_spe,
                                 false,
                                 1, /* inc */
                                 grad);

        for (int g_id = 0; g_id < n_i_groups; g_id++) {

#         pragma omp parallel for
          for (int t_id = 0; t_id < n_i_threads; t_id++) {

            for (cs_lnum_t f_id = i_group_index[(t_id*n_i_groups + g_id)*2];
                 f_id < i_group_index[(t_id*n_i_groups + g_id)*2 + 1];
                 f_id++) {

              const cs_lnum_t c_id0 = i_face_cells[f_id][0];
              const cs_lnum_t c_id1 = i_face_cells[f_id][1];

              const cs_real_t *_dijpf = dijpf[f_id];

              const cs_real_t pnd = weight[f_id];

              /* Computation of II' and JJ' */

              const cs_real_t diipf[3]
                = {  i_face_cog[f_id][0]
                   - (cell_cen[c_id0][0] + (1.0-pnd) * _dijpf[0]),
                     i_face_cog[f_id][1]
                   - (cell_cen[c_id0][1] + (1.0-pnd) * _dijpf[1]),
                     i_face_cog[f_id][2]
                   - (cell_cen[c_id0][2] + (1.0-pnd) * _dijpf[2])};

              const cs_real_t djjpf[3]
                = {  i_face_cog[f_id][0]
                   -  cell_cen[c_id1][0] +  pnd  * _dijpf[0],
                     i_face_cog[f_id][1]
                   -  cell_cen[c_id1][1] +  pnd  * _dijpf[1],
                     i_face_cog[f_id][2]
                   -  cell_cen[c_id1][2] +  pnd  * _dijpf[2]};

              const cs_real_t yip =   yk[c_id0]
                                    + cs_math_3_dot_product(grad[c_id0], diipf);

              const cs_real_t yjp =   yk[c_id1]
                                    + cs_math_3_dot_product(grad[c_id1], djjpf);

              /* Gradient of deduced species */
              grad_dd[f_id] = grad_dd[f_id] - (yjp - yip);

              const cs_real_t flux = i_visck[f_id] * cv * (yip - yjp);

              rhs[c_id0] = rhs[c_id0] + flux;
              rhs[c_id1] = rhs[c_id1] - flux;

            }

          }

        }

      } /* End loop on species */

      /* Diffusion flux for the deduced species */

      int iddgas = -1;
      if (cs_glob_physical_model_flag[CS_GAS_MIX] <= 5)
        iddgas = cs_glob_gas_mix->species_to_field_id[n_species_solved];

      assert(iddgas > -1);
      cs_field_t *f_iddgas = cs_field_by_id(iddgas);
      cs_gas_mix_species_prop_t s_k;
      cs_field_get_key_struct(f_iddgas, k_id, &s_k);

      const cs_real_t mol_mas = s_k.mol_mas;
      const cs_real_t cp = s_k.cp;
      const cs_real_t cv = cp - cs_physical_constants_r / mol_mas;

      for (int g_id = 0; g_id < n_i_groups; g_id++) {

#       pragma omp parallel for
        for (int t_id = 0; t_id < n_i_threads; t_id++) {

          for (cs_lnum_t f_id = i_group_index[(t_id*n_i_groups + g_id)*2];
               f_id < i_group_index[(t_id*n_i_groups + g_id)*2 + 1];
               f_id++) {

            const cs_lnum_t c_id0 = i_face_cells[f_id][0];
            const cs_lnum_t c_id1 = i_face_cells[f_id][1];

            const cs_real_t flux = i_visc[f_id] * grad_dd[f_id] * cv;

            rhs[c_id0] = rhs[c_id0] + flux;
            rhs[c_id1] = rhs[c_id1] - flux;

          }

        }

      }

      CS_FREE(grad_dd);

    } /* End gas mix process */

    /* Assembly based on boundary faces
       for the faces where a flux or a temperature is imposed,
       all is taken into account by the energy diffusion term.
       Hence the contribution of the terms in u2 and e-CvT shouldn't be taken into
       account when ifbet(f_id) != 0. */

    cs_real_3_t  *coefau = (cs_real_3_t  *)f_vel->bc_coeffs->a;
    cs_real_33_t *coefbu = (cs_real_33_t *)f_vel->bc_coeffs->b;

    int *ifbet = cs_cf_boundary_conditions_get_ifbet();

#   pragma omp parallel for
    for (int t_id = 0; t_id < n_b_threads; t_id++) {

      for (cs_lnum_t f_id = b_group_index[t_id*2];
           f_id < b_group_index[t_id*2 + 1];
           f_id++) {
        if (ifbet[f_id] != 0)
          continue;

        const cs_lnum_t c_id = b_face_cells[f_id];

        const cs_real_t flux
          =     b_visc[f_id] * (c_viscs_t[c_id] / b_dist[f_id])
            * (  w9[c_id] - wb[f_id]
                + 0.5 *(  cs_math_pow2(vel[c_id][0])
                        - cs_math_pow2(  coefau[f_id][0]
                                       + coefbu[f_id][0][0] * vel[c_id][0]
                                       + coefbu[f_id][1][0] * vel[c_id][1]
                                       + coefbu[f_id][2][0] * vel[c_id][2])
                        + cs_math_pow2(vel[c_id][1])
                        - cs_math_pow2(  coefau[f_id][1]
                                       + coefbu[f_id][0][1] * vel[c_id][0]
                                       + coefbu[f_id][1][1] * vel[c_id][1]
                                       + coefbu[f_id][2][1] * vel[c_id][2])
                        + cs_math_pow2(vel[c_id][2])
                        - cs_math_pow2(  coefau[f_id][2]
                                       + coefbu[f_id][0][2] * vel[c_id][0]
                                       + coefbu[f_id][1][2] * vel[c_id][1]
                                       + coefbu[f_id][2][2] * vel[c_id][2])));
        rhs[c_id] = rhs[c_id] + flux;
      }

    }

    if (cs_glob_physical_model_flag[CS_GAS_MIX] > 0) {

      cs_real_t *coefat = f_tempk->bc_coeffs->a;
      cs_real_t *coefbt = f_tempk->bc_coeffs->b;

      cs_real_t *grad_dd, *btemp;
      CS_MALLOC(grad_dd, n_b_faces, cs_real_t);
      CS_MALLOC(btemp, n_b_faces, cs_real_t);

      cs_field_gradient_scalar(f_tempk,
                               false,
                               1,
                               grad);

#     pragma omp parallel for if (n_b_faces > CS_THR_MIN)
      for (cs_lnum_t f_id = 0; f_id < n_b_faces; f_id++) {
        grad_dd[f_id] = 0.0;

        const cs_lnum_t c_id = b_face_cells[f_id];

        const cs_real_t tip =   tempk[c_id]
                              + cs_math_3_dot_product(grad[c_id],
                                                      diipb[f_id]);

        btemp[f_id] = coefat[f_id] + coefbt[f_id] * tip;
      }

      for (cs_lnum_t spe_id = 0; spe_id < n_species_solved; spe_id++) {

        const int f_spe_id = cs_glob_gas_mix->species_to_field_id[spe_id];
        cs_field_t *f_spe = cs_field_by_id(f_spe_id);

        cs_real_t *yk = f_spe->val;
        cs_real_t *coefayk = f_spe->bc_coeffs->a;
        cs_real_t *coefbyk = f_spe->bc_coeffs->b;

        cs_gas_mix_species_prop_t s_k;
        cs_field_get_key_struct(f_spe, k_id, &s_k);

        cs_real_t mol_mas = s_k.mol_mas;
        cs_real_t cp = s_k.cp;
        cs_real_t cv = cp - cs_physical_constants_r / mol_mas;

        cs_field_gradient_scalar(f_spe,
                                 false,
                                 1, /* inc */
                                 grad);

#       pragma omp parallel for
        for (int t_id = 0; t_id < n_b_threads; t_id++) {

          for (cs_lnum_t f_id = b_group_index[t_id*2];
               f_id < b_group_index[t_id*2 + 1];
               f_id++) {
            if (ifbet[f_id] != 0)
              continue;

            const cs_lnum_t c_id = b_face_cells[f_id];

            const cs_real_t yip
              = yk[c_id] + cs_math_3_dot_product(grad[c_id], diipb[f_id]);

            const cs_real_t gradnb
              = coefayk[f_id] + (coefbyk[f_id] - 1.0) * yip;

            grad_dd[f_id] = grad_dd[f_id] - gradnb;

            const cs_real_t flux
              =   b_visck[f_id] * c_viscs_t[c_id] * btemp[f_id] * cv
                                / b_dist[f_id] * (-gradnb);

            rhs[c_id] = rhs[c_id] + flux;
          }

        } /* End loops on boundary faces */

      } /* End loop on species */

      /* Boundary diffusion flux for the deduced species */
      int iddgas = -1;

      if (cs_glob_physical_model_flag[CS_GAS_MIX] <= 5)
        iddgas = cs_glob_gas_mix->species_to_field_id[n_species_solved];

      assert(iddgas > -1);
      cs_field_t *f_iddgas = cs_field_by_id(iddgas);
      cs_gas_mix_species_prop_t s_k;
      cs_field_get_key_struct(f_iddgas, k_id, &s_k);

      const cs_real_t mol_mas = s_k.mol_mas;
      const cs_real_t cp = s_k.cp;
      const cs_real_t cv = cp - cs_physical_constants_r/ mol_mas;

#     pragma omp parallel for
      for (int t_id = 0; t_id < n_b_threads; t_id++) {

        for (cs_lnum_t f_id = b_group_index[t_id*2];
             f_id < b_group_index[t_id*2 + 1];
             f_id++) {
          if (ifbet[f_id] != 0)
            continue;

          const cs_lnum_t c_id = b_face_cells[f_id];

          const cs_real_t flux
            =   b_visck[f_id] * c_viscs_t[c_id] * btemp[f_id] * cv
                              / b_dist[f_id] * grad_dd[f_id];

          rhs[c_id] = rhs[c_id] + flux;
        }
      }

      CS_FREE(grad_dd);
      CS_FREE(btemp);
      CS_FREE(i_visck);
      CS_FREE(b_visck);

    } /* End gas mix process */

  } /* End eqp_e->idiff >= 1 */
  else {

    cs_array_real_fill_zero(n_i_faces, i_visc);
    cs_array_real_fill_zero(n_b_faces, b_visc);

  }

  /* Solving
     ------- */

  /* idtvar = 1  => unsteady */

  /* Impose boundary convective at some faces (face indicator icvfli) */
  int icvflb = 1;

  cs_field_bc_coeffs_t *bc_coeffs_sc = f_sc->bc_coeffs;

  cs_equation_param_t eqp_loc = *eqp_e;

  eqp_loc.istat  = -1;
  eqp_loc.icoupl = -1;
  eqp_loc.idifft = -1;
  eqp_loc.idften = CS_ISOTROPIC_DIFFUSION;
  eqp_loc.iswdyn = 0; /* No dynamic relaxation */
  eqp_loc.iwgrec = 0; /* Warning, may be overwritten if a field */
  eqp_loc.blend_st = 0; /* Warning, may be overwritten if a field */

  cs_equation_iterative_solve_scalar(idtvar,
                                     0, /* init */
                                     f_sc->id,
                                     nullptr,
                                     0,      /* iescap */
                                     0,      /* imucpp: not a thermal scalar */
                                     -1.0,   /* normp */
                                     &eqp_loc,
                                     energy_pre, energy_pre,
                                     bc_coeffs_sc,
                                     i_mass_flux, b_mass_flux,
                                     i_visc, b_visc,
                                     i_visc, b_visc,
                                     nullptr,   /* viscel */
                                     nullptr, nullptr, /* weighf, weighb */
                                     icvflb,
                                     icvfli,
                                     rovsdt,
                                     rhs,
                                     energy, dpvar,
                                     nullptr,   /* xcpp */
                                     nullptr);  /* eswork */

  CS_FREE(dpvar);
  CS_FREE(i_visc);
  CS_FREE(b_visc);

  /* Logging and clipping
     -------------------- */

  cs_scalar_clipping(f_sc);

  /* User processing for finer control of boundary
     and any corrective action required. */

  cs_cf_check_internal_energy(energy, n_cells, vel);

  /* Explicit balance (see cs_equation_iterative_solve_scalar:
     the increment is removed) */

  if (eqp_e->verbosity >= 2) {
#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++)
      rhs[c_id] =   rhs[c_id]
                    - eqp_e->istat*(crom[c_id]/dt[c_id])*cell_f_vol[c_id]
                    * (energy[c_id] - energy_pre[c_id])
                    * cs::max(0., cs::min(eqp_e->nswrsm - 2., 1.));

    const cs_real_t sclnor = sqrt(cs_gdot(n_cells, rhs, rhs));

    cs_log_printf(CS_LOG_DEFAULT,
                  _(" %s : EXPLICIT BALANCE = %14.5e"),
                  f_sc->name, sclnor);
  }

  /* Final updating of the pressure (and temperature)
     ------------------------------------------------ */
  /*
                                n+1      n+1  n+1
    The state equation is used P   =P(RHO   ,H   )
  */

  /* Computation of P and T at cell centers */

  cs_cf_thermo_pt_from_de(cpro_cp,
                          cpro_cv,
                          crom,
                          energy,
                          pr,
                          tempk,
                          vel,
                          fracv,
                          fracm,
                          frace,
                          n_cells);

  /*                             n+1      n+1  n+1
     The state equation is used P   =P(rho   ,e   )
  */

  /* Communication of pressure, energy and temperature
     ------------------------------------------------- */

  if (halo != nullptr) {
    cs_halo_sync_var(halo, CS_HALO_STANDARD, pr);
    cs_halo_sync_var(halo, CS_HALO_STANDARD, energy);
    cs_halo_sync_var(halo, CS_HALO_STANDARD, tempk);
  }

  /* Free memory */
  CS_FREE(wb);
  CS_FREE(rhs);
  CS_FREE(rovsdt);
  CS_FREE(grad);
  CS_FREE(c_viscs_t);
  CS_FREE(w7);
  CS_FREE(w9);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
