/*============================================================================
 * Compute the modified convective flux for scalars with a drift.
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
#include <assert.h>
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
#include "alge/cs_balance.h"
#include "base/cs_boundary_conditions.h"
#include "base/cs_boundary_conditions_set_coeffs.h"
#include "alge/cs_convection_diffusion.h"
#include "base/cs_dispatch.h"
#include "alge/cs_divergence.h"
#include "alge/cs_face_viscosity.h"
#include "base/cs_field_default.h"
#include "base/cs_field_pointer.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_physical_constants.h"
#include "turb/cs_turbulence_model.h"
#include "base/cs_volume_mass_injection.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "base/cs_drift_convective_flux.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  ! \file cs_drift_convective_flux.cpp
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

static int class_id_max = 0;

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update boundary flux mass of the mixture
 *
 * \param[in]      m       pointer to associated mesh structure
 * \param[in, out] bmasfl  boundary face mass flux
 */
/*----------------------------------------------------------------------------*/

void
cs_drift_boundary_mass_flux(const cs_mesh_t  *m,
                            cs_real_t         b_mass_flux[])
{
  const cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *)m->b_face_cells;

  const int *bc_type = cs_glob_bc_type;

  //TODO add a return in case no addition
  const int keydri = cs_field_key_id("drift_scalar_model");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");

 /* At walls, if particle classes have a outgoing flux,
  * mixture get the same quantity.
    (rho Vs)_f = sum_classes (rho x2 V2)_f
    Warning in case of ALE or turbomachinary...
   ---------------------------------------------------------------- */

  for (int jcla = 1; jcla < class_id_max; jcla++) {

    char var_name[15];
    snprintf(var_name, 15, "x_p_%02d", jcla);
    var_name[14] = '\0';

    cs_field_t *f_x_p_i = cs_field_by_name_try(var_name);
    cs_real_t *x2 = nullptr;

    if (f_x_p_i != nullptr) {
      x2 = f_x_p_i->val;
      const int iscdri = cs_field_get_key_int(f_x_p_i, keydri);

      /* We have a boundary flux on particle class */
      if (   !(iscdri & CS_DRIFT_SCALAR_IMPOSED_MASS_FLUX)
          && !(iscdri & CS_DRIFT_SCALAR_ZERO_BNDY_FLUX)
          && !(iscdri & CS_DRIFT_SCALAR_ZERO_BNDY_FLUX_AT_WALLS)) {

        int b_flmass_id = cs_field_get_key_int(f_x_p_i, kbmasf);

        assert(b_flmass_id > -1);
        /* Pointer to the Boundary mass flux */
        cs_real_t *b_mass_flux2 = cs_field_by_id(b_flmass_id)->val;

        cs_dispatch_context ctx;

        ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
          /* Only for walls and outgoing values */
          if (   (   bc_type[face_id] != CS_SMOOTHWALL
                  && bc_type[face_id] != CS_ROUGHWALL)
              || b_mass_flux2[face_id] < 0.)
            return; // contine in loop

          cs_lnum_t c_id = b_face_cells[face_id];
          b_mass_flux[face_id] += x2[c_id] * b_mass_flux2[face_id];
        });
      }
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the modified convective flux for scalars with a drift.
 *
 * \param[in]     f_sc          drift scalar field
 * \param[in,out] i_mass_flux   scalar mass flux at interior face centers
 * \param[in,out] b_mass_flux   scalar mass flux at boundary face centers
 * \param[in,out] fimp          implicit term
 * \param[in,out] rhs           right hand side term
 */
/*----------------------------------------------------------------------------*/

void
cs_drift_convective_flux(cs_field_t  *f_sc,
                         cs_real_t    i_mass_flux[],
                         cs_real_t    b_mass_flux[],
                         cs_real_t    fimp[],
                         cs_real_t    rhs[])
{
  const cs_mesh_t  *mesh = cs_glob_mesh;
  cs_mesh_quantities_t *fvq = cs_glob_mesh_quantities;
  const cs_lnum_t n_cells     = mesh->n_cells;
  const cs_lnum_t n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t n_i_faces   = mesh->n_i_faces;
  const cs_lnum_t n_b_faces   = mesh->n_b_faces;

  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *)mesh->b_face_cells;
  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *)mesh->i_face_cells;
  const cs_real_t *cell_vol = fvq->cell_vol;

  const int kivisl = cs_field_key_id("diffusivity_id");
  const int keyccl = cs_field_key_id("scalar_class");
  const int keydri = cs_field_key_id("drift_scalar_model");
  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");

  const int iscdri = cs_field_get_key_int(f_sc, keydri);
  const int icla = cs_field_get_key_int(f_sc, keyccl);

  class_id_max = cs::max(icla, class_id_max);

  const cs_real_t *dt = CS_F_(dt)->val;
  const int model  = cs_glob_turb_model->model;
  const int itytur = cs_glob_turb_model->itytur;
  const cs_real_t *gravity = cs_glob_physical_constants->gravity;
  const int idtvar = cs_glob_time_step_options->idtvar;
  const int *bc_type = cs_glob_bc_type;

  cs_field_t *f_vel = CS_F_(vel);
  cs_equation_param_t *eqp_sc = cs_field_get_equation_param(f_sc);
  cs_equation_param_t *eqp_vel = cs_field_get_equation_param(f_vel);

  /* Pointers to the mass fluxes of the mix (based on mix velocity) */

  const int iflmas_v = cs_field_get_key_int(f_vel, kimasf);
  const int iflmab_v = cs_field_get_key_int(f_vel, kbmasf);
  cs_real_t *i_mass_flux_mix = cs_field_by_id(iflmas_v)->val;
  cs_real_t *b_mass_flux_mix = cs_field_by_id(iflmab_v)->val;

  /* Mass fraction of gas */
  cs_field_t *f_xc = cs_field_by_name_try("x_c");
  cs_real_t *x1 = nullptr, *b_x1 = nullptr;
  cs_real_t *i_mass_flux_gas = nullptr;
  cs_real_t *b_mass_flux_gas = nullptr;

  if (f_xc != nullptr) {
    x1 = f_xc->val;

    /* Mass fraction of the gas at the boundary */
    cs_field_t *f_b_xc = cs_field_by_name("b_x_c");
    b_x1 = f_b_xc->val;
  }
  /* Map field arrays */
  cs_real_3_t *vel = (cs_real_3_t *)f_vel->val;
  cs_real_3_t *vel_pre = (cs_real_3_t *)f_vel->val_pre;

  /* Initialization
     -------------- */

  /* Physical properties */

  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *brom = CS_F_(rho_b)->val;

  cs_field_t *f_rij = CS_F_(rij);
  cs_field_t *f_k = CS_F_(k);

  cs_real_6_t *rij = nullptr;
  cs_real_t *k = nullptr;

  if (f_rij != nullptr)
    rij = (cs_real_6_t *)f_rij->val;
  if (f_k != nullptr)
    k = f_k->val;

  /* Brownian diffusivity */
  cs_real_t *cpro_viscls = nullptr;
  int ifcvsl = cs_field_get_key_int(f_sc, kivisl);
  if (ifcvsl >= 0)
    cpro_viscls = cs_field_by_id(ifcvsl)->val;

  /* Vector containing all the additional convective terms */

  cs_real_t *w1, *viscce;
  cs_real_3_t *dudt;
  CS_MALLOC_HD(w1, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(viscce, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(dudt, n_cells_ext, cs_real_3_t, cs_alloc_mode);

  cs_field_bc_coeffs_t bc_coeffs_loc;
  cs_field_bc_coeffs_init(&bc_coeffs_loc);
  CS_MALLOC_HD(bc_coeffs_loc.a,  n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_loc.b,  n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_loc.af, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_loc.bf, n_b_faces, cs_real_t, cs_alloc_mode);

  cs_real_t *coefap = bc_coeffs_loc.a;
  cs_real_t *coefbp = bc_coeffs_loc.b;
  cs_real_t *cofafp = bc_coeffs_loc.af;
  cs_real_t *cofbfp = bc_coeffs_loc.bf;

  cs_field_bc_coeffs_t bc_coeffs1_loc;
  cs_field_bc_coeffs_init(&bc_coeffs1_loc);
  CS_MALLOC_HD(bc_coeffs1_loc.a, 3*n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs1_loc.b, 9*n_b_faces, cs_real_t, cs_alloc_mode);

  cs_real_3_t  *coefa1 = (cs_real_3_t  *)bc_coeffs1_loc.a;
  cs_real_33_t *coefb1 = (cs_real_33_t *)bc_coeffs1_loc.b;

  cs_real_t *i_visc, *flumas;
  CS_MALLOC_HD(i_visc, n_i_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(flumas, n_i_faces, cs_real_t, cs_alloc_mode);

  cs_real_t *b_visc, *flumab;
  CS_MALLOC_HD(flumab, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(b_visc, n_b_faces, cs_real_t, cs_alloc_mode);

  CS_MALLOC_HD(i_mass_flux_gas, n_i_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(b_mass_flux_gas, n_b_faces, cs_real_t, cs_alloc_mode);

  cs_dispatch_context ctx;

  const cs_real_t gxyz[3] = {gravity[0], gravity[1], gravity[2]};

  if (iscdri & CS_DRIFT_SCALAR_ADD_DRIFT_FLUX) {

    /* Index of the corresponding relaxation time */
    cs_real_t *cpro_taup = nullptr;
    {
      cs_field_t *f_tau
        = cs_field_by_composite_name_try(f_sc->name, "drift_tau");

      if (f_tau != nullptr)
        cpro_taup = f_tau->val;
    }

    /* Index of the corresponding relaxation time (cpro_taup) */
    cs_real_3_t *cpro_drift = nullptr;
    {
      cs_field_t *f_drift_vel
        = cs_field_by_composite_name_try(f_sc->name, "drift_vel");

      if (f_drift_vel != nullptr)
        cpro_drift = (cs_real_3_t *)f_drift_vel->val;
    }

    /* Index of the corresponding interaction time
       particle--eddies (drift_turb_tau) */

    cs_real_t *cpro_taufpt = nullptr;
    if (iscdri & CS_DRIFT_SCALAR_TURBOPHORESIS) {
      cs_field_t *f_drift_turb_tau
        = cs_field_by_composite_name(f_sc->name, "drift_turb_tau");

      cpro_taufpt = f_drift_turb_tau->val;
    }

    /* Initialization of the convection flux for the current particle class */

    cs_array_real_fill_zero(n_i_faces, i_visc);
    cs_array_real_fill_zero(n_i_faces, flumas);

    cs_array_real_fill_zero(n_b_faces, b_visc);
    cs_array_real_fill_zero(n_b_faces, flumab);

    /* Initialization of the gas "class" convective flux by the
       first particle "class":
       it is initialized by the mass flux of the bulk */

    if (icla == 1 && f_xc != nullptr) {
      cs_array_real_copy(n_i_faces, i_mass_flux_mix, i_mass_flux_gas);
      cs_array_real_copy(n_b_faces, b_mass_flux_mix, b_mass_flux_gas);
    }
    /* Initialize the additional convective flux with the gravity term
       --------------------------------------------------------------- */

    /* Test if a deviation velocity of particles class exists */

    if (icla >= 1) {

      char var_name[64];
      snprintf(var_name, 64, "vd_p_%02d", icla);
      var_name[63] = '\0';

      cs_field_t *f_vdp_i = cs_field_by_name_try(var_name);
      cs_real_3_t *vdp_i = nullptr;

      if (f_vdp_i != nullptr) {
        vdp_i = (cs_real_3_t *)f_vdp_i->val;

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          const cs_real_t rho = crom[c_id];
          // FIXME should by multiplied by (1-x2) or x1
          cpro_drift[c_id][0] = rho * vdp_i[c_id][0];
          cpro_drift[c_id][1] = rho * vdp_i[c_id][1];
          cpro_drift[c_id][2] = rho * vdp_i[c_id][2];
        });
      }
    }

    else if (icla >= 0 && cpro_taup != nullptr && cpro_drift != nullptr) {

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        const cs_real_t rho = crom[c_id];
        cpro_drift[c_id][0] = rho * cpro_taup[c_id] * gxyz[0];
        cpro_drift[c_id][1] = rho * cpro_taup[c_id] * gxyz[1];
        cpro_drift[c_id][2] = rho * cpro_taup[c_id] * gxyz[2];
      });

    }

    /* Computation of the turbophoresis and the thermophoresis terms
       ------------------------------------------------------------- */

    /* Initialized to 0 */
    cs_array_real_fill_zero(n_cells, viscce);

    if ((iscdri & CS_DRIFT_SCALAR_TURBOPHORESIS) && model != CS_TURB_NONE) {

      /* The diagonal part is easy to implicit (Grad (K) . n = (K_j - K_i)/IJ)
         Compute the K=1/3*trace(R) coefficient (diffusion of Zaichik) */

      if (itytur == 3) {

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          cs_real_t rtrace = cs_math_6_trace(rij[c_id]);

          /* Correction by Omega */
          const cs_real_t omega = cpro_taup[c_id] / cpro_taufpt[c_id];
          /* FIXME: use idifft or not? */
          viscce[c_id] = 1.0/3.0 * cpro_taup[c_id] / (1.0 + omega) * rtrace;
        });

      }
      else if (itytur == 2 || itytur == 5 || model == CS_TURB_K_OMEGA) {

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          /* Correction by Omega */
          const cs_real_t omega = cpro_taup[c_id] / cpro_taufpt[c_id];
          viscce[c_id] = 2.0/3.0 * cpro_taup[c_id] / (1.0 + omega) * k[c_id];
        });

      }

    } /* End turbophoresis */

    if (iscdri & CS_DRIFT_SCALAR_THERMOPHORESIS) {

      /* cpro_viscls[c_id]: contains the Brownian motion
         ------------------------------------------------ */

      if (ifcvsl >= 0) {

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          viscce[c_id] += cpro_viscls[c_id] / crom[c_id];
        });

      }
      else {

        const int kvisl0 = cs_field_key_id("diffusivity_ref");
        const cs_real_t visls_0 = cs_field_get_key_double(f_sc, kvisl0);

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          viscce[c_id] += visls_0 / crom[c_id];
        });
      }

    } /* End thermophoresis */

    ctx.wait();

    if (   (iscdri & CS_DRIFT_SCALAR_TURBOPHORESIS)
        || (iscdri & CS_DRIFT_SCALAR_THERMOPHORESIS)) {

      /* Face diffusivity of rho to compute rho*(Grad K . n)_face */
      cs_array_real_copy(n_cells, crom, w1);

      if (mesh->halo != nullptr)
        cs_halo_sync_var(mesh->halo, CS_HALO_STANDARD, w1);

      cs_face_viscosity(mesh,
                        fvq,
                        eqp_sc->imvisf,
                        w1,
                        i_visc,
                        b_visc);

      /* Homogeneous Neumann BC */
      {
        /* Code from cs_boundary_conditions_set_neumann_scalar_hmg
           expanded here to allow generation on GPU */
        cs_real_t *a = bc_coeffs_loc.a;
        cs_real_t *b = bc_coeffs_loc.b;
        cs_real_t *af = bc_coeffs_loc.af;
        cs_real_t *bf = bc_coeffs_loc.bf;

        ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
          a[face_id] = 0; b[face_id] = 1;
          af[face_id] = 0; bf[face_id] = 0;
        });
      }

      ctx.wait();

      /* The computed convective flux has the dimension of rho*velocity */

      cs_face_diffusion_potential(-1,
                                  mesh,
                                  fvq,
                                  0, /* init */
                                  1, /* inc */
                                  eqp_sc->imrgra,
                                  eqp_sc->nswrgr,
                                  eqp_sc->imligr,
                                  0, /* iphydr */
                                  0, /* iwgrp */
                                  eqp_sc->verbosity,
                                  eqp_sc->epsrgr,
                                  eqp_sc->climgr,
                                  nullptr, /* frcxt */
                                  viscce,
                                  &bc_coeffs_loc,
                                  i_visc, b_visc,
                                  w1,
                                  flumas, flumab);

      /* TODO add extradiagonal part */

    } /* End turbophoresis or thermophoresis */

    /* Centrifugal force (particular derivative Du/Dt)
       ----------------------------------------------- */

    if (iscdri & CS_DRIFT_SCALAR_CENTRIFUGALFORCE) {

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        const cs_real_t rhovdt = crom[c_id] * cell_vol[c_id] / dt[c_id];

        dudt[c_id][0] = - rhovdt * (vel[c_id][0]-vel_pre[c_id][0]);
        dudt[c_id][1] = - rhovdt * (vel[c_id][1]-vel_pre[c_id][1]);
        dudt[c_id][2] = - rhovdt * (vel[c_id][2]-vel_pre[c_id][2]);
      });

      /* Reset i_visc and b_visc */
      cs_array_real_fill_zero(n_i_faces, i_visc);
      cs_array_real_fill_zero(n_b_faces, b_visc);

      /* Get Boundary conditions of the velocity */
      cs_field_bc_coeffs_t *bc_coeffs_vel = f_vel->bc_coeffs;

      /* The added convective scalar mass flux is:
         (thetap*Y_\face-imasac*Y_\celli)*mf.
         When building the implicit part of the rhs, one
         has to impose 1 on mass accumulation. */

      cs_equation_param_t eqp_loc = *eqp_vel;

      eqp_loc.iconv  = 1;
      eqp_loc.istat  = -1;
      eqp_loc.idiff  = 0;
      eqp_loc.idifft = -1;
      eqp_loc.iswdyn = -1;
      eqp_loc.nswrsm = -1;
      eqp_loc.iwgrec = 0;
      eqp_loc.blend_st = 0; /* Warning, may be overwritten if a field */
      eqp_loc.epsilo = -1;
      eqp_loc.epsrsm = -1;

      cs_balance_vector(idtvar,
                        CS_F_(vel)->id,
                        1, /* imasac */
                        1, /* inc */
                        0, /* ivisep */
                        &eqp_loc,
                        vel, vel,
                        bc_coeffs_vel,
                        nullptr, // bc_coeffs_solve
                        i_mass_flux_mix, b_mass_flux_mix,
                        i_visc, b_visc,
                        nullptr, nullptr, /* secvif, secvib */
                        nullptr, nullptr, nullptr,
                        0, nullptr, /* icvflb, icvfli */
                        nullptr, nullptr,
                        dudt);

      /* Warning: cs_balance_vector adds "-( grad(u) . rho u)" */

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {

        cpro_drift[c_id][0] =   cpro_drift[c_id][0]
                              + cpro_taup[c_id]*dudt[c_id][0]/cell_vol[c_id];

        cpro_drift[c_id][1] =   cpro_drift[c_id][1]
                              + cpro_taup[c_id]*dudt[c_id][1]/cell_vol[c_id];

        cpro_drift[c_id][2] =   cpro_drift[c_id][2]
                              + cpro_taup[c_id]*dudt[c_id][2]/cell_vol[c_id];

      });

    } /* End centrifugalforce */

    /* Electrophoresis term
       -------------------- */

    if (iscdri & CS_DRIFT_SCALAR_ELECTROPHORESIS) {

      /* TODO */
      bft_error(__FILE__, __LINE__, 0,
                _("The drift scalar electrophoresis "
                  "functionality is not yet available"));
    }

    /* Finalization of the mass flux of the current class
       -------------------------------------------------- */

    /* For all scalar with a drift excepted the gas phase which is deduced
       And for those whom mass flux is imposed elsewhere */

    if (icla >= 0 && !(iscdri & CS_DRIFT_SCALAR_IMPOSED_MASS_FLUX)) {

      /* Homogeneous Neumann at the boundary */
      if (iscdri & CS_DRIFT_SCALAR_ZERO_BNDY_FLUX) {

        ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
          for (cs_lnum_t i = 0; i < 3; i++) {
            coefa1[face_id][i] = 0.0;

            for (cs_lnum_t j = 0; j < 3; j++)
              coefb1[face_id][i][j] = 0.0;
          }
        });

      }
      else if (iscdri & CS_DRIFT_SCALAR_ZERO_BNDY_FLUX_AT_WALLS) {

        ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
          for (cs_lnum_t i = 0; i < 3; i++) {
            coefa1[face_id][i] = 0.0;

            for (cs_lnum_t j = 0; j < 3; j++)
              coefb1[face_id][i][j] = 0.0;

            if (   bc_type[face_id] != CS_SMOOTHWALL
                && bc_type[face_id] != CS_ROUGHWALL)
              coefb1[face_id][i][i] = 1.0;

          }
        });
      }
      else {

        ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
          for (cs_lnum_t i = 0; i < 3; i++) {
            coefa1[face_id][i] = 0.0;

            for (cs_lnum_t j = 0; j < 3; j++)
              coefb1[face_id][i][j] = 0.0;

            coefb1[face_id][i][i] = 1.0;
          }
        });

      }
      ctx.wait();

      cs_mass_flux(mesh,
                   fvq,
                   -1,
                   0, /* itypfl: drift has already been multiplied by rho */
                   0, /* iflmb0 */
                   0, /* init */
                   1, /* inc */
                   eqp_sc->imrgra,
                   eqp_sc->nswrgr,
                   static_cast<cs_gradient_limit_t>(eqp_sc->imligr),
                   eqp_sc->verbosity,
                   eqp_sc->epsrgr,
                   eqp_sc->climgr,
                   crom, brom,
                   cpro_drift,
                   &bc_coeffs1_loc,
                   flumas, flumab);

      /* Update the convective flux, exception for the Gas "class" */
      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        i_mass_flux[face_id] = i_mass_flux_mix[face_id] + flumas[face_id];
      });

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        b_mass_flux[face_id] = b_mass_flux_mix[face_id] + flumab[face_id];
      });

    } /* End: not drift scalar imposed mass flux */

    else if (icla == -1 && f_xc != nullptr) {

      /* Deduce the convective flux of the continuous "class" by removing
         the flux of the current particle "class":
         (rho x1 V1)_f = (rho Vs)_f - sum_classes (rho x2 V2)_f
         ---------------------------------------------------------------- */

      /* Initialize continuous phase mass flux as mixture mass flux */
      cs_array_real_copy(n_i_faces, i_mass_flux_mix, i_mass_flux);
      cs_array_real_copy(n_b_faces, b_mass_flux_mix, b_mass_flux);

      for (int jcla = 1; jcla < class_id_max; jcla++) {

        char var_name[64];
        snprintf(var_name, 64, "x_p_%02d", jcla);
        var_name[63] = '\0';

        cs_field_t *f_x_p_i = cs_field_by_name_try(var_name);
        cs_real_t *x2 = nullptr;

        if (f_x_p_i != nullptr) {
          x2 = f_x_p_i->val;

          int i_flmass_id = cs_field_get_key_int(f_x_p_i, kimasf);
          int b_flmass_id = cs_field_get_key_int(f_x_p_i, kbmasf);

          assert(b_flmass_id > -1);
          cs_real_t *i_mass_flux2 = cs_field_by_id(i_flmass_id)->val;
          /* Pointer to the Boundary mass flux */
          cs_real_t *b_mass_flux2 = cs_field_by_id(b_flmass_id)->val;

          ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {

            /* Upwind value of x2 at the face, consistent with the
               other transport equations */

            cs_lnum_t c_id_up = i_face_cells[face_id][1];

            if (i_mass_flux2[face_id] >= 0.0)
              c_id_up = i_face_cells[face_id][0];

            i_mass_flux[face_id] -= x2[c_id_up] * i_mass_flux2[face_id];

          });

          ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
            /* TODO Upwind value of x2 at the face, consistent with the
               other transport equations
               !if (bmasfl[face_id]>=0.d0) */
            cs_lnum_t c_id_up = b_face_cells[face_id];
            b_mass_flux[face_id] -= x2[c_id_up] * b_mass_flux2[face_id];
          });
        }
      }

      /* Finalize the convective flux of the gas "class" by scaling by x1
         (rho x1 V1)_ij = (rho Vs)_ij - sum_classes (rho x2 V2)_ij
         Warning, x1 at the face must be computed so that it is consistent
         with an upwind scheme on (rho V1) */

      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {

        /* Upwind value of x2 at the face, consistent with the
           other transport equations */
        cs_lnum_t c_id_up = i_face_cells[face_id][1];

        if (i_mass_flux[face_id] >= 0.0)
          c_id_up = i_face_cells[face_id][0];

        i_mass_flux[face_id] /= x1[c_id_up];

      });

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {

        /* Upwind value of x1 at the face, consistent with the
           other transport equations */
        const cs_lnum_t c_id_up = b_face_cells[face_id];

        if (b_mass_flux[face_id] < 0.0)
          b_mass_flux[face_id] /= b_x1[face_id];
        else
          b_mass_flux[face_id] /= x1[c_id_up];

      });
    } /* End continuous phase */

  } /* End drift scalar add drift flux */

  /* Mass aggregation term of the additional part "div(rho(u_p-u_f))"
     ---------------------------------------------------------------- */

  if (!(iscdri & CS_DRIFT_SCALAR_NO_MASS_AGGREGATION)) {
    /* Recompute the difference between mixture and the class */
    if (iscdri & CS_DRIFT_SCALAR_IMPOSED_MASS_FLUX) {
      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        flumas[face_id] = - i_mass_flux_mix[face_id];
      });

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        flumab[face_id] = - b_mass_flux_mix[face_id];
      });
    }
    else {
      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        flumas[face_id] = i_mass_flux[face_id] - i_mass_flux_mix[face_id];
      });

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        flumab[face_id] = b_mass_flux[face_id] - b_mass_flux_mix[face_id];
      });
    }
    ctx.wait();

    cs_real_t *divflu;
    CS_MALLOC_HD(divflu, n_cells_ext, cs_real_t, cs_alloc_mode);

    cs_divergence(mesh,
                  1, /* init */
                  flumas,
                  flumab,
                  divflu);

    const int iconvp = eqp_sc->iconv;

    /*  NB: if the porosity module is switched on, the porosity is already
     * taken into account in divflu */

    /* mass aggregation term */
    if (f_sc->dim == 1) {
      cs_real_t *cvara_var = f_sc->val_pre;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        fimp[c_id] += iconvp*divflu[c_id];
        rhs[c_id] -= iconvp*divflu[c_id]*cvara_var[c_id];
      });
    }
    else {
      assert(f_sc->dim == 3);
      cs_real_3_t *cvara_var = (cs_real_3_t *)f_sc->val_pre;
      cs_real_3_t *_rhs= (cs_real_3_t *)rhs;
      cs_real_33_t *_fimp= (cs_real_33_t *)fimp;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t i = 0; i < f_sc->dim; i++) {
          _fimp[c_id][i][i] += iconvp*divflu[c_id];
          _rhs[c_id][i] -= iconvp*divflu[c_id]*cvara_var[c_id][i];
        }
      });
    }
    ctx.wait();

    /* Free memory */
    CS_FREE(divflu);
  }

  CS_FREE(viscce);
  CS_FREE(dudt);
  CS_FREE(w1);
  CS_FREE(i_visc);
  CS_FREE(b_visc);
  CS_FREE(flumas);
  CS_FREE(flumab);

  CS_FREE(i_mass_flux_gas);
  CS_FREE(b_mass_flux_gas);

  CS_FREE(coefap);
  CS_FREE(coefbp);
  CS_FREE(cofafp);
  CS_FREE(cofbfp);

  CS_FREE(coefa1);
  CS_FREE(coefb1);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
