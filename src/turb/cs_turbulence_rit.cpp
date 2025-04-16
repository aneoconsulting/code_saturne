/*============================================================================
 * Turbulence transport equation.
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
#include "alge/cs_divergence.h"
#include "base/cs_equation_iterative_solve.h"
#include "cdo/cs_equation_param.h"
#include "alge/cs_face_viscosity.h"
#include "base/cs_field.h"
#include "base/cs_field_default.h"
#include "base/cs_field_operator.h"
#include "base/cs_field_pointer.h"
#include "base/cs_log_iteration.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_location.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_parall.h"
#include "base/cs_physical_constants.h"
#include "base/cs_prototypes.h"
#include "base/cs_thermal_model.h"
#include "base/cs_solid_zone.h"
#include "base/cs_time_step.h"
#include "turb/cs_turbulence_bc.h"
#include "turb/cs_turbulence_model.h"
#include "base/cs_velocity_pressure.h"

#include "turb/cs_turbulence_rij.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "turb/cs_turbulence_rit.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro Definitions
 *============================================================================*/

/*=============================================================================
 * Local Structure Definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the turbulent flux source terms
 *
 * \param[in]     name       name of current field
 * \param[in]     f_ut       scalar turbulent flux field
 * \param[in]     f_tv       variance of the thermal scalar field, or nullptr
 * \param[in]     n_cells    number of cells
 * \param[in]     l_viscls   0 uniform viscls values, 1 for local values
 * \param[in]     xcpp       \f$ C_p \f$
 * \param[in]     viscl      Molecular viscosity
 * \param[in]     viscls     variable diffusivity field
 * \param[in]     xuta       calculated variables at cell centers
 *                           (at current and previous time steps)
 * \param[in]     gradv      mean velocity gradient
 * \param[in]     gradt      mean scalar gradient
 * \param[in]     grad_al    alpha scalar gradient
 * \param[in]     grav       gravity
 * \param[out]    fimp       implicit part of source term
 * \param[out]    rhs_ut     right-hand side part source term
 */
/*----------------------------------------------------------------------------*/

static void
_turb_flux_st(const char          *name,
              const cs_field_t    *f_ut,
              const cs_field_t    *f_tv,
              const cs_lnum_t      n_cells,
              const cs_lnum_t      l_viscls,
              const cs_real_t      xcpp[],
              const cs_real_t      viscl[],
              const cs_real_t      viscls[],
              const cs_real_33_t   gradv[],
              const cs_real_3_t    gradt[],
              const cs_real_3_t    grad_al[],
              cs_real_33_t         fimp[],
              cs_real_3_t          rhs_ut[])
{
  const cs_real_t *cell_f_vol = cs_glob_mesh_quantities->cell_vol;

  cs_field_t *f = cs_field_by_name(name);

  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *cvar_ep = CS_F_(eps)->val;
  const cs_real_6_t *cvar_rij = (const cs_real_6_t *)CS_F_(rij)->val;

  const cs_real_3_t *xuta = (const cs_real_3_t *)f_ut->val_pre;

  cs_real_t *cpro_beta = nullptr;
  if (cs_field_by_name_try("thermal_expansion") != nullptr)
    cpro_beta = cs_field_by_name_try("thermal_expansion")->val;

  const cs_real_t *cvar_tt = nullptr, *cvara_tt = nullptr, *cvar_al = nullptr;

  const cs_turb_rans_model_t *rans_mdl = cs_glob_turb_rans_model;
  /* Get the turbulent flux model */
  const int kturt = cs_field_key_id("turbulent_flux_model");
  int turb_flux_model =  cs_field_get_key_int(f, kturt);

  if (f_tv != nullptr) {
    cvar_tt = f_tv->val;
    cvara_tt = f_tv->val_pre;
  }

  /* Save production terms if required */

  cs_real_3_t *prod_ut = nullptr;
  cs_field_t *f_ut_prod = cs_field_by_double_composite_name_try
                            ("algo:", f->name, "_turbulent_flux_production");

  if (f_ut_prod != nullptr)
    prod_ut = (cs_real_3_t *)f_ut_prod->val;

  cs_real_3_t *phi_ut = nullptr;
  cs_field_t *f_phi_ut = cs_field_by_double_composite_name_try
                           ("algo:", f->name, "_turbulent_flux_scrambling");
  if (f_phi_ut != nullptr)
    phi_ut = (cs_real_3_t *)f_phi_ut->val;

  cs_real_3_t *prod_by_vel_grad_ut = nullptr;
  cs_field_t *f_ut_prod_by_vel
    = cs_field_by_double_composite_name_try
        ("algo:", f->name, "_turbulent_flux_production_by_velocity_gradient");
  if (f_ut_prod_by_vel != nullptr)
    prod_by_vel_grad_ut = (cs_real_3_t *)f_ut_prod_by_vel->val;

  cs_real_3_t *prod_by_scal_grad_ut = nullptr;
  cs_field_t *f_ut_prod_by_scal
    = cs_field_by_double_composite_name_try
        ("algo:", f->name, "_turbulent_flux_production_by_scalar_gradient");
  if (f_ut_prod_by_scal != nullptr)
    prod_by_scal_grad_ut = (cs_real_3_t *)f_ut_prod_by_scal->val;

  cs_real_3_t *buo_ut = nullptr;
  cs_field_t *f_buo_ut = cs_field_by_double_composite_name_try
                           ("algo:", f->name, "_turbulent_flux_buoyancy");
  if (f_buo_ut != nullptr)
    buo_ut = (cs_real_3_t *)f_buo_ut->val;

  cs_real_3_t *dissip_ut = nullptr;
  cs_field_t *f_dissip_ut = cs_field_by_double_composite_name_try
                              ("algo:", f_ut->name, "_dissipation");
  if (f_dissip_ut != nullptr)
    dissip_ut = (cs_real_3_t *)f_dissip_ut->val;

  if (turb_flux_model == 31)
    cvar_al = cs_field_by_composite_name_try(f->name, "alpha")->val;

  const cs_real_t rhebdfm = 0.5;
  const cs_real_t *grav = cs_glob_physical_constants->gravity;

  const cs_real_t c1trit = cs_turb_c1trit;
  const cs_real_t c2trit = cs_turb_c2trit;
  const cs_real_t c3trit = cs_turb_c3trit;
  const cs_real_t c4trit = cs_turb_c4trit;

  constexpr cs_real_t c_1ov3 = 1./3.;

# pragma omp parallel if(n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {

    cs_real_t xrij[3][3];
    xrij[0][0] = cvar_rij[c_id][0];
    xrij[0][1] = cvar_rij[c_id][3];
    xrij[0][2] = cvar_rij[c_id][5];
    xrij[1][0] = cvar_rij[c_id][3];
    xrij[1][1] = cvar_rij[c_id][1];
    xrij[1][2] = cvar_rij[c_id][4];
    xrij[2][0] = cvar_rij[c_id][5];
    xrij[2][1] = cvar_rij[c_id][4];
    xrij[2][2] = cvar_rij[c_id][2];

    cs_real_t prdtl = viscl[c_id]*xcpp[c_id]/viscls[l_viscls*c_id];

    const cs_real_t tke = 0.5 * cs_math_6_trace(cvar_rij[c_id]);

    /* Compute Durbin time scheme */
    const cs_real_t xttke = tke / cvar_ep[c_id];

    cs_real_t alpha = 1., xttdrbt = xttke, xttdrbw = xttke;
    cs_real_t xxc1 = 0, xxc2 = 0, xxc3 = 0;
    cs_real_t xnal[3] = {0, 0, 0};

    if (turb_flux_model == 31) {
      alpha = cvar_al[c_id];
      /* FIXME Warning / rhebdfm**0.5 compared to F Dehoux
       * And so multiplied by (R/Prandt)^0.5 */
      xttdrbt = xttke * sqrt((1.0-alpha)*prdtl/rhebdfm + alpha);
      xttdrbw = xttdrbt * sqrt(rhebdfm/prdtl);

      /* Compute the unit normal vector */
      cs_real_t xnoral = cs_math_3_norm(grad_al[c_id]);
      const cs_real_t eps = cs_math_epzero/pow(cell_f_vol[c_id], c_1ov3);
      if (xnoral > eps) {
        for (cs_lnum_t i = 0; i < 3; i++)
          xnal[i] = grad_al[c_id][i] / xnoral;
      }

      /* Production and buoyancy for TKE */
      cs_real_t pk = 0;
      for (cs_lnum_t i = 0; i < 3; i++) {
        for (cs_lnum_t j = 0; j < 3; j++)
          pk -= xrij[i][j]*gradv[c_id][i][j];
      }

      cs_real_t gk = 0;
      /* FIXME make buoyant term coherent elsewhere */
      if (cpro_beta != nullptr && rans_mdl->has_buoyant_term == 1)
        gk = - cpro_beta[c_id] * cs_math_3_dot_product(xuta[c_id], grav);

      xxc1 = 1.+2.*(1.-cvar_al[c_id])*(pk+gk)/cvar_ep[c_id];
      xxc2 = 0.5*(1.+1./prdtl)*(1.-0.3*(1.-cvar_al[c_id])
                                      *(pk+gk)/cvar_ep[c_id]);
      xxc3 = xxc2;
    }

    cs_real_t phiith[3], phiitw[3];

    for (cs_lnum_t i = 0; i < 3; i++) {
       phiith[i] = - c1trit / xttdrbt * xuta[c_id][i]
                   + c2trit * cs_math_3_dot_product( gradv[c_id][i], xuta[c_id])
                   + c4trit * (-xrij[0][i] * gradt[c_id][0]
                               -xrij[1][i] * gradt[c_id][1]
                               -xrij[2][i] * gradt[c_id][2]);

       if ((cvar_tt != nullptr) && (cpro_beta != nullptr)
           && rans_mdl->has_buoyant_term == 1)
         phiith[i] += c3trit*(cpro_beta[c_id] * grav[i] * cvar_tt[c_id]);

       phiitw[i] =   -1. / xttdrbw *xxc1   /* FIXME full implicit */
                   * (  xuta[c_id][0]*xnal[0]*xnal[i]
                      + xuta[c_id][1]*xnal[1]*xnal[i]
                      + xuta[c_id][2]*xnal[2]*xnal[i]);

       /* Pressure/thermal fluctuation correlation term
        * --------------------------------------------- */
       const cs_real_t press_correl_i =       alpha  * phiith[i]
                                        + (1.-alpha) * phiitw[i];
       if (f_phi_ut != nullptr) /* Save it if needed */
         phi_ut[c_id][i] = press_correl_i;

       cs_real_t imp_term
         =   cell_f_vol[c_id] * crom[c_id]
           * (      alpha  * (c1trit/xttdrbt - c2trit*gradv[c_id][i][i])
              + (1.-alpha) * (xxc1*xnal[i]*xnal[i]/xttdrbw));

       fimp[c_id][i][i] += cs::max(imp_term, 0);

       /* Production terms
        *----------------- */

       /* Production term due to the mean velocity */
       const cs_real_t prod_by_vel_grad_i =
         - cs_math_3_dot_product(gradv[c_id][i], xuta[c_id]);
       if (prod_by_vel_grad_ut != nullptr) /* Save it if needed */
         prod_by_vel_grad_ut[c_id][i] = prod_by_vel_grad_i;

       /* Production term due to the mean temperature */
      const cs_real_t prod_by_scal_grad_i =  - (   xrij[i][0]*gradt[c_id][0]
                                                 + xrij[i][1]*gradt[c_id][1]
                                                 + xrij[i][2]*gradt[c_id][2]);
       if (prod_by_scal_grad_ut != nullptr) /* Save it if needed */
         prod_by_scal_grad_ut[c_id][i] = prod_by_scal_grad_i;

       /* Production term due to the gravity */
       cs_real_t buoyancy_i = 0.;
       if ((cvar_tt != nullptr) && (cpro_beta != nullptr)
           && rans_mdl->has_buoyant_term == 1)
         buoyancy_i = -grav[i] * cpro_beta[c_id] * cvara_tt[c_id];

       if (buo_ut != nullptr) /* Save it if needed */
         buo_ut[c_id][i] = buoyancy_i;

       /* Dissipation (Wall term only because "h" term is zero */
       const cs_real_t dissip_i =  (1.-alpha)/xttdrbw
                                 * (  xxc2 * xuta[c_id][i]
                                    + xxc3 * (  xuta[c_id][0]*xnal[0]*xnal[i]
                                              + xuta[c_id][1]*xnal[1]*xnal[i]
                                              + xuta[c_id][2]*xnal[2]*xnal[i]));
       if (dissip_ut != nullptr)/* Save it if needed */
         dissip_ut[c_id][i] = dissip_i;

       /* Save production terms for post-processing */
       if (prod_ut != nullptr)
         prod_ut[c_id][i] = prod_by_vel_grad_i + prod_by_scal_grad_i
                          + buoyancy_i - dissip_i;

       rhs_ut[c_id][i] += (  prod_by_vel_grad_i + prod_by_scal_grad_i
                           + buoyancy_i + press_correl_i - dissip_i)
                         * cell_f_vol[c_id]*crom[c_id];

       /* TODO we can implicit more terms */
       imp_term =   cell_f_vol[c_id] * crom[c_id]
                  * (1.-alpha)/xttdrbw * (xxc2+xxc3*xnal[i]*xnal[i]);

       fimp[c_id][i][i] += cs::max(imp_term, 0);

       if ((cvar_tt != nullptr) && (cpro_beta != nullptr)
           && rans_mdl->has_buoyant_term == 1) {

         /* Stable if negative w'T' */
         cs_real_t mez[3];
         cs_math_3_normalize(grav, mez);
         cs_real_t wptp = -cs_math_3_dot_product(mez, xuta[c_id]);
         cs_real_t w2 = cs_math_3_sym_33_3_dot_product(mez,
                                                       cvar_rij[c_id],
                                                       mez);

         if (wptp < - cs_math_epzero * sqrt(cvara_tt[c_id] * w2)) {

           /* Note Cauchy Schwarz implies that
            * T'2/|w'T'| > |w'T'| / w'2
            * */
           imp_term =   cell_f_vol[c_id] * crom[c_id]
             * grav[i] * cpro_beta[c_id] * cvara_tt[c_id] / wptp;

           fimp[c_id][i][i] += cs::max(imp_term, 0);
         }
       }

    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief compute the thermal fluxes and Diffusivity.
 *
 * \param[in]     f                  Current field
 * \param[in]     f_tv               variance of the thermal scalar field,
 *                                   or nullptr
 * \param[in]     n_cells            number of cells
 * \param[in]     n_b_faces          number od boundary faces
 * \param[in]     n_cells_ext        number of cells + gost
 * \param[in]     turb_flux_model    turb_flux_model
 * \param[in]     xcpp               \f$ C_p \f$
 * \param[in]     gradv              mean velocity gradient
 * \param[in]     gradt              mean scalar gradient
 * \param[in]     grad_al            alpha scalar gradient
 * \param[out]    xut                calculated variables at cell centers
 *                                    (at current and previous time steps)
 * \param[out]    thflxf             thermal flux on interior faces
 * \param[out]    thflxb             thermal flux on boundary faces
 * \param[out]    vistet             Diffusivity tensor
 */
/*----------------------------------------------------------------------------*/

static void
_thermal_flux_and_diff(cs_field_t         *f,
                       const cs_field_t   *f_tv,
                       cs_lnum_t           n_cells,
                       cs_lnum_t           n_cells_ext,
                       cs_lnum_t           n_b_faces,
                       int                 turb_flux_model,
                       const cs_real_t     xcpp[],
                       const cs_real_33_t  gradv[],
                       const cs_real_3_t   gradt[],
                       const cs_real_3_t   grad_al[],
                       cs_real_3_t         xut[],
                       cs_real_t           thflxf[],
                       cs_real_t           thflxb[],
                       cs_real_6_t         vistet[])
{
  const cs_real_t *cell_f_vol = cs_glob_mesh_quantities->cell_vol;

  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *viscl  = CS_F_(mu)->val;
  const cs_real_t *brom = CS_F_(rho_b)->val;

  const cs_real_t *cvara_ep = CS_F_(eps)->val_pre;
  const cs_real_6_t *cvara_rij = (const cs_real_6_t *)CS_F_(rij)->val_pre;

  const cs_field_t *f_beta = cs_field_by_name_try("thermal_expansion");
  const cs_turb_rans_model_t *rans_mdl = cs_glob_turb_rans_model;
  const cs_real_t *cpro_beta = nullptr, *cvara_tt = nullptr;
  if (f_beta != nullptr)
    cpro_beta = f_beta->val;

  if (f_tv != nullptr)
    cvara_tt = f_tv->val_pre;

  cs_lnum_t l_viscls = 0; /* stride for uniform/local viscosity access */
  cs_real_t _visls_0 = -1;
  const cs_real_t *viscls = nullptr;
  {
    const int kivisl = cs_field_key_id("diffusivity_id");
    int ifcvsl = cs_field_get_key_int(f, kivisl);
    if (ifcvsl > -1) {
      viscls = cs_field_by_id(ifcvsl)->val;
      l_viscls = 1;
    }
    else {
      const int kvisls0 = cs_field_key_id("diffusivity_ref");
      _visls_0 = cs_field_get_key_double(f, kvisls0);
      viscls = &_visls_0;
      l_viscls = 0;
    }
  }

  cs_real_t *cvar_al = nullptr;
  if (   (turb_flux_model == 11)
      || (turb_flux_model == 21)
      || (turb_flux_model == 31))
    cvar_al = cs_field_by_composite_name(f->name, "alpha")->val;

  const cs_real_t *grav = cs_glob_physical_constants->gravity;

  const int kctheta = cs_field_key_id("turbulent_flux_ctheta");
  cs_real_t ctheta = cs_field_get_key_double(f, kctheta);

  cs_real_3_t *w1;
  CS_MALLOC(w1, n_cells_ext, cs_real_3_t);

  /* loop on cells */

# pragma omp parallel if(n_cells > CS_THR_MIN)
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {

    cs_real_t xnal[3] = {0, 0, 0}, temp[3] = {0, 0, 0};

    /* Rij (copy to local 3x3 tensor to allow loops */

    cs_real_t xrij[3][3];
    xrij[0][0] = cvara_rij[c_id][0];
    xrij[0][1] = cvara_rij[c_id][3];
    xrij[0][2] = cvara_rij[c_id][5];
    xrij[1][0] = cvara_rij[c_id][3];
    xrij[1][1] = cvara_rij[c_id][1];
    xrij[1][2] = cvara_rij[c_id][4];
    xrij[2][0] = cvara_rij[c_id][5];
    xrij[2][1] = cvara_rij[c_id][4];
    xrij[2][2] = cvara_rij[c_id][2];

    /* Epsilon */
    const cs_real_t xe = cvara_ep[c_id];

    /* Kinetic turbulent energy */
    const cs_real_t xk = 0.5 * cs_math_6_trace(cvara_rij[c_id]);

    /* Turbulent time-scale (constant in AFM) */
    const cs_real_t xtt = xk/xe;
    cs_real_t alpha_theta = 0, xpk = 0., xgk = 0;
    cs_real_t eta_ebafm = 0, xi_ebafm = 0, gamma_eb = 0;

    if ((turb_flux_model == 11) || (turb_flux_model == 21)) {

      alpha_theta = cvar_al[c_id];

      /* Production and buoyancy */
      xpk = 0;
      for (cs_lnum_t ii = 0; ii < 3; ii++) {
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          xpk -= xrij[jj][ii] * gradv[c_id][jj][ii];
      }
      if (cpro_beta != nullptr)
        xgk = -cpro_beta[c_id] * cs_math_3_dot_product(xut[c_id], grav);

      /* Thermo-mecanical scales ratio R */
      cs_real_t prdtl = viscl[c_id] * xcpp[c_id] / viscls[l_viscls*c_id];
      cs_real_t xr_h = 0.5;
      cs_real_t xr = (1.0-alpha_theta)*prdtl + alpha_theta*xr_h;

      /* Unit normal vector */
      cs_real_t xnoral = cs_math_3_norm(grad_al[c_id]);

      const cs_real_t eps = cs_math_epzero/pow(cell_f_vol[c_id], 1./3);
      if (xnoral > eps) {
        for (cs_lnum_t i = 0; i < 3; i++)
          xnal[i] = grad_al[c_id][i] / xnoral;
      }

      /* Constants for EB-GGDH and EB-AFM */

      cs_real_t xxc1 = 1. + 2.*(1.-alpha_theta)*(xpk+xgk)/cvara_ep[c_id];
      cs_real_t xxc2 = 0.5*(1.+1./prdtl)*(1. -0.3*(1.-alpha_theta)
                                                 *(xpk+xgk)/cvara_ep[c_id]);

      ctheta =   (0.97*sqrt(xr))/(alpha_theta*(4.15*sqrt(0.5))
               + (1.-alpha_theta)*(sqrt(prdtl))*xxc2);
      gamma_eb = (1.-alpha_theta)*(xxc1 + xxc2);

      /* Constants for EB-AFM */
      if (turb_flux_model == 21) {
        eta_ebafm = 1.0 - alpha_theta*0.6;
        xi_ebafm  = 1.0 - alpha_theta*0.3;
      }

    }

    /* Compute thermal flux u'T' */

    for (cs_lnum_t ii = 0; ii < 3; ii++) {
      temp[ii] = 0;

      /* AFM model
         "-C_theta*k/eps*( xi* uT'.Grad u + eta*beta*g_i*T'^2)" */
      if (turb_flux_model == 20) {
        if ((cvara_tt != nullptr) && (cpro_beta != nullptr)
            && rans_mdl->has_buoyant_term == 1)
          temp[ii] -=   ctheta * xtt * cs_turb_etaafm
                      * cpro_beta[c_id] * grav[ii] * cvara_tt[c_id];

        for (cs_lnum_t jj = 0; jj < 3; jj++) {
          /*  Partial implicitation of "-C_theta*k/eps*(xi* uT'.Grad u)"
           *  Only the i != j  components are added. */
          if (ii != jj)
            temp[ii] -=  ctheta*xtt*cs_turb_xiafm
                        *xut[c_id][jj]*gradv[c_id][ii][jj];
          else
            temp[ii] -= cs::min(  ctheta*xtt*cs_turb_xiafm*xut[c_id][jj]
                                * gradv[c_id][ii][jj],
                                0.);
        }
      }

      /* EB-AFM model
       *  "-C_theta*k/eps*(  xi*uT'.Gradu+eta*beta*g_i*T'^2
       *                   + eps/k gamma uT' ni nj )"
       */
      if (turb_flux_model == 21) {
        if ((cvara_tt != nullptr) && (cpro_beta != nullptr)
            && rans_mdl->has_buoyant_term == 1)
          temp[ii] -=   ctheta * xtt * eta_ebafm
                      * cpro_beta[c_id] * grav[ii] * cvara_tt[c_id];
        for (cs_lnum_t jj = 0; jj < 3; jj++) {
          /* Partial implicitation of
           * "-C_theta*k/eps*( xi* uT'.Grad u + eps/k gamma uT' ni nj)"
           * Only the i.ne.j  components are added. */
          cs_real_t tmp1 = xtt * xi_ebafm * gradv[c_id][ii][jj] * xut[c_id][jj];
          if (ii != jj)
            temp[ii] -=   ctheta * tmp1
                        + ctheta*gamma_eb*xnal[ii]*xnal[jj]*xut[c_id][jj];
          else
            temp[ii] -=   ctheta
                        * cs::min(tmp1 +  gamma_eb*xnal[ii]
                                         *xnal[jj]*xut[c_id][jj],
                                  0);
        }
      }

      /* EB-GGDH model
       *  "-C_theta*k/eps*( eps/k gamma uT' ni nj)" */
      if (turb_flux_model == 11) {
        for (cs_lnum_t jj = 0; jj < 3; jj++) {
          /* Partial implicitation of "-C_theta*k/eps*( eps/k gamma uT' ni nj)"
           * Only the i.ne.j  components are added. */
          if (ii != jj)
            temp[ii] -= ctheta*gamma_eb*xnal[ii]*xnal[jj]*xut[c_id][jj];
        }
      }
    }

    cs_real_t coeff_imp;

    for (cs_lnum_t ii = 0; ii < 3; ii++) {
      /* Add the term in "grad T" which is implicited by the GGDH part in
         cs_solve_equation_scalar.
       *  "-C_theta*k/eps* R.grad T"
       * The resulting XUT array is only use for post processing purpose in
       * (EB)GGDH & (EB)AFM */
      xut[c_id][ii] =   temp[ii]
                      - ctheta*xtt*(  xrij[0][ii]*gradt[c_id][0]
                                    + xrij[1][ii]*gradt[c_id][1]
                                    + xrij[2][ii]*gradt[c_id][2]);
      /* Partial implicitation of "-C_theta*k/eps*( xi* uT'.Grad u )" for
       * EB-GGDH & (EB)-AFM
       * if positive
       * X_i = C*Y_ij*X_j -> X_i = Coeff_imp * Y_ij * X_j for i.ne.j
       * with Coeff_imp = C/(1+C*Y_ii) */

      /* AFM */
      if (turb_flux_model == 20) {
        coeff_imp
          = 1.+cs::max(ctheta*xtt*cs_turb_xiafm*gradv[c_id][ii][ii], 0);

        xut[c_id][ii] = xut[c_id][ii]/coeff_imp;
        temp[ii] = temp[ii]/coeff_imp;
        /* Calculation of the diffusion tensor for the implicited part
         * of the model computed in cs_convection_diffusion_solve.c */
        vistet[c_id][ii] = crom[c_id]*ctheta*xtt*xrij[ii][ii]/coeff_imp;

      }

      /* EB-AFM */
      else if (turb_flux_model == 21) {
        coeff_imp = 1. + cs::max(  ctheta*xtt*xi_ebafm*gradv[c_id][ii][ii]
                                 + ctheta*gamma_eb*xnal[ii]*xnal[ii],
                                 0.);

        xut[c_id][ii] = xut[c_id][ii]/coeff_imp;
        temp[ii] = temp[ii] / coeff_imp;
        /* Calculation of the diffusion tensor for the implicited part
         * of the model computed in cs_convection_diffusion_solve.c */
        vistet[c_id][ii] = crom[c_id]*ctheta*xtt*xrij[ii][ii]/coeff_imp;

      }

      /*! EB-GGDH */
      else if (turb_flux_model == 11) {
        coeff_imp = 1. + ctheta*gamma_eb*xnal[ii]*xnal[ii];

        xut[c_id][ii] = xut[c_id][ii]/coeff_imp;
        temp[ii] = temp[ii]/coeff_imp;
        /* Calculation of the diffusion tensor for the implicited part
         * of the model computed in cs_convection_diffusion_solve.c */
        vistet[c_id][ii] = crom[c_id]*ctheta*xtt*xrij[ii][ii]/coeff_imp;
      }

      /* In the next step, we compute the divergence of:
       *  "-Cp*C_theta*k/eps*( xi* uT'.Grad u + eta*beta*g_i*T'^2)"
       *  The part "-C_theta*k/eps* R.Grad T" is computed by the GGDH part */
      w1[c_id][ii] = xcpp[c_id]*temp[ii];
    }

    /*  Extra diag part of the diffusion tensor
        for cs_convection_diffusion_solve.c */
    if (   (turb_flux_model == 11)
        || (turb_flux_model == 20)
        || (turb_flux_model == 21)) {
      vistet[c_id][3] = crom[c_id]*ctheta*xtt*xrij[1][0];
      vistet[c_id][4] = crom[c_id]*ctheta*xtt*xrij[2][1];
      vistet[c_id][5] = crom[c_id]*ctheta*xtt*xrij[2][0];
    }

  } /* End loop on cells */

  cs_solid_zone_set_zero_on_cells(3, (cs_real_t *)xut);

  /* FIXME the line below would reproduce the previous behavior, which
     is incorrect (see issue #387). Either we should consider
     ctheta here purely local, or we must use an associated field to save it. */

  /* cs_field_set_key_double(f, kctheta, ctheta); */

  cs_field_bc_coeffs_t bc_coeffs_v_loc;
  cs_field_bc_coeffs_init(&bc_coeffs_v_loc);
  CS_MALLOC(bc_coeffs_v_loc.a, 3*n_b_faces, cs_real_t);
  CS_MALLOC(bc_coeffs_v_loc.b, 9*n_b_faces, cs_real_t);

  cs_real_3_t  *coefat = (cs_real_3_t  *)bc_coeffs_v_loc.a;
  cs_real_33_t *coefbt = (cs_real_33_t *)bc_coeffs_v_loc.b;

  cs_array_real_fill_zero(3*n_b_faces, (cs_real_t *)coefat);

  const cs_real_t kr_33[3][3] = {{1., 0., 0.},
                                 {0., 1., 0.},
                                 {0., 0., 1.}};

  for (cs_lnum_t f_id = 0; f_id < n_b_faces; f_id++) {
    for (cs_lnum_t ii = 0; ii < 3; ii++) {
      for (cs_lnum_t jj = 0; jj < 3; jj++)
        coefbt[f_id][jj][ii] = kr_33[ii][jj];
    }
  }

  const cs_equation_param_t *eqp = cs_field_get_equation_param_const(f);;

  cs_mass_flux(cs_glob_mesh,
               cs_glob_mesh_quantities,
               -1,
               1,
               1,
               1,
               1,
               eqp->imrgra,
               eqp->nswrgr,
               (cs_gradient_limit_t)(eqp->imligr),
               eqp->verbosity,
               eqp->epsrgr,
               eqp->climgr,
               crom,
               brom,
               w1,
               &bc_coeffs_v_loc,
               thflxf,
               thflxb);

  CS_FREE(w1);
  CS_FREE(coefat);
  CS_FREE(coefbt);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief This function perform the solving of the transport equation
 * of the turbulent heat fluxes.
 *
 * \param[in]     f             pointer to scalar field
 * \param[in,out] f_ut          pointer to turbulent flux field
 * \param[in]     xcpp          \f$ C_p \f$
 * \param[in]     gradv         mean velocity gradient
 * \param[in]     gradt         mean scalar gradient
 * \param[in]     grad_al       alpha scalar gradient
 */
/*----------------------------------------------------------------------------*/

static void
_solve_rit(const cs_field_t     *f,
           cs_field_t           *f_ut,
           const cs_real_t       xcpp[],
           const cs_real_33_t    gradv[],
           const cs_real_3_t     gradt[],
           const cs_real_3_t     grad_al[])
{
  if (cs_glob_turb_model->order == CS_TURB_FIRST_ORDER)
    bft_error(__FILE__, __LINE__, 0,
              _("%s: use an Rij model with thermal model."),
              __func__);

  const cs_mesh_t *m = cs_glob_mesh;
  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;

  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_real_t *cell_f_vol = mq->cell_vol;

  const cs_real_t *dt = CS_F_(dt)->val;
  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *viscl  = CS_F_(mu)->val;
  const cs_real_t *visct = CS_F_(mu_t)->val;
  const cs_real_6_t *visten
    = (const cs_real_6_t *)cs_field_by_name
                             ("anisotropic_turbulent_viscosity")->val;

  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
  const int iflmas = cs_field_get_key_int(CS_F_(vel), kimasf);
  const int iflmab = cs_field_get_key_int(CS_F_(vel), kbmasf);

  const cs_real_t *imasfl = cs_field_by_id(iflmas)->val;
  const cs_real_t *bmasfl = cs_field_by_id(iflmab)->val;

  const cs_real_3_t *xuta = (cs_real_3_t *)f_ut->val_pre;
  cs_real_3_t *xut = (cs_real_3_t *)f_ut->val;

  /* vcopt */
  const cs_equation_param_t *eqp
    = cs_field_get_equation_param_const(f);

  /* vcopt_ut */
  const cs_equation_param_t *eqp_ut
    = cs_field_get_equation_param_const(f_ut);

  if (eqp->verbosity >= 1)
    bft_printf(" Solving variable %s\n", f_ut->name);

  int kstprv = cs_field_key_id("source_term_prev_id");
  int st_prv_id = cs_field_get_key_int(f_ut, kstprv);
  cs_real_3_t *c_st_prv;
  if (st_prv_id > -1)
    c_st_prv = (cs_real_3_t *)cs_field_by_id(st_prv_id)->val;

  cs_lnum_t l_viscls = 0; /* stride for uniform/local viscosity access */
  cs_real_t _visls_0 = -1;
  const cs_real_t *viscls = nullptr;
  {
    const int kivisl = cs_field_key_id("diffusivity_id");
    int ifcvsl = cs_field_get_key_int(f, kivisl);
    if (ifcvsl > -1) {
      viscls = cs_field_by_id(ifcvsl)->val;
      l_viscls = 1;
    }
    else {
      const int kvisls0 = cs_field_key_id("diffusivity_ref");
      _visls_0 = cs_field_get_key_double(f, kvisls0);
      viscls = &_visls_0;
      l_viscls = 0;
    }
  }

  cs_real_33_t *fimp;
  cs_real_3_t *rhs_ut;

  CS_MALLOC(fimp, n_cells_ext, cs_real_33_t);
  CS_MALLOC(rhs_ut, n_cells_ext, cs_real_3_t);

  cs_array_real_fill_zero(3*n_cells_ext, (cs_real_t *)rhs_ut);
  cs_array_real_fill_zero(9*n_cells_ext, (cs_real_t *)fimp);

  /* Find the corresponding variance of the scalar */

  const cs_real_t *grav = cs_glob_physical_constants->gravity;

  const cs_field_t *f_tv = nullptr;

  if (cs_math_3_norm(grav) > cs_math_epzero)
    f_tv = cs_field_get_variance(f);
  else
    grav = nullptr;

  /* User source terms
     ----------------- */
  cs_user_source_terms(cs_glob_domain,
                       f_ut->id,
                       (cs_real_t *)rhs_ut,
                       (cs_real_t *)fimp);

  const cs_real_t thetv = eqp->theta;
  if (st_prv_id > -1) {
#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      for (cs_lnum_t i = 0; i < 3; i++) {
        for (cs_lnum_t j = 0; j < 3; j++) {
          rhs_ut[c_id][i] = fimp[c_id][i][j]*xuta[c_id][j];
          fimp[c_id][i][j] = -thetv*fimp[c_id][i][j];
        }
      }
    }
  }

  /* If we do not extrapolate the source terms */
  else {
    const cs_real_t zero_threshold = cs_math_zero_threshold;
#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      for (cs_lnum_t i = 0; i < 3; i++) {
        for (cs_lnum_t j = 0; j < 3; j++) {
          /* User source term */
          rhs_ut[c_id][i] += fimp[c_id][i][j]*xuta[c_id][j];
        }
        /* Diagonal */
        fimp[c_id][i][i] = cs::max(-fimp[c_id][i][i],
                                   zero_threshold);
      }
    }
  }

  /* Mass source terms FIXME
   * ----------------------- */


  /* Unsteady term
   * ------------- */

  if (eqp->istat == 1) {
#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      for (cs_lnum_t i = 0; i < 3; i++)
        fimp[c_id][i][i] += (crom[c_id] / dt[c_id]) * cell_f_vol[c_id];
    }
  }

  /* Right Hand Side of the turbulent fluxes:
   *     rho*(Pit + Git + Phi*_it - eps_it)
   * -------------------------------------- */

  _turb_flux_st(f->name, f_ut, f_tv, n_cells, l_viscls,
                xcpp, viscl, viscls, gradv,
                gradt, grad_al, fimp, rhs_ut);

  /* Tensor diffusion
   * ---------------- */

  cs_real_2_t *weighf;
  cs_real_6_t *viscce;
  cs_real_t *w1, *viscf, *viscb, *weighb;

  CS_MALLOC(w1, n_cells_ext, cs_real_t);
  CS_MALLOC(viscf, n_i_faces, cs_real_t);
  CS_MALLOC(viscb, n_b_faces, cs_real_t);
  CS_MALLOC(weighb, n_b_faces, cs_real_t);
  CS_MALLOC(weighf, n_i_faces, cs_real_2_t);
  CS_MALLOC(viscce, n_cells_ext, cs_real_6_t);

  cs_real_t mdifft = (cs_real_t)(eqp_ut->idifft);

  const int kctheta = cs_field_key_id("turbulent_flux_ctheta");
  const cs_real_t ctheta = cs_field_get_key_double(f, kctheta);

  /* Symmetric tensor diffusivity (GGDH) */
  if (eqp_ut->idiff > 0) {
    if (eqp_ut->idften & CS_ANISOTROPIC_RIGHT_DIFFUSION) {

#     pragma omp parallel if(n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        cs_real_t prdtl = viscl[c_id]*xcpp[c_id]/viscls[l_viscls*c_id];

        for (cs_lnum_t i = 0; i < 3; i++)
          viscce[c_id][i] =   0.5*(viscl[c_id]*(1.+1./prdtl))
                            + mdifft*ctheta*visten[c_id][i]/cs_turb_csrij;
        for (cs_lnum_t i = 3; i < 6; i++)
          viscce[c_id][i] = mdifft*ctheta*visten[c_id][i]/cs_turb_csrij;
      }

      cs_face_anisotropic_viscosity_scalar(m,
                                           mq,
                                           viscce,
                                           eqp->verbosity,
                                           weighf,
                                           weighb,
                                           viscf,
                                           viscb);
    }

    /* Scalar diffusivity */
    else {

#     pragma omp parallel if(n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        w1[c_id] = viscl[c_id] + mdifft*(ctheta*visct[c_id]/cs_turb_cmu);
      }

      cs_face_viscosity(m,
                        mq,
                        eqp->imvisf,
                        w1,
                        viscf,
                        viscb);
    }
  }
  /* No diffusion */
  else {
    cs_array_set_value_real(n_i_faces, 1, 0., viscf);
    cs_array_set_value_real(n_b_faces, 1, 0., viscb);
  }

  /* Add Rusanov fluxes */
  if (cs_glob_turb_rans_model->irijnu == 2) {
     cs_real_t *ipro_rusanov = cs_field_by_name("i_rusanov_diff")->val;
     for (cs_lnum_t face_id = 0; face_id < n_i_faces; face_id++) {
       viscf[face_id] = fmax(viscf[face_id], 0.5 * ipro_rusanov[face_id]);
     }

     const cs_real_3_t *restrict b_face_normal
       = (const cs_real_3_t *)mq->b_face_normal;
     cs_real_t *bpro_rusanov = cs_field_by_name("b_rusanov_diff")->val;

     //cs_real_3_t *coefap = (cs_real_3_t *)f_ut->bc_coeffs->a;
     cs_real_33_t *cofbfp = (cs_real_33_t *)f_ut->bc_coeffs->bf;
     for (cs_lnum_t face_id = 0; face_id < n_b_faces; face_id++) {
       cs_real_t n[3];
       cs_math_3_normalize(b_face_normal[face_id], n); /* Warning:
                                                          normalized here */

       for (cs_lnum_t i = 0; i < 3; i++) {
         for (cs_lnum_t j = 0; j < 3; j++) {
           cofbfp[face_id][i][j] +=  bpro_rusanov[face_id] * n[i]*n[j];
           //TODO ?cofafp[face_id][i] -= bf[i][j] * coefap[face_id][j];
         }
       }

     }
   }

  /* Vectorial solving of the turbulent thermal fluxes
   * ------------------------------------------------- */

  if (st_prv_id > -1) {
    const cs_time_scheme_t *time_scheme = cs_glob_time_scheme;
    const cs_real_t thets = time_scheme->thetst;
    const cs_real_t thetp1 = 1.0+thets;
#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      for (cs_lnum_t i = 0; i < 3; i++)
        rhs_ut[c_id][i] += thetp1*c_st_prv[c_id][i];
    }
  }

  cs_equation_param_t eqp_loc = *eqp;
  eqp_loc.iwgrec = 0;     /* Warning, may be overwritten if a field */
  eqp_loc.theta = thetv;
  eqp_loc.blend_st = 0;   /* Warning, may be overwritten if a field */

  cs_equation_iterative_solve_vector(cs_glob_time_step_options->idtvar,
                                     1, // init
                                     f_ut->id,
                                     nullptr,
                                     0,
                                     0,
                                     &eqp_loc,
                                     xuta,
                                     xuta,
                                     f_ut->bc_coeffs,
                                     imasfl,
                                     bmasfl,
                                     viscf,
                                     viscb,
                                     viscf,
                                     viscb,
                                     nullptr,
                                     nullptr,
                                     viscce,
                                     weighf,
                                     weighb,
                                     0,
                                     nullptr,
                                     fimp,
                                     rhs_ut,
                                     xut,
                                     nullptr);

  CS_FREE(w1);
  CS_FREE(viscf);
  CS_FREE(viscb);
  CS_FREE(weighb);
  CS_FREE(weighf);
  CS_FREE(viscce);
  CS_FREE(fimp);
  CS_FREE(rhs_ut);
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*=============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add the divergence of turbulent flux to a scalar transport equation.
 *
 * \param[in]   field_id  transported field id
 * \param[in]   xcpp      Cp
 * \param[out]  vistet    diffusivity tensor
 * \param[out]  smbrs     right hand side to update
 */
/*----------------------------------------------------------------------------*/

void
cs_turbulence_rij_transport_div_tf(const int        field_id,
                                   const cs_real_t  xcpp[],
                                   cs_real_t        vistet[][6],
                                   cs_real_t        smbrs[])
{
  const cs_mesh_t *m = cs_glob_mesh;
  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;

  /* TODO: declare field as const when ctheta issue (#387) is solved */
  cs_field_t *f = cs_field_by_id(field_id);

  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const int kturt = cs_field_key_id("turbulent_flux_model");
  const int turb_flux_model = cs_field_get_key_int(f, kturt);
  const int turb_flux_model_type = turb_flux_model / 10;

  /* Value of the corresponding turbulent flux */
  cs_field_t *f_ut = cs_field_by_composite_name(f->name, "turbulent_flux");
  cs_real_3_t *xut = (cs_real_3_t *)f_ut->val;

  cs_field_t *f_vel = CS_F_(vel);

  /* Compute velocity gradient */

  cs_real_33_t *gradv = nullptr, *_gradv = nullptr;
  {
    cs_field_t *f_vg = cs_field_by_name_try("algo:velocity_gradient");

    if (f_vel->grad != nullptr)
      gradv = (cs_real_33_t *)f_vel->grad;
    else if (f_vg != nullptr)
      gradv = (cs_real_33_t *)f_vg->val;
    else {
      CS_MALLOC(_gradv, n_cells_ext, cs_real_33_t);
      gradv = _gradv;
    }
  }

  cs_field_gradient_vector(f_vel, false, 1, gradv);

  /* Compute scalar gradient */

  cs_real_3_t *gradt = nullptr, *_gradt = nullptr;
  {
    cs_field_t *f_tg = cs_field_by_double_composite_name_try
                         ("algo:", f->name, "_gradient");

    if (f_tg != nullptr)
      gradt = (cs_real_3_t *)f_tg->val;
    else {
      CS_MALLOC(_gradt, n_cells_ext, cs_real_3_t);
      gradt = _gradt;
    }
  }

  cs_field_gradient_scalar(f,
                           true,     /* use previous t   */
                           1,        /* not on increment */
                           gradt);


  /* EB- AFM or EB-DFM: compute the gradient of alpha of the scalar */

  cs_real_3_t *grad_al = nullptr;

  if (   (turb_flux_model == 11)
      || (turb_flux_model == 21)
      || (turb_flux_model == 31)) {

    CS_MALLOC(grad_al, n_cells_ext, cs_real_3_t);

    cs_field_gradient_scalar(cs_field_by_composite_name(f->name, "alpha"),
                             false,       /* use previous t */
                             1,           /* not on increment */
                             grad_al);
  }

  /* Find the corresponding variance of the scalar */

  const cs_field_t *f_tv = nullptr;

  const int irovar = cs_glob_fluid_properties->irovar;
  const int idilat = cs_glob_velocity_pressure_model->idilat;
  const cs_real_t *grav = cs_glob_physical_constants->gravity;
  const cs_turb_rans_model_t *rans_mdl = cs_glob_turb_rans_model;

  const cs_real_t mod_grav = cs_math_3_norm(grav);
  if (   (mod_grav > cs_math_epzero)
      && ((irovar > 0) || (idilat == 0))
      && ((turb_flux_model_type == 2) || (turb_flux_model_type == 3))
      && rans_mdl->has_buoyant_term == 1) {

    f_tv = cs_field_get_variance(f);

    if (f_tv == nullptr)
      bft_error(__FILE__, __LINE__, 0,
                _("%s: the variance field required for\n"
                  "the turbulent transport of \"%s\" is not available."),
                __func__, f->name);

  }
  else
    grav = nullptr;

  /* Agebraic models AFM
   * ------------------- */

  cs_real_t *thflxf, *thflxb;
  CS_MALLOC(thflxf, n_i_faces, cs_real_t);
  CS_MALLOC(thflxb, n_b_faces, cs_real_t);

  if (turb_flux_model_type != 3) {

    cs_array_real_fill_zero(n_i_faces, thflxf);
    cs_array_real_fill_zero(n_b_faces, thflxb);

    _thermal_flux_and_diff(f,
                           f_tv,
                           n_cells,
                           n_cells_ext,
                           n_b_faces,
                           turb_flux_model,
                           xcpp,
                           gradv,
                           gradt,
                           grad_al,
                           xut,
                           thflxf,
                           thflxb,
                           vistet);

  }
  else {

    /* Transport equation on turbulent thermal fluxes (DFM)
     * ---------------------------------------------------- */

    _solve_rit(f, f_ut, xcpp, gradv, gradt, grad_al);

    /*  Clipping of the turbulence flux vector */
    if ((f_tv != nullptr) && (cs_glob_time_step->nt_cur > 1)) {
      const int kclipp = cs_field_key_id("is_clipped");
      const int clprit = cs_field_get_key_int(f_ut, kclipp);
      if (clprit > 0)
        cs_clip_turbulent_fluxes(f_ut->id,
                                 f_tv->id);
    }

    const cs_real_t *crom = CS_F_(rho)->val;
    const cs_real_t *brom = CS_F_(rho_b)->val;

    cs_real_3_t *w1;
    CS_MALLOC(w1, n_cells_ext, cs_real_3_t);

#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells_ext; c_id++)  {
      for (cs_lnum_t ii = 0; ii < 3; ii ++)
        w1[c_id][ii] = xcpp[c_id] * xut[c_id][ii];
    }

    /* Boundary Conditions on T'u' for the divergence term of
     * the thermal transport equation */

    cs_field_bc_coeffs_t bc_coeffs;
    cs_field_bc_coeffs_init(&bc_coeffs);

    bc_coeffs.a = f_ut->bc_coeffs->ad;
    bc_coeffs.b = f_ut->bc_coeffs->bd;

    const cs_equation_param_t *eqp = cs_field_get_equation_param_const(f);

    cs_mass_flux(m,
                 mq,
                 -1, /*f_id */
                 1,
                 1,
                 1,
                 1,
                 eqp->imrgra,
                 eqp->nswrgr,
                 static_cast<cs_gradient_limit_t>(eqp->imligr),
                 eqp->verbosity,
                 eqp->epsrgr,
                 eqp->climgr,
                 crom,
                 brom,
                 w1,
                 &bc_coeffs,
                 thflxf,
                 thflxb);

    CS_FREE(w1);
  }

  /* Add the divergence of the thermal flux to the thermal transport equation
     ------------------------------------------------------------------------ */

  if (   turb_flux_model == 11
      || turb_flux_model_type == 2
      || turb_flux_model_type == 3) {
    cs_real_t *divut = nullptr, *_divut = nullptr;
    cs_field_t *f_dut = cs_field_by_double_composite_name_try
                          ("algo:", f_ut->name, "_divergence");
    if (f_dut != nullptr)
      divut = f_dut->val;
    else {
      CS_MALLOC(_divut, n_cells_ext, cs_real_t);
      divut = _divut;
    }

    cs_divergence(m, 1, thflxf, thflxb, divut);

#   pragma omp parallel if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++)
      smbrs[c_id] -= divut[c_id];

    /* For post-processing intensive quantities */
    if (f_dut != nullptr) {
      int has_disable_flag = mq->has_disable_flag;
      int *c_disable_flag = mq->c_disable_flag;
      const cs_real_t *cell_f_vol = mq->cell_vol;

      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        cs_real_t dvol = 0;
        const int ind = has_disable_flag * c_id;
        const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
        if (c_act == 1)
          dvol = 1.0/cell_f_vol[c_id];
        divut[c_id] *= dvol;
      }
    }
    CS_FREE(_divut);
  }

  CS_FREE(grad_al);
  CS_FREE(_gradt);
  CS_FREE(_gradv);
  CS_FREE(thflxf);
  CS_FREE(thflxb);
}

/*----------------------------------------------------------------------------*/

 END_C_DECLS
