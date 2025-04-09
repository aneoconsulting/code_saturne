/*============================================================================
 * Cooling towers related functions
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

#include "bft/bft_mem.h"
#include "bft/bft_error.h"
#include "bft/bft_printf.h"

#include "atmo/cs_air_props.h"
#include "base/cs_array.h"
#include "atmo/cs_atmo.h"
#include "ctwr/cs_ctwr_physical_properties.h"
#include "base/cs_field.h"
#include "base/cs_field_pointer.h"
#include "lagr/cs_lagr.h"
#include "base/cs_math.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_physical_constants.h"
#include "pprt/cs_physical_model.h"
#include "base/cs_volume_mass_injection.h"
#include "base/cs_volume_zone.h"

#include "ctwr/cs_ctwr.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "ctwr/cs_ctwr_source_terms.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*=============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Compute the Lewis factor used for the evaluation of the heat transfer
 * phase change source terms
 *
 * parameters:
 *   evap_model  <-- Evaporation model: CS_CTWR_POPPE or CS_CTWR_MERKEL
 *   molmassrat  <-- Dry air to water vapor molecular mass ratio
 *   x           <-- Humidity
 *   x_s_tl      <-- Saturation humidity at the temperature of the liquid
 *
 * returns:
 *   le_f        --> Lewis factor
 *----------------------------------------------------------------------------*/

static cs_real_t
_lewis_factor(const int        evap_model,
              const cs_real_t  molmassrat,
              const cs_real_t  x,
              const cs_real_t  x_s_tl)
{
  /* Merkel Model
     Hypothesis of unity Lewis factor */
  cs_real_t le_f = 1.;

  if (evap_model == CS_CTWR_POPPE) {
    /* Poppe evaporation model
       Compute Lewis factor using Bosnjakovic hypothesis
       NB: clippings ensuring xi > 1 and le_f > 0 */
    cs_real_t xi = (molmassrat + x_s_tl)/(molmassrat + cs::min(x, x_s_tl));
    if ((xi - 1.) < 1.e-15)
      le_f = pow(0.866,(2./3.));
    else
      le_f = pow(0.866,(2./3.)) * (xi - 1.) / log(xi);
  }

  return le_f;
}

/*----------------------------------------------------------------------------
 * Compute the evaporation mass source in a packing cell, gives the proper
 * values to beta_x_ai, the global mass source, and its value divided
 * by y_l_packing
 *----------------------------------------------------------------------------*/

static void
_evap_packing(cs_ctwr_zone_t *ct, /*Cooling tower (packing) zone */
              cs_real_t pref,   /* Refrence pressure */
              cs_real_t t_h,    /* Humid air temperature */
              cs_real_t t_l_p,  /* Liquid temperature in packing */
              cs_real_3_t vel_h,  /* Humid air velocity */
              cs_real_t rho_h,  /* Humid air density */
              cs_real_t ym_w,   /* Water mass fraction in air */
              cs_real_t x,      /* Humidity */
              cs_real_t rho_l,  /* Liquid density */
              cs_real_t y_l_p,  /* Liquid volume fraction in packing */
              cs_real_t vel_l,  /* Liquid velocity in packing */
              cs_real_t *beta_x_ai,
              cs_real_t *mass_source,
              cs_real_t *mass_source_oy)
{
  cs_real_t vertical[3], horizontal[3];
  /* Packing zone characteristics */
  cs_real_t a_0 = ct->xap;
  cs_real_t xnp = ct->xnp;
  int zone_type = ct->type;

  /* For correlations, T_h cannot be greater than T_l */
  cs_real_t temp_h = cs::min(t_h, t_l_p);

  /* Saturation humidity at humid air temperature */
  cs_real_t x_s_th = cs_air_x_sat(temp_h, pref);

  /* Saturation humidity at injected liquid temperature */
  cs_real_t x_s_tl = cs_air_x_sat(t_l_p, pref);

  /*--------------------------------------------*
   * Counter or cross flow packing zone         *
   *--------------------------------------------*/

  cs_real_t v_air = 0.;

  cs_real_t mass_flux_h = 0.;

  cs_math_3_normalize(cs_glob_physical_constants->gravity, vertical);

  vertical[0] *= -1;
  vertical[1] *= -1;
  vertical[2] *= -1;

  horizontal[0] = vertical[0] -1.;
  horizontal[1] = vertical[1] -1.;
  horizontal[2] = vertical[2] -1.;


  if (zone_type == CS_CTWR_COUNTER_CURRENT) {
    /* Counter flow packing */
    v_air = cs::abs(cs_math_3_dot_product(vel_h, vertical));
  }
  else if (zone_type == CS_CTWR_CROSS_CURRENT) {
    /* Cross flow packing */
    v_air = cs::abs(cs_math_3_dot_product(vel_h, horizontal));
  }

  /* Dry air flux */
  mass_flux_h = rho_h * v_air * (1. - ym_w);

  /* Liquid mass flux */
  cs_real_t mass_flux_l = rho_l * y_l_p * vel_l;
  cs_real_t mass_flux_l_oy = rho_l * vel_l;

  cs_real_t beta_x_ai_oy = 0.;

  /* There is evaporation only if we have an injected liquid flow */
  if (mass_flux_l > 0.) {
    *beta_x_ai = a_0 * mass_flux_l * pow((mass_flux_h/mass_flux_l), xnp);
    beta_x_ai_oy = a_0 * mass_flux_l_oy * pow((mass_flux_h/mass_flux_l),
        xnp);
  }

  /* Source terms for the different equations */

  /* Humid air mass source term */
  if (x <= x_s_th) {
    *mass_source = *beta_x_ai * (x_s_tl - x);
    *mass_source_oy = beta_x_ai_oy * (x_s_tl - x);
  }
  else {
    *mass_source = *beta_x_ai * (x_s_tl - x_s_th);
    *mass_source_oy = beta_x_ai_oy * (x_s_tl - x_s_th);
  }
  *mass_source = cs::max(*mass_source, 0.);
  *mass_source_oy = cs::max(*mass_source_oy, 0.);
}

/*----------------------------------------------------------------------------
 * Compute the evaporation mass source between rain drops and air in any cell,
 * gives the proper values to beta_x_ai, the global mass source,
 * and its value divided by y_l_rain
 *----------------------------------------------------------------------------*/

static void
_evap_rain(cs_air_fluid_props_t *air_prop,
           cs_real_t visc, /* Air viscosity */
           cs_real_t pref, /* Reference pressure */
           cs_real_t t_h,  /* Humid air temperature */
           cs_real_t t_l_r,/* Liquid rain temperature */
           cs_real_3_t drift_vel_rain, /* Rain drift velocity */
           cs_real_t x, /* Humidity */
           cs_real_t x_s, /* Saturated humidity */
           cs_real_t rho_h, /* Humid air density */
           cs_real_t y_rain,/* Rain mass fraction (if mixture) or mass per kg of air */
           cs_real_t rho_ref, /* Reference density (mixture or humid air) */
           cs_real_t rho_l, /* Liquid density */
           cs_real_t *beta_x_ai,
           cs_real_t *mass_source,
           cs_real_t *mass_source_oy)
{
  /* For correlations, T_h cannot be greater than T_p */
  cs_real_t temp_h = cs::min(t_h, t_l_r);

  /* Saturation humidity at the temperature of the humid air */
  cs_real_t x_s_th = cs_air_x_sat(temp_h, pref);

  /* Saturation humidity at the temperature of the rain drop  */
  cs_real_t x_s_tl = cs_air_x_sat(t_l_r, pref);

  /* Lewis factor computation */
  cs_real_t le_f = _lewis_factor(cs_glob_ctwr_option->evap_model,
      air_prop->molmass_rat,
      x,
      x_s_tl);

  cs_real_t cp_h = cs_air_cp_humidair(x, x_s);

  cs_real_t drift_vel_mag = cs_math_3_norm(drift_vel_rain);

  /* Rain droplets Reynolds number */
  cs_real_t rey = rho_h * drift_vel_mag * air_prop->droplet_diam / visc;

  /* Prandtl number */
  cs_real_t pr = cp_h * visc / air_prop->lambda_h;

  /* Nusselt number correlations */
  /* Ranz-Marshall or Hughmark when rey <= 776.06 && pr <= 250. */
  cs_real_t nusselt = 2. + 0.6 * sqrt(rey) * pow(pr,(1./3.));
  /* Hughmark when rey > 776.06 && pr <= 250. */
  if (rey > 776.06 && pr <= 250.) {
    nusselt = 2. + 0.27 * pow(rey, 0.62) * pow(pr,(1./3.));
  }

  /* Convective exchange coefficient 'a_c' */
  cs_real_t a_c = (nusselt * air_prop->lambda_h) / air_prop->droplet_diam;

  /* beta_x coefficient */
  cs_real_t beta_x = a_c / (le_f * cp_h);

  /* Exchange surface area per unit volume based on the total droplets
   * surface in the cell
   * NOTE: Use rho_h to compute the number of particles per unit volume
   * since conservation equation for Y_p based on rho_h
   *   --> Should really be rho_mixture!?
   * Use the symmetric relationship:
   *   a_i = 6*alpha_p*(1.-alpha_p)/droplet_diam
   * where alpha_p is the droplets volume fraction
   * - this kills transfer when there is only one phase (pure humid air
   *   or pure rain) */
  cs_real_t vol_frac_rain = y_rain * rho_ref / rho_l;
  cs_real_t vol_frac_rain_oy = rho_ref / rho_l;

  if (vol_frac_rain >= 1.0)
    vol_frac_rain = 1.0;

  cs_real_t a_i =  6.0 * vol_frac_rain * (1.0 - vol_frac_rain)
    / air_prop->droplet_diam;
  cs_real_t a_i_oy = 6.0 * vol_frac_rain_oy * (1.0 - vol_frac_rain)
    / air_prop->droplet_diam;

  /* Evaporation coefficient 'Beta_x' times exchange surface 'ai' */
  *beta_x_ai = beta_x * a_i;
  cs_real_t beta_x_ai_oy = beta_x * a_i_oy;

  /* Source terms for the different equations */

  /* Humid air mass source term */
  if (x <= x_s_th) {
    *mass_source = *beta_x_ai * (x_s_tl - x);
    *mass_source_oy = beta_x_ai_oy * (x_s_tl - x);
  }
  else {
    *mass_source = *beta_x_ai * (x_s_tl - x_s_th);
    *mass_source_oy = beta_x_ai_oy * (x_s_tl - x_s_th);
  }
  *mass_source = cs::max(*mass_source, 0.);
  *mass_source_oy = cs::max(*mass_source_oy, 0.);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief cs_dof_func_t function to compute volume mass injection for
 *   pressure (mass) equation resulting from water evaporation in the
 *   packing zones or rain injection.
 *
 * \param[in]      n_elts        number of elements to consider
 * \param[in]      elt_ids       list of elements ids
 * \param[in]      dense_output  perform an indirection in retval or not
 * \param[in]      input         null or pointer to a structure cast on-the-fly
 * \param[in, out] retval        resulting value(s). Must be allocated.
 */
/*----------------------------------------------------------------------------*/

void
cs_ctwr_volume_mass_injection_packing_dof_func
(cs_lnum_t         n_elts,
 const cs_lnum_t  *elt_ids,
 bool              dense_output,
 void             *input,
 cs_real_t        *retval)
{
  cs_fluid_properties_t *fp = cs_get_glob_fluid_properties();
  cs_air_fluid_props_t *air_prop = cs_glob_air_props;

  if (dense_output == false)
    bft_error(__FILE__, __LINE__, 0, _("%s: %s."),
              __func__, _("Not implemented for sparse output."));

  const cs_ctwr_option_t *ct_opt = cs_get_glob_ctwr_option();
  /* Air / fluid properties */
  cs_real_t p0 = fp->p0;

  /* Fields necessary for humid atmosphere model */
  cs_field_t *meteo_pressure = cs_field_by_name_try("meteo_pressure");

  cs_real_t *rho_h = nullptr;
  cs_real_3_t *vel_h = nullptr;

 if (ct_opt->mixture_model) {
    rho_h = cs_field_by_name("rho_humid_air")->val; /* Humid air density */
    vel_h = (cs_real_3_t *)cs_field_by_name("v_c")->val; /* Humid air velocity*/
  }
  else {
    rho_h = CS_F_(rho)->val; /* Humid air density */
    vel_h = (cs_real_3_t *)CS_F_(vel)->val_pre; /* Humid air velocity*/
  }

  cs_real_t *ym_w = (cs_real_t *)CS_F_(ym_w)->val; /* Water mass fraction
                                                     in humid air */

  cs_real_t *t_h = nullptr;
  if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
    t_h = cs_field_by_name("real_temperature")->val; /* Humid air temp */
  }
  else{
    t_h = cs_field_by_name("temperature")->val; /* Humid air temp */
  }
  cs_real_t *x = cs_field_by_name("humidity")->val; /* Humidity in air (bulk) */
  cs_real_t *vel_l = cs_field_by_name("vertvel_l")->val;  /* Liquid velocity
                                                             in packing */

  cs_real_t *y_l_p = CS_F_(y_l_pack)->val;
  cs_real_t *t_l_p = cs_field_by_name("temp_l_packing")->val;

  /* Get ct_zone from input */
  cs_ctwr_zone_t *ct
    = (cs_ctwr_zone_t*)input;

  int evap_model = ct_opt->evap_model;

  /* Fields for source terms post-processing */
  cs_real_t *evap_rate_pack = nullptr;
  evap_rate_pack = cs_field_by_name("evaporation_rate_packing")->val;

  cs_real_t rho_l = air_prop->rho_l;

  if (evap_model != CS_CTWR_NONE) {

    /* =========================================
     * HOT RAIN : INJECTION ZONE
     * ========================================= */

    /* Testing if we are in a rain injection zone */
    if (ct->xleak_fac > 0.0 && ct->type == CS_CTWR_INJECTION) {
      /* Bulk mass injected only if rain is in bulk = mixture model activated */
      if (ct_opt->mixture_model) {
        cs_real_t inj_vol = ct->vol_f;

        for (cs_lnum_t j = 0; j < ct->n_cells; j++) {
          cs_real_t mass_source =  ct->q_l_bc * ct->xleak_fac / inj_vol;

          /* Global mass source term for continuity (pressure) equation
           * Note that rain is already considered in the bulk, so inner
           * mass transfer between liquid and vapor disappears */
          /* Warning: not multiplied by Cell volume! no addition neither */
          /* FIXME : j or cell_id ? zone cell number or mesh cell number ? */
          retval[j] = mass_source;
        }
      }
      return;
    }

    /* =========================================
     * PHASE CHANGE : PACKING ZONES
     * Between the liquid film and the humid air
     * ========================================= */

    assert(n_elts == ct->n_cells);

    for (cs_lnum_t j = 0; j < n_elts; j++) {

      cs_lnum_t cell_id = elt_ids[j];

      /* Compute cell reference pressure */
      cs_real_t pphy = cs_ctwr_compute_reference_pressure(cell_id,
          p0,
          meteo_pressure);


      cs_real_t beta_x_ai = 0.;
      cs_real_t mass_source = 0.;
      cs_real_t mass_source_oy = 0.;

        _evap_packing(ct,
                      pphy,
                      t_h[cell_id],
                      t_l_p[cell_id],
                      vel_h[cell_id],
                      rho_h[cell_id],
                      ym_w[cell_id],
                      x[cell_id],
                      rho_l,
                      y_l_p[cell_id],
                      vel_l[cell_id],
                      &(beta_x_ai),
                      &(mass_source),
                      &(mass_source_oy));

      mass_source = fmax(mass_source, 0.);

      /* Global mass source term for continuity (pressure) equation
       * Note that rain is already considered in the bulk, so inner
       * mass transfer between liquid and vapor disappears */
      /* Warning: not multiplied by Cell volume! no addition neither */
      retval[j] = mass_source;

      /* Saving evaporation rate for post-processing */
      evap_rate_pack[cell_id] = mass_source;
    } /* end loop over the cells of a packing zone */
  } /* End evaporation model active */
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief cs_dof_func_t function to compute volume mass injection for
 *   pressure (mass) equation resulting from water evaporation of the rain.
 *
 * \param[in]      n_elts        number of elements to consider
 * \param[in]      elt_ids       list of elements ids
 * \param[in]      dense_output  perform an indirection in retval or not
 * \param[in]      input         null or pointer to a structure cast on-the-fly
 * \param[in, out] retval        resulting value(s). Must be allocated.
 */
/*----------------------------------------------------------------------------*/

void
cs_ctwr_volume_mass_injection_evap_rain_dof_func
([[maybe_unused]]cs_lnum_t         n_elts,
 [[maybe_unused]]const cs_lnum_t  *elt_ids,
 bool                              dense_output,
 [[maybe_unused]]void             *input,
 cs_real_t                        *retval)
{
  const cs_mesh_t *m = cs_glob_mesh;

  assert(n_elts == m->n_cells); /* all cells zone only */
  if (dense_output == false)
    bft_error(__FILE__, __LINE__, 0, _("%s: %s."),
              __func__, _("Not implemented for sparse output."));

  cs_fluid_properties_t *fp = cs_get_glob_fluid_properties();
  cs_air_fluid_props_t *air_prop = cs_glob_air_props;

  const cs_ctwr_option_t *ct_opt = cs_get_glob_ctwr_option();
  /* Air / fluid properties */
  cs_real_t p0 = fp->p0;

  /* Fields necessary for humid atmosphere model */
  cs_field_t *meteo_pressure = cs_field_by_name_try("meteo_pressure");

  cs_real_t *rho = nullptr;
  cs_real_t *rho_h = nullptr;

 if (ct_opt->mixture_model) {
    rho = (cs_real_t *)CS_F_(rho)->val; /* Mixture density */
    rho_h = cs_field_by_name("rho_humid_air")->val; /* Humid air density */
  }
  else {
    rho = CS_F_(rho)->val; /* Humid air density */
    rho_h = CS_F_(rho)->val; /* Humid air density */
  }

  cs_real_t *t_h = nullptr;
  if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
    t_h = cs_field_by_name("real_temperature")->val; /* Humid air temp */
  }
  else{
    t_h = cs_field_by_name("temperature")->val; /* Humid air temp */
  }
  cs_real_t *x = cs_field_by_name("humidity")->val; /* Humidity in air (bulk) */
  cs_real_t *x_s = cs_field_by_name("x_s")->val;

  /* Variable and properties for rain drops */
  cs_field_t *cfld_yp = nullptr;
  if (ct_opt->mixture_model)
    cfld_yp = cs_field_by_name("x_p_01");  /* Rain mass fraction */
  else
    cfld_yp = cs_field_by_name("ym_l_r");  /* Rain mass fraction */

  /* Rain drift velocity */
  cs_field_t *cfld_drift_vel = cs_field_by_composite_name(cfld_yp->name,"drift_vel");
  cs_real_3_t *drift_vel_rain = (cs_real_3_t *)cfld_drift_vel->val;

  /* Fields for source terms post-processing */
  cs_real_t *evap_rate_rain = nullptr;
  evap_rate_rain = cs_field_by_name("evaporation_rate_rain")->val;


  int evap_model = ct_opt->evap_model;

  if (ct_opt->rain_evap) {
    if (evap_model != CS_CTWR_NONE) {

      /* ========================================
       *  PHASE CHANGE : RAIN ZONES (whole domain)
       *  Between the rain drops and the humid air
       * ========================================= */

      if (cfld_yp != nullptr) {
        cs_real_t *y_rain = (cs_real_t *)cfld_yp->val;
        cs_real_t *t_l_r = (cs_real_t *)cs_field_by_name("temp_l_r")->val;

        for (cs_lnum_t cell_id = 0; cell_id < m->n_cells; cell_id++) {

          if (y_rain[cell_id] > 0.) {

            cs_real_t beta_x_ai = 0.;
            cs_real_t mass_source = 0.;
            cs_real_t mass_source_oy = 0.;

            /* Compute cell reference pressure */
            cs_real_t pphy = cs_ctwr_compute_reference_pressure(cell_id,
                p0,
                meteo_pressure);

            _evap_rain(air_prop,
                fp->viscl0,
                pphy,
                t_h[cell_id],
                t_l_r[cell_id],
                drift_vel_rain[cell_id],
                x[cell_id],
                x_s[cell_id],
                rho_h[cell_id],
                y_rain[cell_id],
                rho[cell_id],
                air_prop->rho_l,
                &(beta_x_ai),
                &(mass_source),
                &(mass_source_oy));

            mass_source = cs::max(mass_source, 0.);

            retval[cell_id] = mass_source;
            /* Saving evaporation rate for post-processing */
            evap_rate_rain[cell_id] = mass_source;

          } /* End if (y_rain > 0) */
        } /* End loop over all the cells of the domain */
      } /* End yp field exists */
    } /* End evaporation model active */
  } /* End rain evaporation active */
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief cs_dof_func_t function to compute volume mass injection for
 *   pressure (mass) equation for the leaking rain form packings
 *   if mixture model is on.
 *
 * \param[in]      n_elts        number of elements to consider
 * \param[in]      elt_ids       list of elements ids
 * \param[in]      dense_output  perform an indirection in retval or not
 * \param[in]      input         null or pointer to a structure cast on-the-fly
 * \param[in, out] retval        resulting value(s). Must be allocated.
 */
/*----------------------------------------------------------------------------*/

void
cs_ctwr_volume_mass_injection_rain_dof_func
([[maybe_unused]] cs_lnum_t         n_elts,
 [[maybe_unused]] const cs_lnum_t  *elt_ids,
 bool                               dense_output,
 [[maybe_unused]] void             *input,
 cs_real_t                         *retval)
{
  const cs_mesh_t *m = cs_glob_mesh;
  const cs_lnum_2_t *i_face_cells
    = (const cs_lnum_2_t *)(m->i_face_cells);

  const cs_real_t *cell_f_vol = cs_glob_mesh_quantities->cell_vol;

  const cs_ctwr_option_t *ct_opt = cs_get_glob_ctwr_option();

  assert(n_elts == m->n_cells); /* all cells zone only */
  if (dense_output == false)
    bft_error(__FILE__, __LINE__, 0, _("%s: %s."),
              __func__, _("Not implemented for sparse output."));

  /* Generate rain from packing zones which are leaking
     ================================================== */

  cs_real_t *liq_vol_frac = CS_F_(y_l_pack)->val; /* Liquid mass fraction
                                                      in packing */
  /* Inner mass flux of liquids (in the packing) */
  cs_real_t *liq_mass_flow
    = cs_field_by_name("inner_mass_flux_y_l_packing")->val;

  /* Cooling tower zones */
  cs_ctwr_zone_t **_ct_zone = cs_get_glob_ctwr_zone();
  const int *_n_ct_zones = cs_get_glob_ctwr_n_zones();

  for (int ict = 0; ict < *_n_ct_zones; ict++) {
    cs_ctwr_zone_t *ct = _ct_zone[ict];

    if (ct->xleak_fac > 0.0 && ct->type != CS_CTWR_INJECTION) {

      /* Rain generation source terms
         ============================ */

      for (cs_lnum_t i = 0; i < ct->n_outlet_faces; i++) {

        /* Leak face_id */
        cs_lnum_t face_id = ct->outlet_faces_ids[i];
        cs_lnum_t cell_id_leak, cell_id_rain;

        /* Convention: outlet is positive mass flux
         * Then upwind cell for liquid is i_face_cells[][0] */
        int sign = 1;
        if (liq_mass_flow[face_id] < 0) {
          sign = -1;
          cell_id_leak = i_face_cells[face_id][1];
          cell_id_rain = i_face_cells[face_id][0];
        }
        else {
          cell_id_leak = i_face_cells[face_id][0];
          cell_id_rain = i_face_cells[face_id][1];
        }

        /* Note: mass_source must be divided by
         * cell_f_vol[cell_id_rain]
         * because mass source computed from liq_mass_flow is
         * already in kg/s associated to the facing rain cell */
        cs_real_t mass_source = ct->xleak_fac / cell_f_vol[cell_id_rain]
          * liq_vol_frac[cell_id_leak] * sign * liq_mass_flow[face_id];

        /* Global bulk mass - continuity */
        /* Warning: not multiplied by Cell volume! */
        retval[cell_id_rain] = mass_source;

      } /* End of loop through outlet cells of the packing zone */

    } /* End of leaking zone test */

    /* Rain - packing interaction
     * ========================== */

    /* Boolean rain_to_packing ctwr model option enables rain liquid water to
     * become liquid water film when it reaches a packing zone */

    /* FIXME: Corrections needed to ensure mass and energy conservation,
     * better not use it for the moment */
    if  (ct_opt->rain_to_packing) {
      /* Rain sink term in packing zones */
      //TODO : Add rain leak portion inside packing

    }

  } /* End of loop over cooling towers */
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief cs_dof_func_t function to compute volume mass injection for
 *   yphp rain equation (enthalpy) if mixture model is on.
 *
 * \param[in]      n_elts        number of elements to consider
 * \param[in]      elt_ids       list of elements ids
 * \param[in]      dense_output  perform an indirection in retval or not
 * \param[in]      input         null or pointer to a structure cast on-the-fly
 * \param[in, out] retval        resulting value(s). Must be allocated.
 */
/*----------------------------------------------------------------------------*/

void
cs_ctwr_volume_mass_injection_yh_rain_dof_func
([[maybe_unused]]cs_lnum_t         n_elts,
 [[maybe_unused]]const cs_lnum_t  *elt_ids,
 bool                              dense_output,
 [[maybe_unused]]void             *input,
 cs_real_t                        *retval)
{
  const cs_mesh_t *m = cs_glob_mesh;
  const cs_lnum_2_t *i_face_cells
    = (const cs_lnum_2_t *)(m->i_face_cells);

  assert(n_elts == m->n_cells); /* all cells zone only */
  if (dense_output == false)
    bft_error(__FILE__, __LINE__, 0, _("%s: %s."),
              __func__, _("Not implemented for sparse output."));

  /* Variable and properties for rain drops */
  cs_real_t *h_l_p = cs_field_by_name("h_l_packing")->val;

  const cs_ctwr_option_t *ct_opt = cs_get_glob_ctwr_option();

  /* Generate rain from packing zones which are leaking
     ================================================== */

  /* Inner mass flux of liquids (in the packing) */
  cs_real_t *liq_mass_flow
    = cs_field_by_name("inner_mass_flux_y_l_packing")->val;

  /* Cooling tower zones */
  cs_ctwr_zone_t **_ct_zone = cs_get_glob_ctwr_zone();
  const int *_n_ct_zones = cs_get_glob_ctwr_n_zones();

  for (int ict = 0; ict < *_n_ct_zones; ict++) {
    cs_ctwr_zone_t *ct = _ct_zone[ict];

    if (ct->xleak_fac > 0.0 && ct->type != CS_CTWR_INJECTION) {

      /* Rain generation source terms
         ============================ */

      for (cs_lnum_t i = 0; i < ct->n_outlet_faces; i++) {

        /* Leak face_id */
        cs_lnum_t face_id = ct->outlet_faces_ids[i];
        cs_lnum_t cell_id_leak, cell_id_rain;

        /* Convention: outlet is positive mass flux
         * Then upwind cell for liquid is i_face_cells[][0] */
        if (liq_mass_flow[face_id] < 0) {
          cell_id_leak = i_face_cells[face_id][1];
          cell_id_rain = i_face_cells[face_id][0];
        }
        else {
          cell_id_leak = i_face_cells[face_id][0];
          cell_id_rain = i_face_cells[face_id][1];
        }

        /* Rain enthalpy */
        /* Warning: not multiplied by Cell volume! */
        // FIXME: There should be a y_p factor in there so that
        // mass and enthalpy are compatible
        retval[cell_id_rain] = h_l_p[cell_id_leak];

      } /* End of loop through outlet cells of the packing zone */

    } /* End of leaking zone test */


    /* Rain - packing interaction
     * ========================== */

    /* Boolean rain_to_packing ctwr model option enables rain liquid water to
     * become liquid water film when it reaches a packing zone */

    /* FIXME: Corrections needed to ensure mass and energy conservation,
     * better not use it for the moment */
    if  (ct_opt->rain_to_packing) {
      /* Rain sink term in packing zones */
      //TODO : Add rain leak portion inside packing

    }

  } /* End of loop over cooling towers */
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Phase change source terms - Exchange terms between the injected
 *        liquid and the water vapor phase in the bulk, humid air
 *
 * \param[in]     f_id          field id
 * \param[in,out] exp_st        Explicit source term
 * \param[in,out] imp_st        Implicit source term
 */
/*----------------------------------------------------------------------------*/

void
cs_ctwr_source_term(int              f_id,
                    cs_real_t        exp_st[],
                    cs_real_t        imp_st[])
{
  const cs_mesh_t *m = cs_glob_mesh;
  const cs_lnum_2_t *i_face_cells
    = (const cs_lnum_2_t *)(m->i_face_cells);
  const cs_lnum_t n_i_faces = m->n_i_faces;

  const cs_real_t *cell_f_vol = cs_glob_mesh_quantities->cell_vol;

  cs_fluid_properties_t *fp = cs_get_glob_fluid_properties();
  cs_air_fluid_props_t *air_prop = cs_glob_air_props;
  const  cs_ctwr_option_t *ct_opt = cs_get_glob_ctwr_option();

  /* Water / air molar mass ratio */
  const cs_real_t molmassrat = air_prop->molmass_rat;

  cs_real_t *rho = nullptr;
  cs_real_t *rho_h = nullptr;
  cs_real_3_t *vel_h = nullptr;

 if (ct_opt->mixture_model) {
    rho = (cs_real_t *)CS_F_(rho)->val; /* Mixture density */
    rho_h = cs_field_by_name("rho_humid_air")->val; /* Humid air density */
    vel_h = (cs_real_3_t *)cs_field_by_name("v_c")->val; /* Humid air velocity*/
  }
  else {
    rho = CS_F_(rho)->val; /* Humid air density */
    rho_h = CS_F_(rho)->val;
    vel_h = (cs_real_3_t *)CS_F_(vel)->val_pre; /* Humid air velocity*/
  }

  cs_real_t *ym_w = (cs_real_t *)CS_F_(ym_w)->val; /* Water mass fraction
                                                     in humid air */

  cs_real_t *t_h = nullptr;
  if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
    t_h = cs_field_by_name("real_temperature")->val; /* Humid air temp */
  }
  else{
    t_h = cs_field_by_name("temperature")->val; /* Humid air temp */
  }
  cs_real_t *x = cs_field_by_name("humidity")->val; /* Humidity in air (bulk) */
  cs_real_t *x_s = cs_field_by_name("x_s")->val;
  cs_real_t *vel_l = cs_field_by_name("vertvel_l")->val;  /* Liquid velocity
                                                             in packing */

  cs_real_t *t_l_p = cs_field_by_name("temp_l_packing")->val;
  cs_real_t *y_l_p = CS_F_(y_l_pack)->val_pre;

  /* Variable and properties for rain drops */
  cs_field_t *cfld_yp = nullptr;
  if (ct_opt->mixture_model)
    cfld_yp = cs_field_by_name("x_p_01");  /* Rain mass fraction */
  else
    cfld_yp = cs_field_by_name("ym_l_r");  /* Rain mass fraction */

  cs_field_t *cfld_yh_rain = cs_field_by_name("ymh_l_r"); /* Yp times Tp */

  /* Rain drift velocity */
  cs_field_t *cfld_drift_vel = cs_field_by_composite_name(cfld_yp->name,"drift_vel");
  cs_real_3_t *drift_vel_rain = (cs_real_3_t *)cfld_drift_vel->val;

  /* Phases volume fractions */
  cs_real_t *vol_f_r = cs_field_by_name("vol_f_r")->val; /* Vol frac. rain */

  /* Rain inner mass flux */
  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  cs_real_t *imasfl_r = cs_field_by_id
                         (cs_field_get_key_int(cfld_yp, kimasf))->val;

  int evap_model = ct_opt->evap_model;

  /* Cooling tower zones */
  cs_ctwr_zone_t **_ct_zone = cs_get_glob_ctwr_zone();
  const int *_n_ct_zones = cs_get_glob_ctwr_n_zones();

  /* Identify the source term formulation for the required field */

  const cs_field_t *f = cs_field_by_id(f_id);

  cs_real_t *f_var = f->val;  /* Field variable */

  /* Compute the source terms */

  /* Fields for source terms post-processing */
  cs_real_t *evap_rate_rain = nullptr;
  evap_rate_rain = cs_field_by_name("evaporation_rate_rain")->val;

  cs_real_t *thermal_power_pack = nullptr;
  cs_real_t *thermal_power_rain = nullptr;
  thermal_power_pack = cs_field_by_name("thermal_power_packing")->val;
  thermal_power_rain = cs_field_by_name("thermal_power_rain")->val;

  /* Table to track cells belonging to packing zones */
  cs_lnum_t  *packing_cell;
  CS_MALLOC(packing_cell, m->n_cells_with_ghosts, int);
# pragma omp parallel for if (m->n_cells> CS_THR_MIN)
  for (cs_lnum_t cell_id = 0; cell_id < m->n_cells_with_ghosts; cell_id++)
    packing_cell[cell_id] = -1;

  /* Air / fluid properties */
  cs_real_t cp_d = fp->cp0;
  cs_real_t rscp = fp->r_pg_cnst / cp_d;
  cs_real_t cp_v = air_prop->cp_v;
  cs_real_t cp_l = air_prop->cp_l;
  cs_real_t hv0 = air_prop->hv0;
  cs_real_t rho_l = air_prop->rho_l;
  cs_real_t visc = fp->viscl0;
  cs_real_t p0 = fp->p0;
  cs_real_t ps = cs_glob_atmo_constants->ps;
  cs_real_t droplet_diam  = air_prop->droplet_diam;
  cs_real_t sigma  = air_prop->sigma;

  /* Fields necessary for humid atmosphere model */
  cs_field_t *meteo_pressure = cs_field_by_name_try("meteo_pressure");

  cs_real_t *lagr_injection_profile = nullptr;
  if (cs_glob_lagr_model->physical_model == CS_LAGR_PHYS_CTWR)
    lagr_injection_profile =
      cs_field_by_name_try("lagr_injection_profile")->val;

  if (evap_model != CS_CTWR_NONE) {

    /* =========================================
     * PHASE CHANGE : PACKING ZONES
     * Between the liquid film and the humid air
     * ========================================= */

    for (int ict = 0; ict < *_n_ct_zones; ict++) {

      cs_ctwr_zone_t *ct = _ct_zone[ict];

      /* We skip this if we are in injection zone */
      if (ct->type == CS_CTWR_INJECTION)
        continue;

      const cs_lnum_t *ze_cell_ids = cs_volume_zone_by_name(ct->name)->elt_ids;

      for (cs_lnum_t j = 0; j < ct->n_cells; j++) {

        cs_lnum_t cell_id = ze_cell_ids[j];

        /* Compute cell reference pressure */
        cs_real_t pphy = cs_ctwr_compute_reference_pressure(cell_id,
            p0,
            meteo_pressure);


        /* Identify packing cells ids */
        if (ct->type != CS_CTWR_INJECTION)
          packing_cell[cell_id] = ict;

        cs_real_t beta_x_ai = 0.;
        cs_real_t mass_source = 0.;
        cs_real_t mass_source_oy = 0.;

        _evap_packing(ct,
                      pphy,
                      t_h[cell_id],
                      t_l_p[cell_id],
                      vel_h[cell_id],
                      rho_h[cell_id],
                      ym_w[cell_id],
                      x[cell_id],
                      rho_l,
                      y_l_p[cell_id],
                      vel_l[cell_id],
                      &(beta_x_ai),
                      &(mass_source),
                      &(mass_source_oy));

        cs_real_t vol_mass_source = mass_source * cell_f_vol[cell_id];
        cs_real_t vol_mass_source_oy = mass_source_oy * cell_f_vol[cell_id];
        cs_real_t vol_beta_x_ai = beta_x_ai * cell_f_vol[cell_id];

        /* Extra parameters for humid air thermal source term */
        cs_real_t x_s_tl = cs_air_x_sat(t_l_p[cell_id], pphy);
        cs_real_t x_s_th = cs_air_x_sat(t_h[cell_id], pphy);

        cs_real_t cp_h = cs_air_cp_humidair(x[cell_id], x_s[cell_id]);


        /* Global mass source term for continuity (pressure) equation
         * is already taken into account using standard volume
         * mass injection (see cs_ctwr_volume_mass_injection_dof_func)
         *
         * Consequent source terms on ym_w are also taken into account
         * using standard mass source term mechanism
         * (defined in cs_ctwr_build_all).
         * */

        /* Injected liquid mass equation (solve in drift model form) */
        if (f_id == (CS_F_(y_l_pack)->id)) {
          exp_st[cell_id] -= vol_mass_source_oy * y_l_p[cell_id];
          imp_st[cell_id] += vol_mass_source_oy;
        }

        /* Humid air temperature equation */
        else if (f_id == (CS_F_(t)->id)) {
          /* Because the writing is in a non-conservative form */
          cs_real_t l_imp_st = vol_mass_source * cp_h;
          cs_real_t l_exp_st = 0.;
          cs_real_t le_f = _lewis_factor(evap_model, molmassrat,
              x[cell_id], x_s_tl);
          cs_real_t coef = 1.;
          cs_real_t t_shift = 0.;
          if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
            /* Coefficient to convert T into theta_l */
            coef = pow((ps / pphy), rscp);
            t_shift = cs_physical_constants_celsius_to_kelvin;
          }

          if (x[cell_id] <= x_s_th) {
            /* Implicit and explicit terms for temperature T */
            l_imp_st += vol_beta_x_ai * (le_f * cp_h
                + (x_s_tl - x[cell_id]) * cp_v
                / (1. + x[cell_id]));
            l_exp_st += l_imp_st * (coef * (t_l_p[cell_id] + t_shift)
                - f_var[cell_id]);
          }
          else {
            /* Implicit and explicit terms for temperature T */
            l_imp_st +=   vol_beta_x_ai
              * (  le_f * cp_h + (x_s_tl - x_s_th) * cp_l
                  / (1. + x[cell_id]));
            l_exp_st +=   vol_beta_x_ai
              * (  le_f * cp_h * coef * (t_l_p[cell_id] + t_shift)
                  +   (x_s_tl - x_s_th)
                  * (cp_v * coef * (t_l_p[cell_id] + t_shift))
                  / (1. + x[cell_id]))
              - l_imp_st * f_var[cell_id];
          }
          /* If humid atmosphere model, temperature is liquid potential
           * temperature theta_l */
          if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
           // l_exp_st -= l_imp_st * coef * (hv0 / cp_d) * yw_liq->val[cell_id];
          }

          imp_st[cell_id] += cs::max(l_imp_st, 0.);
          exp_st[cell_id] += l_exp_st;
        }

        /* Injected liquid enthalpy equation (solve in drift model form)
         * NB: it is in fact "y_l_p x h_l" */
        else if (f_id == (CS_F_(yh_l_pack)->id)) {
          /* Liquid temperature in Kelvin */
          cs_real_t t_l_k = t_l_p[cell_id]
            + cs_physical_constants_celsius_to_kelvin;
          cs_real_t l_exp_st = 0.;

          /* Note: the solved variable is yl_p.hl_p so the source term
           * associated to evaporation is: Gamma/y_lp * (yl_p.h_lp) */
          /* Implicit term */
          cs_real_t l_imp_st = vol_mass_source_oy;

          l_exp_st -= l_imp_st * f_var[cell_id];

          cs_real_t le_f = _lewis_factor(evap_model,
              molmassrat,
              x[cell_id],
              x_s_tl);

          /* Under saturated */
          if (x[cell_id] <= x_s_th) {
            /* Explicit term */
            l_exp_st -= vol_beta_x_ai * ((x_s_tl - x[cell_id])
                * (cp_v * t_l_k + hv0)
                + le_f * cp_h
                * (t_l_p[cell_id] - t_h[cell_id]));
          }
          /* Over saturated */
          else {
            /* Explicit term */
            l_exp_st += vol_beta_x_ai
                        * (le_f * cp_h * (t_h[cell_id] - t_l_p[cell_id])
                           + (x_s_tl - x_s_th) / (1. + x[cell_id])
                             * (cp_l * t_h[cell_id]
                                - (cp_v * t_l_p[cell_id] + hv0)));
          }
          /* Because we deal with an increment */
          exp_st[cell_id] += l_exp_st;
          imp_st[cell_id] += cs::max(l_imp_st, 0.);

          /* Saving thermal power for post-processing */
          thermal_power_pack[cell_id] = -(l_exp_st + l_imp_st * f_var[cell_id])
            / cell_f_vol[cell_id];
        }
      } /* end loop over the cells of a packing zone */

    } /* end loop over all the packing zones */

    /* ========================================
     *  PHASE CHANGE : RAIN ZONES (whole domain)
     *  Between the rain drops and the humid air
     * ========================================= */

    if (cfld_yp != nullptr && ct_opt->rain_evap) {
      cs_real_t *y_rain = (cs_real_t *)cfld_yp->val;
      cs_real_t *t_l_r = (cs_real_t *)cs_field_by_name("temp_l_r")->val;

      for (cs_lnum_t cell_id = 0; cell_id < m->n_cells; cell_id++) {

        if (y_rain[cell_id] > 0.) {

        /* Compute cell reference pressure */
        cs_real_t pphy = cs_ctwr_compute_reference_pressure(cell_id,
            p0,
            meteo_pressure);


          cs_real_t beta_x_ai = 0.;
          cs_real_t mass_source = 0.;
          cs_real_t mass_source_oy = 0.;

          _evap_rain(air_prop,
                     visc,
                     pphy,
                     t_h[cell_id],
                     t_l_r[cell_id],
                     drift_vel_rain[cell_id],
                     x[cell_id],
                     x_s[cell_id],
                     rho_h[cell_id],
                     y_rain[cell_id],
                     rho[cell_id],
                     rho_l,
                     &(beta_x_ai),
                     &(mass_source),
                     &(mass_source_oy));

          cs_real_t vol_mass_source = mass_source * cell_f_vol[cell_id];
          cs_real_t vol_mass_source_oy = mass_source_oy * cell_f_vol[cell_id];
          cs_real_t vol_beta_x_ai = beta_x_ai * cell_f_vol[cell_id];
          /* Note: global bulk mass - continuity is taken with
           * cs_ctwr_volume_mass_injection_dof_func */

        /* Extra parameters for humid air thermal source term */
          cs_real_t x_s_tl = cs_air_x_sat(t_l_r[cell_id], pphy);
          cs_real_t x_s_th = cs_air_x_sat(t_h[cell_id], pphy);
          cs_real_t le_f = _lewis_factor(evap_model,
              molmassrat,
              x[cell_id],
              x_s_tl);

          cs_real_t cp_h = cs_air_cp_humidair(x[cell_id], x_s[cell_id]);

          /* Water mass fraction in air has source terms associated with rain
           * evaporation only if it is not related to a mass injection i.e.,
           * if mixture model is activated */
          if (ct_opt->mixture_model) {
          /* Water (vapor + condensate) in gas mass fraction equation
             except rain */
            if (f_id == (CS_F_(ym_w)->id)) {
              exp_st[cell_id] += vol_mass_source * (1. - f_var[cell_id]);
              imp_st[cell_id] += vol_mass_source;

              /* Saving evaporation rate for post-processing */
              evap_rate_rain[cell_id] = mass_source;
            }
          }
          /* Rain drop mass equation (solve in drift model form) */
          else if (f_id == cfld_yp->id) {
            exp_st[cell_id] -= vol_mass_source_oy * y_rain[cell_id];
            imp_st[cell_id] += vol_mass_source_oy;
          }

          /* Humid air temperature equation */
          else if (f_id == (CS_F_(t)->id)) {
            /* Because the writing is in a non-conservative form */
            cs_real_t l_imp_st = vol_mass_source * cp_h;
            cs_real_t l_exp_st = 0.;

            cs_real_t coef = 1.;
            cs_real_t t_shift = 0.;
            if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
              /* Coefficient to convert T into theta_l */
              coef = pow((ps / pphy), rscp);
              t_shift = cs_physical_constants_celsius_to_kelvin;
            }

            if (x[cell_id] <= x_s_th) {
              /* Implicit and explicit terms for temperature T */
              l_imp_st += vol_beta_x_ai * (le_f * cp_h
                                           + (x_s_tl - x[cell_id]) * cp_v
                                             / (1. + x[cell_id]));
              l_exp_st = l_imp_st * (coef * (t_l_r[cell_id] + t_shift)
                                     - f_var[cell_id]);
            }
            else {
              /* Implicit and explicit terms for temperature T */
              l_imp_st += vol_beta_x_ai * (le_f * cp_h + (x_s_tl - x_s_th) * cp_l
                  / (1. + x[cell_id]));
              l_exp_st =   vol_beta_x_ai
                         * (  le_f * cp_h * coef * (t_l_r[cell_id] + t_shift)
                            + (x_s_tl - x_s_th)
                               * (cp_v * coef * (t_l_r[cell_id] + t_shift ))
                               / (1. + x[cell_id]))
                         - l_imp_st * f_var[cell_id];
            }
            /* If humid atmosphere model, temperature is liquid potential
             * temperature theta_l */
            if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] == CS_ATMO_HUMID) {
              //l_exp_st -= l_imp_st * coef * (hv0 / cp_d) * yw_liq->val[cell_id];
            }
            imp_st[cell_id] += cs::max(l_imp_st, 0.);
            exp_st[cell_id] += l_exp_st;
          }

          /* Rain enthalpy equation (solve in drift model form)
           * NB: The actual variable being solved is y_rain x h_rain */
          else if (f_id == cfld_yh_rain->id) {
            /* Liquid temperature in Kelvin */
            cs_real_t t_l_k = t_l_r[cell_id]
              + cs_physical_constants_celsius_to_kelvin;

            cs_real_t l_exp_st = 0.;
            cs_real_t l_imp_st = 0.;

            if (x[cell_id] <= x_s_th) {
              /* Explicit term */
              l_exp_st -= vol_beta_x_ai * ((x_s_tl - x[cell_id])
                  * (cp_v * t_l_k + hv0)
                  + le_f * cp_h
                  * (t_l_r[cell_id] - t_h[cell_id]));
            }
            /* Over saturated */
            else {
              /* Explicit term */
              l_exp_st += vol_beta_x_ai
                          * (le_f * cp_h * (t_h[cell_id] - t_l_r[cell_id])
                             + (x_s_tl - x_s_th) / (1. + x[cell_id])
                               * (cp_l * t_h[cell_id]
                                  - (cp_v * t_l_r[cell_id] + hv0)));
            }
            /* Because we deal with an increment */
            exp_st[cell_id] += l_exp_st;
            imp_st[cell_id] += cs::max(l_imp_st, 0.);

            /* Saving thermal power for post-processing */
            if (t_l_r[cell_id] > 0.) {
              thermal_power_rain[cell_id]
                = -(l_exp_st + l_imp_st * f_var[cell_id])
                  / cell_f_vol[cell_id];
            }
          }

        } /* End if (y_rain > 0) */

      } /* End loop over all the cells of the domain */
    }
  } /* End evaporation model active */

  if (ct_opt->has_rain) {

    /* Generate rain from packing zones which are leaking
       ================================================== */

    /* Note: global bulk mass - continuity is taken with
     * cs_ctwr_volume_mass_injection_rain_dof_func
     * Injected liquid mass equation for rain zones is taken
     * into account using standard volume mass injection mechanism
     * (cs_equation_add_volume_mass_injection_by_value, with 1 as value)
     * Warning : this function must be used only if mixture model is activated,
     * otherwise rain does not contribute to bulk mass */

    /* Note: Rain enthalpy is treated by
     * cs_ctwr_volume_mass_injection_yh_rain_dof_func */

    /* If mixture model is not activated, then the leaking packing is not
     * treated through volume mass injection functions and must be accounted
     * through source terms for y_rain and yh_rain */
    if (!(ct_opt->mixture_model)) {

      cs_real_t *liq_vol_frac = CS_F_(y_l_pack)->val; /* Liquid mass fraction
                                                          in packing */
      cs_real_t *yh_l_pack = CS_F_(yh_l_pack)->val;
      /* Inner mass flux of liquids (in the packing) */
      cs_real_t *liq_mass_flow
        = cs_field_by_name("inner_mass_flux_y_l_packing")->val;

      for (int ict = 0; ict < *_n_ct_zones; ict++) {

        cs_ctwr_zone_t *ct = _ct_zone[ict];

        if (ct->xleak_fac > 0.0 && ct->type != CS_CTWR_INJECTION) {

          /* Rain generation source terms
             ============================ */

          for (cs_lnum_t i = 0; i < ct->n_outlet_faces; i++) {

            /* Leak face_id */
            cs_lnum_t face_id = ct->outlet_faces_ids[i];
            cs_lnum_t cell_id_leak, cell_id_rain;

            /* Convention: outlet is positive mass flux
             * Then upwind cell for liquid is i_face_cells[][0] */
            int sign = 1;
            if (liq_mass_flow[face_id] < 0) {
              sign = -1;
              cell_id_leak = i_face_cells[face_id][1];
              cell_id_rain = i_face_cells[face_id][0];
            }
            else {
              cell_id_leak = i_face_cells[face_id][0];
              cell_id_rain = i_face_cells[face_id][1];
            }

            /* Note: vol_mass_source must not be multiplied by
             * cell_f_vol[cell_id_rain]
             * because mass source computed from liq_mass_flow is
             * already in kg/s associated to the facing rain cell */
            cs_real_t rain_leak_mass_flow = ct->xleak_fac
              * liq_vol_frac[cell_id_leak] * sign * liq_mass_flow[face_id];

            cs_real_t yh_rain_leak_mass_flow = ct->xleak_fac
              * yh_l_pack[cell_id_leak] * sign * liq_mass_flow[face_id];

            /* Injected liquid mass equation for rain zones
               (solve in drift model form)
               or store it for Lagrangian model for future injection
             */
            if (f_id == cfld_yp->id) {
              if (lagr_injection_profile != nullptr)
                lagr_injection_profile[cell_id_rain] = rain_leak_mass_flow;
              else
                exp_st[cell_id_rain] += rain_leak_mass_flow;
              t_l_p[cell_id_rain] = t_l_p[cell_id_leak];
            }
            /* Rain enthalpy */
            else if (f_id == cfld_yh_rain->id) {
              exp_st[cell_id_rain] += yh_rain_leak_mass_flow;
            }

          } /* End of loop through outlet cells of the packing zone */
        } /* End of leaking packing zone test */

        /* Testing if we are in a rain injection zone */
        if (ct->xleak_fac > 0.0 && ct->type == CS_CTWR_INJECTION) {
          /* Bulk mass injected only if rain is in bulk = mixture model activated */
          const cs_lnum_t *ze_cell_ids
            = cs_volume_zone_by_name(ct->name)->elt_ids;
          cs_real_t inj_vol = ct->vol_f;

          for (cs_lnum_t j = 0; j < ct->n_cells; j++) {
            cs_lnum_t cell_id = ze_cell_ids[j];
            cs_real_t vol_mass_source =  cell_f_vol[cell_id] * ct->q_l_bc
              * ct->xleak_fac / inj_vol;

            cs_real_t t_inj = ct->t_l_bc;
            cs_real_t h_inj = cs_liq_t_to_h(t_inj);

            /* Injected liquid mass equation for rain zones
               (solve in drift model form) */
            if (f_id == cfld_yp->id) {
              exp_st[cell_id] += vol_mass_source;
            }
            /* Rain enthalpy */
            else if (f_id == cfld_yh_rain->id) {
              exp_st[cell_id] += vol_mass_source * h_inj;
            }
          } /* End loop over injection zone cells */
        } /* End test injection zone */

      } /* End loop over cooling tower zones */
    } /* End test mixture model activated */

    /* Rain - packing interaction
     * ========================== */

    /* Boolean rain_to_packing ctwr model option enables rain liquid water to
     * become liquid water film when it reaches a packing zone */

    /* FIXME: Corrections needed to ensure mass and energy conservation,
     * better not use it for the moment */
    if  (ct_opt->rain_to_packing) {
      //TODO create list of inlet_faces as we do for outlet faces
      for (cs_lnum_t face_id = 0; face_id < n_i_faces; face_id++) {
        cs_lnum_t cell_id_0 = i_face_cells[face_id][0];
        cs_lnum_t cell_id_1 = i_face_cells[face_id][1];

        /* one of neigh. cells is in packing */
        if (packing_cell[cell_id_0] != -1 || packing_cell[cell_id_1] != -1) {

          /* Rain sink term in packing zones */
          //TODO : Add rain leak portion inside packing
          if (f_id == cfld_yp->id) {
            if (packing_cell[cell_id_0] != -1) {
              imp_st[cell_id_0] += cs::max(imasfl_r[face_id], 0.);
              exp_st[cell_id_0] -=   cs::max(imasfl_r[face_id], 0)
                                   * f_var[cell_id_0];
            }
            if (packing_cell[cell_id_1] != -1) {
              imp_st[cell_id_1] += cs::max(-imasfl_r[face_id], 0.);
              exp_st[cell_id_1] -=   cs::max(-imasfl_r[face_id], 0)
                                   * f_var[cell_id_1];
            }
          }

          if (f_id == cfld_yh_rain->id) {
            if (packing_cell[cell_id_0] != -1) {
              imp_st[cell_id_0] += cs::max(imasfl_r[face_id], 0.);
              exp_st[cell_id_0] -=   cs::max(imasfl_r[face_id], 0)
                                   * f_var[cell_id_0];
            }
            if (packing_cell[cell_id_1] != -1) {
              imp_st[cell_id_1] += cs::max(-imasfl_r[face_id], 0.);
              exp_st[cell_id_1] -=   cs::max(-imasfl_r[face_id], 0)
                                   * f_var[cell_id_1];
            }
          }

          /* Liquid source term in packing zones from rain */
          if (f_id == CS_F_(y_l_pack)->id) {
            if (packing_cell[cell_id_0] != -1) {
              exp_st[cell_id_0] +=   cs::max(imasfl_r[face_id], 0)
                                   * cfld_yp->val[cell_id_0];
            }
            if (packing_cell[cell_id_1] != -1) {
              exp_st[cell_id_1] +=   cs::max(-imasfl_r[face_id], 0)
                                   * cfld_yp->val[cell_id_1];
            }
          }

          if (f_id == CS_F_(yh_l_pack)->id) {
            if (packing_cell[cell_id_0] != -1) {
              exp_st[cell_id_0] +=   cs::max(imasfl_r[face_id], 0)
                                   * cfld_yh_rain->val[cell_id_0];
            }

            if (packing_cell[cell_id_1] != -1) {
              exp_st[cell_id_1] +=   cs::max(-imasfl_r[face_id], 0)
                                   * cfld_yh_rain->val[cell_id_1];
            }
          }
        }
      }
    }

  } /* End of test on whether to generate rain */

  /*  Adding Lagrangian source terms when twoway_coupling is activated
      ================================================================ */

  /* General comment: Since all Lagrangian source terms have been calculated
     using the values of the current and previous time step (see the
     cs_lagr_coupling.c subroutine), only explicit source terms have to be
     accounted for in the corresponding transport equations ?! */

  if (cs_glob_lagr_time_scheme->iilagr == CS_LAGR_TWOWAY_COUPLING) {

    if (cs_glob_lagr_model->physical_model == CS_LAGR_PHYS_CTWR) {

      cs_lnum_t ncel    = cs_glob_mesh->n_cells;

      /* verifying if a mass source term is activated in the Lagrangian module*/
      if (cs_glob_lagr_source_terms->ltsmas == 1) {
        cs_real_t *lag_st_m = cs_field_by_name("lagr_st_pressure")->val;
        for (cs_lnum_t cell_id = 0; cell_id < ncel; cell_id++) {
          /* Since there is only evaporation accounting for a liquid - gas phase
             mass transfer, the transferred must be water vapor -> ym_w*/
          if (f_id == (CS_F_(ym_w)->id)) {
            /* Since THETSS = 0, the explicit source can only be a
               function of ym_w(n).
               However, this dependency has already been accounted for in
               the cs_lagr_sde_model.c
               subroutine, since the mass transfer rate has been calculated as
               a function of the local humidity (and thus ym_w)*/
            // FIXME it is kept as it was coded but the mass source terms must
            // be written on mass equation and not on ym_w
            exp_st[cell_id] += lag_st_m[cell_id];
          }
        }
      }
    }
  }

  /* Source terms for rain drops velocity
   * ==================================== */

  if (ct_opt->solve_rain_velocity) {
    /* Drops terminal relative velocity fields */
    int class_id = 1;
    char vg_lim_name[80];
    sprintf(vg_lim_name, "vg_lim_p_%02d", class_id);
    // cs_real_3_t *vg_lim_p = (cs_real_3_t *)cs_field_by_name(vg_lim_name)->val;

    /* Droplets velocity relaxation time */
    // cs_field_t *cfld_taup = cs_field_by_name_try("ym_l_r_drift_tau");
    // cs_real_t *cpro_taup = nullptr;
    // if (cfld_taup != nullptr)
    //   cpro_taup = cfld_taup->val;

    /* Continuous phase drift velocity */
    // cs_real_3_t *vd_c = (cs_real_3_t *)cs_field_by_name("vd_c")->val;

    char f_name[80];
    sprintf(f_name, "v_p_%02d", class_id);
    /* Rain drops velocity field */
    cs_field_t *f_vp = cs_field_by_name(f_name);

    /* Gravity */
    const cs_real_t *gravity = cs_glob_physical_constants->gravity;

    if (f_id == f_vp->id) {
      /* Casting implicit source term on a 3x3 symmetric matrix */
      // cs_real_33_t *_imp_st = (cs_real_33_t *)imp_st;
      cs_real_3_t *_exp_st = (cs_real_3_t *)exp_st;

      /* Rain drops velocity */
      cs_real_3_t *vp = (cs_real_3_t *)f_vp->val;

      /* Continuous phase velocity */
      cs_real_3_t *v_c = nullptr;
      if (ct_opt->mixture_model)
        v_c = (cs_real_3_t *)cs_field_by_name("v_c")->val;
      else
        v_c = (cs_real_3_t *)CS_F_(vel)->val;

      /* Bulk mixture velocity */
      // const cs_real_3_t *vel = (const cs_real_3_t *)CS_F_(vel)->val;

      /* Rain mass fraction field */
      // cs_real_t *y_rain = (cs_real_t *)cfld_yp->val;

      for (cs_lnum_t cell_id = 0; cell_id < m->n_cells; cell_id++) {

        /* Air / droplets interfacial area density calculation */
        cs_real_t ai_o_yp = 6.0 * rho_h[cell_id] / rho_l
                                * (1.0 - vol_f_r[cell_id])
                                / droplet_diam;

        /* Relative velocity calculation */
        cs_real_t u_rel[] = {vp[cell_id][0] - v_c[cell_id][0],
                             vp[cell_id][1] - v_c[cell_id][1],
                             vp[cell_id][2] - v_c[cell_id][2]};

        /* Rain velocity calculation */
        for (cs_lnum_t i = 0; i < 3; i++) {
          /* Explicit source term */
          /* Gravity term */
          _exp_st[cell_id][i] +=   rho_h[cell_id] * gravity[i]
                                 * cell_f_vol[cell_id];

          /* Drag term*/
          cs_real_t re_p =   rho_h[cell_id] * cs_math_3_norm(u_rel)
                           * droplet_diam / visc;
          cs_real_t cd = 0.;
          if (re_p > 1.e-9) {
            cd = (24./ re_p) * (1. + 0.15 * pow(re_p, 0.687));
          }

          _exp_st[cell_id][i] += - 1./8. * ai_o_yp * rho_h[cell_id] * cd
                                 * cs_math_3_norm(u_rel) * u_rel[i]
                                 * cell_f_vol[cell_id];
        }
      }
    }
    /* If mixture model not activated, humid air velocity equation must
     * have source terms for interfacial friction with rain drops */
    if (!(ct_opt->mixture_model)) {
      /* Interfacial pressure drop due to air / rain friction */
      if (f_id == (CS_F_(vel)->id)) {
        /* Casting implicit source term on a 3x3 symmetric matrix */
        cs_real_33_t *_imp_st = (cs_real_33_t *)imp_st;
        cs_real_3_t *_exp_st = (cs_real_3_t *)exp_st;

        /* Rain mass fraction field */
        cs_real_t *y_rain = (cs_real_t *)cfld_yp->val;
        /* Rain drift and velocity fields */
        sprintf(f_name, "vd_p_%02d", class_id);
        cs_real_3_t *cfld_drift
          = (cs_real_3_t *)cs_field_by_composite_name
                             (cfld_yp->name, "drift_vel")->val;
        cs_real_3_t *vp = (cs_real_3_t *)f_vp->val;

        /* Gravity norm */
        cs_real_t g = cs_math_3_norm(cs_glob_physical_constants->gravity);
        for (cs_lnum_t cell_id = 0; cell_id < m->n_cells; cell_id++) {
          if (y_rain[cell_id] > 0.) {
            /* Droplet drift and absolute velocity */
            cs_real_t drift = cs_math_3_norm(cfld_drift[cell_id]);
            cs_real_t v_drop = cs_math_3_norm(vp[cell_id]);

            /* Droplet Reynolds and Eotvos number */
            cs_real_t re_d = rho_h[cell_id] * drift * droplet_diam / visc;
            cs_real_t e_o = g * droplet_diam * (rho_l - rho_h[cell_id]) / sigma;
            /* Sphere drag coefficient */
            cs_real_t cd = 0.;
            if (re_d > 0.) {
              cd = (24. / re_d) * (1. + 0.15 * pow(re_d, 0.685));

              /* Droplet terminal velocity */
              cs_real_t v_term = sqrt((4. * rho_l * droplet_diam * g)
                                      / (3. * cd * rho_h[cell_id]));
              /* Droplet deformation / elongation */
              cs_real_t e_tau = 1. / (1. + 0.148 * pow(e_o, 0.85));
              //FIXME : check positivity of E
              cs_real_t e =   1. - cs_math_pow2(cs::min(v_drop / v_term, 1.))
                            * (1. - e_tau);

              /* Total drag coefficient for deformed drop */
              cs_real_t cd_tot = cd * (1. - 0.17185 * (1. - e)
                                       + 6.692 * cs_math_pow2(1. - e)
                                       - 6.605 * cs_math_pow3(1. - e));

              /* Air / droplets interfacial area density calculation */
              cs_real_t vol_frac_rain = y_rain[cell_id] * rho_h[cell_id] / rho_l;
              if (vol_frac_rain >= 1.0)
                vol_frac_rain = 1.0;
              cs_real_t a_i =  6.0 * vol_frac_rain * (1.0 - vol_frac_rain)
                              / droplet_diam;

              /* Droplet relaxation time */
              cs_real_t tau_d = rho_l * cs_math_pow2(droplet_diam)
                                      / (18. * visc);
              /* Final head loss coefficient */
              cs_real_t k_drop = rho_l * (cd_tot * re_d / 24.)
                                       * droplet_diam * a_i
                                       / (6. * tau_d);
              for (int k = 0; k < 3; k++) {
                _imp_st[cell_id][k][k] += -cell_f_vol[cell_id] * k_drop;
                _exp_st[cell_id][k] +=   cell_f_vol[cell_id] * k_drop
                                       * vp[cell_id][k];
              }
            }
          }
        }
      }
    }

  } /* End of solve_rain variable check */

  CS_FREE(packing_cell);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
