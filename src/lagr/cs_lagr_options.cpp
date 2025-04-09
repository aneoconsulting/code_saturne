/*============================================================================
 * Lagrangian module options setting
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
  Street, Fifth Floor, Boston, MA 02110-1301, USA. */

/*----------------------------------------------------------------------------*/

/*============================================================================
 * Functions dealing with Lagrangian module options
 *============================================================================*/

#include "base/cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <stdio.h>
#include <assert.h>
#include <string.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_mem.h"
#include "bft/bft_printf.h"

#include "base/cs_base.h"
#include "base/cs_field.h"
#include "base/cs_file.h"
#include "gui/cs_gui_particles.h"
#include "gui/cs_gui_util.h"
#include "mesh/cs_mesh_location.h"
#include "base/cs_parameters.h"
#include "base/cs_parameters_check.h"
#include "pprt/cs_physical_model.h"
#include "turb/cs_turbulence_model.h"
#include "base/cs_volume_zone.h"

#include "lagr/cs_lagr.h"
#include "lagr/cs_lagr_event.h"
#include "lagr/cs_lagr_particle.h"
#include "lagr/cs_lagr_post.h"
#include "lagr/cs_lagr_prototypes.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "lagr/cs_lagr_options.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro definitions
 *============================================================================*/

/*============================================================================
 * Local type definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*=============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Create source term fields for lagrangian module
 *
 * \param[in]  name            source term field name
 * \param[in]  dim             source term field dimension
 * \param[in]  have_previous   source term field dimension
 */
/*----------------------------------------------------------------------------*/

static void
_define_st_field(const char  *name,
                 int          dim,
                 bool         have_previous)
{
  int field_type = CS_FIELD_INTENSIVE | CS_FIELD_PROPERTY;
  int location_id = CS_MESH_LOCATION_CELLS;

  cs_field_create(name,
                  field_type,
                  location_id,
                  dim,
                  have_previous);
}

/*-----------------------------------------------------------------------------
 * Copy a variable name to the boundary variable names array
 *
 * parameters:
 *   ipp     <-- index from the fortran array associated to varname
 *   varname <-- name or label of the variable/scalar/property
 *----------------------------------------------------------------------------*/

static void
_copy_boundary_varname(int          ipp,
                       const char  *varname)
{
  size_t  l;
  assert(ipp >= 0);

  int nvplmx = 50+4*cs_glob_lagr_const_dim->nlayer;

  if (cs_glob_lagr_boundary_interactions->nombrd == nullptr) {

    CS_MALLOC(cs_glob_lagr_boundary_interactions->nombrd,
              nvplmx,
              char *);
    for (int i = 0; i < nvplmx; i++)
      cs_glob_lagr_boundary_interactions->nombrd[i] = nullptr;
  }

  l = strlen(varname);

  CS_REALLOC(cs_glob_lagr_boundary_interactions->nombrd[ipp], l + 1, char);

  strcpy(cs_glob_lagr_boundary_interactions->nombrd[ipp], varname);
}

/*----------------------------------------------------------------------------
 * Initialize Encrustation pointers.
 *----------------------------------------------------------------------------*/

static void
_init_lagr_encrustation_pointers(void)
{
  if (cs_glob_lagr_encrustation->enc1 == nullptr)
    CS_MALLOC(cs_glob_lagr_encrustation->enc1,
              cs_glob_lagr_const_dim->ncharm2,
              cs_real_t);
  if (cs_glob_lagr_encrustation->enc2 == nullptr)
    CS_MALLOC(cs_glob_lagr_encrustation->enc2,
              cs_glob_lagr_const_dim->ncharm2,
              cs_real_t);
  if (cs_glob_lagr_encrustation->tprenc == nullptr)
    CS_MALLOC(cs_glob_lagr_encrustation->tprenc,
              cs_glob_lagr_const_dim->ncharm2,
              cs_real_t);
  if (cs_glob_lagr_encrustation->visref == nullptr)
    CS_MALLOC(cs_glob_lagr_encrustation->visref,
              cs_glob_lagr_const_dim->ncharm2,
              cs_real_t);

  for (int icha = 0; icha < cs_glob_lagr_const_dim->ncharm2; icha++) {

    cs_glob_lagr_encrustation->tprenc[icha] = -999.0;
    cs_glob_lagr_encrustation->visref[icha] = -999.0;
    cs_glob_lagr_encrustation->enc1[icha] = -999.0;
    cs_glob_lagr_encrustation->enc2[icha] = -999.0;

  }
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Lagrangian module options definition.
 *
 * - default initialization
 * - read user settings
 * - check settings coherency
 * - initialize some structures relative to Lagrangian module
 *
 * \param[in]       is_restart
 * \param[in]       have_thermal_model
 * \param[in]       dtref
 * \param[in, out]  iccvfg
 */
/*----------------------------------------------------------------------------*/

void
cs_lagr_options_definition(int         is_restart,
                           int         have_thermal_model,
                           cs_real_t   dtref,
                           int        *iccvfg)
{
  /* Short-name and write access pointers to global variables */

  const cs_lagr_const_dim_t *const_dim = cs_glob_lagr_const_dim;

  cs_lagr_model_t *lagr_model = cs_glob_lagr_model;
  cs_lagr_time_scheme_t * lagr_time_scheme = cs_glob_lagr_time_scheme;
  cs_lagr_extra_module_t *extra_i = cs_glob_lagr_extra_module;
  cs_lagr_extra_module_t *extra = extra_i;
  cs_lagr_dim_t *lagdim = cs_glob_lagr_dim;

  /* Default initializations for Lagrangian module. */

  lagr_time_scheme->iilagr = CS_LAGR_OFF;
  lagr_time_scheme->isuila = 0;

  cs_glob_lagr_stat_options->isuist = 1;

  lagr_model->physical_model = CS_LAGR_PHYS_OFF;

  cs_glob_lagr_specific_physics->solve_diameter = 0;

  cs_glob_lagr_specific_physics->solve_temperature_seen = 0;

  cs_glob_lagr_specific_physics->solve_temperature = 0;

  cs_glob_lagr_specific_physics->solve_mass = 0;

  cs_glob_lagr_specific_physics->tpart = -999.0;

  cs_glob_lagr_specific_physics->cppart = -999.0;

  lagr_model->fouling = 0;

  /* Initializations for physical models */
  _init_lagr_encrustation_pointers();

  lagr_time_scheme->isttio = 0;

  cs_glob_lagr_source_terms->nstits = 1;
  cs_glob_lagr_source_terms->ltsdyn = 0;
  cs_glob_lagr_source_terms->ltsmas = 0;
  cs_glob_lagr_source_terms->ltsthe = 0;

  cs_glob_lagr_boundary_interactions->nombrd = nullptr;

  lagr_time_scheme->t_order = 2;
  lagr_model->idistu = -1;
  lagr_model->idiffl = -1;
  lagr_time_scheme->ilapoi = 0;
  lagr_time_scheme->iadded_mass = 0;
  lagr_time_scheme->added_mass_const = 1.0;

  cs_glob_lagr_boundary_interactions->has_part_impact_nbr = 0;

  /* User setup
     ---------- */

  cs_gui_particles_model();

  cs_user_lagr_model();

  if (lagr_time_scheme->iilagr == CS_LAGR_OFF) {

    cs_lagr_finalize();

    return;
  }

  /* Check user initializations of Lagrangian module
     ----------------------------------------------- */

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_time_scheme->iilagr",
                                lagr_time_scheme->iilagr,
                                CS_LAGR_OFF, CS_LAGR_FROZEN_CONTINUOUS_PHASE + 1);

  /* Restart needed if computation on frozen field.
     Note that for the Lagrangian module, frozen field also includes scalars. */

  if (   lagr_time_scheme->iilagr == CS_LAGR_FROZEN_CONTINUOUS_PHASE
      && is_restart != 1)
    cs_parameters_error
      (CS_ABORT_DELAYED,
       _("in Lagrangian module"),
       _("The specified Lagrangian time scheme requires frozen fields\n"
         "(cs_glob_lagr_time_scheme->iilagr == %d)\n"
         "but the background Eulerian computation is not a restart.\n"),
       lagr_time_scheme->iilagr);

  if (lagr_time_scheme->iilagr == CS_LAGR_FROZEN_CONTINUOUS_PHASE)
    *iccvfg = 1;

  if (lagr_time_scheme->iilagr == CS_LAGR_TWOWAY_COUPLING
      && (cs_glob_time_step->is_local || cs_glob_time_step->is_variable))
    cs_parameters_error
      (CS_ABORT_DELAYED,
       _("in Lagrangian module"),
       _("The two-way coupling model is incompatible with a\n"
         "local or variable time step.\n"));

  if (lagr_time_scheme->isuila < 0)
    lagr_time_scheme->isuila = 0;
  else if (lagr_time_scheme->isuila > 1)
    lagr_time_scheme->isuila = 1;

  if (lagr_time_scheme->isuila == 1 && is_restart == 0)
    lagr_time_scheme->isuila = 0;

  if (cs_glob_lagr_stat_options->isuist < 0)
    cs_glob_lagr_stat_options->isuist = 0;

  if (lagr_time_scheme->isuila == 1) {
    if (cs_glob_lagr_stat_options->isuist > 1)
      cs_glob_lagr_stat_options->isuist = 1;
  }
  else
    cs_glob_lagr_stat_options->isuist = 0;

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_model->physical_model",
                                lagr_model->physical_model,
                                0, 4);

  cs_parameters_error_barrier();

  /* solve_diameter solve_temperature solve_mass
   * Return coupling only towards continuous phase */

  if (lagr_model->physical_model == CS_LAGR_PHYS_HEAT) {
    /* If the particle temperature is computed so does the temperature seen */
    if (cs_glob_lagr_specific_physics->solve_temperature > 0)
      cs_glob_lagr_specific_physics->solve_temperature_seen = 1;

    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_specific_physics->solve_diameter",
                                  cs_glob_lagr_specific_physics->solve_diameter,
                                  0, 2);
    int resol_temp_seen = cs_glob_lagr_specific_physics->solve_temperature_seen;
    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_specific_physics->"
                                    "solve_temperature_seen",
                                  resol_temp_seen,
                                  0, 2);
    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_specific_physics->solve_temperature",
                                  cs_glob_lagr_specific_physics->solve_temperature,
                                  0, 2);
    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_specific_physics->solve_mass",
                                  cs_glob_lagr_specific_physics->solve_mass,
                                  0, 2);

    if (   cs_glob_lagr_specific_physics->solve_temperature == 1
        && have_thermal_model == 0)
      cs_parameters_error
        (CS_ABORT_DELAYED,
         _("in Lagrangian module"),
         _("The resolution of the particles temperature is activated\n"
           "(cs_glob_lagr_specific_physics->solve_temperature == %d)\n"
           "but the background Eulerian computation has no thermal scalar."),
         cs_glob_lagr_specific_physics->solve_temperature);

  }
  else {

    cs_glob_lagr_specific_physics->solve_temperature_seen = 0;
    cs_glob_lagr_specific_physics->solve_temperature = 0;
    cs_glob_lagr_specific_physics->solve_mass = 0;
    cs_glob_lagr_specific_physics->solve_diameter = 0;

  }

  if (lagr_time_scheme->isuila == 1 &&
      lagr_model->physical_model == CS_LAGR_PHYS_HEAT &&
      cs_glob_lagr_specific_physics->solve_temperature == 1) {
    cs_parameters_is_greater_double(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_specific_physics->cppart",
                                    cs_glob_lagr_specific_physics->cppart,
                                    0);

    cs_parameters_is_greater_double(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_specific_physics->tpart",
                                    cs_glob_lagr_specific_physics->tpart,
                                    -273.15);

  }

  cs_parameters_error_barrier();

  if (lagr_model->physical_model == CS_LAGR_PHYS_COAL) {

    if (lagr_time_scheme->t_order == 2) {
      lagr_time_scheme->t_order = 1;
      cs_parameters_error
        (CS_WARNING,
         _("in Lagrangian module"),
         _("Lagrangian transport of coal particles is not implemented in\n"
           "second-order integration scheme, "
           "so first-order scheme will be used.\n"));
    }

    if (cs_glob_lagr_source_terms->ltsthe == 1)
      cs_parameters_error
        (CS_ABORT_DELAYED,
         _("in Lagrangian module"),
         _("Lagrangian transport of coal particles is not implemented with\n"
           "thermal return coupling (cs_glob_lagr_source_terms->ltsthe = %d)\n"),
         cs_glob_lagr_source_terms->ltsthe);

    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_model->fouling",
                                  lagr_model->fouling,
                                  0, 2);

    if (lagr_model->fouling == 1) {

      for (int icha = 0; icha < extra->ncharb; icha++) {
        if (cs_glob_lagr_encrustation->visref[icha] <= 0)
          cs_parameters_error
            (CS_ABORT_DELAYED,
             _("in Lagrangian module"),
             _("Particle fouling is active (lagr_model->fouling = %d)\n"
               "with an incorrect critical viscosity for coal %d.\n"
               "cs_glob_lagr_encrustation->visref[%d] = %g "
               "but should be > 0.\n"),
             lagr_model->fouling, icha, icha,
             cs_glob_lagr_encrustation->visref[icha]);

        if (cs_glob_lagr_encrustation->tprenc[icha] < 150.)
          cs_parameters_error
            (CS_ABORT_DELAYED,
             _("in Lagrangian module"),
             _("Particle fouling is active (lagr_model->fouling = %d)\n"
               "with an incorrect temperature threshold for coal %d.\n"
               "cs_glob_lagr_encrustation->tprenc[%d] = %g degrees Celcius\n"
               "but should be > %g.\n"),
             lagr_model->fouling, icha, icha,
             cs_glob_lagr_encrustation->tprenc[icha], 150.);
      }
    }

    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "const_dim->nlayer",
                                  const_dim->nlayer,
                                  1, 99);
  }
  else
    lagr_model->fouling = 0;

  if (   lagr_model->physical_model == CS_LAGR_PHYS_COAL
      && cs_glob_physical_model_flag[CS_COMBUSTION_COAL] < 0)
    cs_parameters_error
      (CS_ABORT_DELAYED,
       _("in Lagrangian module"),
       _("Coal particle transport is activated "
         "(lagr_model->physical_model = %d)\n"
         "but the matching model coupling is not active:\n"
         " cs_glob_physical_model_flag[CS_COMBUSTION_COAL] = %d\n"),
       lagr_model->physical_model,
       cs_glob_physical_model_flag[CS_COMBUSTION_COAL]);

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_time_scheme->isuila",
                                lagr_time_scheme->isuila,
                                0, 2);

  if (cs_glob_lagr_stat_options->isuist > 0) {
    if (cs_glob_time_step->nt_prev > 0) {
      if (!cs_file_isreg("restart/lagrangian_stats")) {
        cs_parameter_error_behavior_t err_type = CS_WARNING;
        if (cs_glob_lagr_stat_options->isuist > 1) {
          err_type = CS_ABORT_DELAYED;
          cs_parameters_error
            (err_type,
             _("in Lagrangian module"),
             _("Restart of lagrangian statistics and source terms is requested\n"
               "(cs_glob_lagr_stat_options->isuist = %d), but matching file\n"
               "is not present in the checkpoint.\n"),
           cs_glob_lagr_stat_options->isuist);
        }
        else { /* isuist = 1 allows reset */
          cs_glob_lagr_stat_options->isuist = 0;
          bft_printf(_("\nReset statitics and source terms.\n"));
        }
      }
    }
  }
  if (cs_glob_lagr_stat_options->isuist == 0) {
    if (cs_glob_time_step->nt_prev >= cs_glob_lagr_stat_options->idstnt)
      cs_glob_lagr_stat_options->idstnt = cs_glob_time_step->nt_prev + 1;
    if (cs_glob_time_step->nt_prev >= cs_glob_lagr_stat_options->nstist)
      cs_glob_lagr_stat_options->nstist = cs_glob_time_step->nt_prev + 1;
  }

  cs_parameters_error_barrier();

  /* ISTTIO NSTITS LTSDYN LTSMAS LTSTHE  */

  if (lagr_time_scheme->iilagr == CS_LAGR_FROZEN_CONTINUOUS_PHASE)
    lagr_time_scheme->isttio = 1;

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_time_scheme->isttio",
                                lagr_time_scheme->isttio,
                                0, 2);

  if (lagr_time_scheme->iilagr == CS_LAGR_TWOWAY_COUPLING) {

    if (   lagr_time_scheme->isttio == 1
        && cs_glob_lagr_source_terms->nstits < 1)
      cs_glob_lagr_source_terms->nstits = 1;

    cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                  _("in Lagrangian module"),
                                  "cs_glob_lagr_source_terms->ltsdyn",
                                  cs_glob_lagr_source_terms->ltsdyn,
                                  0, 2);

    if (     lagr_model->physical_model == CS_LAGR_PHYS_HEAT
        && (   cs_glob_lagr_specific_physics->solve_mass == 1
            || cs_glob_lagr_specific_physics->solve_diameter == 1)) {
      cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_source_terms->ltsmas",
                                    cs_glob_lagr_source_terms->ltsmas,
                                    0, 2);
    }

    else if (lagr_model->physical_model == CS_LAGR_PHYS_CTWR) {
      cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_source_terms->ltsmas",
                                    cs_glob_lagr_source_terms->ltsmas,
                                    0, 2);

    }
    else
      cs_glob_lagr_source_terms->ltsmas = 0;

    if (   (   lagr_model->physical_model == CS_LAGR_PHYS_HEAT
            && cs_glob_lagr_specific_physics->solve_temperature == 1)
        || lagr_model->physical_model == CS_LAGR_PHYS_COAL) {

      cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_source_terms->ltsthe",
                                    cs_glob_lagr_source_terms->ltsthe,
                                    0, 2);

    }

    else if (   lagr_model->physical_model == CS_LAGR_PHYS_CTWR) {

      cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                    _("in Lagrangian module"),
                                    "cs_glob_lagr_source_terms->ltsthe",
                                    cs_glob_lagr_source_terms->ltsthe,
                                    0, 2);

    }
    else
      cs_glob_lagr_source_terms->ltsthe = 0;

    if (cs_glob_lagr_source_terms->ltsdyn == 1 && *iccvfg == 1)
      cs_parameters_error
        (CS_ABORT_DELAYED,
         _("in Lagrangian module"),
         _("The return coupling on the flow field is activated\n"
           "(cs_glob_lagr_source_terms->ltsdyn = %d)\n"
           "but the carrier field is frozen (iccvfg = %d)?\n"),
         cs_glob_lagr_source_terms->ltsdyn,
         *iccvfg);

    if (   cs_glob_lagr_source_terms->ltsdyn != 1
        && cs_glob_lagr_source_terms->ltsthe != 1
        && cs_glob_lagr_source_terms->ltsmas != 1)
      cs_parameters_error
        (CS_WARNING,
         _("in Lagrangian module"),
         _("The two-way coupling option is activated\n"
           "but all coupling sub-options are deactivated:\n"
           "  cs_glob_lagr_source_terms->ltsdyn = %d\n"
           "  cs_glob_lagr_source_terms->ltsthe = %d\n"
           "  cs_glob_lagr_source_terms->ltsmas = %d\n"),
         cs_glob_lagr_source_terms->ltsdyn,
         cs_glob_lagr_source_terms->ltsthe,
         cs_glob_lagr_source_terms->ltsmas);

  }
  else {
    cs_glob_lagr_source_terms->ltsdyn = 0;
    cs_glob_lagr_source_terms->ltsmas = 0;
    cs_glob_lagr_source_terms->ltsthe = 0;
  }

  if (cs_glob_lagr_stat_options->idstnt < 1)
    cs_glob_lagr_stat_options->idstnt = 1;

  {
    if (lagr_time_scheme->isttio == 1) {
      if (  cs_glob_lagr_stat_options->nstist
          < cs_glob_lagr_stat_options->idstnt)
        cs_glob_lagr_stat_options->idstnt = cs_glob_lagr_stat_options->nstist;
    }
  }

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_time_scheme->t_order",
                                lagr_time_scheme->t_order,
                                1, 3);

  /* Ensure complete model option has valid values */

  if (lagr_model->modcpl < 0)
    lagr_model->modcpl = 0;
  /* When activating the complete turbulent dispersion model,
   * statistics are required, so activate it after the start time of
   * statistics.
   */
  else if (lagr_model->modcpl > 0)
    lagr_model->modcpl = 1;

  /* Default diffusion model depending on complete model or fluid particles */

  /* Fluid particle: activate turbulent dispersion,
   * deactivate crossing effect */
  if (lagr_model->modcpl == 0) {

    if (lagr_model->idistu < 0)
      lagr_model->idistu = 1;
    if (lagr_model->idiffl < 0)
      lagr_model->idiffl = 1;

  }
  /* Full model: turbulent dispersion and crossing effect */
  else {

    if (lagr_model->idistu < 0)
      lagr_model->idistu = 1;
    if (lagr_model->idiffl < 0)
      lagr_model->idiffl = 1;//See Minier 2016

    /* Velocity statistics are needed for this model */
    cs_lagr_stat_activate_attr(CS_LAGR_VELOCITY);
    cs_lagr_stat_activate_attr(CS_LAGR_VELOCITY_SEEN);
    if (!cs_glob_lagr_model->cs_used)
      cs_lagr_stat_activate_attr(CS_LAGR_VELOCITY_SEEN_VELOCITY_COV);

    /* Force immediate activation of volume statistics
       (may be adjusted later based on restart time step) */
    if (cs_glob_lagr_stat_options->idstnt > 1)
      cs_glob_lagr_stat_options->idstnt = 1;

  }

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_model->idistu",
                                lagr_model->idistu,
                                0, 2);

  if (   lagr_model->idistu == 1
      && extra->itytur != 2
      && extra->itytur != 3
      && extra->itytur != 4
      && extra->itytur != 5
      && extra->iturb != CS_TURB_K_OMEGA) {
    cs_parameters_error
      (CS_ABORT_DELAYED,
       _("in Lagrangian module"),
       _("The turbulent dispersion model is not implemented for the selected\n"
         "turbulence model (%d).\n\n"
         "Only k-epsilon, LES, Rij-epsilon, v2f, and k-omega are supported."),
       extra->iturb);

  }
  else if (   lagr_model->idistu == 0
           && extra->iturb != 0
           && extra->itytur!= 2
           && extra->itytur!= 3
           && extra->itytur!= 4
           && extra->itytur!= 5
           && extra->iturb != CS_TURB_K_OMEGA) {
    cs_parameters_error
      (CS_ABORT_DELAYED,
       _("in Lagrangian module"),
       _("The Lagrangian module is not implemented for the selected\n"
         "turbulence model (%d).\n\n"
         "Only laminar, LES, k-epsilon, Rij-epsilon, v2f, and "
         "k-omega are supported."),
       extra->iturb);

  }

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_model->idiffl",
                                lagr_model->idiffl,
                                0, 2);

  cs_parameters_is_in_range_int(CS_ABORT_DELAYED,
                                _("in Lagrangian module"),
                                "cs_glob_lagr_time_scheme->ilapoi",
                                lagr_time_scheme->ilapoi,
                                0, 2);

  cs_parameters_is_in_range_int
    (CS_ABORT_DELAYED,
     _("in Lagrangian module"),
     "cs_glob_lagr_boundary_interactions->has_part_impact_nbr",
     cs_glob_lagr_boundary_interactions->has_part_impact_nbr,
     0, 2);

  cs_parameters_error_barrier();

  /* Initialization which must not be changed by the user
     ==================================================== */

  /* Lagrangian time step (by default, the continuous phase time step) */
  cs_glob_lagr_time_step->dtp = dtref;

  /* Lagrangian current physical time */
  cs_glob_lagr_time_step->ttclag = 0.0;

  /* Boundary statistics */
  cs_glob_lagr_boundary_interactions->npstf = 0;
  cs_glob_lagr_boundary_interactions->npstft = 0;
  cs_glob_lagr_boundary_interactions->tstatp = 0.0;

  /* Return coupling */
  cs_glob_lagr_source_terms->npts = 0;

  /* Definition of pointers related to boundary statistics
   * has_part_impact_nbr: activate stats on particle/boundary interaction

   * nvisbr: total number of interactions to track */

  int irf = -1;

  if (lagr_model->clogging == 1) {

    irf++;
    cs_glob_lagr_boundary_interactions->inclg = irf;
    _copy_boundary_varname(irf, "Part_deposited_number");
    irf++;
    cs_glob_lagr_boundary_interactions->inclgt = irf;
    _copy_boundary_varname(irf, "Part_deposited_part");
    irf++;
    cs_glob_lagr_boundary_interactions->iclogt = irf;
    _copy_boundary_varname(irf, "Part_deposited_time");
    irf++;
    cs_glob_lagr_boundary_interactions->iclogh = irf;
    _copy_boundary_varname(irf, "Part_consolidation_height");
    irf++;
    cs_glob_lagr_boundary_interactions->iscovc = irf;
    _copy_boundary_varname(irf, "Part_surf_coverage");
    irf++;
    cs_glob_lagr_boundary_interactions->ihdepm = irf;
    _copy_boundary_varname(irf, "Part_dep_height_mean");
    irf++;
    cs_glob_lagr_boundary_interactions->ihdiam = irf;
    _copy_boundary_varname(irf, "Part_dep_diameter_mean");
    irf++;
    cs_glob_lagr_boundary_interactions->ihsum = irf;
    _copy_boundary_varname(irf, "Part_dep_diameter_sum");
    irf++;
    cs_glob_lagr_boundary_interactions->ihdepv = irf;
    _copy_boundary_varname(irf, "Part_dep_height_variance");

  }

  /* With the reentrainment model, all cells (i.e. the root zone)
     may involve head losses */
  if (cs_glob_lagr_reentrained_model->iflow == 1)
    cs_volume_zone_set_type(0, CS_VOLUME_ZONE_HEAD_LOSS);

  /* If there is any boundary stat, activate the number of particle impact */
  if (irf > -1)
    cs_glob_lagr_boundary_interactions->has_part_impact_nbr = 1;

  if (cs_glob_lagr_boundary_interactions->has_part_impact_nbr == 1) {
    irf++;
    cs_glob_lagr_boundary_interactions->inbr = irf;
    _copy_boundary_varname(irf, "Part_impact_number");
  }

  lagdim->n_boundary_stats = irf + 1;

  /* Definition of pointers related to Lagrangian source terms
     for return coupling. */

  irf = -1;

  /* Dynamics: velocity + turbulence */
  if (cs_glob_lagr_source_terms->ltsdyn == 1) {

    bool have_previous = (lagr_time_scheme->cell_wise_integ == 1);
    _define_st_field("lagr_st_velocity", 3, have_previous);

    _define_st_field("lagr_st_imp_velocity", 1, false);

    /* K-eps, LES, v2f and k-omega */
    if (   extra->itytur == 2
        || extra->itytur == 4
        || extra->itytur == 5
        || extra->iturb == CS_TURB_K_OMEGA)
      _define_st_field("lagr_st_k", 1, false);

    /* Rij */
    else if (extra->itytur == 3)
      _define_st_field("lagr_st_rij", 6, false);

    else
      cs_parameters_error
        (CS_ABORT_IMMEDIATE,
         _("in Lagrangian module"),
         _("The return coupling is not implemented fo the current "
           "turbulence model (%d).\n"
           "It is compatible with k-epsilon, LES, Rij-epsilon,\n"
           "v2f, and k-omega."),
         extra->iturb);

  }

  /* Deposition model */
  if (   lagr_model->deposition == 1
      && lagr_time_scheme->t_order == 2)
      cs_parameters_error
        (CS_ABORT_IMMEDIATE,
         _("in Lagrangian module"),
         _("The deposition model (Guingo & Minier, 2008) is not implemented\n"
           "with the second order scheme (%s)."),
         "cs_glob_lagr_time_scheme->t_order == 2");

  /* Mass: associated to pressure equation  */
  if (cs_glob_lagr_source_terms->ltsmas == 1)
    _define_st_field("lagr_st_pressure", 1, false);

  /* Thermal model */
  if (cs_glob_lagr_source_terms->ltsthe == 1) {

    if ((lagr_model->physical_model == CS_LAGR_PHYS_HEAT
          /* Temperature */
        && cs_glob_lagr_specific_physics->solve_temperature == 1)
       || lagr_model->physical_model == CS_LAGR_PHYS_COAL
       || lagr_model->physical_model == CS_LAGR_PHYS_CTWR
        ) {

      _define_st_field("lagr_st_temperature", 1, false);
      _define_st_field("lagr_st_imp_temperature", 1, false);

    }
  }

  /* Now define particle map */
  cs_lagr_particle_attr_initialize();
  cs_lagr_event_initialize();

  if (lagr_model->deposition > 0)
    cs_field_find_or_create("boundary_ustar",
                            CS_FIELD_INTENSIVE | CS_FIELD_PROPERTY,
                            CS_MESH_LOCATION_BOUNDARY_FACES,
                            1,
                            false); /* has previous */

  /* Now activate basic statistics */

#if 0
  if (   cs_glob_lagr_model->modcpl > 0
      || cs_glob_lagr_time_scheme->ilapoi == 1)
    cs_lagr_stat_activate(CS_LAGR_STAT_CUMULATIVE_WEIGHT);
#endif

  cs_lagr_stat_initialize();

  cs_base_at_finalize(cs_lagr_finalize);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
