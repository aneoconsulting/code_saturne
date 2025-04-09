/*============================================================================
 * Log field and other array statistics at relevant time steps.
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_ale.h"
#include "atmo/cs_atmo.h"
#include "atmo/cs_atmo_chemistry.h"
#include "base/cs_base.h"
#include "base/cs_boundary.h"
#include "base/cs_boundary_zone.h"
#include "cfbl/cs_cf_model.h"
#include "ctwr/cs_ctwr.h"
#include "comb/cs_coal.h"
#include "cogz/cs_combustion_gas.h"
#include "cdo/cs_domain.h"
#include "base/cs_fan.h"
#include "base/cs_field.h"
#include "base/cs_function.h"
#include "base/cs_log.h"
#include "base/cs_log_iteration.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_mobile_structures.h"
#include "base/cs_notebook.h"
#include "base/cs_parameters.h"
#include "base/cs_physical_constants.h"
#include "base/cs_restart.h"
#include "alge/cs_sles.h"
#include "alge/cs_sles_default.h"
#include "base/cs_syr_coupling.h"
#include "base/cs_thermal_model.h"
#include "base/cs_time_moment.h"
#include "base/cs_turbomachinery.h"
#include "rayt/cs_rad_transfer_options.h"
#include "base/cs_rotation.h"
#include "turb/cs_turbulence_model.h"
#include "lagr/cs_lagr_log.h"
#include "base/cs_velocity_pressure.h"
#include "base/cs_vof.h"
#include "base/cs_volume_zone.h"
#include "base/cs_wall_distance.h"

/*----------------------------------------------------------------------------
 * Header for the current file
 *----------------------------------------------------------------------------*/

#include "base/cs_log_setup.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_log_setup.cpp

  \brief Setup info at the end of the setup stage.
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Type and macro definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Log various global model options.
 *----------------------------------------------------------------------------*/

static void
_log_error_estimators(void)
{
  int ee_count = 0;

  const char *name[] = {"est_error_pre_2",
                        "est_error_der_2",
                        "est_error_cor_2",
                        "est_error_tot_2"};

  const char *desc[] = {"prediction",
                        "drift",
                        "correction",
                        "total"};

  for (int i = 0; i < 4; i++) {

    const cs_field_t *f = cs_field_by_name_try(name[i]);
    if (f != nullptr) {
      if (ee_count == 0)
        cs_log_printf(CS_LOG_SETUP,
                      _("\n"
                        "Error estimators for Navier-Stokes\n"
                        "----------------------------------\n\n"));

      cs_log_printf(CS_LOG_SETUP,
                    _("  %s: %s\n"), name[i], desc[i]);

      ee_count += 1;
    }
  }
}

/*----------------------------------------------------------------------------
 * Log various global model options.
 *----------------------------------------------------------------------------*/

static void
_log_global_model_options(void)
{
  /* Mesh quantity options */

  cs_mesh_quantities_log_setup();

  /* Notebook parameters */

  cs_notebook_log_setup();

  cs_log_printf(CS_LOG_SETUP,
                _("\n"
                  "Physical model options\n"
                  "----------------------\n"));

  /* Physical properties */

  cs_physical_constants_log_setup();
  cs_fluid_properties_log_setup();

  /* Thermal model */

  cs_thermal_model_log_setup();

  /* Turbulence */

  cs_turb_model_log_setup();
  cs_turb_constants_log_setup();

  /* Time discretization */

  cs_time_step_log_setup();
  cs_time_scheme_log_setup();

  cs_log_iteration_log_setup();

  /* Velocity-pressure coupling */

  cs_velocity_pressure_model_log_setup();
  cs_velocity_pressure_param_log_setup();

  _log_error_estimators();

  /* Compressible model */

  cs_cf_model_log_setup();

  /* Atmospheric */

  cs_atmo_log_setup();

  /* Atmospheric chemistry */

  cs_atmo_chemistry_log_setup();

  /* Atmospheric aerosols */

  cs_atmo_aerosol_log_setup();

  /* VoF and cavitation */

  cs_vof_log_setup();

  /* Combustion */

  cs_combustion_gas_log_setup();
  cs_combustion_coal_log_setup();

  /* TODO: iroext, etc... */

  /* Face viscosity */

  cs_space_disc_log_setup();

  /* Wall distance computation mode */

  if (cs_glob_wall_distance_options->need_compute) {
    cs_log_printf(CS_LOG_SETUP, _("\nWall distance computation\n"
                                  "---------------------------\n\n"));
    int method = cs_glob_wall_distance_options->method;
    cs_log_printf(CS_LOG_SETUP,
                  _("  method: %d"), method);
    switch(method) {
    case 1:
      cs_log_printf(CS_LOG_SETUP,
                    _(" (based on diffusion equation)"));
      break;
    case 2:
      cs_log_printf(CS_LOG_SETUP,
                    _(" (brute force, serial only)"));
      break;
    }
    cs_log_printf(CS_LOG_SETUP, "\n");
  }

  /* ALE */

  cs_ale_log_setup();

  if (cs_glob_ale != CS_ALE_NONE)
    cs_mobile_structures_log_setup();

  /* Rotation info */

  if (cs_turbomachinery_get_model() == CS_TURBOMACHINERY_NONE) {
    const cs_rotation_t  *r = cs_glob_rotation;

    cs_log_printf(CS_LOG_SETUP, _("\nSubdomain rotation\n"
                                  "------------------\n\n"));

    cs_log_printf(CS_LOG_SETUP,
                  _("  Global domain rotation:\n"
                    "    axis:             [%g, %g, %g]\n"
                    "    invariant point:  [%g, %g, %g]\n"
                    "    angular velocity:  %g radians/s\n"),
                  r->axis[0], r->axis[1], r->axis[2],
                  r->invariant[0], r->invariant[1], r->invariant[2],
                  r->omega);

  }

  cs_syr_coupling_log_setup();

  /* Zone information */

  cs_volume_zone_log_setup();
  cs_boundary_zone_log_setup();

  /* BC information */

  cs_boundary_log_setup(cs_glob_domain->boundaries);
  cs_boundary_log_setup(cs_glob_domain->ale_boundaries);
}

/*============================================================================
 * Fortran wrapper function definitions
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Log setup options and define the default setup for SLES.
 */
/*----------------------------------------------------------------------------*/

void
cs_log_setup(void)
{
  cs_field_log_defs();
  cs_field_log_key_defs();
  cs_field_log_all_key_vals(false);

  cs_time_moment_log_setup();

  cs_function_log_defs();
  cs_function_log_all_settings();

  cs_sles_default_setup();

  cs_restart_log_setup();
  cs_log_printf(CS_LOG_SETUP,
                _("  read auxiliary:       %d\n"),
                cs_glob_restart_auxiliary->read_auxiliary);
  cs_log_printf(CS_LOG_SETUP,
                _("  write auxiliary:      %d\n"),
                cs_glob_restart_auxiliary->write_auxiliary);

  _log_global_model_options();

  cs_rad_transfer_log_setup();

  cs_lagr_log_setup();

  cs_fan_log_setup();

  cs_ctwr_log_setup();

  cs_log_printf_flush(CS_LOG_SETUP);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
