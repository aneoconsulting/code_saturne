#ifndef __CS_CDOFB_MONOLITHIC_H__
#define __CS_CDOFB_MONOLITHIC_H__

/*============================================================================
 * Build an algebraic CDO face-based system for the Navier-Stokes equations
 * and solved it as one block (monolithic approach of the velocity-pressure
 * coupling)
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

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"
#include "cdo/cs_cdo_connect.h"
#include "cdo/cs_cdo_quantities.h"
#include "cdo/cs_equation.h"
#include "mesh/cs_mesh.h"
#include "cdo/cs_navsto_param.h"
#include "base/cs_time_step.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*============================================================================
 * Static inline public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Retrieve the values of the velocity on the faces
 *
 * \param[in] previous        retrieve the previous state (true/false)
 *
 * \return a pointer to an array of \ref cs_real_t
 */
/*----------------------------------------------------------------------------*/

inline static cs_real_t *
cs_cdofb_monolithic_get_face_velocity(bool     previous)
{
  return cs_equation_get_face_values(cs_equation_by_name("momentum"),
                                     previous);
}

/*============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set shared pointers from the main domain members
 *
 * \param[in] mom_eqp    equation parameter settings
 * \param[in] mesh       pointer to a cs_mesh_t structure
 * \param[in] quant      additional mesh quantities struct.
 * \param[in] connect    pointer to a \ref cs_cdo_connect_t struct.
 * \param[in] time_step  pointer to a \ref cs_time_step_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic_init_sharing(const cs_equation_param_t  *mom_eqp,
                                 const cs_mesh_t            *mesh,
                                 const cs_cdo_quantities_t  *quant,
                                 const cs_cdo_connect_t     *connect,
                                 const cs_time_step_t       *time_step);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Free shared pointers with lifecycle dedicated to this file
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic_finalize_common(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize a \ref cs_cdofb_monolithic_t structure
 *
 * \param[in] nsp          pointer to a \ref cs_navsto_param_t structure
 * \param[in] adv_field    pointer to \ref cs_adv_field_t structure
 * \param[in] mflux        current values of the mass flux across primal faces
 * \param[in] mflux_pre    current values of the mass flux across primal faces
 * \param[in] bf_type      type of boundary for each boundary face
 * \param[in] cc_context   pointer to a \ref cs_navsto_monolithic_t structure
 *
 * \return a pointer to a new allocated \ref cs_cdofb_monolithic_t structure
 */
/*----------------------------------------------------------------------------*/

void *
cs_cdofb_monolithic_init_scheme_context(const cs_navsto_param_t  *nsp,
                                        cs_adv_field_t           *adv_field,
                                        cs_real_t                *mflux,
                                        cs_real_t                *mflux_pre,
                                        cs_boundary_type_t       *bf_type,
                                        void                     *cc_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Destroy a \ref cs_cdofb_monolithic_t structure
 *
 * \param[in] scheme_context   pointer to a scheme context structure to free
 *
 * \return a null pointer
 */
/*----------------------------------------------------------------------------*/

void *
cs_cdofb_monolithic_free_scheme_context(void   *scheme_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Solve the steady Navier-Stokes system with a CDO face-based scheme
 *        using a monolithic approach.
 *
 * \param[in] mesh            pointer to a \ref cs_mesh_t structure
 * \param[in] nsp             pointer to a \ref cs_navsto_param_t structure
 * \param[in] scheme_context  pointer to a structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic_steady(const cs_mesh_t            *mesh,
                           const cs_navsto_param_t    *nsp,
                           void                       *scheme_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Solve the steady Navier-Stokes system with a CDO face-based scheme
 *        using a monolithic approach and Picard iterations to solve the
 *        non-linearities arising from the advection term
 *
 * \param[in]      mesh            pointer to a \ref cs_mesh_t structure
 * \param[in]      nsp             pointer to a \ref cs_navsto_param_t structure
 * \param[in, out] scheme_context  pointer to a structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic_steady_nl(const cs_mesh_t           *mesh,
                              const cs_navsto_param_t   *nsp,
                              void                      *scheme_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Solve the unsteady Navier-Stokes system with a CDO face-based scheme
 *        using a monolithic approach.
 *        According to the settings, this function can handle either an
 *        implicit Euler time scheme or more generally a theta time scheme.
 *
 * \param[in] mesh            pointer to a \ref cs_mesh_t structure
 * \param[in] nsp             pointer to a \ref cs_navsto_param_t structure
 * \param[in] scheme_context  pointer to a structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic(const cs_mesh_t          *mesh,
                    const cs_navsto_param_t  *nsp,
                    void                     *scheme_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Solve the unsteady Navier-Stokes system with a CDO face-based scheme
 *        using a monolithic approach.
 *        According to the settings, this function can handle either an
 *        implicit Euler time scheme or more generally a theta time scheme.
 *        Rely on Picard iterations to solve the non-linearities arising from
 *        the advection term
 *
 * \param[in]      mesh            pointer to a \ref cs_mesh_t structure
 * \param[in]      nsp             pointer to a \ref cs_navsto_param_t structure
 * \param[in, out] scheme_context  pointer to a structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_monolithic_nl(const cs_mesh_t           *mesh,
                       const cs_navsto_param_t   *nsp,
                       void                      *scheme_context);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_CDOFB_MONOLITHIC_H__ */
