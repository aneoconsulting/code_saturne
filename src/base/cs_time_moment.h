#ifndef __CS_TIME_MOMENT_H__
#define __CS_TIME_MOMENT_H__

/*============================================================================
 * Moments management
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

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"
#include "base/cs_field.h"
#include "base/cs_restart.h"
#include "base/cs_function.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/* Moment type */

typedef enum {

  CS_TIME_MOMENT_MEAN,
  CS_TIME_MOMENT_VARIANCE

} cs_time_moment_type_t;

/* Moment restart behavior */

typedef enum {

  CS_TIME_MOMENT_RESTART_RESET,
  CS_TIME_MOMENT_RESTART_AUTO,
  CS_TIME_MOMENT_RESTART_EXACT

} cs_time_moment_restart_t;

/*----------------------------------------------------------------------------
 * Function pointer for computation of data values for moments computation.
 *
 * If the matching values are multidimensional, they must be interleaved.
 *
 * Note: if the input pointer is non-null, it must point to valid data
 * when the selection function is called, so either:
 * - that value or structure should not be temporary (i.e. local);
 * - when a single integer identifier is needed, the input pointer can be
 *   set to that value instead of an actual address;
 *
 * parameters:
 *   input <-- pointer to optional (untyped) value or structure.
 *   vals  --> pointer to values (size: n_local elements*dimension)
 *----------------------------------------------------------------------------*/

typedef void
(cs_time_moment_data_t) (const void  *input,
                         cs_real_t   *vals);

/*=============================================================================
 * Global variables
 *============================================================================*/

/* Names associated with moment types */

extern const char  *cs_time_moment_type_name[];

/*============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Destroy all moments management metadata.
 *----------------------------------------------------------------------------*/

void
cs_time_moment_destroy_all(void);

/*----------------------------------------------------------------------------
 * Map time step values array for temporal moments.
 *
 * If this function is not called, the field referenced by field pointer
 * CS_F_(dt) will be used instead.
 *----------------------------------------------------------------------------*/

void
cs_time_moment_map_cell_dt(const cs_real_t  *dt);

/*----------------------------------------------------------------------------
 * Update all moment accumulators.
 *----------------------------------------------------------------------------*/

void
cs_time_moment_update_all(void);

/*----------------------------------------------------------------------------
 * Return 1 if a moment is active, 0 if it is not active yet.
 *
 * parameters:
 *   moment_id <-- id of associated moment
 *
 * returns:
 *    0 in case of success, 1 if moment accumulation has not started yet
 *----------------------------------------------------------------------------*/

int
cs_time_moment_is_active(int moment_id);

/*----------------------------------------------------------------------------
 * Define a moment of a product of existing field components
 *
 * Moments will involve the tensor products of their component fields,
 * and only scalar, vector, or rank-2 tensors are handled (for
 * post-processing output reasons), so a moment may not involve more than
 * 2 vectors or 1 tensor, unless single components are specified.
 *
 * If of dimension > 1, the moment array is always interleaved.
 *
 * parameters:
 *   name         <-- name of associated moment
 *   n_fields     <-- number of associated fields
 *   field_id     <-- ids of associated fields
 *   component_id <-- ids of matching field components (-1 for all)
 *   type         <-- moment type
 *   nt_start     <-- starting time step (or -1 to use t_start)
 *   t_start      <-- starting time
 *   restart_mode <-- behavior in case of restart (reset, auto, or strict)
 *   restart_name <-- if non-null, previous name in case of restart
 *
 * returns:
 *   id of new moment in case of success, -1 in case of error.
 *----------------------------------------------------------------------------*/

int
cs_time_moment_define_by_field_ids(const char                *name,
                                   int                        n_fields,
                                   const int                  field_id[],
                                   const int                  component_id[],
                                   cs_time_moment_type_t      type,
                                   int                        nt_start,
                                   double                     t_start,
                                   cs_time_moment_restart_t   restart_mode,
                                   const char                *restart_name);

/*----------------------------------------------------------------------------
 * Define a time moment of an existing field.
 *
 * Moments will involve the tensor products of their component fields,
 * and only scalar, vector, or rank-2 tensors are handled (for
 * post-processing output reasons), so a 1st-order moment (i.e. mean) may
 * involve a scalar, vector, or tensor, while a second-order moment
 * (i.e. variance) may only involve a scalar or vector.
 *
 * parameters:
 *   name         <-- name of associated moment
 *   f            <-- pointer to associated field
 *   type         <-- moment type
 *   nt_start     <-- starting time step (or -1 to use t_start)
 *   t_start      <-- starting time
 *   restart_mode <-- behavior in case of restart (reset, auto, or strict)
 *   restart_name <-- if non-null, previous name in case of restart
 *
 * returns:
 *   id of new moment in case of success, -1 in case of error.
 *----------------------------------------------------------------------------*/

int
cs_time_moment_define_by_field(const char                *name,
                               const cs_field_t          *f,
                               cs_time_moment_type_t      type,
                               int                        nt_start,
                               double                     t_start,
                               cs_time_moment_restart_t   restart_mode,
                               const char                *restart_name);

/*----------------------------------------------------------------------------
 * Define a time moment based on an evaluation function.
 *
 * Moments will involve the tensor products of their component fields,
 * and only scalar, vector, or rank-2 tensors are handled (for
 * post-processing output reasons), so a 1st-order moment (i.e. mean) may
 * involve a scalar, vector, or tensor, while a second-order moment
 * (i.e. variance) may only involve a scalar or vector.
 *
 * parameters:
 *   name         <-- name of associated moment
 *   f            <-- pointer to function object
 *   type         <-- moment type
 *   nt_start     <-- starting time step (or -1 to use t_start)
 *   t_start      <-- starting time
 *   restart_mode <-- behavior in case of restart (reset, auto, or strict)
 *   restart_name <-- if non-null, previous name in case of restart
 *
 * returns:
 *   id of new moment in case of success, -1 in case of error.
 *----------------------------------------------------------------------------*/

int
cs_time_moment_define_by_function(const char                *name,
                                  cs_function_t             *f,
                                  cs_time_moment_type_t      type,
                                  int                        nt_start,
                                  double                     t_start,
                                  cs_time_moment_restart_t   restart_mode,
                                  const char                *restart_name);

/*----------------------------------------------------------------------------
 * Define a moment whose data values will be computed using a
 * specified function.
 *
 * If of dimension > 1, the moment array is always interleaved.
 *
 * parameters:
 *   name         <-- name of associated moment
 *   location_id  <-- id of associated mesh location
 *   dim          <-- dimension associated with element data
 *   is_intensive <-- is the time moment intensive?
 *   data_func    <-- function used to define data values
 *   data_input   <-- pointer to optional (untyped) value or structure
 *                    to be used by data_func
 *   weight_func  <-- function used to define weight values
 *   weight_input <-- pointer to optional (untyped) value or structure
 *                    to be used by weight_func
 *   type         <-- moment type
 *   nt_start     <-- starting time step (or -1 to use t_start)
 *   t_start      <-- starting time
 *   restart_mode <-- behavior in case of restart (reset, auto, or strict)
 *   restart_name <-- if non-null, previous name in case of restart
 *
 * returns:
 *   id of new moment in case of success, -1 in case of error.
 *----------------------------------------------------------------------------*/

int
cs_time_moment_define_by_func(const char                *name,
                              int                        location_id,
                              int                        dim,
                              bool                       is_intensive,
                              cs_time_moment_data_t     *data_func,
                              const void                *data_input,
                              cs_time_moment_data_t     *w_data_func,
                              void                      *w_data_input,
                              cs_time_moment_type_t      type,
                              int                        nt_start,
                              double                     t_start,
                              cs_time_moment_restart_t   restart_mode,
                              const char                *restart_name);

/*----------------------------------------------------------------------------
 * Return the number of defined time moments.
 *
 * returns:
 *   number of defined time moments
 *----------------------------------------------------------------------------*/

int
cs_time_moment_n_moments(void);

/*----------------------------------------------------------------------------
 * Return the number of time moments in the restart file, if any
 *
 * returns:
 *   number of defined moments in restart file, or 0
 *----------------------------------------------------------------------------*/

int
cs_time_moment_n_moments_restart(void);

/*----------------------------------------------------------------------------
 * Define a moment restart mode and name by an id.
 *
 * This is a utility function, to allow simplification of automatic setups.
 * It must be called just before defining a moment to work properly if
 * restart_id < -1 (automatic mode).
 *
 * parameters:
 *   restart_id   <--  -2: automatic, -1: reset, >= 0: id of
 *                     matching moment in restart data
 *   restart_mode -->  matching restart mode
 *   restart_name -->  matching restart name
 *----------------------------------------------------------------------------*/

void
cs_time_moment_restart_options_by_id(int                         restart_id,
                                     cs_time_moment_restart_t   *restart_mode,
                                     const char                **restart_name);

/*----------------------------------------------------------------------------
 * Return name of a given time moments in the restart file, if any
 *   (check also \ref cs_time_moment_n_moments_restart).
 *
 * parameters:
 *   restart_id <-- id of time moment in restart data
 *
 * returns:
 *   name of defined moment in restart file, or null
 *----------------------------------------------------------------------------*/

const char *
cs_time_moment_restart_name(int  restart_id);

/*----------------------------------------------------------------------------
 * Return pointer to field associated with a given moment.
 *
 * For moments defined automatically to assist computation of higher order
 * moments, which do not have an associated field, a null pointer is returned.
 *
 * parameters:
 *   moment_id <-- id of associated moment
 *
 * returns:
 *   pointer to field associated with given moment, or null
 *----------------------------------------------------------------------------*/

cs_field_t *
cs_time_moment_get_field(int  moment_id);

/*----------------------------------------------------------------------------
 * Return 1 if moment is active, 0 if it is not active yet.
 *
 * parameters:
 *   moment_id <-- id of associated moment
 *
 * returns:
 *   1 if moment is active, 0 if it is not active yet
 *----------------------------------------------------------------------------*/

int
cs_time_moment_is_active(int  moment_id);

END_C_DECLS

#ifdef __cplusplus

/*----------------------------------------------------------------------------*/
/*
 * \brief Reset selected time step for starting time step of selected moment.
 *
 * All other time moments sharing the same start time will also start
 * at the same time step.
 *
 * \param[in]   moment_id  id of associated moment, or -1 for all
 * \param[in]   nt_start   starting time step
 */
/*----------------------------------------------------------------------------*/

void
cs_time_moment_set_start_time(int   moment_id,
                              int   nt_start);

/*----------------------------------------------------------------------------*/
/*
 * \brief Reset selected time step for starting time step of selected moment.
 *
 * All other time moments sharing the same start time will also start
 * at the same time step.
 *
 * \param[in]   moment_id  id of associated moment, or -1 for all
 * \param[in]   nt_start   starting time value
 */
/*----------------------------------------------------------------------------*/

void
cs_time_moment_set_start_time(int     moment_id,
                              double  t_start);

#endif //ifdef __cplusplus

BEGIN_C_DECLS

/*----------------------------------------------------------------------------*/
/*
 * \brief Set current iteration as starting time step of selected moment.
 *
 * All other time moments sharing the same start time should also be reset.
 *
 * \param[in]   moment_id  id of associated moment, or -1 for all.
 */
/*----------------------------------------------------------------------------*/

void
cs_time_moment_reset(int   moment_id);

/*----------------------------------------------------------------------------
 * Update all moment accumulators.
 ----------------------------------------------------------------------------*/

void
cs_time_moment_update_all(void);

/*----------------------------------------------------------------------------
 * Log moment definition setup information.
 *----------------------------------------------------------------------------*/

void
cs_time_moment_log_setup(void);

/*----------------------------------------------------------------------------
 * Log moment definition information for a given iteration.
 *----------------------------------------------------------------------------*/

void
cs_time_moment_log_iteration(void);

/*----------------------------------------------------------------------------
 * Indicate if restart API should use "main" instead of "auxiliary" file.
 *
 * parameters:
 *   use_main <-- use "main" restart if nonzero, "auxiliary" otherwise
 *----------------------------------------------------------------------------*/

void
cs_time_moment_restart_use_main(int  use_main);

/*----------------------------------------------------------------------------
 * Read restart moment data
 *
 * parameters:
 *   <-> restart  associated restart file pointer
 *----------------------------------------------------------------------------*/

void
cs_time_moment_restart_read(cs_restart_t  *restart);

/*----------------------------------------------------------------------------
 * Checkpoint moment data
 *
 * parameters:
 *   <-> restart  associated restart file pointer
 *----------------------------------------------------------------------------*/

void
cs_time_moment_restart_write(cs_restart_t  *restart);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_TIME_MOMENT_H__ */
