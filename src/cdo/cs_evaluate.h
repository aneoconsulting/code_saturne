#ifndef __CS_EVALUATE_H__
#define __CS_EVALUATE_H__

/*============================================================================
 * Functions and structures to deal with evaluation of quantities
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
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "base/cs_base.h"
#include "cdo/cs_cdo_connect.h"
#include "cdo/cs_cdo_local.h"
#include "cdo/cs_cdo_quantities.h"
#include "cdo/cs_xdef.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*============================================================================
 * Inline public function prototypes
 *============================================================================*/

/*============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set shared pointers to main domain members
 *
 * \param[in] quant    pointer to additional mesh quantities for CDO schemes
 * \param[in] connect  pointer to additional mesh connectivities for CDO schemes
 * \param[in] mesh     pointer to the shared mesh structure
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_init_sharing(const cs_cdo_quantities_t    *quant,
                         const cs_cdo_connect_t       *connect,
                         const cs_mesh_t              *mesh);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute reduced quantities for an array of size equal to dim * n_x
 *        The computed quantities are synchronized in parallel.
 *
 * \param[in]      dim    local array dimension (max: 3)
 * \param[in]      n_x    number of elements
 * \param[in]      array  array to analyze
 * \param[in]      w_x    weight to apply (may be set to  nullptr)
 * \param[in, out] min    resulting min array (size: dim, or 4 if dim = 3)
 * \param[in, out] max    resulting max array (size: dim, or 4 if dim = 3)
 * \param[in, out] wsum   (weighted) sum array (size: dim, or 4 if dim = 3)
 * \param[in, out] asum   (weighted) sum of absolute values (same size as wsum)
 * \param[in, out] ssum   (weighted) sum of squared values (same size as wsum)
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_array_reduction(int              dim,
                            cs_lnum_t        n_x,
                            const cs_real_t *array,
                            const cs_real_t *w_x,
                            cs_real_t       *min,
                            cs_real_t       *max,
                            cs_real_t       *wsum,
                            cs_real_t       *asum,
                            cs_real_t       *ssum);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Compute reduced quantities for an array attached to either vertex,
 *         face or edge DoFs
 *         The weight to apply to each entity x is scanned using the adjacency
 *         structure. array size is equal to dim * n_x
 *         The computed quantities are synchronized in parallel.
 *
 * \param[in]      dim     local array dimension (max: 3)
 * \param[in]      n_x     number of elements
 * \param[in]      array   array to analyze
 * \param[in]      c2x     pointer to the associated cs_adjacency_t structure
 * \param[in]      w_x     weight to apply (may be set to  null)
 * \param[in, out] min     resulting min array (size: dim, or 4 if dim = 3)
 * \param[in, out] max     resulting max array (size: dim, or 4 if dim = 3)
 * \param[in, out] wsum    (weighted) sum array (size: dim, or 4 if dim = 3)
 * \param[in, out] asum    (weighted) sum of absolute values (same size as vsum)
 * \param[in, out] ssum    (weighted) sum of squared values (same size as vsum)
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_scatter_array_reduction(int                     dim,
                                    cs_lnum_t               n_x,
                                    const cs_real_t        *array,
                                    const cs_adjacency_t   *c2x,
                                    const cs_real_t        *w_x,
                                    cs_real_t              *min,
                                    cs_real_t              *max,
                                    cs_real_t              *wsum,
                                    cs_real_t              *asum,
                                    cs_real_t              *ssum);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity defined by a value in the case of a density
 *         field for all the degrees of freedom
 *         Accessor to the value is by unit of volume and the return values are
 *         integrated over a volume
 *
 * \param[in]      dof_flag  indicate where the evaluation has to be done
 * \param[in]      def       pointer to a cs_xdef_t structure
 * \param[in, out] retval    pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_density_by_value(cs_flag_t          dof_flag,
                             const cs_xdef_t   *def,
                             cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Compute the value related to each DoF in the case of a density field
 *         The value defined by the analytic function is by unity of volume and
 *         the return values are integrated over a volume
 *
 * \param[in]      dof_flag    indicate where the evaluation has to be done
 * \param[in]      def         pointer to a cs_xdef_t structure
 * \param[in]      time_eval   physical time at which one evaluates the term
 * \param[in, out] retval      pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_density_by_analytic(cs_flag_t           dof_flag,
                                const cs_xdef_t    *def,
                                cs_real_t           time_eval,
                                cs_real_t           retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate a potential field at vertices from a definition by a
 *         constant value
 *
 * \param[in]      def             pointer to a cs_xdef_t pointer
 * \param[in]      n_v_selected    number of selected vertices
 * \param[in]      selected_lst    list of selected vertices
 * \param[in, out] retval          pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_vertices_by_value(const cs_xdef_t   *def,
                                           const cs_lnum_t    n_v_selected,
                                           const cs_lnum_t   *selected_lst,
                                           cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity attached to a potential field at vertices
 *         when the definition relies on an analytic expression
 *
 * \param[in]      def           pointer to a cs_xdef_t pointer
 * \param[in]      time_eval     physical time at which one evaluates the term
 * \param[in]      n_v_selected  number of selected vertices
 * \param[in]      selected_lst  list of selected vertices
 * \param[in, out] retval        pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_vertices_by_analytic(const cs_xdef_t   *def,
                                              const cs_real_t    time_eval,
                                              const cs_lnum_t    n_v_selected,
                                              const cs_lnum_t   *selected_lst,
                                              cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity attached to a potential field at vertices
 *         when the definition relies on a DoF function (Degrees of freedom)
 *
 * \param[in]      def           pointer to a cs_xdef_t pointer
 * \param[in]      n_v_selected  number of selected vertices
 * \param[in]      selected_lst  list of selected vertices
 * \param[in, out] retval        pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_vertices_by_dof_func(const cs_xdef_t   *def,
                                              const cs_lnum_t    n_v_selected,
                                              const cs_lnum_t   *selected_lst,
                                              cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate a potential field at face centers from a definition by a
 *         constant value
 *
 * \param[in]      def             pointer to a cs_xdef_t pointer
 * \param[in]      n_f_selected    number of selected faces
 * \param[in]      selected_lst    list of selected faces
 * \param[in, out] retval          pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_faces_by_value(const cs_xdef_t   *def,
                                        const cs_lnum_t    n_f_selected,
                                        const cs_lnum_t   *selected_lst,
                                        cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity attached to a potential field at face centers
 *         when the definition relies on an analytic expression
 *
 * \param[in]      def           pointer to a cs_xdef_t pointer
 * \param[in]      time_eval     physical time at which one evaluates the term
 * \param[in]      n_f_selected  number of selected faces
 * \param[in]      selected_lst  list of selected faces
 * \param[in, out] retval        pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_faces_by_analytic(const cs_xdef_t   *def,
                                           const cs_real_t    time_eval,
                                           const cs_lnum_t    n_f_selected,
                                           const cs_lnum_t   *selected_lst,
                                           cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate a potential field at cell centers from a definition by
 *         array
 *
 * \param[in]      def       pointer to a cs_xdef_t pointer
 * \param[in, out] retval    pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_cells_by_array(const cs_xdef_t   *def,
                                        cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate a potential field at cell centers from a definition by
 *         value
 *
 * \param[in]      def       pointer to a cs_xdef_t pointer
 * \param[in, out] retval    pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_cells_by_value(const cs_xdef_t   *def,
                                        cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity attached to a potential field at cell centers
 *         when the definition relies on an analytic expression
 *
 * \param[in]      def         pointer to a cs_xdef_t pointer
 * \param[in]      time_eval   physical time at which one evaluates the term
 * \param[in, out] retval      pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_cells_by_analytic(const cs_xdef_t    *def,
                                           const cs_real_t     time_eval,
                                           cs_real_t           retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the quantity attached to a potential field at cells
 *         when the definition relies on a DoF function (Degrees of freedom)
 *
 * \param[in]      def           pointer to a cs_xdef_t pointer
 * \param[in, out] retval        pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_at_cells_by_dof_func(const cs_xdef_t   *def,
                                           cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Define a value to each DoF in the case of a potential field in order
 *         to put a given quantity inside the volume associated to the zone
 *         related to the given definition
 *         wvals may be null.
 *
 * \param[in]      dof_flag  indicate where the evaluation has to be done
 * \param[in]      def       pointer to a cs_xdef_t pointer
 * \param[in, out] vvals     pointer to the first array of computed values
 * \param[in, out] wvals     pointer to the second array of computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_potential_by_qov(cs_flag_t          dof_flag,
                             const cs_xdef_t   *def,
                             cs_real_t          vvals[],
                             cs_real_t          wvals[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the circulation along a selection of (primal) edges.
 *         Circulation is defined thanks to a constant vector field (by value)
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      n_e_selected   number of selected edges
 * \param[in]      selected_lst   list of selected edges
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_circulation_along_edges_by_value(const cs_xdef_t   *def,
                                             const cs_lnum_t    n_e_selected,
                                             const cs_lnum_t   *selected_lst,
                                             cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the circulation along a selection of (primal) edges.
 *         Circulation is defined thanks to an array
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      n_e_selected   number of selected edges
 * \param[in]      selected_lst   list of selected edges
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_circulation_along_edges_by_array(const cs_xdef_t   *def,
                                             const cs_lnum_t    n_e_selected,
                                             const cs_lnum_t   *selected_lst,
                                             cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the circulation along a selection of (primal) edges.
 *         Circulation is defined by an analytical function.
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      time_eval      physical time at which one evaluates the term
 * \param[in]      n_e_selected   number of selected edges
 * \param[in]      selected_lst   list of selected edges
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_circulation_along_edges_by_analytic(const cs_xdef_t   *def,
                                                const cs_real_t    time_eval,
                                                const cs_lnum_t    n_e_selected,
                                                const cs_lnum_t   *selected_lst,
                                                cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average of a function on the faces
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      n_f_selected   number of selected faces
 * \param[in]      selected_lst   list of selected faces
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_faces_by_value(const cs_xdef_t   *def,
                                      const cs_lnum_t    n_f_selected,
                                      const cs_lnum_t   *selected_lst,
                                      cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average of a function on the faces.
 *         Warning: retval has to be initialize before calling this function.
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      time_eval      physical time at which one evaluates the term
 * \param[in]      n_f_selected   number of selected faces
 * \param[in]      selected_lst   list of selected faces
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_faces_by_analytic(const cs_xdef_t   *def,
                                         const cs_real_t    time_eval,
                                         const cs_lnum_t    n_f_selected,
                                         const cs_lnum_t   *selected_lst,
                                         cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average value on faces following the given definition
 *
 * \param[in]      def            pointer to a cs_xdef_t pointer
 * \param[in]      time_eval      physical time at which one evaluates the term
 * \param[in]      n_f_selected   number of selected faces
 * \param[in]      selected_lst   list of selected faces
 * \param[in, out] retval         pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_faces(const cs_xdef_t   *def,
                             cs_real_t          time_eval,
                             const cs_lnum_t    n_f_selected,
                             const cs_lnum_t   *selected_lst,
                             cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average of a function on the cells
 *
 * \param[in]      def       pointer to a cs_xdef_t pointer
 * \param[in, out] retval    pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_cells_by_value(const cs_xdef_t   *def,
                                      cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average of a function on the cells
 *
 * \param[in]      def       pointer to a cs_xdef_t pointer
 * \param[in, out] retval    pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_cells_by_array(const cs_xdef_t   *def,
                                      cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average of a function on the cells
 *         Warning: retval has to be initialize before calling this function.
 *
 * \param[in]      def        pointer to a cs_xdef_t pointer
 * \param[in]      time_eval  physical time at which one evaluates the term
 * \param[in, out] retval     pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_cells_by_analytic(const cs_xdef_t   *def,
                                         cs_real_t          time_eval,
                                         cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the average value on cells following the given definition
 *         The cells associated to this definition (through the related zone)
 *         are all considered.
 *
 * \param[in]      def        pointer to a cs_xdef_t pointer
 * \param[in]      time_eval  physical time at which one evaluates the term
 * \param[in, out] retval     pointer to the computed values
 */
/*----------------------------------------------------------------------------*/

void
cs_evaluate_average_on_cells(const cs_xdef_t   *def,
                             cs_real_t          time_eval,
                             cs_real_t          retval[]);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Evaluate the integral over the full computational domain of a
 *         quantity defined by an array. The parallel sum reduction is
 *         performed inside this function.
 *
 * \param[in]  array_loc   flag indicating where are located values
 * \param[in]  array_val   array of values
 *
 * \return the value of the integration (parallel sum reduction done)
 */
/*----------------------------------------------------------------------------*/

cs_real_t
cs_evaluate_scal_domain_integral_by_array(cs_flag_t         array_loc,
                                          const cs_real_t  *array_val);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_EVALUATE_H__ */
