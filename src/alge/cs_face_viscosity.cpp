/*============================================================================
 * Face viscosity
 *============================================================================*/

/* This file is part of code_saturne, a general-purpose CFD tool.

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

#include "base/cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <float.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_error.h"
#include "bft/bft_mem.h"
#include "bft/bft_printf.h"

#include "alge/cs_blas.h"
#include "base/cs_dispatch.h"
#include "base/cs_ext_neighborhood.h"
#include "base/cs_field.h"
#include "base/cs_field_default.h"
#include "base/cs_field_pointer.h"
#include "alge/cs_gradient.h"
#include "base/cs_halo.h"
#include "base/cs_halo_perio.h"
#include "base/cs_internal_coupling.h"
#include "base/cs_log.h"
#include "base/cs_math.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_parall.h"
#include "base/cs_parameters.h"
#include "base/cs_physical_constants.h"
#include "pprt/cs_physical_model.h"
#include "base/cs_porous_model.h"
#include "base/cs_prototypes.h"
#include "base/cs_timer.h"
#include "turb/cs_turbulence_model.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "alge/cs_face_viscosity.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional Doxygen documentation
 *============================================================================*/

/*! \file  cs_face_viscosity.cpp
 *
 *  \brief Face viscosity.
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local type definitions
 *============================================================================*/

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Computes the secondary viscosity contribution \f$\kappa
 * -\dfrac{2}{3} \mu\f$ in order to compute:
 * \f[
 * \grad\left( (\kappa -\dfrac{2}{3} \mu) \trace( \gradt(\vect{u})) \right)
 * \f]
 * with:
 *   - \f$ \mu = \mu_{laminar} + \mu_{turbulent} \f$
 *   - \f$ \kappa \f$ is the volume viscosity (generally zero)
 *
 * \remark
 * In LES, the tensor
 * \f$\overline{\left(\vect{u}-\overline{\vect{u}}\right)\otimes\left(\vect{u}
 *-\overline{\vect{u}}\right)}\f$
 * is modeled by \f$\mu_t \overline{\tens{S}}\f$
 * and not by
 * \f$\mu_t\overline{\tens{S}}-\dfrac{2}{3}\mu_t
 * \trace\left(\overline{\tens{S}}\right)\tens{1}+\dfrac{2}{3}k\tens{1}\f$
 * so that no term
 * \f$\mu_t \dive \left(\overline{\vect{u}}\right)\f$ is needed.
 *
 * Please refer to the
 * <a href="../../theory.pdf#visecv"><b>visecv</b></a> section
 * of the theory guide for more informations.
 *
 * \param[in,out] secvif        lambda*surface at interior faces
 * \param[in,out] secvib        lambda*surface at boundary faces
 */
/*----------------------------------------------------------------------------*/

void
cs_face_viscosity_secondary(cs_real_t  secvif[],
                            cs_real_t  secvib[])
{
  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t *fvq = cs_glob_mesh_quantities;

  const cs_lnum_t n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t n_cells = mesh->n_cells;
  const cs_lnum_t n_b_faces = mesh->n_b_faces;
  const cs_lnum_t n_i_faces = mesh->n_i_faces;

  const cs_lnum_2_t *i_face_cells
    = (const cs_lnum_2_t *)mesh->i_face_cells;
  const cs_lnum_t *b_face_cells = mesh->b_face_cells;
  const cs_real_t *weight = fvq->weight;

  const int itytur = cs_glob_turb_model->itytur;

  cs_dispatch_context ctx;

  /* Initialization
     -------------- */

  /* Allocate temporary arrays */
  cs_real_t *secvis;
  CS_MALLOC_HD(secvis, n_cells_ext, cs_real_t, cs_alloc_mode);

  cs_field_t *vel = CS_F_(vel);
  cs_equation_param_t *eqp_vel = cs_field_get_equation_param(vel);
  const cs_real_t *viscl = CS_F_(mu)->val;
  const cs_real_t *visct = CS_F_(mu_t)->val;

  cs_real_t *cpro_viscv = nullptr;
  cs_field_t *f_viscv = cs_field_by_name_try("volume_viscosity");

  if (f_viscv != nullptr)
    cpro_viscv = cs_field_by_name_try("volume_viscosity")->val;

  /* Time extrapolation ? */
  int key_t_ext_id = cs_field_key_id("time_extrapolated");

  /* Computation of the second viscosity: lambda = K -2/3 mu
     ------------------------------------------------------- */

  /* For order 2 in time, everything should be taken in n...*/

  const cs_real_t d2s3m = -2.0/3.0;

  const int iviext = cs_field_get_key_int(CS_F_(mu), key_t_ext_id);
  const int iviext_t = cs_field_get_key_int(CS_F_(mu_t), key_t_ext_id);

  /* Laminar viscosity */

  if (cs_glob_time_scheme->isno2t > 0 && iviext > 0) {
    cs_real_t *cpro_viscl_pre = CS_F_(mu)->val_pre;
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      secvis[c_id] = d2s3m * cpro_viscl_pre[c_id];
    });
  }
  else {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      secvis[c_id] = d2s3m * viscl[c_id];
    });
  }

  /* Volume viscosity if present */
  if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] >= 0) {
    if (f_viscv != nullptr) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        secvis[c_id] += cpro_viscv[c_id];
      });
    }
    else {
      const cs_real_t viscv0 = cs_glob_fluid_properties->viscv0;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        secvis[c_id] += viscv0;
      });
    }
  }

  /* Turbulent viscosity (if not in Rij or LES) */

  if (itytur != 3 && itytur != 4) {

    if (cs_glob_time_scheme->isno2t > 0 && iviext_t > 0) {
      cs_real_t *cpro_visct_pre = CS_F_(mu_t)->val_pre;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        secvis[c_id] += d2s3m * cpro_visct_pre[c_id];
      });
    }
    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        secvis[c_id] += d2s3m * visct[c_id];
      });
    }

  }

  /* With porosity */
  if (cs_glob_porous_model == 1 || cs_glob_porous_model == 2) {
    cs_real_t *porosity = CS_F_(poro)->val;
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      secvis[c_id] *= porosity[c_id];
    });
  }

  /* Parallelism and periodicity process */

  if (mesh->halo != nullptr) {
    ctx.wait(); // needed for the next synchronization
    cs_halo_sync(mesh->halo, CS_HALO_STANDARD, ctx.use_gpu(), secvis);
  }

  /* Interior faces
     TODO we should (re)test the weight walue for imvisf=0 */

  if (eqp_vel->imvisf == 0) {
    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      const cs_lnum_t c_id1 = i_face_cells[f_id][0];
      const cs_lnum_t c_id2 = i_face_cells[f_id][1];

      const cs_real_t secvsi = secvis[c_id1];
      const cs_real_t secvsj = secvis[c_id2];

      secvif[f_id] = 0.5 * (secvsi + secvsj);
    });
  }
  else {
    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      const cs_lnum_t c_id1 = i_face_cells[f_id][0];
      const cs_lnum_t c_id2 = i_face_cells[f_id][1];

      const cs_real_t secvsi = secvis[c_id1];
      const cs_real_t secvsj = secvis[c_id2];
      const cs_real_t pnd = weight[f_id];

      secvif[f_id] = secvsi * secvsj / (pnd * secvsi + (1.0 - pnd) * secvsj);
    });
  }

  /* Boundary faces
     TODO shall we extrapolate this value? */

  ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
    const cs_lnum_t c_id = b_face_cells[f_id];
    secvib[f_id] = secvis[c_id];
  });

  ctx.wait(); // (temporary) needed for the CPU function cs_solve_navier_stokes

  /* TODO stresses at the wall? */

  /* Free memory */
  CS_FREE_HD(secvis);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the diffusion velocity at faces.
 *
 * i_visc,b_visc = viscosity*surface/distance, homogeneous to a rate of flow
 * in kg/s.
 *
 * <a name="viscfa"></a>
 *
 * Please refer to the
 * <a href="../../theory.pdf#viscfa"><b>viscfa</b></a> section of the theory
 * guide for more informations.
 *
 * \remark: a priori, no need of reconstruction techniques
 * (to improve if necessary).
 *
 * \param[in]     m              pointer to mesh
 * \param[in]     fvq            pointer to finite volume quantities
 * \param[in]     visc_mean_type method to compute the viscosity at faces:
 *                                - 0 arithmetical
 *                                - 1 harmonic
 * \param[in]     c_visc         cell viscosity (scalar)
 * \param[out]    i_visc         inner face viscosity
 *                                (times surface divided by distance)
 * \param[out]    b_visc         boundary face viscosity
 *                                (surface, must be consistent with flux BCs)
 */
/*----------------------------------------------------------------------------*/

void
cs_face_viscosity(const cs_mesh_t               *m,
                  const cs_mesh_quantities_t    *fvq,
                  const int                      visc_mean_type,
                  cs_real_t            *restrict c_visc,
                  cs_real_t            *restrict i_visc,
                  cs_real_t            *restrict b_visc)
{
  const cs_halo_t  *halo = m->halo;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *)m->b_face_cells;
  const cs_real_t *restrict weight = fvq->weight;
  const cs_real_t *restrict i_dist = fvq->i_dist;
  const cs_real_t *restrict i_f_face_surf = fvq->i_face_surf;
  const cs_real_t *restrict b_f_face_surf = fvq->b_face_surf;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_i_faces = m->n_i_faces;

  cs_dispatch_context ctx, ctx_b;
#if defined(HAVE_CUDA)
  ctx_b.set_cuda_stream(cs_cuda_get_stream(1));
#endif

  /* Porosity field */
  cs_field_t *fporo = cs_field_by_name_try("porosity");

  cs_real_t *porosi = nullptr;

  if (cs_glob_porous_model == 1 || cs_glob_porous_model == 2) {
    porosi = fporo->val;
  }

  /* ---> Periodicity and parallelism treatment */
  if (halo != nullptr) {
    cs_halo_type_t halo_type = CS_HALO_STANDARD;
    const bool on_device = ctx.use_gpu();
    cs_halo_sync(halo, halo_type, on_device, c_visc);
    if (porosi != nullptr)
      cs_halo_sync(halo, halo_type, on_device, porosi);
  }

  /* Without porosity */
  if (porosi == nullptr) {

    if (visc_mean_type == 0) {

      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

        cs_lnum_t ii = i_face_cells[f_id][0];
        cs_lnum_t jj = i_face_cells[f_id][1];

        cs_real_t visci = c_visc[ii];
        cs_real_t viscj = c_visc[jj];

        i_visc[f_id] = 0.5*(visci+viscj)
                         *i_f_face_surf[f_id]/i_dist[f_id];

      });
    }
    else {

      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

        cs_lnum_t ii = i_face_cells[f_id][0];
        cs_lnum_t jj = i_face_cells[f_id][1];

        cs_real_t visci = c_visc[ii];
        cs_real_t viscj = c_visc[jj];
        cs_real_t pnd   = weight[f_id];

        i_visc[f_id] = visci*viscj/cs::max(pnd*visci+(1.-pnd)*viscj,
                                           DBL_MIN)
                         *i_f_face_surf[f_id]/i_dist[f_id];
      });
    }

    ctx_b.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      b_visc[f_id] = b_f_face_surf[f_id];
    });

  /* With porosity */
  }
  else {

    if (visc_mean_type == 0) {

      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

        cs_lnum_t ii = i_face_cells[f_id][0];
        cs_lnum_t jj = i_face_cells[f_id][1];

        cs_real_t visci = c_visc[ii] * porosi[ii];
        cs_real_t viscj = c_visc[jj] * porosi[jj];

        i_visc[f_id] = 0.5*(visci+viscj)
                         *i_f_face_surf[f_id]/i_dist[f_id];

      });
    }
    else {

      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

        cs_lnum_t ii = i_face_cells[f_id][0];
        cs_lnum_t jj = i_face_cells[f_id][1];

        cs_real_t visci = c_visc[ii] * porosi[ii];
        cs_real_t viscj = c_visc[jj] * porosi[jj];
        cs_real_t pnd   = weight[f_id];

        i_visc[f_id] = visci*viscj/cs::max(pnd*visci+(1.-pnd)*viscj,
                                           DBL_MIN)
                         *i_f_face_surf[f_id]/i_dist[f_id];

      });
    }

    ctx_b.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      cs_lnum_t ii = b_face_cells[f_id];
      b_visc[f_id] = b_f_face_surf[f_id]*porosi[ii];
    });

  }

  /* Force face viscosity (and thus matrix extradiagonal terms)
     to 0 when both cells are disabled. This is especially useful for
     the multigrid solvers, which can then handle disabled cells as
     penalized rows, and build an aggregation ignoring those. */

  if (fvq->has_disable_flag) {

    int *c_disable_flag = fvq->c_disable_flag;

    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      cs_lnum_t ii = i_face_cells[f_id][0];
      cs_lnum_t jj = i_face_cells[f_id][1];

      if (c_disable_flag[ii] + c_disable_flag[jj] == 2)
        i_visc[f_id] = 0;
    });

  }

  // guaranteed results for the CPU outside functions
  ctx.wait();
  ctx_b.wait();
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the equivalent tensor viscosity at faces for a 3x3 symetric
 * tensor.
 *
 * \param[in]     m              pointer to mesh
 * \param[in]     fvq            pointer to finite volume quantities
 * \param[in]     visc_mean_type method to compute the viscosity at faces:
 *                                - 0: arithmetic
 *                                - 1: harmonic
 * \param[in]     c_visc         cell viscosity symmetric tensor
 * \param[out]    i_visc         inner face tensor viscosity
 *                                (times surface divided by distance)
 * \param[out]    b_visc         boundary face viscosity
 *                                (surface, must be consistent with flux BCs)
 */
/*----------------------------------------------------------------------------*/

void
cs_face_anisotropic_viscosity_vector(const cs_mesh_t             *m,
                                     const cs_mesh_quantities_t  *fvq,
                                     const int                    visc_mean_type,
                                     cs_real_6_t        *restrict c_visc,
                                     cs_real_33_t       *restrict i_visc,
                                     cs_real_t          *restrict b_visc)
{
  const cs_halo_t  *halo = m->halo;

  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *)m->b_face_cells;
  const cs_real_t *restrict weight = fvq->weight;
  const cs_real_t *restrict i_dist = fvq->i_dist;
  const cs_real_t *restrict i_f_face_surf = fvq->i_face_surf;
  const cs_real_t *restrict b_f_face_surf = fvq->b_face_surf;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_i_faces = m->n_i_faces;

  /* Parallel or device dispatch */
  cs_dispatch_context ctx, ctx_c;
#if defined(HAVE_CUDA)
  ctx_c.set_cuda_stream(cs_cuda_get_stream(1));
#endif

  cs_real_6_t *c_poro_visc = nullptr;
  cs_real_6_t *w2 = nullptr;

  /* Porosity fields */
  cs_field_t *fporo = cs_field_by_name_try("porosity");
  cs_field_t *ftporo = cs_field_by_name_try("tensorial_porosity");

  cs_real_t *porosi = nullptr;
  cs_real_6_t *porosf = nullptr;

  if (cs_glob_porous_model == 1 || cs_glob_porous_model == 2) {
    porosi = fporo->val;
    if (ftporo != nullptr) {
      porosf = (cs_real_6_t *)ftporo->val;
    }
  }

  /* Without porosity */
  if (porosi == nullptr) {

    c_poro_visc = c_visc;

  /* With porosity */
  }
  else if (porosi != nullptr && porosf == nullptr) {

    CS_MALLOC_HD(w2, n_cells_ext, cs_real_6_t, cs_alloc_mode);
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (int isou = 0; isou < 6; isou++) {
        w2[c_id][isou] = porosi[c_id]*c_visc[c_id][isou];
      }
    });
    ctx.wait(); // needed before pointer egality

    c_poro_visc = w2;

  /* With tensorial porosity */
  }
  else if (porosi != nullptr && porosf != nullptr) {

    CS_MALLOC_HD(w2, n_cells_ext, cs_real_6_t, cs_alloc_mode);
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cs_math_sym_33_product(porosf[c_id],
                             c_visc[c_id],
                             w2[c_id]);
    });
    ctx.wait(); // needed before pointer egality

    c_poro_visc = w2;

  }

  /* ---> Periodicity and parallelism treatment */
  if (halo != nullptr)
    cs_halo_sync_r(halo, ctx.use_gpu(), c_poro_visc);

  /* Arithmetic mean */
  if (visc_mean_type == 0) {

    ctx_c.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = i_face_cells[f_id][0];
      cs_lnum_t jj = i_face_cells[f_id][1];

      cs_real_t visci[3][3];
      visci[0][0] = c_poro_visc[ii][0];
      visci[1][1] = c_poro_visc[ii][1];
      visci[2][2] = c_poro_visc[ii][2];
      visci[1][0] = c_poro_visc[ii][3];
      visci[0][1] = c_poro_visc[ii][3];
      visci[2][1] = c_poro_visc[ii][4];
      visci[1][2] = c_poro_visc[ii][4];
      visci[2][0] = c_poro_visc[ii][5];
      visci[0][2] = c_poro_visc[ii][5];

      cs_real_t viscj[3][3];
      viscj[0][0] = c_poro_visc[jj][0];
      viscj[1][1] = c_poro_visc[jj][1];
      viscj[2][2] = c_poro_visc[jj][2];
      viscj[1][0] = c_poro_visc[jj][3];
      viscj[0][1] = c_poro_visc[jj][3];
      viscj[2][1] = c_poro_visc[jj][4];
      viscj[1][2] = c_poro_visc[jj][4];
      viscj[2][0] = c_poro_visc[jj][5];
      viscj[0][2] = c_poro_visc[jj][5];

      for (int isou = 0; isou < 3; isou++) {
        for (int jsou = 0; jsou < 3; jsou++) {
          i_visc[f_id][jsou][isou] =  0.5*(visci[jsou][isou]
                                             +viscj[jsou][isou])
                                       * i_f_face_surf[f_id]/i_dist[f_id];
        }
      }

    });

    /* Harmonic mean: Kf = Ki . (pnd Ki +(1-pnd) Kj)^-1 . Kj */
  }
  else {

    ctx_c.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = i_face_cells[f_id][0];
      cs_lnum_t jj = i_face_cells[f_id][1];

      cs_real_t pnd = weight[f_id];

      cs_real_t s1[6], s2[6];
      for (int isou = 0; isou < 6; isou++) {
        s1[isou] = pnd*c_poro_visc[ii][isou] + (1.-pnd)*c_poro_visc[jj][isou];
      }

      cs_math_sym_33_inv_cramer(s1, s2);

      cs_math_sym_33_product(s2, c_poro_visc[jj], s1);

      cs_math_sym_33_product(c_poro_visc[ii], s1, s2);

      cs_real_t srfddi = i_f_face_surf[f_id]/i_dist[f_id];

      i_visc[f_id][0][0] = s2[0]*srfddi;
      i_visc[f_id][1][1] = s2[1]*srfddi;
      i_visc[f_id][2][2] = s2[2]*srfddi;
      i_visc[f_id][1][0] = s2[3]*srfddi;
      i_visc[f_id][0][1] = s2[3]*srfddi;
      i_visc[f_id][2][1] = s2[4]*srfddi;
      i_visc[f_id][1][2] = s2[4]*srfddi;
      i_visc[f_id][2][0] = s2[5]*srfddi;
      i_visc[f_id][0][2] = s2[5]*srfddi;

    });

  }

  /* Without porosity */
  if (porosi == nullptr) {

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      b_visc[f_id] = b_f_face_surf[f_id];
    });

  /* With porosity */
  }
  else if (porosi != nullptr && porosf == nullptr) {

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = b_face_cells[f_id];

      b_visc[f_id] = b_f_face_surf[f_id]*porosi[ii];

    });

  /* With anisotropic porosity */
  }
  else if (porosi != nullptr && porosf != nullptr) {

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = b_face_cells[f_id];

      b_visc[f_id] = b_f_face_surf[f_id]*porosi[ii];

    });

  }

  ctx.wait();
  ctx_c.wait();

  CS_FREE_HD(w2);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute the equivalent viscosity at faces for a 3x3 symetric tensor,
 * always using a harmonic mean.
 *
 * \param[in]     m             pointer to mesh
 * \param[in]     fvq           pointer to finite volume quantities
 * \param[in]     c_visc        cell viscosity symmetric tensor
 * \param[in]     iwarnp        verbosity
 * \param[out]    weighf        inner face weight between cells i and j
 *                              \f$ \frac{\vect{IF} \cdot \tens{K}_\celli}
 *                               {\norm{\tens{K}_\celli \cdot \vect{S}}^2} \f$
 *                              and
 *                              \f$ \frac{\vect{FJ} \cdot \tens{K}_\cellj}
 *                               {\norm{\tens{K}_\cellj \cdot \vect{S}}^2} \f$
 * \param[out]    weighb        boundary face weight
 *                              \f$ \frac{\vect{IF} \cdot \tens{K}_\celli}
 *                               {\norm{\tens{K}_\celli \cdot \vect{S}}^2} \f$
 * \param[out]    i_visc        inner face viscosity
 *                               (times surface divided by distance)
 * \param[out]    b_visc        boundary face viscosity
 *                               (surface, must be consistent with flux BCs)
 */
/*----------------------------------------------------------------------------*/

void
cs_face_anisotropic_viscosity_scalar(const cs_mesh_t               *m,
                                     const cs_mesh_quantities_t    *fvq,
                                     cs_real_6_t          *restrict c_visc,
                                     const int                      iwarnp,
                                     cs_real_2_t          *restrict weighf,
                                     cs_real_t            *restrict weighb,
                                     cs_real_t            *restrict i_visc,
                                     cs_real_t            *restrict b_visc)
{
  const cs_halo_t  *halo = m->halo;

  cs_mesh_quantities_t *mq_g = cs_glob_mesh_quantities_g;

  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *)m->b_face_cells;
  const cs_real_t *restrict weight = mq_g->weight;
  const cs_real_t *restrict i_dist = mq_g->i_dist;
  const cs_real_t *restrict b_dist = mq_g->b_dist;
  const cs_real_t *restrict b_f_face_surf = mq_g->b_face_surf;
  const cs_real_3_t *restrict cell_cen
    = (const cs_real_3_t *)mq_g->cell_cen;
  const cs_real_3_t *restrict i_face_normal
    = (const cs_real_3_t *)mq_g->i_face_normal;
  const cs_real_t *restrict i_face_surf
    = (const cs_real_t *)mq_g->i_face_surf;
  const cs_real_t *restrict i_f_face_surf
    = (const cs_real_t *)fvq->i_face_surf;
  const cs_real_3_t *restrict b_face_normal
    = (const cs_real_3_t *)mq_g->b_face_normal;
  const cs_real_3_t *restrict i_face_cog
    = (const cs_real_3_t *)mq_g->i_face_cog;
  const cs_real_3_t *restrict b_face_cog
    = (const cs_real_3_t *)mq_g->b_face_cog;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_i_faces = m->n_i_faces;

  /* Parallel or device dispatch */
  cs_dispatch_context ctx, ctx_c;
#if defined(HAVE_CUDA)
  ctx_c.set_cuda_stream(cs_cuda_get_stream(1));
#endif

  short *i_clip = nullptr, *b_clip = nullptr;
  if (iwarnp >= 3) {
    CS_MALLOC_HD(i_clip, n_i_faces, short, cs_alloc_mode);
    CS_MALLOC_HD(b_clip, n_b_faces, short, cs_alloc_mode);

    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      i_clip[f_id] = 0;
    });
    ctx_c.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      b_clip[f_id] = 0;
    });
  }

  const cs_real_t eps = 0.1;

  cs_real_6_t *c_poro_visc = nullptr;
  cs_real_6_t *w2 = nullptr;

  /* Porosity fields */
  cs_field_t *fporo = cs_field_by_name_try("porosity");
  cs_field_t *ftporo = cs_field_by_name_try("tensorial_porosity");

  cs_real_t *porosi = nullptr;
  cs_real_6_t *porosf = nullptr;

  if (cs_glob_porous_model == 1 || cs_glob_porous_model == 2) {
    porosi = fporo->val;
    if (ftporo != nullptr) {
      porosf = (cs_real_6_t *)ftporo->val;
    }
  }

  /* Without porosity */
  if (porosi == nullptr) {

    c_poro_visc = c_visc;

  /* With porosity */
  }
  else if (porosi != nullptr && porosf == nullptr) {

    CS_MALLOC_HD(w2, n_cells_ext, cs_real_6_t, cs_alloc_mode);

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (int isou = 0; isou < 6; isou++) {
        w2[c_id][isou] = porosi[c_id]*c_visc[c_id][isou];
      }
    });

    ctx.wait();

    c_poro_visc = w2;

  /* With tensorial porosity */
  }
  else if (porosi != nullptr && porosf != nullptr) {

    CS_MALLOC_HD(w2, n_cells_ext, cs_real_6_t, cs_alloc_mode);

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cs_math_sym_33_product(porosf[c_id],
                             c_visc[c_id],
                             w2[c_id]);
    });

    ctx.wait();

    c_poro_visc = w2;

  }

  /* ---> Periodicity and parallelism treatment */
  if (halo != nullptr)
    cs_halo_sync_r(halo, ctx.use_gpu(), c_poro_visc);

  /* Always Harmonic mean */
  ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

    cs_lnum_t ii = i_face_cells[f_id][0];
    cs_lnum_t jj = i_face_cells[f_id][1];

    /* ||Ki.S||^2 */
    cs_real_t viscisv[3];
    cs_math_sym_33_3_product(c_poro_visc[ii], i_face_normal[f_id], viscisv);
    cs_real_t viscis = cs_math_3_square_norm(viscisv);

    /* IF */
    cs_real_t fi[3];
    for (cs_lnum_t kk = 0; kk < 3; kk++)
      fi[kk] = i_face_cog[f_id][kk]-cell_cen[ii][kk];

    /* IF.Ki.S */
    cs_real_t fiki[3];
    cs_math_sym_33_3_product(c_poro_visc[ii], fi, fiki);
    cs_real_t fikis = cs_math_3_dot_product(fiki, i_face_normal[f_id]);

    cs_real_t distfi = (1. - weight[f_id])*i_dist[f_id];

    /* Take I" so that I"F= eps*||FI||*Ki.n when I" is in cell rji */
    cs_real_t temp = eps*sqrt(viscis)*distfi;
    if (fikis < temp) {
      fikis = temp;
      if (i_clip != nullptr)
        i_clip[f_id] = 1;
    }

    /* ||Kj.S||^2 */
    cs_real_t viscjsv[3];
    cs_math_sym_33_3_product(c_poro_visc[jj], i_face_normal[f_id], viscjsv);
    cs_real_t viscjs = cs_math_3_square_norm(viscjsv);

    /* FJ */
    cs_real_t fj[3];
    for (int kk = 0; kk < 3; kk++)
      fj[kk] = cell_cen[jj][kk]-i_face_cog[f_id][kk];

    /* FJ.Kj.S */
    cs_real_t fjkj[3];
    cs_math_sym_33_3_product(c_poro_visc[jj], fj, fjkj);
    cs_real_t fjkjs = cs_math_3_dot_product(fjkj, i_face_normal[f_id]);

    cs_real_t distfj = weight[f_id]*i_dist[f_id];

    /* Take J" so that FJ"= eps*||FJ||*Kj.n when J" is in cell i */
    temp = eps*sqrt(viscjs)*distfj;
    if (fjkjs < temp) {
      fjkjs = temp;
      if (i_clip != nullptr)
        i_clip[f_id] += 1;
    }

    weighf[f_id][0] = fikis/viscis;
    weighf[f_id][1] = fjkjs/viscjs;

    i_visc[f_id] = 1./(weighf[f_id][0] + weighf[f_id][1]);

  });

  /* For the porous modelling based on integral formulation Section and fluid
   * Section are different */
  if (cs_glob_porous_model == 3) {
     ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      i_visc[f_id] *= i_f_face_surf[f_id] / i_face_surf[f_id];
     });
  }

  ctx_c.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

    cs_lnum_t ii = b_face_cells[f_id];

    /* ||Ki.S||^2 */
    cs_real_t viscisv[3];
    cs_math_sym_33_3_product(c_poro_visc[ii], b_face_normal[f_id], viscisv);
    cs_real_t viscis = cs_math_3_square_norm(viscisv);

    /* IF */
    cs_real_t fi[3];
    for (int kk = 0; kk < 3; kk++)
      fi[kk] = b_face_cog[f_id][kk]-cell_cen[ii][kk];

    /* IF.Ki.S */
    cs_real_t fiki[3];
    cs_math_sym_33_3_product(c_poro_visc[ii], fi, fiki);
    cs_real_t fikis = cs_math_3_dot_product(fiki, b_face_normal[f_id]);

    cs_real_t distfi = b_dist[f_id];

    /* Take I" so that I"F= eps*||FI||*Ki.n when J" is in cell rji */
    cs_real_t temp = eps*sqrt(viscis)*distfi;
    if (fikis < temp) {
      fikis = temp;
      if (b_clip != nullptr)
        b_clip[f_id] = 1;
    }

    weighb[f_id] = fikis/viscis;

  });

  /* Without porosity */
  if (porosi == nullptr) {

    ctx_c.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      /* Warning: hint must be ||Ki.n||/I"F */
      b_visc[f_id] = b_f_face_surf[f_id];
    });

  /* With porosity */
  }
  else if (porosi != nullptr && porosf == nullptr) {

    ctx_c.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = b_face_cells[f_id];

      /* Warning: hint must be ||Ki.n||/I"F */
      b_visc[f_id] = b_f_face_surf[f_id]*porosi[ii];

    });

  /* With tensorial porosity */
  }
  else if (porosi != nullptr && porosf != nullptr) {

    ctx_c.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

      cs_lnum_t ii = b_face_cells[f_id];

      /* Warning: hint must be ||Ki.n||/I"F */
      b_visc[f_id] = b_f_face_surf[f_id]*porosi[ii];

    });

  }

  ctx.wait();
  ctx_c.wait();

  if (iwarnp >= 3) {
    cs_gnum_t n_i_clip = 0, n_b_clip = 0;

#     pragma omp parallel for reduction(+:n_i_clip)
    for (cs_lnum_t i = 0; i < n_i_faces; i++) {
      n_i_clip += i_clip[i];
    }
#     pragma omp parallel for reduction(+:n_b_clip)
    for (cs_lnum_t i = 0; i < n_b_faces; i++) {
      n_b_clip += b_clip[i];
    }

    cs_gnum_t count_clip[2] = {n_i_clip, n_b_clip};
    cs_parall_counter(count_clip, 2);

    bft_printf("Computing the face viscosity from the tensorial viscosity:\n"
               "   Number of internal clippings: %lu\n"
               "   Number of boundary clippings: %lu\n",
               (unsigned long)count_clip[0], (unsigned long)count_clip[1]);

    CS_FREE_HD(i_clip);
    CS_FREE_HD(b_clip);
  }

  CS_FREE_HD(w2);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
