/*============================================================================
 * Management of the post-processing
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_printf.h"

#include "fvm/fvm_nodal.h"
#include "fvm/fvm_nodal_append.h"
#include "fvm/fvm_nodal_extract.h"

#include "base/cs_array.h"
#include "base/cs_base.h"
#include "base/cs_boundary_zone.h"
#include "base/cs_field.h"
#include "base/cs_field_operator.h"
#include "base/cs_file.h"
#include "base/cs_function.h"
#include "lagr/cs_lagr_extract.h"
#include "base/cs_log.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "lagr/cs_lagr_query.h"
#include "meg/cs_meg_prototypes.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_connect.h"
#include "mesh/cs_mesh_location.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_parall.h"
#include "base/cs_prototypes.h"
#include "base/cs_selector.h"
#include "base/cs_time_control.h"
#include "base/cs_timer.h"
#include "base/cs_timer_stats.h"
#include "base/cs_volume_zone.h"

/*----------------------------------------------------------------------------
 * Header for the current file
 *----------------------------------------------------------------------------*/

#include "base/cs_post.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_post.cpp

  \brief Post-processing management.

  \var  CS_POST_ON_LOCATION
        postprocess variables on their base location (volume for variables)
  \var  CS_POST_BOUNDARY_NR
        postprocess boundary without reconstruction

  \enum cs_post_type_t

  \brief Postprocessing input variable type

  \var CS_POST_TYPE_cs_real_t
       Fortran double precision
  \var CS_POST_TYPE_int
       integer
  \var CS_POST_TYPE_float
       single precision floating-point value
  \var CS_POST_TYPE_double
       double precision floating-point value

  \typedef  cs_post_elt_select_t

  \brief  Function pointer to elements selection definition

  Each function of this sort may be used to select a given type of element,
  usually cells, interior faces, or boundary faces.

  If non-empty and not containing all elements, a list of elements of the
  main mesh should be allocated (using CS_MALLOC) and defined by this
  function when called. This list's lifecycle is then managed by the
  postprocessing subsystem.

   Note: if the input pointer is non-null, it must point to valid data
   when the selection function is called, so either:
   - that value or structure should not be temporary (i.e. local);
   - post-processing output must be ensured using cs_post_write_meshes()
   with a fixed-mesh writer before the data pointed to goes out of scope;

  \param[in, out]  input     pointer to optional (untyped) value or structure
  \param[out]      n_elts    number of selected elements
  \param[out]      elt_list  list of selected elements (0 to n-1 numbering)

  \typedef  cs_post_time_dep_output_t

  Function pointer associated with a specific post-processing output.

  Such functions are registered using the \ref cs_post_add_time_dep_output,
  and all registered functions are automatically called by
  \ref cs_post_write_vars.

  Note: if the input pointer is non-null, it must point to valid data
  when the output function is called, so either:
  - that value or structure should not be temporary (i.e. local);
  - post-processing output must be ensured using cs_post_write_var()
  or similar before the data pointed to goes out of scope.

  \param[in, out]  input       pointer to optional (untyped) value or structure
  \param[in]       ts          time step status structure, or null

  \typedef cs_post_time_mesh_dep_output_t

  Function pointer associated with a specific post-processing output
  on multiple meshes.

  Such functions are registered using the cs_post_add_time_mesh_dep_vars(),
  and all registered functions are automatically called by
  cs_post_write_vars().

  Note: if the input pointer is non-null, it must point to valid data
  when the output function is called, so either:
  - that value or structure should not be temporary (i.e. local);
  - post-processing output must be ensured using cs_post_write_var()
  or similar before the data pointed to goes out of scope.

  \param[in, out]  input       pointer to optional (untyped) value or structure
  \param[in]       mesh_id     id of the output mesh for the current call
  \param[in]       cat_id      category id of the output mesh for the
                               current call
  \param[in]       ent_flag    indicate global presence of cells
                               (ent_flag[0]), interior faces (ent_flag[1]),
                               boundary faces (ent_flag[2]), particles
                               (ent_flag[3]) or probes (ent_flag[4])
  \param[in]       n_cells     local number of cells of post_mesh
  \param[in]       n_i_faces   local number of interior faces of post_mesh
  \param[in]       n_b_faces   local number of boundary faces of post_mesh
  \param[in]       cell_ids    list of cells (0 to n-1) of post-processing mesh
  \param[in]       i_face_ids  list of interior faces (0 to n-1) of
                               post-processing mesh
  \param[in]       b_face_ids  list of boundary faces (0 to n-1) of
                               post-processing mesh
  \param[in]       ts          time step status structure, or null
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro Definitions
 *============================================================================*/

#define _MIN_RESERVED_MESH_ID    CS_POST_MESH_PROBES
#define _MIN_RESERVED_WRITER_ID  CS_POST_WRITER_HISTOGRAMS

/*============================================================================
 * Type definitions
 *============================================================================*/

/* Specific (forced) writer output times */
/*---------------------------------------*/

typedef struct {

  int      n_t_steps_max ;   /* Max. number of forced time steps */
  int      n_t_vals_max;     /* Max. number of forced time values */

  int      n_t_steps;        /* Number of forced time steps */
  int      n_t_vals;         /* Number of forced time values */

  int     *t_steps;          /* Forced output time steps (unordered) */
  double  *t_vals;           /* Forced output time values (unordered) */

} cs_post_writer_times_t;

/* Writer structure definition parameters */
/*----------------------------------------*/

typedef struct {

  fvm_writer_time_dep_t   time_dep;     /* Time dependency */
  int                     fmt_id;       /* format id */
  char                   *case_name;    /* Case (writer) name */
  char                   *dir_name;     /* Associated directory name */
  char                   *fmt_opts;     /* Format options */

} cs_post_writer_def_t;

/* Mesh location type */
/*--------------------*/

typedef enum {

  CS_POST_LOCATION_CELL,         /* Values located at cells */
  CS_POST_LOCATION_I_FACE,       /* Values located at interior faces */
  CS_POST_LOCATION_B_FACE,       /* Values located at boundary faces */
  CS_POST_LOCATION_VERTEX,       /* Values located at vertices */
  CS_POST_LOCATION_PARTICLE,     /* Values located at particles */

} cs_post_location_t;

/* Writer structure */
/*------------------*/

/* This object is based on a choice of a case, directory, and format,
   as well as a flag for associated mesh's time dependency, and the default
   output interval for associated variables. */

typedef struct {

  int            id;            /* Identifier (< 0 for "reservable" writer,
                                 * > 0 for user writer */
  int            active;        /* -1 if blocked at this stage,
                                   0 if no output at current time step,
                                   1 in case of output */
  cs_time_control_t        tc;  /* Time control sub-structure */


  cs_post_writer_times_t  *ot;  /* Specific output times */
  cs_post_writer_def_t    *wd;  /* Associated writer definition */

  fvm_writer_t  *writer;        /* Associated FVM writer */

} cs_post_writer_t;

/* Post-processing mesh structure */
/*--------------------------------*/

/* This object manages the link between an exportable mesh and
   associated writers. */

typedef struct {

  int                     id;            /* Identifier (< 0 for "reservable"
                                            mesh, > 0 for user mesh */

  char                   *name;          /* Mesh name */
  char                   *criteria[5];   /* Base selection criteria for
                                            cells, interior faces,
                                            boundary faces, and particles
                                            respectively */
  cs_post_elt_select_t   *sel_func[5];   /* Advanced selection functions for
                                            cells, interior faces,
                                            boundary faces, particles and
                                            probes respectively */
  void                   *sel_input[5];  /* Advanced selection input for
                                            matching selection functions */
  int                     ent_flag[5];   /* Presence of cells (ent_flag[0],
                                            interior faces (ent_flag[1]),
                                            boundary faces (ent_flag[2]),
                                            or particles (ent_flag[3] = 1
                                            for particles, 2 for trajectories),
                                            probes (ent_flag[4] = 1 for
                                            monitoring probes, 2 for profile)
                                            on one processor at least */

  int                     location_id;   /* Associated location id if defined
                                            by location id, or -1 */
  int                     cat_id;        /* Optional category id as regards
                                            variables output (CS_POST_MESH_...,
                                            0 by default) */

  int                     edges_ref;     /* Base mesh for edges mesh */
  int                     locate_ref;    /* Base mesh for location mesh */

  bool                    add_groups;    /* Add group information if present */
  bool                    post_domain;   /* Output domain number in parallel
                                            if true */
  bool                    time_varying;  /* Time varying if associated writers
                                            allow it */
  bool                    centers_only;  /* Build only associated centers,
                                            not full elements. */

  int                     n_writers;     /* Number of associated writers */
  int                    *writer_id;     /* Array of associated writer ids */
  int                    *nt_last;       /* Time step number for the last
                                            output (-2 before first output,
                                            -1 for time-indepedent output)
                                            for each associated writer */

  cs_lnum_t               n_i_faces;     /* N. associated interior faces */
  cs_lnum_t               n_b_faces;     /* N. associated boundary faces */

  double                  density;       /* Particles density in case
                                            of particle mesh */

  const fvm_nodal_t      *exp_mesh;      /* Associated exportable mesh */
  fvm_nodal_t            *_exp_mesh;     /* Associated exportable mesh,
                                            if owner */

  fvm_writer_time_dep_t   mod_flag_min;  /* Minimum mesh time dependency */
  fvm_writer_time_dep_t   mod_flag_max;  /* Maximum mesh time dependency */

  int                     n_a_fields;    /* Number of additional fields
                                            (in addition to those output
                                            through "cat_id */
  int                   *a_field_info;   /* For each additional field,
                                            associated writer id, field id,
                                            and component id */

} cs_post_mesh_t;

/*============================================================================
 * Static global variables
 *============================================================================*/

/* Default output format and options */

static int _cs_post_default_format_id = 0;
static char *_cs_post_default_format_options = nullptr;

/* Minimum global mesh time dependency */

fvm_writer_time_dep_t  _cs_post_mod_flag_min = FVM_WRITER_FIXED_MESH;

/* Flag for stable numbering of particles */

static bool        _number_particles_by_coord = false;

/* Array of exportable meshes associated with post-processing;
   free ids start under the last CS_POST_MESH_* definition,
   currently at -5) */

static int              _cs_post_min_mesh_id = _MIN_RESERVED_MESH_ID;
static int              _cs_post_n_meshes = 0;
static int              _cs_post_n_meshes_max = 0;
static cs_post_mesh_t  *_cs_post_meshes = nullptr;

/* Array of writers for post-processing; */
/* writers CS_POST_WRITER_... are reserved */

static int                _cs_post_min_writer_id = _MIN_RESERVED_WRITER_ID;
static int                _cs_post_n_writers = 0;
static int                _cs_post_n_writers_max = 0;
static cs_post_writer_t  *_cs_post_writers = nullptr;

/* Array of registered variable output functions and instances */

static int                _cs_post_n_output_tp = 0;
static int                _cs_post_n_output_tp_max = 0;

static int                _cs_post_n_output_mtp = 0;
static int                _cs_post_n_output_mtp_max = 0;

static cs_post_time_dep_output_t  **_cs_post_f_output_tp = nullptr;
static void                       **_cs_post_i_output_tp = nullptr;

static cs_post_time_mesh_dep_output_t  **_cs_post_f_output_mtp = nullptr;
static void                            **_cs_post_i_output_mtp = nullptr;

/* Default directory name */

static const char  _cs_post_dirname[] = "postprocessing";

/* Local flag for field synchronization */

static char *_field_sync = nullptr;

/* Timer statistics */

static int  _post_out_stat_id = -1;

/*============================================================================
 * Prototypes for functions intended for use only by Fortran wrappers.
 * (descriptions follow, with function bodies).
 *============================================================================*/

void
cs_f_post_activate_by_time_step(void);

void
cs_f_post_write_var(int               mesh_id,
                    const char       *var_name,
                    int               var_dim,
                    bool              interlace,
                    bool              use_parent,
                    int               nt_cur_abs,
                    double            t_cur_abs,
                    const cs_real_t  *cel_vals,
                    const cs_real_t  *i_face_vals,
                    const cs_real_t  *b_face_vals);

/*============================================================================
 * Prototypes for user functions called only by functions from this module.
 *============================================================================*/

/*----------------------------------------------------------------------------
 * User function for output of values on a post-processing mesh.
 *
 * \param[in]       mesh_name    name of the output mesh for the current call
 * \param[in]       mesh_id      id of the output mesh for the current call
 * \param[in]       cat_id       category id of the output mesh for the
 *                               current call
 * \param[in]       probes       pointer to associated probe set structure if
 *                               the mesh is a probe set, null otherwise
 * \param[in]       n_cells      local number of cells of post_mesh
 * \param[in]       n_i_faces    local number of interior faces of post_mesh
 * \param[in]       n_b_faces    local number of boundary faces of post_mesh
 * \param[in]       n_vertices   local number of vertices faces of post_mesh
 * \param[in]       cell_list    list of cells (0 to n-1) of post-processing mesh
 * \param[in]       i_face_list  list of interior faces (0 to n-1) of
 *                               post-processing mesh
 * \param[in]       b_face_list  list of boundary faces (0 to n-1) of
 *                               post-processing mesh
 * \param[in]       vertex_list  list of vertices (0 to n-1) of
 *                               post-processing mesh
 * \param[in]       ts           time step status structure, or null
 *----------------------------------------------------------------------------*/

void
cs_user_postprocess_values(const char            *mesh_name,
                           int                    mesh_id,
                           int                    cat_id,
                           cs_probe_set_t        *probes,
                           cs_lnum_t              n_cells,
                           cs_lnum_t              n_i_faces,
                           cs_lnum_t              n_b_faces,
                           cs_lnum_t              n_vertice,
                           const cs_lnum_t        cell_list[],
                           const cs_lnum_t        i_face_list[],
                           const cs_lnum_t        b_face_list[],
                           const cs_lnum_t        vertex_list[],
                           const cs_time_step_t  *ts);

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Transform time-independent values into time-dependent
 *        values for transient meshes.
 *
 * \param[in]       writer  pointer to associated writer
 * \param[in, out]  nt_cur  associated time step (-1 initially for
 *                          time-independent values)
 * \param[in, out]  t_cur   associated time value
 */
/*----------------------------------------------------------------------------*/

static inline void
_check_non_transient(const cs_post_writer_t  *writer,
                     int                     *nt_cur,
                     double                  *t_cur)
{
  assert(writer->active > 0);
  assert(writer->writer != nullptr);

  fvm_writer_time_dep_t time_dep = fvm_writer_get_time_dep(writer->writer);

  if (time_dep == FVM_WRITER_TRANSIENT_CONNECT) {
    *nt_cur = writer->tc.last_nt;
    *t_cur = writer->tc.last_t;
  }
}

/*----------------------------------------------------------------------------
 * Clear temporary writer definition information.
 *
 * parameters:
 *   writer <-> pointer to writer structure
 *----------------------------------------------------------------------------*/

static void
_destroy_writer_def(cs_post_writer_t  *writer)
{
  assert(writer != nullptr);
  if (writer->wd != nullptr) {
    cs_post_writer_def_t  *wd = writer->wd;
    CS_FREE(wd->case_name);
    CS_FREE(wd->dir_name);
    CS_FREE(wd->fmt_opts);
    CS_FREE(writer->wd);
  }
}

/*----------------------------------------------------------------------------
 * Print writer information to log file
 *----------------------------------------------------------------------------*/

static void
_writer_info(void)
{
  if (cs_glob_rank_id < 1) {

    cs_log_t log = CS_LOG_SETUP;

    cs_log_printf(log, _("\n"
                         "Postprocessing output writers:\n"
                         "------------------------------\n\n"));

    for (int i = 0; i < _cs_post_n_writers; i++) {

      int fmt_id = 0, n_fmt_str = 0;
      fvm_writer_time_dep_t   time_dep = FVM_WRITER_FIXED_MESH;
      const char  *fmt_name, *fmt_opts = nullptr;
      const char  *case_name = nullptr, *dir_name = nullptr;
      const char empty[] = "";
      char interval_s[128] = "";

      const cs_post_writer_t  *writer = _cs_post_writers + i;

      if (writer->wd != nullptr) {
        const cs_post_writer_def_t *wd = writer->wd;
        fmt_id = wd->fmt_id;
        time_dep = wd->time_dep;
        fmt_opts = wd->fmt_opts;
        case_name = wd->case_name;
        dir_name = wd->dir_name;
      }
      else if (writer->writer != nullptr) {
        const fvm_writer_t *w = writer->writer;
        fmt_id = fvm_writer_get_format_id(fvm_writer_get_format(w));
        time_dep = fvm_writer_get_time_dep(w);
        case_name = fvm_writer_get_name(w);
        fmt_opts = fvm_writer_get_options(w);
        dir_name = fvm_writer_get_path(w);
      }
      if (fmt_opts == nullptr)
        fmt_opts = empty;

      n_fmt_str = fvm_writer_n_version_strings(fmt_id);
      if (n_fmt_str == 0)
        fmt_name = fvm_writer_format_name(fmt_id);
      else
        fmt_name = fvm_writer_version_string(fmt_id, 0, 0);

      cs_time_control_get_description(&(writer->tc), interval_s, 128);

      cs_log_printf(log,
                    _("  %2d: name: %s\n"
                      "      directory: %s\n"
                      "      format: %s\n"
                      "      options: %s\n"
                      "      time dependency: %s\n"
                      "      output: %s\n\n"),
                    writer->id, case_name, dir_name, fmt_name, fmt_opts,
                    _(fvm_writer_time_dep_name[time_dep]), interval_s);
    }
  }
}

/*----------------------------------------------------------------------------
 * Initialize a writer; this creates the FVM writer structure, and
 * clears the temporary writer definition information.
 *
 * parameters:
 *   writer <-> pointer to writer structure
 *----------------------------------------------------------------------------*/

static void
_init_writer(cs_post_writer_t  *writer)
{
  assert(writer != nullptr);

  if (writer->writer == nullptr) {

    cs_post_writer_def_t  *wd = writer->wd;

    /* Sanity checks */
    assert(writer->wd != nullptr);
    if (wd->fmt_id >= fvm_writer_n_formats())
      bft_error(__FILE__, __LINE__, 0,
                _(" Invalid format name for writer (case: %s, dirname: %s)."),
                wd->case_name, wd->dir_name);

    writer->writer = fvm_writer_init(wd->case_name,
                                     wd->dir_name,
                                     fvm_writer_format_name(wd->fmt_id),
                                     wd->fmt_opts,
                                     wd->time_dep);
    _destroy_writer_def(writer);

  }

}

/*----------------------------------------------------------------------------
 * Free a writer's forced output time values.
 *
 * parameters:
 *   w <-> pointer to writer structure
 *----------------------------------------------------------------------------*/

static void
_free_writer_times(cs_post_writer_t  *w)
{
  assert(w != nullptr);

  if (w->ot == nullptr) {
    CS_FREE(w->ot->t_vals);
    CS_FREE(w->ot->t_steps);
    CS_FREE(w->ot);
  }
}

/*----------------------------------------------------------------------------
 * Create a specific writer output times structure.
 *
 * returns:
 *   structure for handling of specific output times
 *----------------------------------------------------------------------------*/

static cs_post_writer_times_t *
_writer_times_create(void)
{
  cs_post_writer_times_t  *ot;
  CS_MALLOC(ot, 1, cs_post_writer_times_t);

  ot->n_t_steps_max = 0;
  ot->n_t_vals_max = 0;

  ot->n_t_steps = 0;
  ot->n_t_vals = 0;

  ot->t_steps = nullptr;
  ot->t_vals = nullptr;

  return ot;
}

/*----------------------------------------------------------------------------
 * Add an activation time step for a specific writer.
 *
 * If a negative value is provided, a previously added activation time
 * step matching that absolute value will be removed, if present.
 *
 * parameters:
 *   writer_id <-- writer id, or 0 for all writers
 *   nt        <-- time step value to add (or remove)
 *----------------------------------------------------------------------------*/

static void
_add_writer_ts(cs_post_writer_t  *w,
               int                nt)
{
  int prev_id;
  int nt_abs = cs::abs(nt);

  if (w->ot == nullptr)
    w->ot = _writer_times_create();

  /* Search for previous value */

  for (prev_id = 0; prev_id < w->ot->n_t_steps; prev_id++) {
    if (w->ot->t_steps[prev_id] == nt_abs)
      break;
  }

  /* If value already present */

  if (prev_id < w->ot->n_t_steps) {

    /* Remove previous value from unsorted list (swap with last, remove last) */

    if (nt < 0) {
      w->ot->t_steps[prev_id] = w->ot->t_steps[w->ot->n_t_steps - 1];
      w->ot->n_t_steps -= 1;
    }

  }

  /* If values not already present */

  else if (nt > -1) {

    if (w->ot->n_t_steps_max < w->ot->n_t_steps + 1) {
      if (w->ot->n_t_steps_max == 0)
        w->ot->n_t_steps_max = 1;
      else
        w->ot->n_t_steps_max *= 2;
      CS_REALLOC(w->ot->t_steps, w->ot->n_t_steps_max, int);
    }

    w->ot->t_steps[w->ot->n_t_steps] = nt;
    w->ot->n_t_steps += 1;

  }
}

/*----------------------------------------------------------------------------
 * Add an activation time value for a specific writer.
 *
 * If a negative value is provided, a previously added activation time
 * step matching that absolute value will be removed, if present.
 *
 * parameters:
 *   writer_id <-- writer id, or 0 for all writers
 *   t         <-- time value to add (or remove)
 *----------------------------------------------------------------------------*/

static void
_add_writer_tv(cs_post_writer_t  *w,
               double             t)
{
  int prev_id;
  double t_abs = cs::abs(t);

  if (w->ot == nullptr)
    w->ot = _writer_times_create();

  /* Search for previous value */

  for (prev_id = 0; prev_id < w->ot->n_t_steps; prev_id++) {
    double td = w->ot->t_vals[prev_id] - t_abs;
    if (td > -1.e-35 && td < 1.e-35)
      break;
  }

  /* If value already present */

  if (prev_id < w->ot->n_t_vals) {

    /* Remove previous value from unsorted list (swap with last, remove last) */

    if (t < 0.) {
      w->ot->t_vals[prev_id] = w->ot->t_vals[w->ot->n_t_vals - 1];
      w->ot->n_t_vals -= 1;
    }

  }

  /* If values not already present */

  else if (t >= 0.) {

    if (w->ot->n_t_vals_max < w->ot->n_t_vals + 1) {
      if (w->ot->n_t_vals_max == 0)
        w->ot->n_t_vals_max = 1;
      else
        w->ot->n_t_vals_max *= 2;
      CS_REALLOC(w->ot->t_vals, w->ot->n_t_vals_max, double);
    }

    w->ot->t_vals[w->ot->n_t_vals] = t;
    w->ot->n_t_vals += 1;

  }
}

/*----------------------------------------------------------------------------
 * Update "active" or "inactive" flag of a writer based on specified
 * output lists.
 *
 * parameters:
 *   w  <-> pointer to writer structure
 *   ts <-- time step status structure
 *----------------------------------------------------------------------------*/

static void
_activate_if_listed(cs_post_writer_t      *w,
                    const cs_time_step_t  *ts)
{
  int  i;
  bool force_status = false;
  int prev_status = w->active;

  cs_post_writer_times_t *ot = w->ot;

  /* If no output times list is provided, nothing to do */

  if (ot == nullptr)
    return;

  /* In case of previous calls for a given time step,
     do not change status (which must have been forced otherwise),
     but update lists so as not to provoke an output at the next
     time step (so as to be consistent with the forcing that must have
     been done prior to entering here for this situation to exist). */

  if (w->tc.last_nt == ts->nt_cur)
    force_status = true;

  /* Test for listed time steps */

  i = 0;
  while (i < ot->n_t_steps) {
    /* Activate, then remove current or previous time steps from list */
    if (ot->t_steps[i] <= ts->nt_cur) {
      if (w->active > -1)
        w->active = 1;
      ot->t_steps[i] = ot->t_steps[ot->n_t_steps - 1];
      ot->n_t_steps -= 1;
    }
    else
      i++;
  }

  /* Test for listed time values */

  i = 0;
  while (i < ot->n_t_vals) {
    /* Activate, then remove current or previous time values from list */
    if (ot->t_vals[i] <= ts->t_cur) {
      if (w->active > -1)
        w->active = 1;
      ot->t_vals[i] = ot->t_vals[ot->n_t_steps - 1];
      ot->n_t_vals -= 1;
    }
    else
      i++;
  }

  if (force_status)
    w->active = prev_status;
}

/*----------------------------------------------------------------------------
 * Search for position in the array of writers of a writer with a given id.
 *
 * parameters:
 *   writer_id <-- id of writer
 *
 * returns:
 *   position in the array of writers
 *----------------------------------------------------------------------------*/

static int
_cs_post_writer_id(const int  writer_id)
{
  int  id;

  cs_post_writer_t  *writer = nullptr;

  /* Search for requested writer */

  for (id = 0; id < _cs_post_n_writers; id++) {
    writer = _cs_post_writers + id;
    if (writer->id == writer_id)
      break;
  }
  if (id >= _cs_post_n_writers)
    bft_error(__FILE__, __LINE__, 0,
              _("The requested post-processing writer number\n"
                "%d is not defined.\n"), (int)(writer_id));

  return id;
}

/*----------------------------------------------------------------------------
 * Search for position in the array of writers of a writer with a given id,
 * allowing the writer not to be present.
 *
 * parameters:
 *   writer_id <-- id of writer
 *
 * returns:
 *   position in the array of writers, or -1
 *----------------------------------------------------------------------------*/

static int
_cs_post_writer_id_try(const int  writer_id)
{
  int  id;

  cs_post_writer_t  *writer = nullptr;

  /* Search for requested writer */

  for (id = 0; id < _cs_post_n_writers; id++) {
    writer = _cs_post_writers + id;
    if (writer->id == writer_id)
      break;
  }
  if (id >= _cs_post_n_writers)
    id = -1;

  return id;
}

/*----------------------------------------------------------------------------
 * Search for position in the array of meshes of a mesh with a given id.
 *
 * parameters:
 *   mesh_id <-- id of mesh
 *
 * returns:
 *   position in the array of meshes
 *----------------------------------------------------------------------------*/

static int
_cs_post_mesh_id(int  mesh_id)
{
  int id;
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Search for requested mesh */

  for (id = 0; id < _cs_post_n_meshes; id++) {
    post_mesh = _cs_post_meshes + id;
    if (post_mesh->id == mesh_id)
      break;
  }
  if (id >= _cs_post_n_meshes)
    bft_error(__FILE__, __LINE__, 0,
              _("The requested post-processing mesh number\n"
                "%d is not defined.\n"), (int)mesh_id);

  return id;
}

/*----------------------------------------------------------------------------
 * Search for position in the array of meshes of a mesh with a given id,
 * allowing the id not to be present
 *
 * parameters:
 *   mesh_id <-- id of mesh
 *
 * returns:
 *   position in the array of meshes, or -1
 *----------------------------------------------------------------------------*/

static int
_cs_post_mesh_id_try(int  mesh_id)
{
  int id;
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Search for requested mesh */

  for (id = 0; id < _cs_post_n_meshes; id++) {
    post_mesh = _cs_post_meshes + id;
    if (post_mesh->id == mesh_id)
      break;
  }
  if (id >= _cs_post_n_meshes)
    id = -1;

  return id;
}

/*----------------------------------------------------------------------------
 * Return indicator base on Lagrangian calculation status:
 *
 * parameters:
 *   ts            <-- time step structure, or null
 *
 * returns:
 *   0 if Lagrangian model is not active
 *   1 if Lagrangian model is active but no particle data is ready
 *   2 if current but not previous particle data is present
 *   3 if current and previous particle data is present
 *----------------------------------------------------------------------------*/

static int
_lagrangian_needed(const cs_time_step_t  *ts)
{
  int retval = 0;

  int _model = cs_lagr_model_type();

  if (_model != 0) {

    retval = 1;

    if (ts != nullptr) {
      int _restart = cs_lagr_particle_restart();
      int _nt_start = (_restart) ? ts->nt_prev : ts->nt_prev + 1;
      if (ts->nt_cur == _nt_start)
        retval = 2;
      else if (ts->nt_cur > _nt_start)
        retval = 3;
    }

  }

  return retval;
}

/*----------------------------------------------------------------------------
 * Update mesh attributes related to writer association.
 *
 * parameters:
 *   post_mesh  <-> pointer to postprocessing mesh
 *----------------------------------------------------------------------------*/

static void
_update_mesh_writer_associations(cs_post_mesh_t  *post_mesh)
{
  /* Minimum and maximum time dependency flags initially inverted,
     will be recalculated after mesh - writer associations */

  if (post_mesh->time_varying)
    post_mesh->mod_flag_min = FVM_WRITER_TRANSIENT_CONNECT;
  else
    post_mesh->mod_flag_min = _cs_post_mod_flag_min;
  post_mesh->mod_flag_max = FVM_WRITER_FIXED_MESH;

  int   n_writers = post_mesh->n_writers;

  if (post_mesh->ent_flag[3] == 0) { /* Non-Lagrangian mesh */

    for (int i = 0; i < n_writers; i++) {

      fvm_writer_time_dep_t mod_flag;
      const int _writer_id = post_mesh->writer_id[i];
      cs_post_writer_t  *writer = _cs_post_writers + _writer_id;

      if (writer->wd != nullptr)
        mod_flag = writer->wd->time_dep;
      else
        mod_flag = fvm_writer_get_time_dep(writer->writer);

      if (mod_flag < post_mesh->mod_flag_min)
        post_mesh->mod_flag_min = mod_flag;
      if (mod_flag > post_mesh->mod_flag_max)
        post_mesh->mod_flag_max = mod_flag;

    }

  }
  else { /* Lagrangian mesh: post_mesh->ent_flag[3] != 0 */

    int mode = post_mesh->ent_flag[3];
    fvm_writer_time_dep_t mod_type = (mode == 2) ?
      FVM_WRITER_FIXED_MESH : FVM_WRITER_TRANSIENT_CONNECT;

    post_mesh->mod_flag_min = FVM_WRITER_TRANSIENT_CONNECT;
    post_mesh->mod_flag_max = FVM_WRITER_TRANSIENT_CONNECT;

    int i, j;
    for (i = 0, j = 0; i < n_writers; i++) {

      fvm_writer_time_dep_t mod_flag;
      const int _writer_id = post_mesh->writer_id[i];
      cs_post_writer_t  *writer = _cs_post_writers + _writer_id;

      if (writer->wd != nullptr)
        mod_flag = writer->wd->time_dep;
      else
        mod_flag = fvm_writer_get_time_dep(writer->writer);

      if (mod_flag == mod_type) {
        post_mesh->writer_id[j] = _writer_id;
        post_mesh->nt_last[j] = post_mesh->nt_last[i];
        j++;
      }

    }

    if (j < n_writers) {
      post_mesh->n_writers = j;
      CS_REALLOC(post_mesh->writer_id, j, int);
      CS_REALLOC(post_mesh->nt_last, j, int);
    }

  }
}

/*----------------------------------------------------------------------------
 * Add or select a post-processing mesh, do basic initialization, and return
 * a pointer to the associated structure.
 *
 * parameters:
 *   mesh_id      <-- requested mesh id
 *   time_varying <-- if true, mesh may be redefined over time if associated
 *                    writers allow it
 *   mode         <-- 0 for standard mesh, 1 for particles, 2 for trajectories,
 *                    3 for probes, 4 for profiles
 *   n_writers    <-- number of associated writers
 *   writer_ids   <-- ids of associated writers
 *
 * returns:
 *   pointer to associated structure
 *----------------------------------------------------------------------------*/

static cs_post_mesh_t *
_predefine_mesh(int        mesh_id,
                bool       time_varying,
                int        mode,
                int        n_writers,
                const int  writer_ids[])
{
  /* local variables */

  int  i, j;

  cs_post_mesh_t  *post_mesh = nullptr;

  /* Check that the requested mesh is available */

  if (mesh_id == 0)
      bft_error(__FILE__, __LINE__, 0,
                _("The requested post-processing mesh number\n"
                  "must be < 0 (reserved) or > 0 (user).\n"));

  for (i = 0; i < _cs_post_n_meshes; i++) {
    if ((_cs_post_meshes + i)->id == mesh_id) {

      post_mesh = _cs_post_meshes + i;

      CS_FREE(post_mesh->name);
      for (j = 0; j < 5; j++)
        CS_FREE(post_mesh->criteria[j]);
      CS_FREE(post_mesh->writer_id);
      CS_FREE(post_mesh->nt_last);

      post_mesh->exp_mesh = nullptr;
      if (post_mesh->_exp_mesh != nullptr)
        post_mesh->_exp_mesh = fvm_nodal_destroy(post_mesh->_exp_mesh);

      break;

    }
  }

  if (i == _cs_post_n_meshes) {

    /* Resize global array of exportable meshes */

    if (_cs_post_n_meshes == _cs_post_n_meshes_max) {
      if (_cs_post_n_meshes_max == 0)
        _cs_post_n_meshes_max = 8;
      else
        _cs_post_n_meshes_max *= 2;
      CS_REALLOC(_cs_post_meshes,
                 _cs_post_n_meshes_max,
                 cs_post_mesh_t);
    }

    post_mesh = _cs_post_meshes + i;

    _cs_post_n_meshes += 1;
  }

  if (mesh_id < _cs_post_min_mesh_id)
    _cs_post_min_mesh_id = mesh_id;

  /* Assign newly created mesh to the structure */

  post_mesh->id = mesh_id;
  post_mesh->name = nullptr;
  post_mesh->cat_id = mesh_id;
  post_mesh->location_id = -1;
  post_mesh->edges_ref = -1;
  post_mesh->locate_ref = -1;

  post_mesh->n_writers = 0;
  post_mesh->writer_id = nullptr;
  post_mesh->nt_last = nullptr;

  post_mesh->add_groups = false;
  post_mesh->post_domain = false;

  post_mesh->time_varying = time_varying;
  post_mesh->centers_only = false;

  for (j = 0; j < 5; j++) {
    post_mesh->criteria[j] = nullptr;
    post_mesh->sel_func[j] = nullptr;
    post_mesh->sel_input[j] = nullptr;
    post_mesh->ent_flag[j] = 0;
  }

  post_mesh->n_i_faces = 0;
  post_mesh->n_b_faces = 0;

  post_mesh->density = 1.;

  post_mesh->exp_mesh = nullptr;
  post_mesh->_exp_mesh = nullptr;

  /* Minimum and maximum time dependency flags initially inverted,
     will be recalculated after mesh - writer associations */

  if (post_mesh->time_varying)
    post_mesh->mod_flag_min = FVM_WRITER_TRANSIENT_CONNECT;
  else
    post_mesh->mod_flag_min = _cs_post_mod_flag_min;
  post_mesh->mod_flag_max = FVM_WRITER_FIXED_MESH;

  /* Associate mesh with writers */

  post_mesh->n_writers = n_writers;
  CS_MALLOC(post_mesh->writer_id, n_writers, int);
  CS_MALLOC(post_mesh->nt_last, n_writers, int);

  for (i = 0; i < n_writers; i++) {
    post_mesh->writer_id[i] = _cs_post_writer_id(writer_ids[i]);
    post_mesh->nt_last[i] = -2;
  }

  if (mode == 1 || mode == 2)          /* Lagrangian mesh */
    post_mesh->ent_flag[3] = mode;

  else if (mode == 3 || mode == 4)     /* Probe or profile mesh */
    post_mesh->ent_flag[4] = mode - 2; /* 1 = probe monitoring,
                                          2 = profile */

  _update_mesh_writer_associations(post_mesh);

  /* Additional field output */

  post_mesh->n_a_fields = 0;
  post_mesh->a_field_info = nullptr;

  return post_mesh;
}

/*----------------------------------------------------------------------------
 * Free a postprocessing mesh's data.
 *
 * parameters:
 *   _mesh_id <-- local id of mesh to remove
 *----------------------------------------------------------------------------*/

static void
_free_mesh(int _mesh_id)
{
  int i;
  cs_post_mesh_t  *post_mesh = _cs_post_meshes + _mesh_id;

  if (post_mesh->_exp_mesh != nullptr)
    post_mesh->_exp_mesh = fvm_nodal_destroy(post_mesh->_exp_mesh);

  CS_FREE(post_mesh->writer_id);
  CS_FREE(post_mesh->nt_last);
  post_mesh->n_writers = 0;

  for (i = 0; i < 5; i++)
    CS_FREE(post_mesh->criteria[i]);

  CS_FREE(post_mesh->name);
  CS_FREE(post_mesh->a_field_info);

  /* Shift remaining meshes */

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->locate_ref > _mesh_id)
      post_mesh->locate_ref -= 1;
    else if (post_mesh->locate_ref == _mesh_id)
      post_mesh->locate_ref = -1;
    if (post_mesh->edges_ref >= _mesh_id) {
      assert(post_mesh->edges_ref != _mesh_id);
      post_mesh->edges_ref -= 1;
    }
  }

  for (i = _mesh_id + 1; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    _cs_post_meshes[i-1] = _cs_post_meshes[i];
  }
  _cs_post_n_meshes -= 1;
}

/*----------------------------------------------------------------------------
 * Check and possibly fix postprocessing mesh category id once we have
 * knowledge of entity counts.
 *
 * parameters:
 *   post_mesh <-> pointer to partially initialized post-processing mesh
 *----------------------------------------------------------------------------*/

static void
_check_mesh_cat_id(cs_post_mesh_t  *post_mesh)
{
  if (   post_mesh->cat_id == CS_POST_MESH_VOLUME
      || post_mesh->cat_id == CS_POST_MESH_BOUNDARY
      || post_mesh->cat_id == CS_POST_MESH_SURFACE) {
    const int *ef = post_mesh->ent_flag;
    if (ef[0] == 1 && ef[1] == 0 && ef[2] == 0)
      post_mesh->cat_id = CS_POST_MESH_VOLUME;
    else if (ef[0] == 0 && ef[1] == 0 && ef[2] == 1)
      post_mesh->cat_id = CS_POST_MESH_BOUNDARY;
    else if (ef[0] == 0 && (ef[1] == 1 || ef[2] == 1))
      post_mesh->cat_id = CS_POST_MESH_SURFACE;
  }
}

/*----------------------------------------------------------------------------
 * Create a post-processing mesh; lists of cells or faces to extract are
 * sorted upon exit, whether they were sorted upon calling or not.
 *
 * The list of associated cells is only necessary if the number of cells
 * to extract is strictly greater than 0 and less than the number of cells
 * of the computational mesh.
 *
 * Lists of faces are ignored if the number of extracted cells is nonzero;
 * otherwise, if the number of boundary faces to extract is equal to the
 * number of boundary faces in the computational mesh, and the number of
 * interior faces to extract is zero, then we extract by default the boundary
 * mesh, and the list of associated boundary faces is thus not necessary.
 *
 * parameters:
 *   post_mesh   <-> pointer to partially initialized post-processing mesh
 *   n_cells     <-- number of associated cells
 *   n_i_faces   <-- number of associated interior faces
 *   n_b_faces   <-- number of associated boundary faces
 *   cell_list   <-> list of associated cells
 *   i_face_list <-> list of associated interior faces
 *   b_face_list <-> list of associated boundary faces
 *----------------------------------------------------------------------------*/

static void
_define_export_mesh(cs_post_mesh_t  *post_mesh,
                    cs_lnum_t        n_cells,
                    cs_lnum_t        n_i_faces,
                    cs_lnum_t        n_b_faces,
                    cs_lnum_t        cell_list[],
                    cs_lnum_t        i_face_list[],
                    cs_lnum_t        b_face_list[])
{
  /* local variables */

  fvm_nodal_t  *exp_mesh = nullptr;

  /* Create associated structure */

  if (post_mesh->centers_only == false) {

    if (post_mesh->ent_flag[0] == 1) {

      if (n_cells >= cs_glob_mesh->n_cells)
        exp_mesh = cs_mesh_connect_cells_to_nodal(cs_glob_mesh,
                                                  post_mesh->name,
                                                  post_mesh->add_groups,
                                                  cs_glob_mesh->n_cells,
                                                  nullptr);
      else
        exp_mesh = cs_mesh_connect_cells_to_nodal(cs_glob_mesh,
                                                  post_mesh->name,
                                                  post_mesh->add_groups,
                                                  n_cells,
                                                  cell_list);

    }
    else {

      if (   n_b_faces >= cs_glob_mesh->n_b_faces
          && n_i_faces == 0)
        exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                  post_mesh->name,
                                                  post_mesh->add_groups,
                                                  0,
                                                  cs_glob_mesh->n_b_faces,
                                                  nullptr,
                                                  nullptr);
      else
        exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                  post_mesh->name,
                                                  post_mesh->add_groups,
                                                  n_i_faces,
                                                  n_b_faces,
                                                  i_face_list,
                                                  b_face_list);

    }

  }

  /* Create only associated points if requested */

  else {

    cs_lnum_t n_elts = 0;
    cs_lnum_t *elt_ids = nullptr;
    const cs_real_3_t *elt_coords = nullptr;
    const cs_gnum_t *elt_gnum = nullptr;

    if (post_mesh->ent_flag[0] == 1) {

      if (n_cells >= cs_glob_mesh->n_cells) {
        n_elts = cs_glob_mesh->n_cells;
      }
      else {
        n_elts = n_cells;
        elt_ids = cell_list;
      }

      elt_coords = cs_glob_mesh_quantities->cell_cen;
      elt_gnum = cs_glob_mesh->global_cell_num;

    }
    else {

      if (post_mesh->ent_flag[1] == 0) {

        if (n_b_faces >= cs_glob_mesh->n_b_faces) {
          n_elts = cs_glob_mesh->n_b_faces;
        }
        else {
          n_elts = n_b_faces;
          elt_ids = b_face_list;
        }

        elt_coords = cs_glob_mesh_quantities->b_face_cog;
        elt_gnum = cs_glob_mesh->global_b_face_num;

      }
      else if (post_mesh->ent_flag[2] == 0) {

        if (n_i_faces >= cs_glob_mesh->n_i_faces) {
          n_elts = cs_glob_mesh->n_i_faces;
        }
        else {
          n_elts = n_i_faces;
          elt_ids = i_face_list;
        }

        elt_coords = cs_glob_mesh_quantities->i_face_cog;
        elt_gnum = cs_glob_mesh->global_i_face_num;

      }
      else {

        bft_error(__FILE__, __LINE__, 0,
                  _("%s: Mixed interior and boundary faces not currently handled "
                    "with 'centers only' option."),
                  __func__);

      }

      exp_mesh = fvm_nodal_create(post_mesh->name, 3);

      fvm_nodal_define_vertex_list(exp_mesh, n_elts, elt_ids);
      fvm_nodal_set_shared_vertices(exp_mesh, (const cs_real_t *)elt_coords);

      fvm_nodal_init_io_num(exp_mesh, elt_gnum, 0);

    }

  }

  /* Fix category id now that we have knowledge of entity counts */

  _check_mesh_cat_id(post_mesh);

  /* Local dimensions */

  post_mesh->n_i_faces = n_i_faces;
  post_mesh->n_b_faces = n_b_faces;

  /* As faces might be split, ensure the number of faces is correct */

  /* Link to newly created mesh */

  post_mesh->exp_mesh = exp_mesh;
  post_mesh->_exp_mesh = exp_mesh;
}

/*----------------------------------------------------------------------------
 * Create a particles post-processing mesh;
 *
 * parameters:
 *   post_mesh     <-> pointer to partially initialized post-processing mesh
 *   n_particles   <-- number of associated particles
 *   particle_list <-> list of associated particles
 *   ts            <-- time step structure
 *----------------------------------------------------------------------------*/

static void
_define_particle_export_mesh(cs_post_mesh_t        *post_mesh,
                             cs_lnum_t              n_particles,
                             cs_lnum_t              particle_list[],
                             const cs_time_step_t  *ts)
{
  /* local variables */

  fvm_nodal_t  *exp_mesh = nullptr;

  assert(ts != nullptr);

  /* Create associated structure */

  {
    cs_gnum_t *global_num = nullptr;
    cs_coord_3_t *coords = nullptr;
    fvm_io_num_t  *io_num = nullptr;

    cs_lagr_particle_set_t  *p_set = cs_lagr_get_particle_set();

    if (p_set == nullptr)
      return;

    /* Particle positions */

    if (post_mesh->ent_flag[3] == 1) {

      assert(ts->nt_cur > -1);

      exp_mesh = fvm_nodal_create(post_mesh->name, 3);

      CS_MALLOC(coords, n_particles, cs_coord_3_t);

      cs_lagr_get_particle_values(p_set,
                                  CS_LAGR_COORDS,
                                  CS_REAL_TYPE,
                                  3,
                                  -1,
                                  n_particles,
                                  particle_list,
                                  coords);

      fvm_nodal_define_vertex_list(exp_mesh, n_particles, nullptr);
      fvm_nodal_transfer_vertices(exp_mesh, (cs_coord_t *)coords);

    }

    /* Particle trajectories */

    else if (post_mesh->ent_flag[3] == 2) {

      cs_lnum_t i;
      cs_lnum_t  *vertex_num;
      char *mesh_name;

      assert(ts->nt_cur > 0);

      CS_MALLOC(mesh_name, strlen(post_mesh->name) + 32, char);
      sprintf(mesh_name, "%s_%05d", post_mesh->name, ts->nt_cur);

      exp_mesh = fvm_nodal_create(mesh_name, 3);

      CS_FREE(mesh_name);

      CS_MALLOC(vertex_num, n_particles*2, cs_lnum_t);

      for (i = 0; i < n_particles*2; i++)
        vertex_num[i] = i+1;

      CS_MALLOC(coords, n_particles*2, cs_coord_3_t);

      cs_lagr_get_trajectory_values(p_set,
                                    CS_LAGR_COORDS,
                                    CS_REAL_TYPE,
                                    3,
                                    -1,
                                    n_particles,
                                    particle_list,
                                    coords);

      fvm_nodal_append_by_transfer(exp_mesh,
                                   n_particles,
                                   FVM_EDGE,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vertex_num,
                                   nullptr);

      fvm_nodal_transfer_vertices(exp_mesh, (cs_coord_t *)coords);

      for (int j = 0; j < post_mesh->n_writers; j++) {
        if (post_mesh->nt_last[j] < ts->nt_cur)
          post_mesh->nt_last[j] = -2;
      }
    }

    /* Build global numbering if required */

    if (_number_particles_by_coord)
      io_num = fvm_io_num_create_from_sfc((const cs_coord_t *)coords,
                                          3,
                                          n_particles,
                                          FVM_IO_NUM_SFC_MORTON_BOX);
    else if (cs_glob_n_ranks > 1)
      io_num = fvm_io_num_create_from_scan(n_particles);

    if (io_num != nullptr) {

      global_num = fvm_io_num_transfer_global_num(io_num);
      fvm_io_num_destroy(io_num);

      if (post_mesh->ent_flag[3] == 1) {

        fvm_nodal_init_io_num(exp_mesh, global_num, 0);
        CS_FREE(global_num);

      }
      else if (post_mesh->ent_flag[3] == 2) {

        cs_lnum_t i;
        cs_gnum_t *g_coord_num;

        fvm_nodal_init_io_num(exp_mesh, global_num, 1);

        CS_MALLOC(g_coord_num, n_particles*2, cs_gnum_t);
        for (i = 0; i < n_particles; i++) {
          g_coord_num[i*2] = global_num[i]*2 - 1;
          g_coord_num[i*2+1] = global_num[i]*2;
        }
        CS_FREE(global_num);

        fvm_nodal_init_io_num(exp_mesh, g_coord_num, 0);

        CS_FREE(g_coord_num);

      }

    }

    /* Drop trajectory sub-mesh if it is empty
       (otherwise, EnSight is OK, but ParaView can't display variables) */

    if (   post_mesh->ent_flag[3] == 2
        && fvm_nodal_get_n_g_elements(exp_mesh, FVM_EDGE) == 0)
      exp_mesh = fvm_nodal_destroy(exp_mesh);

  }

  /* Fix category id */

  if (post_mesh->cat_id < 0)
    post_mesh->cat_id = CS_POST_MESH_PARTICLES;

  /* Link to newly created mesh */

  post_mesh->exp_mesh = exp_mesh;
  post_mesh->_exp_mesh = exp_mesh;
}

/*----------------------------------------------------------------------------
 * Initialize a volume or surface post-processing mesh based on its
 * selection criteria or selection functions.
 *
 * parameters:
 *   post_mesh <-> pointer to partially initialized post-processing mesh
 *   ts        <-- time step structure
 *----------------------------------------------------------------------------*/

static void
_define_regular_mesh(cs_post_mesh_t  *post_mesh)
{
  const cs_mesh_t *mesh = cs_glob_mesh;

  assert(post_mesh != nullptr);

  assert(post_mesh->exp_mesh == nullptr);

  cs_lnum_t n_cells = 0, n_i_faces = 0, n_b_faces = 0;
  cs_lnum_t *cell_list = nullptr, *i_face_list = nullptr, *b_face_list = nullptr;

  /* Define element lists based on mesh location, selection criteria,
     or selection function. */

  if (post_mesh->location_id > -1) {
    cs_mesh_location_type_t loc_type
      = cs_mesh_location_get_type(post_mesh->location_id);

    const cs_lnum_t n_elts
      = cs_mesh_location_get_n_elts(post_mesh->location_id)[0];
    const cs_lnum_t *elt_ids
      = cs_mesh_location_get_elt_ids_try(post_mesh->location_id);

    cs_lnum_t *elt_list = nullptr;
    if (elt_ids != nullptr) {
      CS_MALLOC(elt_list, n_elts, cs_lnum_t);
      for (cs_lnum_t i = 0; i < n_elts; i++)
        elt_list[i] = elt_ids[i];
    }

    switch(loc_type) {
    case CS_MESH_LOCATION_CELLS:
      n_cells = n_elts;
      cell_list = elt_list;
      elt_list = nullptr;
      break;
    case CS_MESH_LOCATION_INTERIOR_FACES:
      n_i_faces = n_elts;
      i_face_list = elt_list;
      elt_list = nullptr;
      break;
    case CS_MESH_LOCATION_BOUNDARY_FACES:
      n_b_faces = n_elts;
      b_face_list = elt_list;
      elt_list = nullptr;
      break;
    default:
      CS_FREE(elt_list);
      assert(0);
      break;
    }
  }

  else if (post_mesh->criteria[0] != nullptr) {
    const char *criteria = post_mesh->criteria[0];
    if (!strcmp(criteria, "all[]"))
      n_cells = mesh->n_cells;
    else {
      CS_MALLOC(cell_list, mesh->n_cells, cs_lnum_t);
      cs_selector_get_cell_list(criteria, &n_cells, cell_list);
    }
  }
  else if (post_mesh->sel_func[0] != nullptr) {
    cs_post_elt_select_t *sel_func = post_mesh->sel_func[0];
    sel_func(post_mesh->sel_input[0], &n_cells, &cell_list);
  }

  if (post_mesh->criteria[1] != nullptr) {
    const char *criteria = post_mesh->criteria[1];
    if (!strcmp(criteria, "all[]"))
      n_i_faces = mesh->n_i_faces;
    else {
      CS_MALLOC(i_face_list, mesh->n_i_faces, cs_lnum_t);
      cs_selector_get_i_face_list(criteria, &n_i_faces, i_face_list);
    }
  }
  else if (post_mesh->sel_func[1] != nullptr) {
    cs_post_elt_select_t *sel_func = post_mesh->sel_func[1];
    sel_func(post_mesh->sel_input[1], &n_i_faces, &i_face_list);
  }

  if (post_mesh->criteria[2] != nullptr) {
    const char *criteria = post_mesh->criteria[2];
    if (!strcmp(criteria, "all[]"))
      n_b_faces = mesh->n_b_faces;
    else {
      CS_MALLOC(b_face_list, mesh->n_b_faces, cs_lnum_t);
      cs_selector_get_b_face_list(criteria, &n_b_faces, b_face_list);
    }
  }
  else if (post_mesh->sel_func[2] != nullptr) {
    cs_post_elt_select_t *sel_func = post_mesh->sel_func[2];
    sel_func(post_mesh->sel_input[2], &n_b_faces, &b_face_list);
  }

  /* Define mesh based on current arguments */

  _define_export_mesh(post_mesh,
                      n_cells,
                      n_i_faces,
                      n_b_faces,
                      cell_list,
                      i_face_list,
                      b_face_list);

  CS_FREE(cell_list);
  CS_FREE(i_face_list);
  CS_FREE(b_face_list);
}

/*----------------------------------------------------------------------------
 * Create a post-processing mesh for probes
 *
 * parameters:
 *   post_mesh     <-> pointer to partially initialized post-processing mesh
 *----------------------------------------------------------------------------*/

static void
_define_probe_export_mesh(cs_post_mesh_t  *post_mesh)
{
  /* Sanity checks */
  assert(post_mesh != nullptr);

  cs_probe_set_t     *pset = (cs_probe_set_t *)post_mesh->sel_input[4];
  cs_post_mesh_t     *post_mesh_loc = nullptr;
  const fvm_nodal_t  *location_mesh = nullptr;

  /* First step: locate probes and update their coordinates */

  if (post_mesh->locate_ref > -1) {
    post_mesh_loc = _cs_post_meshes + post_mesh->locate_ref;
    if (post_mesh_loc->exp_mesh == nullptr)
      _define_regular_mesh(post_mesh_loc);
    location_mesh = post_mesh_loc->exp_mesh;
  }

  cs_probe_set_locate(pset, location_mesh);

  /* Create associated structure */

  fvm_nodal_t *exp_mesh
    = cs_probe_set_export_mesh(pset, cs_probe_set_get_name(pset));

  /* Link to newly created mesh */

  post_mesh->exp_mesh = exp_mesh;
  post_mesh->_exp_mesh = exp_mesh;

  /* Unassign matching location mesh ids for non-transient probe sets
     to allow freeing them */

  bool time_varying;

  int  n_writers = 0;
  int  *writer_id = nullptr;

  cs_probe_set_get_post_info(pset,
                             &time_varying,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             &n_writers,
                             &writer_id);

  if (time_varying == false)
    post_mesh->locate_ref = -1;
  else if (post_mesh_loc->mod_flag_max < FVM_WRITER_TRANSIENT_COORDS)
    post_mesh_loc->mod_flag_max = FVM_WRITER_TRANSIENT_COORDS;
}

/*----------------------------------------------------------------------------
 * Initialize a post-processing mesh based on its selection criteria
 * or selection functions.
 *
 * parameters:
 *   post_mesh <-> pointer to partially initialized post-processing mesh
 *   ts        <-- time step structure
 *----------------------------------------------------------------------------*/

static void
_define_mesh(cs_post_mesh_t        *post_mesh,
             const cs_time_step_t  *ts)

{
  const cs_mesh_t *mesh = cs_glob_mesh;

  assert(post_mesh != nullptr);

  assert(post_mesh->exp_mesh == nullptr);

  /* Edges mesh */

  if (post_mesh->edges_ref > -1) {

    fvm_nodal_t *exp_edges = nullptr;
    cs_post_mesh_t *post_base
      = _cs_post_meshes + _cs_post_mesh_id(post_mesh->edges_ref);

    /* if base mesh structure is not built yet, force its build now */

    if (post_base->exp_mesh == nullptr)
      _define_mesh(post_base, ts);

    /* Copy mesh edges to new mesh structure */

    exp_edges = fvm_nodal_copy_edges(post_mesh->name, post_mesh->exp_mesh);

    /* Create mesh and assign to structure */

    post_mesh->exp_mesh = exp_edges;
    post_mesh->_exp_mesh = exp_edges;
  }

  /* Particle (Lagrangian) mesh */

  else if (post_mesh->ent_flag[3] != 0 && ts != nullptr) {

    cs_lnum_t n_post_particles = 0, n_particles = cs_lagr_get_n_particles();
    cs_lnum_t *particle_list = nullptr;

    if (post_mesh->criteria[3] != nullptr) {

      cs_lnum_t n_cells = 0;
      cs_lnum_t *cell_list = nullptr;
      const char *criteria = post_mesh->criteria[3];

      if (!strcmp(criteria, "all[]"))
        n_cells = mesh->n_cells;
      else {
        CS_MALLOC(cell_list, mesh->n_cells, cs_lnum_t);
        cs_selector_get_cell_list(criteria, &n_cells, cell_list);
      }
      if (n_cells < mesh->n_cells || post_mesh->density < 1.) {
        CS_MALLOC(particle_list, n_particles, cs_lnum_t);
        cs_lagr_get_particle_list(n_cells,
                                  cell_list,
                                  post_mesh->density,
                                  &n_post_particles,
                                  particle_list);
        CS_REALLOC(particle_list, n_post_particles, cs_lnum_t);
      }
      else
        n_post_particles = n_particles;
      CS_FREE(cell_list);
    }

    else if (post_mesh->sel_func[3] != nullptr) {
      cs_post_elt_select_t *sel_func = post_mesh->sel_func[3];
      sel_func(post_mesh->sel_input[0], &n_post_particles, &particle_list);
    }

    _define_particle_export_mesh(post_mesh,
                                 n_post_particles,
                                 particle_list,
                                 ts);

    CS_FREE(particle_list);
  }

  /* Probe mesh */

  else if (post_mesh->ent_flag[4] != 0) {

    _define_probe_export_mesh(post_mesh);

  }

  /* Standard (non-particle) meshes */

  else
    _define_regular_mesh(post_mesh);
}

/*----------------------------------------------------------------------------
 * Modify an existing post-processing mesh.
 *
 * It is not necessary to use this function if a mesh is simply deformed.
 *
 * parameters:
 *   post_mesh <-- pointer to postprocessing mesh structure
 *   ts        <-- time step structure
 *----------------------------------------------------------------------------*/

static void
_redefine_mesh(cs_post_mesh_t        *post_mesh,
               const cs_time_step_t  *ts)
{
  /* Remove previous base structure (return if we do not own the mesh) */

  if (post_mesh->exp_mesh != nullptr) {
    if (post_mesh->_exp_mesh == nullptr)
      return;
    else
      post_mesh->_exp_mesh = fvm_nodal_destroy(post_mesh->_exp_mesh);
  }
  post_mesh->exp_mesh = nullptr;

  /* Define new mesh */

  _define_mesh(post_mesh, ts);
}

/*----------------------------------------------------------------------------
 * Remove meshes which are associated with no writer
 *----------------------------------------------------------------------------*/

static void
_clear_unused_meshes(void)
{
  int  i;
  int *discard = nullptr;

  cs_post_mesh_t  *post_mesh;

  /* Mark used meshes */

  CS_MALLOC(discard, _cs_post_n_meshes, int);

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->n_writers == 0)
      discard[i] = 1;
    else
      discard[i] = 0;
  }

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->locate_ref > -1) {
      if (post_mesh->n_writers > 0)
        discard[post_mesh->locate_ref] = 0;
    }
  }

  /* Discard meshes not required, compacting array */

  for (i = _cs_post_n_meshes - 1; i >= 0; i--) {
    if (discard[i] == 1)
      _free_mesh(i);  /* shifts other meshes and reduces _cs_post_n_meshes */
  }

  CS_FREE(discard);
}

/*----------------------------------------------------------------------------
 * Divide polygons or polyhedra in simpler elements if necessary.
 *
 * parameters:
 *   post_mesh <-> pointer to post-processing mesh
 *   writer    <-- pointer to associated writer
 *----------------------------------------------------------------------------*/

static void
_divide_poly(cs_post_mesh_t    *post_mesh,
             cs_post_writer_t  *writer)
{
  if (fvm_writer_needs_tesselation(writer->writer,
                                   post_mesh->exp_mesh,
                                   FVM_CELL_POLY) > 0)
    fvm_nodal_tesselate(post_mesh->_exp_mesh, FVM_CELL_POLY, nullptr);

  if (fvm_writer_needs_tesselation(writer->writer,
                                   post_mesh->exp_mesh,
                                   FVM_FACE_POLY) > 0)
    fvm_nodal_tesselate(post_mesh->_exp_mesh, FVM_FACE_POLY, nullptr);
}

/*----------------------------------------------------------------------------
 * Write parallel domain (rank) number to post-processing mesh
 *
 * parameters:
 *   writer     <-- FVM writer
 *   exp_mesh   <-- exportable mesh
 *   nt_cur_abs <-- current time step number
 *   t_cur_abs  <-- current physical time
 *---------------------------------------------------------------------------*/

static void
_cs_post_write_domain(fvm_writer_t       *writer,
                      const fvm_nodal_t  *exp_mesh,
                      int                 nt_cur_abs,
                      double              t_cur_abs)
{
  int  dim_ent;
  cs_lnum_t  i, n_elts;

  cs_lnum_t  parent_num_shift[1]  = {0};
  int32_t  *domain = nullptr;

  int _nt_cur_abs = -1;
  double _t_cur_abs = 0.;

  const int32_t   *var_ptr[1] = {nullptr};

  if (cs_glob_n_ranks < 2)
    return;

  dim_ent = fvm_nodal_get_max_entity_dim(exp_mesh);
  n_elts = fvm_nodal_get_n_entities(exp_mesh, dim_ent);

  /* Prepare domain number */

  CS_MALLOC(domain, n_elts, int32_t);

  for (i = 0; i < n_elts; i++)
    domain[i] = cs_glob_rank_id;

  /* Prepare post-processing output */

  var_ptr[0] = domain;

  if (fvm_writer_get_time_dep(writer) != FVM_WRITER_FIXED_MESH) {
    _nt_cur_abs = nt_cur_abs;
    _t_cur_abs = t_cur_abs;
  }

  fvm_writer_export_field(writer,
                          exp_mesh,
                          "mpi_rank_id",
                          FVM_WRITER_PER_ELEMENT,
                          1,
                          CS_INTERLACE,
                          0,
                          parent_num_shift,
                          CS_INT32,
                          _nt_cur_abs,
                          _t_cur_abs,
                          (const void * *)var_ptr);

  /* Free memory */

  CS_FREE(domain);
}

/*----------------------------------------------------------------------------
 * Output fixed zone information if needed.
 *
 * This function is called when we know some writers are active
 *
 * parameters:
 *   writer     <-- FVM writer
 *   post_mesh  <-- postprocessing mesh
 *   nt_cur_abs <-- current time step number
 *   t_cur_abs  <-- current physical time
 *----------------------------------------------------------------------------*/

static void
_cs_post_write_fixed_zone_info(fvm_writer_t          *writer,
                               const cs_post_mesh_t  *post_mesh,
                               int                    nt_cur_abs,
                               double                 t_cur_abs)
{
  assert(post_mesh->exp_mesh != nullptr);

  bool output = false;
  const int   *var_ptr[1] = {nullptr};
  const char  *name = nullptr;

  if (post_mesh->id == CS_POST_MESH_VOLUME) {

    /* Ignore cases where all zones include all cells */

    int n_zones = cs_volume_zone_n_zones();
    int z_id = 0;

    for (z_id = 0; z_id < n_zones; z_id++) {
      const cs_zone_t  *z = cs_volume_zone_by_id(z_id);
      if (z->location_id != CS_MESH_LOCATION_CELLS)
        break;
    }
    if (z_id >= n_zones)
      return;

    const int *zone_id = cs_volume_zone_cell_zone_id();
    name = "volume zone id";

    if (cs_volume_zone_n_zones_time_varying() == 0) {
      output = true;
      var_ptr[0] = zone_id;
    }

  }

  else if (post_mesh->id == CS_POST_MESH_BOUNDARY) {

    /* Ignore cases where all zones include all boundary faces */

    int n_zones = cs_boundary_zone_n_zones();
    int z_id = 0;

    for (z_id = 0; z_id < n_zones; z_id++) {
      const cs_zone_t  *z = cs_boundary_zone_by_id(z_id);
      if (z->location_id != CS_MESH_LOCATION_BOUNDARY_FACES)
        break;
    }
    if (z_id >= n_zones)
      return;

    const int *zone_id = cs_boundary_zone_face_zone_id();
    name = "boundary zone id";

    if (cs_boundary_zone_n_zones_time_varying() == 0) {
      output = true;
      var_ptr[0] = zone_id;
    }

  }

  if (output) {

    cs_lnum_t  parent_num_shift[1]  = {0};
    int _nt_cur_abs = -1;
    double _t_cur_abs = 0.;

    if (fvm_writer_get_time_dep(writer) != FVM_WRITER_FIXED_MESH) {
      _nt_cur_abs = nt_cur_abs;
      _t_cur_abs = t_cur_abs;
    }

    fvm_writer_export_field(writer,
                            post_mesh->exp_mesh,
                            name,
                            FVM_WRITER_PER_ELEMENT,
                            1,
                            CS_INTERLACE,
                            1,
                            parent_num_shift,
                            CS_INT_TYPE,
                            _nt_cur_abs,
                            _t_cur_abs,
                            (const void * *)var_ptr);

  }
}

/*----------------------------------------------------------------------------
 * Output varying zone information if needed.
 *
 * This function is called when we know some writers are active
 *
 * parameters:
 *   ts <-- time step structure, or nullptr
 *----------------------------------------------------------------------------*/

static void
_cs_post_write_transient_zone_info(const cs_post_mesh_t  *post_mesh,
                                   const cs_time_step_t  *ts)
{
  if (post_mesh->id == CS_POST_MESH_VOLUME) {
    if (cs_volume_zone_n_zones_time_varying() > 0) {
      cs_post_write_var(post_mesh->id,
                        CS_POST_WRITER_ALL_ASSOCIATED,
                        "volume zone id",
                        1,       /* var_dim */
                        true,    /* interlace */
                        true,    /* use_parent */
                        CS_POST_TYPE_int,
                        cs_volume_zone_cell_zone_id(),
                        nullptr,
                        nullptr,
                        ts);
    }
  }

  else if (post_mesh->id == CS_POST_MESH_BOUNDARY) {
    if (cs_boundary_zone_n_zones_time_varying() > 0)
      cs_post_write_var(post_mesh->id,
                        CS_POST_WRITER_ALL_ASSOCIATED,
                        "boundary zone id",
                        1,       /* var_dim */
                        true,    /* interlace */
                        true,    /* use_parent */
                        CS_POST_TYPE_int,
                        nullptr,
                        nullptr,
                        cs_boundary_zone_face_zone_id(),
                        ts);
  }
}

/*----------------------------------------------------------------------------
 * Output a post-processing mesh using associated writers.
 *
 * If the time step structure argument passed is null, a time-independent
 * output will be assumed.
 *
 * parameters:
 *   post_mesh  <-> pointer to post-processing mesh
 *   ts         <-- time step structure, or null
 *----------------------------------------------------------------------------*/

static void
_cs_post_write_mesh(cs_post_mesh_t        *post_mesh,
                    const cs_time_step_t  *ts)
{
  const int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
  const double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

  /* Special case: particle trajectories can not be
     output before time stepping starts */

  if (post_mesh->ent_flag[3] == 2 && nt_cur < 1)
    return;

  /* Loop on writers */

  for (int j = 0; j < post_mesh->n_writers; j++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[j];

    fvm_writer_time_dep_t  time_dep;
    if (writer->wd != nullptr)
      time_dep = writer->wd->time_dep;
    else
      time_dep = fvm_writer_get_time_dep(writer->writer);

    bool  write_mesh = false;

    if (   time_dep == FVM_WRITER_FIXED_MESH
        && writer->active > -1
        && post_mesh->ent_flag[3] != 2) {
      if (post_mesh->nt_last[j] < -1)
        write_mesh = true;
    }
    else {
      if (post_mesh->nt_last[j] < nt_cur && writer->active == 1)
        write_mesh = true;
    }

    if (write_mesh == true) {

      if (writer->writer == nullptr)
        _init_writer(writer);

      if (post_mesh->exp_mesh == nullptr)
        _define_mesh(post_mesh, ts);

      if (post_mesh->exp_mesh == nullptr)
        continue;

      _divide_poly(post_mesh, writer);

      if (nt_cur >= 0 && time_dep != FVM_WRITER_FIXED_MESH)
        fvm_writer_set_mesh_time(writer->writer, nt_cur, t_cur);

      fvm_writer_export_nodal(writer->writer, post_mesh->exp_mesh);

      if (nt_cur >= 0 && time_dep != FVM_WRITER_FIXED_MESH) {
        writer->tc.last_nt = nt_cur;
        writer->tc.last_t = t_cur;
      }

      if (post_mesh->post_domain)
        _cs_post_write_domain(writer->writer,
                              post_mesh->exp_mesh,
                              nt_cur,
                              t_cur);

      _cs_post_write_fixed_zone_info(writer->writer,
                                     post_mesh,
                                     nt_cur,
                                     t_cur);

    }

    if (write_mesh == true)
      post_mesh->nt_last[j] = nt_cur;
  }
}

/*----------------------------------------------------------------------------
 * Assemble variable values defined on a mix of interior and boundary
 * faces (with no indirection) into an array defined on a single faces set.
 *
 * The resulting variable is not interlaced.
 *
 * parameters:
 *   exp_mesh    <-- exportable mesh
 *   n_i_faces   <-- number of interior faces
 *   n_b_faces   <-- number of boundary faces
 *   var_dim     <-- variable dimension
 *   interlace   <-- for vector, interlace if 1, no interlace if 0
 *   i_face_vals <-- values at interior faces
 *   b_face_vals <-- values at boundary faces
 *   var_tmp[]   --> assembled values
 *----------------------------------------------------------------------------*/

static void
_cs_post_assmb_var_faces(const fvm_nodal_t  *exp_mesh,
                         cs_lnum_t           n_i_faces,
                         cs_lnum_t           n_b_faces,
                         cs_lnum_t           var_dim,
                         cs_interlace_t      interlace,
                         const cs_real_t     i_face_vals[],
                         const cs_real_t     b_face_vals[],
                         cs_real_t           var_tmp[])
{
  cs_lnum_t  i, j, stride_1, stride_2;

  cs_lnum_t  n_elts = n_i_faces + n_b_faces;

  assert(exp_mesh != nullptr);

  /* The variable is defined on interior and boundary faces of the
     post-processing mesh, and has been built using values
     at the corresponding interior and boundary faces */

  /* Boundary faces contribution */

  if (interlace == CS_INTERLACE) {
    stride_1 = var_dim;
    stride_2 = 1;
  }
  else {
    stride_1 = 1;
    stride_2 = n_b_faces;
  }

  for (i = 0; i < n_b_faces; i++) {
    for (j = 0; j < var_dim; j++)
      var_tmp[i + j*n_elts] = b_face_vals[i*stride_1 + j*stride_2];
  }

  /* Interior faces contribution */

  if (interlace == CS_INTERLACE) {
    stride_1 = var_dim;
    stride_2 = 1;
  }
  else {
    stride_1 = 1;
    stride_2 = n_i_faces;
  }

  for (i = 0; i < n_i_faces; i++) {
    for (j = 0; j < var_dim; j++)
      var_tmp[i + n_b_faces + j*n_elts] = i_face_vals[i*stride_1 + j*stride_2];
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Check if post-processing is activated and then update post-processing
 *        of meshes if there is a time-dependent mesh
 *
 * \param[in]  ts  time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

static void
_update_meshes(const cs_time_step_t  *ts)
{
  bool  active;

  /* Loop on writers to check if something must be done */
  /*----------------------------------------------------*/

  int  w;
  for (w = 0; w < _cs_post_n_writers; w++) {
    cs_post_writer_t  *writer = _cs_post_writers + w;
    if (writer->active == 1)
      break;
  }
  if (w == _cs_post_n_writers)
    return;

  int t_top_id = cs_timer_stats_switch(_post_out_stat_id);

  /* Possible modification of post-processing meshes */
  /*-------------------------------------------------*/

  for (int i = 0; i < _cs_post_n_meshes; i++) {

    cs_post_mesh_t  *post_mesh = _cs_post_meshes + i;

    active = false;

    for (int j = 0; j < post_mesh->n_writers; j++) {
      cs_post_writer_t  *writer = _cs_post_writers + post_mesh->writer_id[j];
      if (writer->active == 1)
        active = true;
    }

    if (active == false)
      continue;

    /* Modifiable user mesh, active at this time step */

    if (post_mesh->mod_flag_min == FVM_WRITER_TRANSIENT_CONNECT)
      _redefine_mesh(post_mesh, ts);

    else if (post_mesh->ent_flag[4] != 0) {
      bool time_varying;
      cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];
      cs_probe_set_get_post_info(pset,
                                 &time_varying,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr);
      if (time_varying) {
        cs_post_mesh_t *post_mesh_loc = _cs_post_meshes + post_mesh->locate_ref;
        cs_probe_set_locate(pset, post_mesh_loc->exp_mesh);

        /* Update associated mesh */
        fvm_nodal_t *exp_mesh
          = cs_probe_set_export_mesh(pset, cs_probe_set_get_name(pset));
        if (post_mesh->_exp_mesh != nullptr)
          post_mesh->_exp_mesh = fvm_nodal_destroy(post_mesh->_exp_mesh);
        post_mesh->_exp_mesh = exp_mesh;
        post_mesh->exp_mesh = exp_mesh;
      }
    }

  }

  /* Output of meshes or vertex displacement field if necessary */
  /*------------------------------------------------------------*/

  cs_post_write_meshes(ts);

  cs_timer_stats_switch(t_top_id);
}

/*----------------------------------------------------------------------------
 * Generate global group flags array from local family flags
 *
 * The caller should free the returned array once it is no longer needed.
 *
 * parameters:
 *   mesh     <-- pointer to mesh structure
 *   fam_flag <-- flag values (size: mesh->n_families + 1)
 *
 * returns:
 *   group flag (size: mesh->n_groups)
 *----------------------------------------------------------------------------*/

static char *
_build_group_flag(const cs_mesh_t  *mesh,
                  int              *fam_flag)
{
  int i, j;

  char *group_flag = nullptr;

  CS_MALLOC(group_flag, mesh->n_groups, char);
  memset(group_flag, 0, mesh->n_groups);

#if defined(HAVE_MPI)
  if (cs_glob_n_ranks > 1) {
    int *_fam_flag = nullptr;
    CS_MALLOC(_fam_flag, mesh->n_families + 1, int);
    MPI_Allreduce(fam_flag, _fam_flag, mesh->n_families + 1,
                  MPI_INT, MPI_MAX, cs_glob_mpi_comm);
    memcpy(fam_flag, _fam_flag, (mesh->n_families + 1)*sizeof(int));
    CS_FREE(_fam_flag);
  }
#endif /* defined(HAVE_MPI) */

  for (i = 0; i < mesh->n_families; i++) {
    if (fam_flag[(i+1)] != 0) {
      char mask = fam_flag[i+1];
      for (j = 0; j < mesh->n_max_family_items; j++) {
        int g_id = - mesh->family_item[mesh->n_families*j + i] - 1;
        if (g_id >= 0)
          group_flag[g_id] = group_flag[g_id] | mask;
      }
    }
  }

  return group_flag;
}

/*----------------------------------------------------------------------------
 * Set a family flags array to 1 for families containg a given group,
 * and to 0 for others.
 *
 * parameters:
 *   mesh     <-- pointer to mesh structure
 *   g_id     <-- group id
 *   fam_flag --> flag values (size: mesh->n_families)
 *----------------------------------------------------------------------------*/

static void
_set_fam_flags(const cs_mesh_t  *mesh,
               int               g_id,
               int              *fam_flag)
{
  int j, k;
  memset(fam_flag, 0, mesh->n_families*sizeof(int));
  for (j = 0; j < mesh->n_families; j++) {
    for (k = 0; k < mesh->n_max_family_items; k++) {
      int _g_id = - mesh->family_item[mesh->n_families*k + j] - 1;
      if (_g_id == g_id)
        fam_flag[j] = 1;
    }
  }
}

/*----------------------------------------------------------------------------
 * Output volume sub-meshes by group
 *
 * parameters:
 *   mesh      <-- base mesh
 *   fmt_name  <-- format name
 *   fmt_opts  <-- format options
 *---------------------------------------------------------------------------*/

static void
_vol_submeshes_by_group(const cs_mesh_t  *mesh,
                        const char       *fmt_name,
                        const char       *fmt_opts)
{
  cs_lnum_t i, j;
  cs_lnum_t n_cells, n_i_faces, n_b_faces;
  char part_name[81];
  int max_null_family = 0;
  int *fam_flag = nullptr;
  char *group_flag = nullptr;
  cs_lnum_t *cell_list = nullptr, *i_face_list = nullptr, *b_face_list = nullptr;
  fvm_writer_t *writer = nullptr;
  fvm_nodal_t *exp_mesh = nullptr;

  if (mesh->n_families == 0)
    return;

  /* Families should be sorted, so if a nonzero family is empty,
     it is family 1 */

  if (mesh->family_item[0] == 0)
    max_null_family = 1;

  if (mesh->n_families <= max_null_family)
    return;

  /* Get writer info */

  /* Default values */

  /* Create default writer */

  writer = fvm_writer_init("mesh_groups",
                           _cs_post_dirname,
                           fmt_name,
                           fmt_opts,
                           FVM_WRITER_FIXED_MESH);

  /* Now detect which groups may be referenced */

  CS_MALLOC(fam_flag, (mesh->n_families + 1), int);
  memset(fam_flag, 0, (mesh->n_families + 1) * sizeof(int));

  if (mesh->cell_family != nullptr) {
    for (i = 0; i < mesh->n_cells; i++)
      fam_flag[mesh->cell_family[i]]
        = fam_flag[mesh->cell_family[i]] | 1;
  }
  if (mesh->i_face_family != nullptr) {
    for (i = 0; i < mesh->n_i_faces; i++)
      fam_flag[mesh->i_face_family[i]]
        = fam_flag[mesh->i_face_family[i]] | 2;
  }
  if (mesh->b_face_family != nullptr) {
    for (i = 0; i < mesh->n_b_faces; i++)
      fam_flag[mesh->b_face_family[i]]
        = fam_flag[mesh->b_face_family[i]] | 4;
  }

  group_flag = _build_group_flag(mesh, fam_flag);

  /* Now extract volume elements by groups.
     Note that selector structures may not have been initialized yet,
     so to avoid issue, we use a direct selection here. */

  CS_REALLOC(fam_flag, mesh->n_families, int);

  CS_MALLOC(cell_list, mesh->n_cells, cs_lnum_t);

  for (i = 0; i < mesh->n_groups; i++) {

    if (group_flag[i] & '\1') {

      const char *g_name = mesh->group + mesh->group_idx[i];

      _set_fam_flags(mesh, i, fam_flag);

      for (j = 0, n_cells = 0; j < mesh->n_cells; j++) {
        int f_id = mesh->cell_family[j];
        if (f_id > 0 && fam_flag[f_id - 1])
          cell_list[n_cells++] = j;
      }
      strcpy(part_name, "vol: ");
      strncat(part_name, g_name, 80 - strlen(part_name));
      exp_mesh = cs_mesh_connect_cells_to_nodal(mesh,
                                                part_name,
                                                false,
                                                n_cells,
                                                cell_list);

      if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_CELL_POLY) > 0)
        fvm_nodal_tesselate(exp_mesh, FVM_CELL_POLY, nullptr);

      fvm_writer_set_mesh_time(writer, -1, 0);
      fvm_writer_export_nodal(writer, exp_mesh);

      exp_mesh = fvm_nodal_destroy(exp_mesh);
    }

  }

  /* Now export cells with no groups */

  if (mesh->cell_family != nullptr) {
    for (j = 0, n_cells = 0; j < mesh->n_cells; j++) {
      if (mesh->cell_family[j] <= max_null_family)
        cell_list[n_cells++] = j;
    }
  }
  else {
    for (j = 0, n_cells = 0; j < mesh->n_cells; j++)
      cell_list[n_cells++] = j;
  }

  i = n_cells;
  cs_parall_counter_max(&i, 1);

  if (i > 0) {
    exp_mesh = cs_mesh_connect_cells_to_nodal(mesh,
                                              "vol: no_group",
                                              false,
                                              n_cells,
                                              cell_list);

    if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_CELL_POLY) > 0)
      fvm_nodal_tesselate(exp_mesh, FVM_CELL_POLY, nullptr);

    fvm_writer_set_mesh_time(writer, -1, 0);
    fvm_writer_export_nodal(writer, exp_mesh);

    exp_mesh = fvm_nodal_destroy(exp_mesh);
  }

  CS_FREE(cell_list);

  /* Now extract faces by groups */

  CS_MALLOC(i_face_list, mesh->n_i_faces, cs_lnum_t);
  CS_MALLOC(b_face_list, mesh->n_b_faces, cs_lnum_t);

  for (i = 0; i < mesh->n_groups; i++) {

    if ((group_flag[i] & '\2') || (group_flag[i] & '\4')) {

      const char *g_name = mesh->group + mesh->group_idx[i];

      _set_fam_flags(mesh, i, fam_flag);

      n_i_faces = 0;
      if (mesh->i_face_family != nullptr) {
        for (j = 0; j < mesh->n_i_faces; j++) {
          int f_id = mesh->i_face_family[j];
          if (f_id > 0 && fam_flag[f_id - 1])
            i_face_list[n_i_faces++] = j;
        }
      }
      n_b_faces = 0;
      if (mesh->b_face_family != nullptr) {
        for (j = 0; j < mesh->n_b_faces; j++) {
          int f_id = mesh->b_face_family[j];
          if (f_id > 0 && fam_flag[f_id - 1])
            b_face_list[n_b_faces++] = j;
        }
      }

      strcpy(part_name, "surf: ");
      strncat(part_name, g_name, 80 - strlen(part_name));
      exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                part_name,
                                                false,
                                                n_i_faces,
                                                n_b_faces,
                                                i_face_list,
                                                b_face_list);

      if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
        fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

      fvm_writer_set_mesh_time(writer, -1, 0);
      fvm_writer_export_nodal(writer, exp_mesh);

      exp_mesh = fvm_nodal_destroy(exp_mesh);
    }

  }

  writer = fvm_writer_finalize(writer);

  CS_FREE(b_face_list);
  CS_FREE(i_face_list);

  CS_FREE(fam_flag);
  CS_FREE(group_flag);
}

/*----------------------------------------------------------------------------
 * Output boundary sub-meshes by group, if it contains multiple groups.
 *
 * parameters:
 *   mesh        <-- base mesh
 *   fmt_name    <-- format name
 *   fmt_opts    <-- format options
 *---------------------------------------------------------------------------*/

static void
_boundary_submeshes_by_group(const cs_mesh_t   *mesh,
                             const char        *fmt_name,
                             const char        *fmt_opts)
{
  cs_lnum_t i, j;
  cs_lnum_t n_b_faces;
  cs_gnum_t n_no_group = 0;
  int max_null_family = 0;
  int *fam_flag = nullptr;
  char *group_flag = nullptr;
  cs_lnum_t *b_face_list = nullptr;
  fvm_writer_t *writer = nullptr;
  fvm_nodal_t *exp_mesh = nullptr;

  if (mesh->n_families == 0)
    return;

  /* Families should be sorted, so if a nonzero family is empty,
     it is family 1 */

  if (mesh->family_item[0] == 0)
    max_null_family = 1;

  if (mesh->n_families <= max_null_family)
    return;

  /* Check how many boundary faces belong to no group */

  if (mesh->b_face_family != nullptr) {
    for (j = 0, n_b_faces = 0; j < mesh->n_b_faces; j++) {
      if (mesh->b_face_family[j] <= max_null_family)
        n_no_group += 1;
    }
  }
  else
    n_no_group = mesh->n_b_faces;

  cs_parall_counter(&n_no_group, 1);

  if (n_no_group == mesh->n_g_b_faces)
    return;

  /* Get writer info */

  /* Default values */

  /* Create default writer */

  writer = fvm_writer_init("boundary_groups",
                           _cs_post_dirname,
                           fmt_name,
                           fmt_opts,
                           FVM_WRITER_FIXED_MESH);

  /* Now detect which groups may be referenced */

  CS_MALLOC(fam_flag, mesh->n_families + 1, int);
  memset(fam_flag, 0, (mesh->n_families + 1)*sizeof(int));

  if (mesh->b_face_family != nullptr) {
    for (i = 0; i < mesh->n_b_faces; i++)
      fam_flag[mesh->b_face_family[i]] = 1;
  }

  group_flag = _build_group_flag(mesh, fam_flag);

  /* Now extract boundary faces by groups.
     Note that selector structures may not have been initialized yet,
     so to avoid issue, we use a direct selection here. */

  CS_REALLOC(fam_flag, mesh->n_families, int);

  CS_MALLOC(b_face_list, mesh->n_b_faces, cs_lnum_t);

  for (i = 0; i < mesh->n_groups; i++) {

    if (group_flag[i] != 0) {

      const char *g_name = mesh->group + mesh->group_idx[i];

      _set_fam_flags(mesh, i, fam_flag);

      n_b_faces = 0;
      if (mesh->b_face_family != nullptr) {
        for (j = 0; j < mesh->n_b_faces; j++) {
          int f_id = mesh->b_face_family[j];
          if (f_id > 0 && fam_flag[f_id - 1])
            b_face_list[n_b_faces++] = j;
        }
      }

      exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                g_name,
                                                false,
                                                0,
                                                n_b_faces,
                                                nullptr,
                                                b_face_list);

      if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
        fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

      fvm_writer_set_mesh_time(writer, -1, 0);
      fvm_writer_export_nodal(writer, exp_mesh);

      exp_mesh = fvm_nodal_destroy(exp_mesh);
    }

  }

  /* Output boundary faces belonging to no group */

  if (n_no_group > 0) {

    if (mesh->b_face_family != nullptr) {
      for (j = 0, n_b_faces = 0; j < mesh->n_b_faces; j++) {
        if (mesh->b_face_family[j] <= max_null_family)
          b_face_list[n_b_faces++] = j;
      }
    }
    else {
      for (j = 0, n_b_faces = 0; j < mesh->n_b_faces; j++)
        b_face_list[n_b_faces++] = j;
    }

    exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                              "no_group",
                                              false,
                                              0,
                                              n_b_faces,
                                              nullptr,
                                              b_face_list);

    if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
      fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

    fvm_writer_set_mesh_time(writer, -1, 0);
    fvm_writer_export_nodal(writer, exp_mesh);

    exp_mesh = fvm_nodal_destroy(exp_mesh);
  }

  CS_FREE(b_face_list);

  writer = fvm_writer_finalize(writer);

  CS_FREE(fam_flag);
  CS_FREE(group_flag);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Check if an active writer associated to a mesh has a transient
 *        connectivity.
 *
 * \param[in]       mesh  pointer to associated postprocessing mesh
 *
 * \return  true if an active associated writer has transent connectivity,
 *          false otherwise
 */
/*----------------------------------------------------------------------------*/

static bool
_post_mesh_have_active_transient(const cs_post_mesh_t  *post_mesh)
{
  bool have_transient_connect = false;

  for (int j = 0; j < post_mesh->n_writers; j++) {
    cs_post_writer_t  *writer = _cs_post_writers + post_mesh->writer_id[j];
    if (writer->active == 1) {
      if (   fvm_writer_get_time_dep(writer->writer)
          == FVM_WRITER_TRANSIENT_CONNECT)
        have_transient_connect = true;
    }
  }

  return have_transient_connect;
}

/*----------------------------------------------------------------------------
 * Check if a given field matches the profile for post_write_var.
 *
 * parameters:
 *   post_mesh      <-- postprocessing mesh
 *   field_loc_type <-- associated field location type
 *
 * return:
 *   true if the field matches the profile for post_writer_var
 *----------------------------------------------------------------------------*/

static bool
_cs_post_match_post_write_var(const cs_post_mesh_t     *post_mesh,
                              cs_mesh_location_type_t   field_loc_type)
{
  bool match = false;

  if (post_mesh->ent_flag[CS_POST_LOCATION_CELL] == 1) {
    if (   field_loc_type == CS_MESH_LOCATION_CELLS
        || field_loc_type == CS_MESH_LOCATION_VERTICES)
      match = true;
  }

  else if (post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {
    if (field_loc_type == CS_MESH_LOCATION_VERTICES)
      match = true;
    else if (   field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES
             && post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 0)
      match = true;
  }

  else if (post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1) {
    if (field_loc_type == CS_MESH_LOCATION_VERTICES)
      match = true;
    else if (field_loc_type == CS_MESH_LOCATION_INTERIOR_FACES)
      match = true;
  }

  return match;
}

/*----------------------------------------------------------------------------
 * Extract component from field values.
 *
 * The caller is responsible for freeing the returned array.
 *
 * parameters:
 *   f            <-- pointer to associated field
 *   comp_id      <-- id of component to extract
 *   name         <-- base name
 *   name_buf     --> name with dimension extension added
 *
 * return:
 *   array restricted to selected component
 *----------------------------------------------------------------------------*/

static cs_real_t *
_extract_field_component(const cs_field_t  *f,
                         cs_lnum_t          comp_id,
                         const char        *name,
                         char               name_buf[96])
{
  strncpy(name_buf, name, 90);
  name_buf[90] = '\0';
  switch (f->dim) {
  case 3:
    strncat(name_buf, cs_glob_field_comp_name_3[comp_id], 5);
    break;
  case 6:
    strncat(name_buf, cs_glob_field_comp_name_6[comp_id], 5);
    break;
  case 9:
    strncat(name_buf, cs_glob_field_comp_name_9[comp_id], 5);
    break;
  default:
    snprintf(name_buf + strlen(name_buf), 5, "[%d]", (int)comp_id);
  }
  name_buf[95] = '\0';

  const cs_lnum_t dim = f->dim;
  const cs_lnum_t n_elts = cs_mesh_location_get_n_elts(f->location_id)[0];
  const cs_real_t *src_val = f->val;

  cs_real_t *val;
  CS_MALLOC(val, n_elts, cs_real_t);

  for (cs_lnum_t i = 0; i < n_elts; i++)
    val[i] = src_val[i*dim + comp_id];

  return val;
}

/*----------------------------------------------------------------------------
 * Output coordinates for profiles if requested.
 *
 * parameters:
 *   post_mesh   <-- pointer to post-processing mesh structure
 *   ts          <-- time step status structure, or null
 *----------------------------------------------------------------------------*/

static void
_cs_post_output_profile_coords(cs_post_mesh_t        *post_mesh,
                               const cs_time_step_t  *ts)
{
  assert(post_mesh != nullptr);

  const cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  bool auto_curve_coo, auto_cart_coo;

  cs_probe_set_get_post_info(pset,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             &auto_curve_coo,
                             &auto_cart_coo,
                             nullptr,
                             nullptr);

  if (auto_curve_coo) {
    cs_real_t *s = cs_probe_set_get_loc_curvilinear_abscissa(pset);
    cs_post_write_probe_values(post_mesh->id,
                               CS_POST_WRITER_ALL_ASSOCIATED,
                               "s",
                               1,
                               CS_POST_TYPE_cs_real_t,
                               0,
                               nullptr,
                               nullptr,
                               s,
                               ts);
    CS_FREE(s);
  }

  if (auto_cart_coo) {

    int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
    double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

    const cs_lnum_t  n_points = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0);
    cs_coord_t  *point_coords;
    CS_MALLOC(point_coords, n_points*3, cs_coord_t);
    fvm_nodal_get_vertex_coords(post_mesh->exp_mesh,
                                CS_INTERLACE,
                                point_coords);

    for (int i = 0; i < post_mesh->n_writers; i++) {

      cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

      if (writer->active == 1 && writer->writer != nullptr) {
        const char *fmt = fvm_writer_get_format(writer->writer);
        if (strcmp(fmt, "plot"))
          continue;

        cs_lnum_t  parent_num_shift[1] = {0};
        const void  *var_ptr[1] = {point_coords};

        fvm_writer_export_field(writer->writer,
                                post_mesh->exp_mesh,
                                "",  /* var_name */
                                FVM_WRITER_PER_NODE,
                                3,
                                CS_INTERLACE,
                                0, /* n_parent_lists */
                                parent_num_shift,
                                CS_COORD_TYPE,
                                nt_cur,
                                t_cur,
                                (const void **)var_ptr);
      }

    } /* End of loop on writers */

    CS_FREE(point_coords);

  }
}

/*----------------------------------------------------------------------------
 * Main post-processing output of variables.
 *
 * parameters:
 *   post_mesh   <-- pointer to post-processing mesh structure
 *   ts          <-- time step status structure, or null
 *----------------------------------------------------------------------------*/

static void
_cs_post_output_fields(cs_post_mesh_t        *post_mesh,
                       const cs_time_step_t  *ts)
{
  int  pset_interpolation = 0;
  bool pset_on_boundary = false;
  cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  if (pset != nullptr) {
    cs_probe_set_get_post_info(pset,
                               nullptr,
                               &pset_on_boundary,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
    if (pset_on_boundary == false && cs_probe_set_get_interpolation(pset) == 1)
      pset_interpolation = 1;
  }

  /* Base output for cell and boundary meshes */
  /*------------------------------------------*/

  if (   post_mesh->cat_id == CS_POST_MESH_VOLUME
      || post_mesh->cat_id == CS_POST_MESH_BOUNDARY
      || post_mesh->cat_id == CS_POST_MESH_SURFACE) {

    const int n_fields = cs_field_n_fields();
    const int vis_key_id = cs_field_key_id("post_vis");
    const int label_key_id = cs_field_key_id("label");

    /* Loop on fields */

    for (int f_id = 0; f_id < n_fields; f_id++) {

      const bool  use_parent = true;

      const cs_field_t  *f = cs_field_by_id(f_id);

      if (! (cs_field_get_key_int(f, vis_key_id) & CS_POST_ON_LOCATION))
        continue;

      const cs_mesh_location_type_t field_loc_type
        = cs_mesh_location_get_type(f->location_id);

      if (_cs_post_match_post_write_var(post_mesh, field_loc_type) == false)
        continue;

      const char *name = cs_field_get_key_str(f, label_key_id);
      if (name == nullptr)
        name = f->name;

      if (pset != nullptr) {
        char interpolate_input[96];
        strncpy(interpolate_input, f->name, 95); interpolate_input[95] = '\0';

        cs_interpolate_from_location_t
          *interpolate_func = cs_interpolate_from_location_p0;
        if (   field_loc_type == CS_MESH_LOCATION_CELLS
            && pset_interpolation == 1)
          interpolate_func = cs_interpolate_from_location_p1;

        cs_post_write_probe_values(post_mesh->id,
                                   CS_POST_WRITER_ALL_ASSOCIATED,
                                   name,
                                   f->dim,
                                   CS_POST_TYPE_cs_real_t,
                                   f->location_id,
                                   interpolate_func,
                                   interpolate_input,
                                   f->val,
                                   ts);
      }

      else if (   field_loc_type == CS_MESH_LOCATION_CELLS
               || field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES
               || field_loc_type == CS_MESH_LOCATION_INTERIOR_FACES) {

        const cs_real_t *f_val = f->val;
        const cs_real_t *cell_val = nullptr, *b_face_val = nullptr, *i_face_val = nullptr;
        cs_real_t *tmp_val = nullptr;

        if (f->location_id != (int)field_loc_type) {

          /* Fields can be output on parent or sibling locations,
             with a value  of 0 outside the actual field location. */

          cs_lnum_t n_elts = cs_mesh_location_get_n_elts(f->location_id)[0];
          cs_lnum_t f_dim = f->dim;
          cs_lnum_t n_elts_p = cs_mesh_location_get_n_elts(field_loc_type)[0];
          cs_lnum_t n_vals_p = n_elts_p * f_dim;

          const cs_lnum_t *elt_ids
            = cs_mesh_location_get_elt_ids_try(f->location_id);

          bool field_and_mesh_ids_match = false;

          /* TODO: check if a given field and postprocessing mesh
             share the same mesh location, and in that case, if
             their respective parent_ids match. If this is the case,
             the projection to the parent mesh below is not necessary,
             and we can simply write the variable with
             use_parent = false.

             In the general case, since fvm_nodal_t structures order
             elements by types, the parent element ids of such a
             structure may be a permutation of the original parent ids,
             so may not be used directly. In that case, either we need
             to project values to the parent mesh first, or apply a
             reverse permutation to the values extracted in the
             output buffer.
          */

          if (field_and_mesh_ids_match == false) {

            CS_MALLOC(tmp_val, n_vals_p, cs_real_t);

            /* Remark: in case we decide to output values on the parent mesh,
               we need to initialize values to a default for elements not in
               the field location subset.

               Outputting values on a smaller subset of the mesh location
               could seem more natural, but we would have to determine
               correctly that we are indeed on a smaller subset, which
               is not trivial as all sub-locations are defined relative to
               a root location type, not in a recursive manner.

               We assign a default of 0, but an associated field keyword
               to define another value could be useful here. */

            for (cs_lnum_t i = 0; i < n_vals_p; i++)
              tmp_val[i] = 0.;

            /* Now scatter values from subset to parent location. */

            cs_array_real_copy_subset(n_elts, f_dim, elt_ids,
                                      CS_ARRAY_SUBSET_OUT, /* elt_ids on dest */
                                      f->val,              /* ref */
                                      tmp_val);            /* dest <-- ref */

            f_val = tmp_val;

          }

        } /* End of case for field on sub-location */

        if (field_loc_type == CS_MESH_LOCATION_CELLS)
          cell_val = f_val;
        else if (field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES)
          b_face_val = f_val;
        else if (field_loc_type == CS_MESH_LOCATION_INTERIOR_FACES)
          i_face_val = f_val;

        cs_post_write_var(post_mesh->id,
                          CS_POST_WRITER_ALL_ASSOCIATED,
                          name,
                          f->dim,
                          true,
                          use_parent,
                          CS_POST_TYPE_cs_real_t,
                          cell_val,
                          i_face_val,
                          b_face_val,
                          ts);

        CS_FREE(tmp_val);
      }

      else if (field_loc_type == CS_MESH_LOCATION_VERTICES)
        cs_post_write_vertex_var(post_mesh->id,
                                 CS_POST_WRITER_ALL_ASSOCIATED,
                                 name,
                                 f->dim,
                                 true,
                                 use_parent,
                                 CS_POST_TYPE_cs_real_t,
                                 f->val,
                                 ts);

    } /* End of loop on fields */

  } /* End of main output for cell or boundary mesh or submesh */

  /* Base output for probes */
  /*------------------------*/

  else if (post_mesh->cat_id == CS_POST_MESH_PROBES) {

    const int n_fields = cs_field_n_fields();
    const int vis_key_id = cs_field_key_id("post_vis");
    const int label_key_id = cs_field_key_id("label");

    /* Loop on fields */

    for (int f_id = 0; f_id < n_fields; f_id++) {

      cs_field_t  *f = cs_field_by_id(f_id);

      const cs_mesh_location_type_t field_loc_type
        = cs_mesh_location_get_type(f->location_id);

      if (pset_on_boundary) {
        if (   field_loc_type != CS_MESH_LOCATION_CELLS
            && field_loc_type != CS_MESH_LOCATION_BOUNDARY_FACES
            && field_loc_type != CS_MESH_LOCATION_VERTICES)
          continue;
      }
      else {
        if (   field_loc_type != CS_MESH_LOCATION_CELLS
            && field_loc_type != CS_MESH_LOCATION_VERTICES)
          continue;
      }

      if (! (cs_field_get_key_int(f, vis_key_id) & CS_POST_MONITOR))
        continue;

      const char *name = cs_field_get_key_str(f, label_key_id);
      if (name == nullptr)
        name = f->name;

      cs_interpolate_from_location_t
        *interpolate_func = cs_interpolate_from_location_p0;
      if (   field_loc_type == CS_MESH_LOCATION_CELLS
          && pset_interpolation == 1) {
        interpolate_func = cs_interpolate_from_location_p1;
        if (_field_sync != nullptr) {
          if (_field_sync[f->id] == 0) {
            if (f->dim == 1 || f->dim == 3 || f->dim == 6 || f->dim == 9)
              cs_field_synchronize(f, CS_HALO_EXTENDED);
            _field_sync[f->id] = 1;
          }
        }
      }

      char interpolate_input[96];
      strncpy(interpolate_input, f->name, 95); interpolate_input[95] = '\0';

      cs_post_write_probe_values(post_mesh->id,
                                 CS_POST_WRITER_ALL_ASSOCIATED,
                                 name,
                                 f->dim,
                                 CS_POST_TYPE_cs_real_t,
                                 f->location_id,
                                 interpolate_func,
                                 interpolate_input,
                                 f->val,
                                 ts);

    } /* End of loop on fields */

  } /* End of main output for probes */

  /* Special case for mesh displacement even when mesh category does
     not indicate automatic propagation to sub-meshes
     --------------------------------------------------------------- */

  else if (   post_mesh->ent_flag[0]
           || post_mesh->ent_flag[1]
           || post_mesh->ent_flag[2]) {

    const cs_field_t  *f = cs_field_by_name_try("mesh_displacement");

    if (   f != nullptr
        && fvm_nodal_get_parent(post_mesh->exp_mesh) == cs_glob_mesh) {

      const cs_mesh_location_type_t field_loc_type
         = cs_mesh_location_get_type(f->location_id);

      if (field_loc_type == CS_MESH_LOCATION_VERTICES) {

        const int vis_key_id = cs_field_key_id("post_vis");

        if (cs_field_get_key_int(f, vis_key_id) & CS_POST_ON_LOCATION) {

          const int label_key_id = cs_field_key_id("label");
          const char *name = cs_field_get_key_str(f, label_key_id);
          if (name == nullptr)
            name = f->name;

          cs_post_write_vertex_var(post_mesh->id,
                                   CS_POST_WRITER_ALL_ASSOCIATED,
                                   name,
                                   f->dim,
                                   true,
                                   true, /* use_parent */
                                   CS_POST_TYPE_cs_real_t,
                                   f->val,
                                   ts);

        }
      }

    }

  } /* End of special output for mesh displacement */
}

/*----------------------------------------------------------------------------
 * Post-processing output of additional associated fields.
 *
 * parameters:
 *   post_mesh   <-- pointer to post-processing mesh structure
 *   ts          <-- time step status structure, or null
 *----------------------------------------------------------------------------*/

static void
_cs_post_output_attached_fields(cs_post_mesh_t        *post_mesh,
                                const cs_time_step_t  *ts)
{
  const int label_key_id = cs_field_key_id("label");

  bool pset_on_boundary = false;
  cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  cs_interpolate_from_location_t
    *interpolate_func = cs_interpolate_from_location_p0;

  int pset_interpolation = 0;

  if (pset != nullptr) {
    cs_probe_set_get_post_info(pset,
                               nullptr,
                               &pset_on_boundary,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
    if (pset_on_boundary == false && cs_probe_set_get_interpolation(pset) == 1){
      pset_interpolation = 1;
      interpolate_func = cs_interpolate_from_location_p1;
    }
  }

  for (int i = 0; i < post_mesh->n_a_fields; i++) {

    bool  use_parent = true;

    char name_buf[96];

    int writer_id = post_mesh->a_field_info[i*3];
    int f_id = post_mesh->a_field_info[i*3+1];
    int comp_id = post_mesh->a_field_info[i*3+2];

    cs_field_t  *f = cs_field_by_id(f_id);

    const cs_mesh_location_type_t field_loc_type
      = cs_mesh_location_get_type(f->location_id);

    const char *name = cs_field_get_key_str(f, label_key_id);
    if (name == nullptr)
      name = f->name;

    cs_interpolate_from_location_t
      *_interpolate_func = interpolate_func;

    int f_dim = f->dim;
    cs_real_t *_val = nullptr;
    const cs_real_t *f_val = f->val;

    if (f->dim > 1 && comp_id > -1) {
      if (comp_id >= f->dim) /* only if input is incorrect */
        continue;
      else {
        _val = _extract_field_component(f, comp_id, name, name_buf);
        f_dim = 1;
        f_val = _val;
        name = name_buf;
        _interpolate_func = cs_interpolate_from_location_p0;
      }
    }

    /* Volume or surface mesh */

    if (_cs_post_match_post_write_var(post_mesh, field_loc_type)) {

      if (   field_loc_type == CS_MESH_LOCATION_CELLS
          || field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES
          || field_loc_type == CS_MESH_LOCATION_INTERIOR_FACES) {

        const cs_real_t *cell_val = nullptr, *b_face_val = nullptr;

        if (field_loc_type == CS_MESH_LOCATION_CELLS)
          cell_val = f_val;
        else /* if (field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES) */
          b_face_val = f_val;

        cs_post_write_var(post_mesh->id,
                          writer_id,
                          name,
                          f_dim,
                          true,
                          use_parent,
                          CS_POST_TYPE_cs_real_t,
                          cell_val,
                          nullptr,
                          b_face_val,
                          ts);
      }

      else if (field_loc_type == CS_MESH_LOCATION_VERTICES)
        cs_post_write_vertex_var(post_mesh->id,
                                 writer_id,
                                 name,
                                 f_dim,
                                 true,
                                 use_parent,
                                 CS_POST_TYPE_cs_real_t,
                                 f_val,
                                 ts);

    }

    /* Probe or profile mesh */

    else if (   pset != nullptr
             && (   field_loc_type == CS_MESH_LOCATION_CELLS
                 || field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES
                 || field_loc_type == CS_MESH_LOCATION_VERTICES)) {

      if (! pset_on_boundary
          && field_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES)
        continue;

      char interpolate_input[96];
      strncpy(interpolate_input, f->name, 95); interpolate_input[95] = '\0';

      if (   field_loc_type == CS_MESH_LOCATION_CELLS
          && pset_interpolation == 1) {
        if (_field_sync != nullptr) {
          if (_field_sync[f->id] == 0) {
            cs_field_synchronize(f, CS_HALO_EXTENDED);
            _field_sync[f->id] = 1;
          }
        }
      }

      cs_post_write_probe_values(post_mesh->id,
                                 writer_id,
                                 name,
                                 f_dim,
                                 CS_POST_TYPE_cs_real_t,
                                 f->location_id,
                                 _interpolate_func,
                                 interpolate_input,
                                 f_val,
                                 ts);

    }

    CS_FREE(_val);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Transfer field output info from probe set to associated
 *        postprocessing mesh.
 *
 * \param[in]  post_mesh  postprocessing mesh structure
 */
/*----------------------------------------------------------------------------*/

static void
_attach_probe_set_fields(cs_post_mesh_t  *post_mesh)
{
  cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  if (pset == nullptr)
    return;

  int  ps_naf = 0;
  int *ps_afi = nullptr;

  cs_probe_set_transfer_associated_field_info(pset, &ps_naf, &ps_afi);

  post_mesh->n_a_fields = 0;
  CS_REALLOC(post_mesh->a_field_info, 3*ps_naf, int);

  const int vis_key_id = cs_field_key_id("post_vis");
  int vis_key_mask = 0;
  if (   post_mesh->cat_id == CS_POST_MESH_BOUNDARY
      || post_mesh->cat_id == CS_POST_MESH_VOLUME)
    vis_key_mask = CS_POST_ON_LOCATION;
  else if (post_mesh->cat_id == CS_POST_MESH_PROBES)
    vis_key_mask = CS_POST_MONITOR;

  for (int i = 0; i < ps_naf; i++) {

    int writer_id = ps_afi[i*3];
    int field_id = ps_afi[i*3 + 1];
    int comp_id = ps_afi[i*3 + 2];

    const cs_field_t  *f = cs_field_by_id(field_id);

    if (f == nullptr)
      continue;

    /* Check that the field is not already output automatically */

    bool redundant = false;

    if (cs_field_get_key_int(f, vis_key_id) & vis_key_mask)
      redundant = true;

    if (! redundant) {
      for (int j = 0; j < post_mesh->n_a_fields; j++) {
        int *afi = post_mesh->a_field_info + 3*j;
        if (   afi[0] == writer_id && afi[1] == field_id
            && (afi[2] == comp_id || f->dim == 1)) {
          redundant = true;
          break;
        }
        afi += 3;
      }
    }

    if (! redundant) {
      int *afi = post_mesh->a_field_info + 3*post_mesh->n_a_fields;
      afi[0] = writer_id;
      afi[1] = field_id;
      afi[2] = comp_id;
      post_mesh->n_a_fields += 1;
    }

  }

  CS_FREE(ps_afi);
  CS_REALLOC(post_mesh->a_field_info, 3*post_mesh->n_a_fields, int);
}

/*----------------------------------------------------------------------------
 * Main post-processing output of additional function data
 *
 * parameters:
 *   post_mesh   <-- pointer to post-processing mesh structure
 *   ts          <-- time step status structure, or null
 *----------------------------------------------------------------------------*/

static void
_output_function_data(cs_post_mesh_t        *post_mesh,
                      const cs_time_step_t  *ts)
{
  const int n_functions = cs_function_n_functions();
  if (n_functions == 0 || post_mesh->n_writers < 1)
    return;

  int  pset_interpolation = 0;
  bool pset_on_boundary = false;
  cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  if (pset != nullptr) {
    cs_probe_set_get_post_info(pset,
                               nullptr,
                               &pset_on_boundary,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
    if (pset_on_boundary == false && cs_probe_set_get_interpolation(pset) == 1)
      pset_interpolation = 1;
  }

  bool have_transient = _post_mesh_have_active_transient(post_mesh);
  bool may_have_time_independent = false;
  for (int i = 1; i < post_mesh->n_writers; i++) {
    if (post_mesh->nt_last[i] <= 0)
      may_have_time_independent = true;
  }

  /* Base output for cell and boundary meshes */
  /*------------------------------------------*/

  if (   post_mesh->cat_id == CS_POST_MESH_VOLUME
      || post_mesh->cat_id == CS_POST_MESH_BOUNDARY
      || post_mesh->cat_id == CS_POST_MESH_SURFACE) {

    /* Loop on functions */

    for (int f_id = 0; f_id < n_functions; f_id++) {

      const cs_function_t *f = cs_function_by_id(f_id);
      const cs_time_step_t  *_ts = ts;

      if (! (f->post_vis & CS_POST_ON_LOCATION))
        continue;

      if (f->type & CS_FUNCTION_TIME_INDEPENDENT) {
        if (   may_have_time_independent == false
            && have_transient == false)
          continue;
        _ts = nullptr;
      }

      const cs_mesh_location_type_t f_loc_type
        = cs_mesh_location_get_type(f->location_id);

      if (_cs_post_match_post_write_var(post_mesh, f_loc_type) == false)
        continue;

      const char *name = f->label;
      if (name == nullptr)
        name = f->name;

      if (pset != nullptr) {
        char interpolate_input[96];
        strncpy(interpolate_input, f->name, 95); interpolate_input[95] = '\0';

        cs_interpolate_from_location_t
          *interpolate_func = cs_interpolate_from_location_p0;
        if (   f_loc_type == CS_MESH_LOCATION_CELLS
            && pset_interpolation == 1)
          interpolate_func = cs_interpolate_from_location_p1;

        cs_post_write_probe_function(post_mesh->id,
                                     CS_POST_WRITER_ALL_ASSOCIATED,
                                     f,
                                     f->location_id,
                                     interpolate_func,
                                     interpolate_input,
                                     _ts);
      }

      else if (f_loc_type == CS_MESH_LOCATION_CELLS)
        cs_post_write_function(post_mesh->id,
                               CS_POST_WRITER_ALL_ASSOCIATED,
                               f, nullptr, nullptr,
                               _ts);

      else if (f_loc_type == CS_MESH_LOCATION_BOUNDARY_FACES)
        cs_post_write_function(post_mesh->id,
                               CS_POST_WRITER_ALL_ASSOCIATED,
                               nullptr, nullptr, f,
                               _ts);

      else if (f_loc_type == CS_MESH_LOCATION_INTERIOR_FACES)
        cs_post_write_function(post_mesh->id,
                               CS_POST_WRITER_ALL_ASSOCIATED,
                               nullptr, f, nullptr,
                               _ts);
      else if (f_loc_type == CS_MESH_LOCATION_VERTICES)
        cs_post_write_vertex_function(post_mesh->id,
                                      CS_POST_WRITER_ALL_ASSOCIATED,
                                      f,
                                      _ts);

    } /* End of loop on functions */

  } /* End of main output for cell or boundary mesh or submesh */

  /* Base output for probes */
  /*------------------------*/

  else if (post_mesh->cat_id == CS_POST_MESH_PROBES) {

    /* Loop on functions */

    for (int f_id = 0; f_id < n_functions; f_id++) {

      const cs_function_t  *f = cs_function_by_id(f_id);
      const cs_time_step_t  *_ts = ts;

      if (! (f->post_vis & CS_POST_MONITOR))
        continue;

      if (f->type & CS_FUNCTION_TIME_INDEPENDENT) {
        if (may_have_time_independent)
          _ts = nullptr;
        else if (have_transient == false)
          continue;
      }

      const cs_mesh_location_type_t f_loc_type
        = cs_mesh_location_get_type(f->location_id);

      if (pset_on_boundary) {
        if (   f_loc_type != CS_MESH_LOCATION_CELLS
            && f_loc_type != CS_MESH_LOCATION_BOUNDARY_FACES
            && f_loc_type != CS_MESH_LOCATION_VERTICES)
          continue;
      }
      else {
        if (   f_loc_type != CS_MESH_LOCATION_CELLS
            && f_loc_type != CS_MESH_LOCATION_VERTICES)
          continue;
      }

      const char *name = f->label;
      if (name == nullptr)
        name = f->name;

      cs_interpolate_from_location_t
        *interpolate_func = cs_interpolate_from_location_p0;
      if (   f_loc_type == CS_MESH_LOCATION_CELLS
          && pset_interpolation == 1)
        interpolate_func = cs_interpolate_from_location_p1;

      char interpolate_input[96];
      strncpy(interpolate_input, f->name, 95); interpolate_input[95] = '\0';

      cs_post_write_probe_function(post_mesh->id,
                                   CS_POST_WRITER_ALL_ASSOCIATED,
                                   f,
                                   f->location_id,
                                   interpolate_func,
                                   interpolate_input,
                                   _ts);

    } /* End of loop on functions */

  } /* End of main output for probes */
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a post-processing mesh for probes (set of probes should have
 *        been already defined)
 *
 * \param[in] mesh_id        id of mesh to define (<0 reserved, >0 for user)
 * \param[in] pset           pointer to a cs_probe_set_t structure
 * \param[in] time_varying   true if probe coords may change during computation
 * \param[in] is_profile     true if probe set is related to a profile
 * \param[in] on_boundary    true if probes are located on boundary
 * \param[in] auto_variable  true if the set of variables to post is predefined
 * \param[in] n_writers      number of associated writers
 * \param[in] writer_ids     ids of associated writers
 */
/*----------------------------------------------------------------------------*/

static void
_cs_post_define_probe_mesh(int                    mesh_id,
                           cs_probe_set_t        *pset,
                           bool                   time_varying,
                           bool                   is_profile,
                           bool                   on_boundary,
                           bool                   auto_variable,
                           int                    n_writers,
                           const int              writer_ids[])
{
  assert(pset != nullptr); /* Sanity check */

  /* Common initializations */

  int  mode = (is_profile == true) ? 4 : 3;
  cs_post_mesh_t *post_mesh = _predefine_mesh(mesh_id, time_varying, mode,
                                              n_writers, writer_ids);

  /* Define mesh based on current arguments */

  const char  *mesh_name = cs_probe_set_get_name(pset);
  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  post_mesh->sel_func[4] = nullptr;
  post_mesh->sel_input[4] = pset;

  post_mesh->add_groups = false;

  if (auto_variable) {
    if (is_profile) {
      if (on_boundary)
        post_mesh->cat_id = CS_POST_MESH_BOUNDARY;
      else
        post_mesh->cat_id = CS_POST_MESH_VOLUME;
    }
    else
      post_mesh->cat_id = CS_POST_MESH_PROBES;
  }

  _attach_probe_set_fields(post_mesh);

  /* Try to assign probe location mesh */

  const char _select_all[] = "all[]";
  const char *sel_criteria = cs_probe_set_get_location_criteria(pset);
  if (sel_criteria == nullptr)
    sel_criteria = _select_all;

  /* Check for existing meshes with the same selection criteria */

  int  ent_flag_id = (on_boundary) ? 2 : 0;
  int  match_partial[2] = {-1, -1};
  bool all_elts = (strcmp(sel_criteria, _select_all) == 0) ? true : false;

  if (all_elts) {
    if (on_boundary)
      post_mesh->location_id = CS_MESH_LOCATION_BOUNDARY_FACES;
    else
      post_mesh->location_id = CS_MESH_LOCATION_CELLS;
  }

  for (int i = 0; i < _cs_post_n_meshes; i++) {

    cs_post_mesh_t *post_mesh_cmp = _cs_post_meshes + i;

    if (   all_elts
        && (time_varying == false || post_mesh_cmp->time_varying == true)) {
      if (post_mesh->ent_flag[ent_flag_id] > 0) {
        post_mesh->locate_ref = i;
        break;
      }
    }

    if (post_mesh_cmp->criteria[ent_flag_id] != nullptr) {
      if (strcmp(sel_criteria, post_mesh_cmp->criteria[ent_flag_id]) == 0) {
        if (time_varying == false || post_mesh_cmp->time_varying == true)
          post_mesh->locate_ref = i;
        break;
      }
      else {
        if (post_mesh_cmp->n_writers == 0)
          match_partial[1] = i;
        else {
          for (int j = 0; j < n_writers && match_partial[0] == -1; j++) {
            for (int k = 0; k < post_mesh_cmp->n_writers; k++)
              if (writer_ids[j] == post_mesh_cmp->writer_id[k])
                match_partial[0] = i;
          }
        }
      }
    }

  }

  if (post_mesh->locate_ref < 0) {
    if (match_partial[0] >= 0)
      post_mesh->locate_ref = match_partial[0];
    else if (match_partial[1] >= 0)
      post_mesh->locate_ref = match_partial[1];
  }

  /* Add (define) location mesh if none found */

  if (post_mesh->locate_ref == -1) {
    int new_id = cs_post_get_free_mesh_id();
    if (on_boundary)
      cs_post_define_surface_mesh(new_id,
                                  "probe_set_location_mesh",
                                  nullptr,
                                  sel_criteria,
                                  false,
                                  false,
                                  0,
                                  nullptr);
    else
      cs_post_define_volume_mesh(new_id,
                                 "probe_set_location_mesh",
                                 sel_criteria,
                                 false,
                                 false,
                                 0,
                                 nullptr);

    /* In case the mesh array has been reallocated, reset pointer */
    int _mesh_id = _cs_post_mesh_id_try(mesh_id);
    post_mesh = _cs_post_meshes + _mesh_id;

    post_mesh->locate_ref = _cs_post_mesh_id(new_id);

    (_cs_post_meshes + post_mesh->locate_ref)->time_varying = true;
  }
}

/*============================================================================
 * Fortran wrapper function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Update "active" or "inactive" flag of writers based on the time step.
 *
 * Writers are activated if their output interval is a divisor of the
 * current time step, or if their optional time step and value output lists
 * contain matches for the current time step.
 *----------------------------------------------------------------------------*/

void
cs_f_post_activate_by_time_step(void)
{
  cs_post_activate_by_time_step(cs_glob_time_step);
}

/*----------------------------------------------------------------------------
 * Output a floating point variable defined at cells or faces of a
 * post-processing mesh using associated writers.
 *
 * parameters:
 *   mesh_id     <-- id of associated mesh
 *   var_name    <-- name of variable to output
 *   var_dim     <-- 1 for scalar, 3 for vector
 *   interlace   <-- if a vector, true for interlaced values, false otherwise
 *   use_parent  <-- true if values are defined on "parent" mesh,
 *                   false if values are defined on post-processing mesh
 *   nt_cur_abs  <-- current time step number
 *   t_cur_abs   <-- current physical time
 *   cel_vals    <-- cell values
 *   i_face_vals <-- interior face values
 *   b_face_vals <-- boundary face values
 *----------------------------------------------------------------------------*/

void
cs_f_post_write_var(int               mesh_id,
                    const char       *var_name,
                    int               var_dim,
                    bool              interlace,
                    bool              use_parent,
                    int               nt_cur_abs,
                    double            t_cur_abs,
                    const cs_real_t  *cel_vals,
                    const cs_real_t  *i_face_vals,
                    const cs_real_t  *b_face_vals)
{
  CS_UNUSED(t_cur_abs);

  cs_post_type_t var_type
    = (sizeof(cs_real_t) == 8) ? CS_POST_TYPE_double : CS_POST_TYPE_float;

  const cs_time_step_t  *ts = cs_glob_time_step;

  if (nt_cur_abs < 0) /* Allow forcing of time-independent output */
    ts = nullptr;

  cs_post_write_var(mesh_id,
                    CS_POST_WRITER_ALL_ASSOCIATED,
                    var_name,
                    var_dim,
                    interlace,
                    use_parent,
                    var_type,
                    cel_vals,
                    i_face_vals,
                    b_face_vals,
                    ts);
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a writer; this objects manages a case's name, directory,
 *        and format, as well as associated mesh's time dependency, and the
 *        default output interval for associated variables.
 *
 * This function must be called before the time loop. If a writer with a
 * given id is defined multiple times, the last definition supersedes the
 * previous ones.
 *
 * Current reserved ids are the following: CS_POST_WRITER_DEFAULT
 * for main/default output, CS_POST_WRITER_ERRORS for error visualization,
 * CS_POST_WRITER_PROBES for main probes, CS_POST_WRITER_PARTICLES for
 * particles, CS_POST_WRITER_TRAJECTORIES for trajectories. Other negative
 * ids may be dynamically reserved by the code depending on options.
 * Positive ids identify user-defined writers.
 *
 * \warning depending on the chosen format, the \em case_name may be
 * shortened (maximum number of characters: 32 for \em MED, 19 for \em EnSight,
 * or modified automatically (white-space or forbidden characters will be
 * replaced by "_").
 *
 * The \c \b format_name argument is used to choose the output format, and the
 * following values are allowed (assuming the matching
 * support was built):
 *
 * - \c \b EnSight \c \b Gold (\c \b EnSight also accepted)
 * - \c \b MED
 * - \c \b CGNS
 * - \c \b Catalyst (in-situ visualization)
 * - \c \b MEDCoupling (in-memory structure, to be used from other code)
 * - \c \b Melissa (in-situ statistics)
 * - \c \b histogram (comma or whitespace separated 2d plot files)
 * - \c \b plot (comma or whitespace separated 2d plot files)
 * - \c \b time_plot (comma or whitespace separated time plot files)
 * - \c \b CCM (only for the full volume and boundary meshes)
 *
 * The format name is case-sensitive, so \c \b ensight or \c \b cgns are also valid.
 *
 * The optional \c \b fmt_opts character string contains a list of options related
 * to the format, separated by spaces or commas; these options include:
 *
 * - \c \b binary for a binary format version (default)
 * - \c \b big_endian to force outputs to be in \c \b big-endian mode
 *         (for \c \b EnSight).
 * - \c \b text for a text format version (for \c \b EnSight).
 * - \c \b adf for ADF file type (for \c \b CGNS).
 * - \c \b hdf5 for HDF5 file type (for \c \b CGNS, normally the default if
 *         HDF5 support is available).
 * - \c \b discard_polygons to prevent from exporting faces with more than
 *         four edges (which may not be recognized by some post-processing
 *         tools); such faces will therefore not appear in the post-processing
 *         mesh.
 * - \c \b discard_polyhedra to prevent from exporting elements which are
 *         neither tetrahedra, prisms, pyramids nor hexahedra (which may not
 *         be recognized by some post-processing tools); such elements will
 *         therefore not appear in the post-processing mesh.
 * - \c \b divide_polygons to divide faces with more than four edges into
 *         triangles, so that any post-processing tool can recognize them.
 * - \c \b divide_polyhedra} to divide elements which are neither tetrahedra,
 *         prisms, pyramids nor hexahedra into simpler elements (tetrahedra and
 *         pyramids), so that any post-processing tool can recognize them.
 * - \c \b separate_meshes to multiple meshes and associated fields to
 *         separate outputs.
 *
 * Note that the white-spaces in the beginning or in the end of the
 * character strings given as arguments here are suppressed automatically.
 *
 * \param[in]  writer_id        id of writer to create. (< 0 reserved,
 *                              > 0 for user); even for reserved ids,
 *                              the matching writer's options
 *                              may be redefined by calls to this function
 * \param[in]  case_name        associated case name
 * \param[in]  dir_name         associated directory name
 * \param[in]  fmt_name         associated format name
 * \param[in]  fmt_opts         associated format options string
 * \param[in]  time_dep         \ref FVM_WRITER_FIXED_MESH if mesh definitions
 *                              are fixed, \ref FVM_WRITER_TRANSIENT_COORDS if
 *                              coordinates change,
 *                              \ref FVM_WRITER_TRANSIENT_CONNECT if
 *                              connectivity changes
 * \param[in]  output_at_start  force output at calculation start if true
 * \param[in]  output_at_end    force output at calculation end if true
 * \param[in]  interval_n       default output interval in time-steps, or < 0
 * \param[in]  interval_t       default output interval in seconds, or < 0
 *                              (has priority over interval_n)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_writer(int                     writer_id,
                      const char             *case_name,
                      const char             *dir_name,
                      const char             *fmt_name,
                      const char             *fmt_opts,
                      fvm_writer_time_dep_t   time_dep,
                      bool                    output_at_start,
                      bool                    output_at_end,
                      int                     interval_n,
                      double                  interval_t)
{
  /* local variables */

  int    i;

  cs_post_writer_t  *w = nullptr;
  cs_post_writer_def_t  *wd = nullptr;

  /* Initialize timer statistics if necessary */

  if (_post_out_stat_id < 0)
    _post_out_stat_id =  cs_timer_stats_id_by_name("postprocessing_output");

  /* Check if the required writer already exists */

  if (writer_id == 0)
    bft_error(__FILE__, __LINE__, 0,
              _("The requested post-processing writer number\n"
                "must be < 0 (reserved) or > 0 (user).\n"));

  for (i = 0; i < _cs_post_n_writers; i++) {
    if ((_cs_post_writers + i)->id == writer_id) {
      w = _cs_post_writers + i;
      CS_FREE(w->ot);
      wd = w->wd;
      assert(wd != nullptr);
      CS_FREE(wd->case_name);
      CS_FREE(wd->dir_name);
      CS_FREE(wd->fmt_opts);
      break;
    }
  }

  if (i == _cs_post_n_writers) { /* New definition */

    /* Resize global writers array */

    if (_cs_post_n_writers == _cs_post_n_writers_max) {
      if (_cs_post_n_writers_max == 0)
        _cs_post_n_writers_max = 4;
      else
        _cs_post_n_writers_max *= 2;
      CS_REALLOC(_cs_post_writers,
                 _cs_post_n_writers_max,
                 cs_post_writer_t);
    }

    if (writer_id < _cs_post_min_writer_id)
      _cs_post_min_writer_id = writer_id;
    _cs_post_n_writers += 1;

    w = _cs_post_writers + i;
    CS_MALLOC(w->wd, 1, cs_post_writer_def_t);
    wd = w->wd;

  }

  /* Assign writer definition to the structure */

  w->id = writer_id;
  w->active = 0;

  if (interval_t >= 0)
    cs_time_control_init_by_time(&(w->tc),
                                 -1,
                                 -1,
                                 interval_t,
                                 output_at_start,
                                 output_at_end);
  else
    cs_time_control_init_by_time_step(&(w->tc),
                                      -1,
                                      -1,
                                      interval_n,
                                      output_at_start,
                                      output_at_end);

  w->tc.last_nt = -2;
  w->tc.last_t = cs_glob_time_step->t_prev;
  if (w->tc.type == CS_TIME_CONTROL_TIME) {
    int n_steps = w->tc.last_t / interval_t;
    if (n_steps * interval_t > w->tc.last_t)
      n_steps -= 1;
    double t_prev = n_steps * interval_t;
    if (t_prev < cs_glob_time_step->t_prev)
      w->tc.last_t = t_prev;
  }
  w->ot = nullptr;

  wd->time_dep = time_dep;

  CS_MALLOC(wd->case_name, strlen(case_name) + 1, char);
  strcpy(wd->case_name, case_name);

  CS_MALLOC(wd->dir_name, strlen(dir_name) + 1, char);
  strcpy(wd->dir_name, dir_name);

  wd->fmt_id = fvm_writer_get_format_id(fmt_name);

  if (fmt_opts != nullptr) {
    CS_MALLOC(wd->fmt_opts, strlen(fmt_opts) + 1, char);
    strcpy(wd->fmt_opts, fmt_opts);
  }
  else {
    CS_MALLOC(wd->fmt_opts, 1, char);
    wd->fmt_opts[0] = '\0';
  }

  w->writer = nullptr;

  /* If writer is the default writer, update defaults */

  if (writer_id == CS_POST_WRITER_DEFAULT) {
    _cs_post_default_format_id = wd->fmt_id;
    if (wd->fmt_opts != nullptr) {
      CS_REALLOC(_cs_post_default_format_options,
                 strlen(wd->fmt_opts)+ 1,
                 char);
      strcpy(_cs_post_default_format_options, wd->fmt_opts);
    }
    else
      CS_FREE(_cs_post_default_format_options);
    /* Remove possible "separate_writers" option from default format */
    fvm_writer_filter_option(_cs_post_default_format_options,
                             "separate_meshes");
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a volume post-processing mesh.
 *
 * \param[in]  mesh_id         id of mesh to define
 *                             (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name       associated mesh name
 * \param[in]  cell_criteria   selection criteria for cells
 * \param[in]  add_groups      if true, add group information if present
 * \param[in]  auto_variables  if true, automatic output of main variables
 * \param[in]  n_writers       number of associated writers
 * \param[in]  writer_ids      ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_volume_mesh(int          mesh_id,
                           const char  *mesh_name,
                           const char  *cell_criteria,
                           bool         add_groups,
                           bool         auto_variables,
                           int          n_writers,
                           const int    writer_ids[])
{
  /* Call common initialization */

  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, true, 0, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  if (cell_criteria != nullptr) {
    CS_MALLOC(post_mesh->criteria[0], strlen(cell_criteria) + 1, char);
    strcpy(post_mesh->criteria[0], cell_criteria);
    if (!strcmp(cell_criteria, "all[]"))
      post_mesh->location_id = CS_MESH_LOCATION_CELLS;
  }
  else
    post_mesh->location_id = CS_MESH_LOCATION_CELLS;

  post_mesh->ent_flag[0] = 1;

  post_mesh->add_groups = (add_groups) ? true : false;
  if (auto_variables)
    post_mesh->cat_id = CS_POST_MESH_VOLUME;

  if (post_mesh->cat_id == CS_POST_MESH_VOLUME)
    post_mesh->post_domain = true;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a volume post-processing mesh using a selection function.
 *
 * The selection may be updated over time steps if both the time_varying
 * flag is set to true and the mesh is only associated with writers defined
 * with the FVM_WRITER_TRANSIENT_CONNECT option.
 *
 * Note: if the cell_select_input pointer is non-null, it must point
 * to valid data when the selection function is called, so either:
 * - that value or structure should not be temporary (i.e. local);
 * - post-processing output must be ensured using cs_post_write_meshes()
 *   with a fixed-mesh writer before the data pointed to goes out of scope;
 *
 * \param[in]  mesh_id            id of mesh to define
 *                                (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name          associated mesh name
 * \param[in]  cell_select_func   pointer to cells selection function
 * \param[in]  cell_select_input  pointer to optional input data for the cell
 *                                selection function, or null
 * \param[in]  time_varying       if true, try to redefine mesh at each
 *                                output time
 * \param[in]  add_groups         if true, add group information if present
 * \param[in]  auto_variables     if true, automatic output of main variables
 * \param[in]  n_writers          number of associated writers
 * \param[in]  writer_ids         ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_volume_mesh_by_func(int                    mesh_id,
                                   const char            *mesh_name,
                                   cs_post_elt_select_t  *cell_select_func,
                                   void                  *cell_select_input,
                                   bool                   time_varying,
                                   bool                   add_groups,
                                   bool                   auto_variables,
                                   int                    n_writers,
                                   const int              writer_ids[])
{
  /* Call common initialization */

  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, time_varying, 0, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  post_mesh->sel_func[0] = cell_select_func;
  post_mesh->sel_input[0] = cell_select_input;
  post_mesh->ent_flag[0] = 1;

  post_mesh->add_groups = (add_groups) ? true : false;
  if (auto_variables)
    post_mesh->cat_id = CS_POST_MESH_VOLUME;

  if (post_mesh->cat_id == CS_POST_MESH_VOLUME)
    post_mesh->post_domain = true;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a surface post-processing mesh.
 *
 * \param[in]  mesh_id          id of mesh to define
 *                              (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name        associated mesh name
 * \param[in]  i_face_criteria  selection criteria for interior faces
 * \param[in]  b_face_criteria  selection criteria for boundary faces
 * \param[in]  add_groups       if true, add group information if present
 * \param[in]  auto_variables   if true, automatic output of main variables
 * \param[in]  n_writers        number of associated writers
 * \param[in]  writer_ids       ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_surface_mesh(int          mesh_id,
                            const char  *mesh_name,
                            const char  *i_face_criteria,
                            const char  *b_face_criteria,
                            bool         add_groups,
                            bool         auto_variables,
                            int          n_writers,
                            const int    writer_ids[])
{
  /* Call common initialization */

  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, true, 0, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  if (i_face_criteria != nullptr) {
    CS_MALLOC(post_mesh->criteria[1], strlen(i_face_criteria) + 1, char);
    strcpy(post_mesh->criteria[1], i_face_criteria);
    post_mesh->ent_flag[1] = 1;
  }

  if (b_face_criteria != nullptr) {
    CS_MALLOC(post_mesh->criteria[2], strlen(b_face_criteria) + 1, char);
    strcpy(post_mesh->criteria[2], b_face_criteria);
    post_mesh->ent_flag[2] = 1;
    if (!strcmp(b_face_criteria, "all[]") && i_face_criteria == nullptr)
      post_mesh->location_id = CS_MESH_LOCATION_BOUNDARY_FACES;
  }

  post_mesh->add_groups = (add_groups != 0) ? true : false;
  if (auto_variables) {
    if (post_mesh->ent_flag[1] == 0)
      post_mesh->cat_id = CS_POST_MESH_BOUNDARY;
    else
      post_mesh->cat_id = CS_POST_MESH_SURFACE;
  }

  if (post_mesh->cat_id == CS_POST_MESH_BOUNDARY)
    post_mesh->post_domain = true;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a surface post-processing mesh using selection functions.
 *
 * The selection may be updated over time steps if both the time_varying
 * flag is set to true and the mesh is only associated with writers defined
 * with the FVM_WRITER_TRANSIENT_CONNECT option.
 *
 * Note: if i_face_select_input or b_face_select_input pointer is non-null,
 * it must point to valid data when the selection function is called,
 * so either:
 * - that value or structure should not be temporary (i.e. local);
 * - post-processing output must be ensured using cs_post_write_meshes()
 *   with a fixed-mesh writer before the data pointed to goes out of scope;
 *
 * \param[in]  mesh_id              id of mesh to define
 *                                  (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name            associated mesh name
 * \param[in]  i_face_select_func   pointer to interior faces selection function
 * \param[in]  b_face_select_func   pointer to boundary faces selection function
 * \param[in]  i_face_select_input  pointer to optional input data for the
 *                                  interior faces selection function, or null
 * \param[in]  b_face_select_input  pointer to optional input data for the
 *                                  boundary faces selection function, or null
 * \param[in]  time_varying         if true, try to redefine mesh at each
 *                                  output time
 * \param[in]  add_groups           if true, add group information if present
 * \param[in]  auto_variables       if true, automatic output of main variables
 * \param[in]  n_writers            number of associated writers
 * \param[in]  writer_ids          ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_surface_mesh_by_func(int                    mesh_id,
                                    const char            *mesh_name,
                                    cs_post_elt_select_t  *i_face_select_func,
                                    cs_post_elt_select_t  *b_face_select_func,
                                    void                  *i_face_select_input,
                                    void                  *b_face_select_input,
                                    bool                   time_varying,
                                    bool                   add_groups,
                                    bool                   auto_variables,
                                    int                    n_writers,
                                    const int              writer_ids[])
{
  /* Call common initialization */

  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, time_varying, 0, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  post_mesh->sel_func[1] = i_face_select_func;
  post_mesh->sel_func[2] = b_face_select_func;

  post_mesh->sel_input[1] = i_face_select_input;
  post_mesh->sel_input[2] = b_face_select_input;

  post_mesh->add_groups = (add_groups != 0) ? true : false;

  if (post_mesh->sel_func[1] != nullptr)
    post_mesh->ent_flag[1] = 1;
  if (post_mesh->sel_func[2] != nullptr)
    post_mesh->ent_flag[2] = 1;

  if (auto_variables)
    post_mesh->cat_id = CS_POST_MESH_BOUNDARY;

  if (post_mesh->cat_id == CS_POST_MESH_BOUNDARY)
    post_mesh->post_domain = true;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a volume or surface post-processing mesh by associated
 *        mesh location id.
 *
 * \param[in]  mesh_id         id of mesh to define
 *                             (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name       associated mesh name
 * \param[in]  location_id     associated mesh location id
 * \param[in]  add_groups      if true, add group information if present
 * \param[in]  auto_variables  if true, automatic output of main variables
 * \param[in]  n_writers       number of associated writers
 * \param[in]  writer_ids      ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_mesh_by_location(int          mesh_id,
                                const char  *mesh_name,
                                int          location_id,
                                bool         add_groups,
                                bool         auto_variables,
                                int          n_writers,
                                const int    writer_ids[])
{
  /* Call common initialization */

  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, true, 0, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  post_mesh->location_id = location_id;

  cs_mesh_location_type_t loc_type = cs_mesh_location_get_type(location_id);

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  switch(loc_type) {
  case CS_MESH_LOCATION_CELLS:
    post_mesh->ent_flag[0] = 1;
    if (auto_variables) {
      post_mesh->cat_id = CS_POST_MESH_VOLUME;
      post_mesh->post_domain = true;
    }
   break;
  case CS_MESH_LOCATION_INTERIOR_FACES:
    post_mesh->ent_flag[1] = 1;
    break;
  case CS_MESH_LOCATION_BOUNDARY_FACES:
    post_mesh->ent_flag[2] = 1;
    if (auto_variables) {
      post_mesh->cat_id = CS_POST_MESH_BOUNDARY;
      post_mesh->post_domain = true;
    }
    break;
  default:
    bft_error(__FILE__, __LINE__, 0,
              _("%s: mesh locations of type %s not handled."),
              __func__, cs_mesh_location_type_name[loc_type]);
    break;
  }

  post_mesh->add_groups = (add_groups) ? true : false;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a particles post-processing mesh.
 *
 * Such a mesh is always time-varying, and will only be output by writers
 * defined with the FVM_WRITER_TRANSIENT_CONNECT option.
 *
 * If the trajectory_mode argument is set to true, this logic is reversed,
 * and output will only occur for writers defined with the
 * FVM_WRITER_FIXED_MESH option. In this case, a submesh consisting of
 * trajectory segments for the current time step will be added to
 * the output at each output time step.
 *
 * \param[in]  mesh_id         id of mesh to define
 *                             (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name       associated mesh name
 * \param[in]  cell_criteria   selection criteria for cells containing
 *                             particles, or null.
 * \param[in]  density         fraction of the particles in the selected area
 *                             which should be output (0 < density <= 1)
 * \param[in]  trajectory      if true, activate trajectory mode
 * \param[in]  auto_variables  if true, automatic output of main variables
 * \param[in]  n_writers       number of associated writers
 * \param[in]  writer_ids      ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_particles_mesh(int          mesh_id,
                              const char  *mesh_name,
                              const char  *cell_criteria,
                              double       density,
                              bool         trajectory,
                              bool         auto_variables,
                              int          n_writers,
                              const int    writer_ids[])
{
  /* Call common initialization */

  int flag = (trajectory) ? 2 : 1;
  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, true, flag, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  if (cell_criteria != nullptr) {
    CS_MALLOC(post_mesh->criteria[3], strlen(cell_criteria) + 1, char);
    strcpy(post_mesh->criteria[3], cell_criteria);
  }

  post_mesh->add_groups = false;

  post_mesh->density = cs::min(density, 1.);
  post_mesh->density = cs::max(post_mesh->density, 0.);

  if (auto_variables)
    post_mesh->cat_id = CS_POST_MESH_VOLUME;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define a particles post-processing mesh using a selection function.
 *
 * The selection may be updated over time steps.
 *
 * Such a mesh is always time-varying, and will only be output by writers
 * defined with the FVM_WRITER_TRANSIENT_CONNECT option.
 *
 * If the trajectory_mode argument is set to true, this logic is reversed,
 * and output will only occur for writers defined with the
 * FVM_WRITER_FIXED_MESH option. In this case, a submesh consisting of
 * trajectory segments for the current time step will be added to
 * the output at each output time step.
 *
 * Note: if the p_select_input pointer is non-null, it must point
 * to valid data when the selection function is called, so
 * that value or structure should not be temporary (i.e. local);
 *
 * \param[in]  mesh_id         id of mesh to define
 *                             (< 0 reserved, > 0 for user)
 * \param[in]  mesh_name       associated mesh name
 * \param[in]  p_select_func   pointer to particles selection function
 * \param[in]  p_select_input  pointer to optional input data for the particles
 *                             selection function, or null
 * \param[in]  trajectory      if true, activate trajectory mode
 * \param[in]  auto_variables  if true, automatic output of main variables
 * \param[in]  n_writers       number of associated writers
 * \param[in]  writer_ids      ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_particles_mesh_by_func(int                    mesh_id,
                                      const char            *mesh_name,
                                      cs_post_elt_select_t  *p_select_func,
                                      void                  *p_select_input,
                                      bool                   trajectory,
                                      bool                   auto_variables,
                                      int                    n_writers,
                                      const int              writer_ids[])
{
  /* Call common initialization */

  int flag = (trajectory) ? 2 : 1;
  cs_post_mesh_t *post_mesh = nullptr;

  post_mesh = _predefine_mesh(mesh_id, true, flag, n_writers, writer_ids);

  /* Define mesh based on current arguments */

  CS_MALLOC(post_mesh->name, strlen(mesh_name) + 1, char);
  strcpy(post_mesh->name, mesh_name);

  post_mesh->sel_func[3] = p_select_func;
  post_mesh->sel_input[3] = p_select_input;
  post_mesh->ent_flag[3] = 1;

  post_mesh->add_groups = false;

  post_mesh->density = 1.;

  if (auto_variables)
    post_mesh->cat_id = CS_POST_MESH_PARTICLES;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Create a post-processing mesh associated with an existing exportable
 * mesh representation.
 *
 * If the exportable mesh is not intended to be used elsewhere, one can choose
 * to transfer its property to the post-processing mesh, which will then
 * manage its lifecycle based on its own requirements.
 *
 * If the exportable mesh must still be shared, one must be careful to
 * maintain consistency between this mesh and the post-processing output.
 *
 * The mesh in exportable dimension may be of a lower dimension than
 * its parent mesh, if it has been projected. In this case, a
 * dim_shift value of 1 indicates that parent cells are mapped to
 * exportable faces, and faces to edges, while a dim_shift value of 2
 * would indicate that parent cells are mapped to edges.
 * This is important when variables values are exported.
 *
 * \param[in]  mesh_id         id of mesh to define
 *                             (< 0 reserved, > 0 for user)
 * \param[in]  exp_mesh        mesh in exportable representation
 *                             (i.e. fvm_nodal_t)
 * \param[in]  dim_shift       nonzero if exp_mesh has been projected
 * \param[in]  transfer        if true, ownership of exp_mesh is transferred
 *                             to the post-processing mesh
 * \param[in]  auto_variables  if true, automatic output of main variables
 * \param[in]  n_writers       number of associated writers
 * \param[in]  writer_ids      ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_existing_mesh(int           mesh_id,
                             fvm_nodal_t  *exp_mesh,
                             int           dim_shift,
                             bool          transfer,
                             bool          auto_variables,
                             int           n_writers,
                             const int     writer_ids[])
{
  /* local variables */

  int        i;
  int        glob_flag[3];
  cs_lnum_t  b_f_num_shift, ind_fac;

  int    loc_flag[3] = {1, 1, 1};  /* Flags 0 to 2 "inverted" compared
                                      to others so as to use a single
                                      call to
                                      MPI_Allreduce(..., MPI_MIN, ...) */

  int         dim_ent = 0;
  int         dim_ext_ent = 0;
  bool        maj_ent_flag = false;
  cs_lnum_t   n_elts = 0;

  cs_lnum_t       *num_ent_parent = nullptr;
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Initialization of base structure */

  post_mesh = _predefine_mesh(mesh_id, true, 0, n_writers, writer_ids);

  /* Assign mesh to structure */

  post_mesh->exp_mesh = exp_mesh;

  if (transfer == true)
    post_mesh->_exp_mesh = exp_mesh;

  /* Compute number of cells and/or faces */

  dim_ext_ent = fvm_nodal_get_max_entity_dim(exp_mesh);
  dim_ent = dim_ext_ent + dim_shift;
  n_elts = fvm_nodal_get_n_entities(exp_mesh, dim_ext_ent);

  if (dim_ent == 3 && n_elts > 0)
    loc_flag[0] = 0;

  else if (dim_ent == 2 && n_elts > 0) {

    CS_MALLOC(num_ent_parent, n_elts, cs_lnum_t);

    fvm_nodal_get_parent_num(exp_mesh, dim_ext_ent, num_ent_parent);

    b_f_num_shift = cs_glob_mesh->n_b_faces;
    for (ind_fac = 0; ind_fac < n_elts; ind_fac++) {
      if (num_ent_parent[ind_fac] > b_f_num_shift)
        post_mesh->n_i_faces += 1;
      else
        post_mesh->n_b_faces += 1;
    }

    CS_FREE(num_ent_parent);

    if (post_mesh->n_i_faces > 0)
      loc_flag[1] = 0;
    else if (post_mesh->n_b_faces > 0)
      loc_flag[2] = 0;

  }

  for (i = 0; i < 3; i++)
    glob_flag[i] = loc_flag[i];

#if defined(HAVE_MPI)
  if (cs_glob_n_ranks > 1)
    MPI_Allreduce (loc_flag, glob_flag, 3, MPI_INT, MPI_MIN,
                   cs_glob_mpi_comm);
#endif

  /* Global indicators of mesh entity type presence;
     updated only if the mesh is not totally empty (for time-depending
     meshes, empty at certain times, we want to know the last type
     of entity used) */

  for (i = 0; i < 3; i++) {
    if (glob_flag[i] == 0)
      maj_ent_flag = true;
  }

  if (maj_ent_flag == true) {
    for (i = 0; i < 3; i++) {
      if (glob_flag[i] == 0)         /* Inverted glob_flag 0 to 2 logic */
        post_mesh->ent_flag[i] = 1;  /* (c.f. remark above) */
      else
        post_mesh->ent_flag[i] = 0;
    }
  }

  if (auto_variables) {
    post_mesh->cat_id = CS_POST_MESH_VOLUME;
    _check_mesh_cat_id(post_mesh);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Create a mesh based upon the extraction of edges from an existing mesh.
 *
 * The newly created edges have no link to their parent elements, so
 * no variable referencing parent elements may be output to this mesh,
 * whose main use is to visualize "true" face edges when polygonal faces
 * are subdivided by the writer. In this way, even highly non-convex
 * faces may be visualized correctly if their edges are overlaid on
 * the surface mesh with subdivided polygons.
 *
 * \param[in]  mesh_id       id of edges mesh to create
 *                           (< 0 reserved, > 0 for user)
 * \param[in]  base_mesh_id  id of existing mesh (< 0 reserved, > 0 for user)
 * \param[in]  n_writers     number of associated writers
 * \param[in]  writer_ids    ids of associated writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_define_edges_mesh(int        mesh_id,
                          int        base_mesh_id,
                          int        n_writers,
                          const int  writer_ids[])
{
  /* local variables */

  cs_post_mesh_t *post_mesh = nullptr;

  cs_post_mesh_t *post_base
    = _cs_post_meshes + _cs_post_mesh_id(base_mesh_id);

  /* Add and initialize base structure */

  post_mesh = _predefine_mesh(mesh_id, true, 0, n_writers, writer_ids);

  CS_MALLOC(post_mesh->name,
            strlen(post_base->name) + strlen(_(" edges")) + 1,
            char);
  strcpy(post_mesh->name, post_base->name);
  strcat(post_mesh->name, _(" edges"));
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set restriction of a postprocessing mesh to element centers
 *
 * This allow simply using element centers instead of full representations.
 *
 * This function must be called during the postprocessing output definition
 * stage, before any output actually occurs.
 *
 * If called with a non-existing mesh or writer id, or if the writer was not
 * previously associated, no setting is changed, and this function
 * returns silently.
 *
 * \param[in]  mesh_id       id of mesh to define
 *                           (< 0 reserved, > 0 for user)
 * \param[in]  centers_only  true if only element centers sould be output,
 *                           false fo normal connectivity.
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_set_element_centers_only(int   mesh_id,
                                      bool  centers_only)
{
  int _mesh_id = _cs_post_mesh_id_try(mesh_id);

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;

  post_mesh->centers_only = centers_only;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Associate a writer to a postprocessing mesh.
 *
 * This function must be called during the postprocessing output definition
 * stage, before any output actually occurs.
 *
 * If called with a non-existing mesh or writer id, or if the writer is
 * already associated, no setting is changed, and this function
 * returns silently.
 *
 * \param[in]  mesh_id      id of mesh to define
 *                          (< 0 reserved, > 0 for user)
 * \param[in]  writer_id    id of writer to associate
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_attach_writer(int  mesh_id,
                           int  writer_id)
{
  int _mesh_id = _cs_post_mesh_id_try(mesh_id);
  int _writer_id = _cs_post_writer_id_try(writer_id);

  if (_mesh_id < 0 || _writer_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;

  /* Ignore if writer id already associated */

  for (int i = 0; i < post_mesh->n_writers; i++) {
    if (post_mesh->writer_id[i] == _writer_id)
      return;
  }

  CS_REALLOC(post_mesh->writer_id, post_mesh->n_writers + 1, int);
  CS_REALLOC(post_mesh->nt_last, post_mesh->n_writers + 1, int);
  post_mesh->writer_id[post_mesh->n_writers] = _writer_id;
  post_mesh->nt_last[post_mesh->n_writers] = -2;
  post_mesh->n_writers += 1;

  _update_mesh_writer_associations(post_mesh);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief De-associate a writer from a postprocessing mesh.
 *
 * This function must be called during the postprocessing output definition
 * stage, before any output actually occurs.
 *
 * If called with a non-existing mesh or writer id, or if the writer was not
 * previously associated, no setting is changed, and this function
 * returns silently.
 *
 * \param[in]  mesh_id      id of mesh to define
 *                          (< 0 reserved, > 0 for user)
 * \param[in]  writer_id    id of writer to associate
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_detach_writer(int  mesh_id,
                           int  writer_id)
{
  int _mesh_id = _cs_post_mesh_id_try(mesh_id);
  int _writer_id = _cs_post_writer_id_try(writer_id);

  if (_mesh_id < 0 || _writer_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;

  /* Check we have not output this mesh yet for this writer */

  for (int i = 0; i < post_mesh->n_writers; i++) {
    if (post_mesh->writer_id[i] == _writer_id) {
      if (post_mesh->nt_last[i] > -2)
        bft_error(__FILE__, __LINE__, 0,
                  _("Error unassociating writer %d from mesh %d:"
                    "output has already been done for this mesh, "
                    "so mesh-writer association is locked."),
                  writer_id, mesh_id);
    }
  }

  /* Ignore if writer id already associated */

  int i, j;
  for (i = 0, j = 0; i < post_mesh->n_writers; i++) {
    if (post_mesh->writer_id[i] != _writer_id) {
      post_mesh->writer_id[j] = post_mesh->writer_id[i];
      post_mesh->nt_last[j] = post_mesh->nt_last[i];
      j++;
    }
  }

  if (j < post_mesh->n_writers) {

    post_mesh->n_writers = j;
    CS_REALLOC(post_mesh->writer_id, post_mesh->n_writers, int);
    CS_REALLOC(post_mesh->nt_last, post_mesh->n_writers, int);

    _update_mesh_writer_associations(post_mesh);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Associate a field to a writer and postprocessing mesh combination.
 *
 * This function must be called during the postprocessing output definition
 * stage, before any output actually occurs.
 *
 * If the field should already be output automatically based on the mesh
 * category and field output keywords, it will not be added.
 *
 * \param[in]  mesh_id    id of associated mesh
 * \param[in]  writer_id  id of specified associated writer,
 *                        or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  field_id   id of field to attach
 * \param[in]  comp_id    id of field component (-1 for all)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_attach_field(int  mesh_id,
                          int  writer_id,
                          int  field_id,
                          int  comp_id)
{
  const int _mesh_id = _cs_post_mesh_id_try(mesh_id);
  const cs_field_t  *f = cs_field_by_id(field_id);

  if (f == nullptr || _mesh_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;

  /* Check that the field is not already output automatically */

  bool redundant = false;

  if (   post_mesh->cat_id == CS_POST_MESH_VOLUME
      || post_mesh->cat_id == CS_POST_MESH_BOUNDARY
      || post_mesh->cat_id == CS_POST_MESH_SURFACE) {
    const int vis_key_id = cs_field_key_id("post_vis");
    if (cs_field_get_key_int(f, vis_key_id) & CS_POST_ON_LOCATION)
      redundant = true;
  }

  if (! redundant) {
    int *afi = post_mesh->a_field_info;
    for (int i = 0; i < post_mesh->n_a_fields; i++) {
      if (   afi[0] == writer_id && afi[1] == field_id
          && (afi[2] == comp_id || f->dim == 1)) {
        redundant = true;
        break;
      }
      afi += 3;
    }
  }

  if (! redundant) {
    CS_REALLOC(post_mesh->a_field_info, 3*(post_mesh->n_a_fields+1), int);
    int *afi = post_mesh->a_field_info + 3*post_mesh->n_a_fields;
    afi[0] = writer_id;
    afi[1] = field_id;
    afi[2] = comp_id;
    post_mesh->n_a_fields += 1;
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing meshes entity presence flag.
 *
 * This flag is an array of 5 integers, indicating the presence of elements
 * of given types on at least one subdomain (i.e. rank):
 *   0: presence of cells
 *   1: presence of interior faces
 *   2: presence of boundary faces
 *   3: presence of particles
 *   4: presence of probes
 *
 * \param[in]  mesh_id  postprocessing mesh id
 *
 * \return  pointer to entity presence flag
 */
/*----------------------------------------------------------------------------*/

const int *
cs_post_mesh_get_ent_flag(int  mesh_id)
{
  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  return mesh->ent_flag;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's number of cells
 *
 * \param[in]  mesh_id  postprocessing mesh id
 *
 * \return  number of cells of postprocessing mesh.
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_post_mesh_get_n_cells(int  mesh_id)
{
  cs_lnum_t retval = 0;

  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr)
    retval = fvm_nodal_get_n_entities(mesh->exp_mesh, 3);
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's list of cells
 *
 * The array of cell ids must be of at least size
 * cs_post_mesh_get_n_cells(mesh_id).
 *
 * \param[in]   mesh_id   postprocessing mesh id
 * \param[out]  cell_ids  array of associated cell ids (0 to n-1 numbering,
 *                        relative to main mesh)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_get_cell_ids(int         mesh_id,
                          cs_lnum_t  *cell_ids)
{
  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr) {
    cs_lnum_t i;
    cs_lnum_t n_cells = fvm_nodal_get_n_entities(mesh->exp_mesh, 3);
    fvm_nodal_get_parent_num(mesh->exp_mesh, 3, cell_ids);
    for (i = 0; i < n_cells; i++)
      cell_ids[i] -= 1;
  }
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's number of interior faces
 *
 * \param[in]  mesh_id  postprocessing mesh id
 *
 * \return  number of cells of postprocessing mesh.
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_post_mesh_get_n_i_faces(int  mesh_id)
{
  cs_lnum_t retval = 0;

  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr)
    retval = mesh->n_i_faces;
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's list of boundary faces.
 *
 * The array of boundary face ids must be of at least size
 * cs_post_mesh_get_n_b_faces(mesh_id).
 *
 * \param[in]   mesh_id     postprocessing mesh id
 * \param[out]  i_face_ids  array of associated interior faces ids
 *                          (0 to n-1 numbering, relative to main mesh)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_get_i_face_ids(int        mesh_id,
                            cs_lnum_t  i_face_ids[])
{
  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr) {
    cs_lnum_t i;
    cs_lnum_t n_faces = fvm_nodal_get_n_entities(mesh->exp_mesh, 2);
    const cs_lnum_t num_shift = cs_glob_mesh->n_b_faces + 1;
    if (mesh->n_b_faces == 0) {
      fvm_nodal_get_parent_num(mesh->exp_mesh, 3, i_face_ids);
      for (i = 0; i < n_faces; i++)
        i_face_ids[i] -= num_shift;
    }
    else {
      cs_lnum_t n_i_faces = 0;
      cs_lnum_t *tmp_ids = nullptr;
      CS_MALLOC(tmp_ids, n_faces, cs_lnum_t);
      fvm_nodal_get_parent_num(mesh->exp_mesh, 3, tmp_ids);
      for (i = 0; i < n_faces; i++) {
        if (tmp_ids[i] > cs_glob_mesh->n_b_faces)
          i_face_ids[n_i_faces++] = tmp_ids[i] - num_shift;
      }
      CS_FREE(tmp_ids);
    }
  }
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's number of boundary faces
 *
 * \param[in]  mesh_id  postprocessing mesh id
 *
 * \return  number of cells of postprocessing mesh.
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_post_mesh_get_n_b_faces(int  mesh_id)
{
  cs_lnum_t retval = 0;

  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr)
    retval = mesh->n_b_faces;
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's list of boundary faces.
 *
 * The array of boundary face ids must be of at least size
 * cs_post_mesh_get_n_b_faces(mesh_id).
 *
 * \param[in]   mesh_id     postprocessing mesh id
 * \param[out]  b_face_ids  array of associated boundary faces ids
 *                          (0 to n-1 numbering, relative to main mesh)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_get_b_face_ids(int        mesh_id,
                            cs_lnum_t  b_face_ids[])
{
  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr) {
    cs_lnum_t i;
    cs_lnum_t n_faces = fvm_nodal_get_n_entities(mesh->exp_mesh, 2);
    if (mesh->n_i_faces == 0) {
      fvm_nodal_get_parent_num(mesh->exp_mesh, 3, b_face_ids);
      for (i = 0; i < n_faces; i++)
        b_face_ids[i] -= 1;
    }
    else {
      cs_lnum_t n_b_faces = 0;
      cs_lnum_t *tmp_ids = nullptr;
      CS_MALLOC(tmp_ids, n_faces, cs_lnum_t);
      fvm_nodal_get_parent_num(mesh->exp_mesh, 3, tmp_ids);
      for (i = 0; i < n_faces; i++) {
        if (tmp_ids[i] > cs_glob_mesh->n_b_faces)
          b_face_ids[n_b_faces++] = tmp_ids[i] - 1;
      }
      CS_FREE(tmp_ids);
    }
  }
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's number of vertices
 *
 * \param[in]  mesh_id  postprocessing mesh id
 *
 * \return  number of vertices of postprocessing mesh.
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_post_mesh_get_n_vertices(int  mesh_id)
{
  cs_lnum_t retval = 0;

  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr)
    retval = fvm_nodal_get_n_entities(mesh->exp_mesh, 0);
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Get a postprocessing mesh's list of vertices
 *
 * The array of vertex ids must be of at least size
 * cs_post_mesh_get_n_vertices(mesh_id).
 *
 * \param[in]   mesh_id     postprocessing mesh id
 * \param[out]  vertex_ids  array of associated vertex ids (0 to n-1 numbering,
 *                          relative to main mesh)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_get_vertex_ids(int         mesh_id,
                            cs_lnum_t  *vertex_ids)
{
  const cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  if (mesh->exp_mesh != nullptr) {
    cs_lnum_t i;
    cs_lnum_t n_vertices = fvm_nodal_get_n_entities(mesh->exp_mesh, 0);
    fvm_nodal_get_parent_num(mesh->exp_mesh, 0, vertex_ids);
    for (i = 0; i < n_vertices; i++)
      vertex_ids[i] -= 1;
  }
  else
    bft_error(__FILE__, __LINE__, 0,
              _("%s called before post-processing meshes are built."),
              __func__);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Set whether postprocessing mesh's parallel domain should be output.
 *
 * \param[in]  mesh_id      postprocessing mesh id
 * \param[in]  post_domain  true if parallel domain should be output,
 *                          false otherwise.
 */
/*----------------------------------------------------------------------------*/

void
cs_post_mesh_set_post_domain(int   mesh_id,
                             bool  post_domain)
{
  cs_post_mesh_t  *mesh = _cs_post_meshes + _cs_post_mesh_id(mesh_id);

  mesh->post_domain = post_domain;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Remove a post-processing mesh.
 *
 * No further post-processing output will be allowed on this mesh,
 * so the associated structures may be freed.
 *
 * A post-processing mesh that has been associated with a time-varying
 * writer may not be removed.
 *
 * \param[in]  mesh_id  postprocessing mesh id
 */
/*----------------------------------------------------------------------------*/

void
cs_post_free_mesh(int  mesh_id)
{
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Search for requested mesh */

  int _mesh_id = _cs_post_mesh_id(mesh_id);

  /* Check if mesh was referenced for probe location */

  for (int i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->locate_ref == _mesh_id)
      bft_error(__FILE__, __LINE__, 0,
                _("Post-processing mesh number %d has been referenced\n"
                  "by probe set mesh %d, so it may not be freed.\n"),
                mesh_id, post_mesh->id);
  }

  /* Now set pointer to mesh and check for time dependency */

  post_mesh = _cs_post_meshes + _mesh_id;

  for (int i = 0; i < post_mesh->n_writers; i++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

    fvm_writer_time_dep_t time_dep = fvm_writer_get_time_dep(writer->writer);

    if (post_mesh->nt_last[i] > -2 && time_dep != FVM_WRITER_FIXED_MESH)
      bft_error(__FILE__, __LINE__, 0,
                _("Post-processing mesh number %d has been associated\n"
                  "to writer %d which allows time-varying meshes, so\n"
                  "it may not be freed.\n"),
                mesh_id, writer->id);

  }

  /* Remove mesh if allowed */

  _free_mesh(_mesh_id);

  /* Finally, update free mesh ids */

  int min_id = _MIN_RESERVED_MESH_ID;
  for (int i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->id < min_id)
      min_id = post_mesh->id;
  }
  _cs_post_min_mesh_id = min_id;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Check for the existence of a writer of the given id.
 *
 * \param[in]  writer_id  writer id to check
 *
 * \return  true if writer with this id exists, false otherwise
 */
/*----------------------------------------------------------------------------*/

bool
cs_post_writer_exists(int  writer_id)
{
  /* local variables */

  int id;
  cs_post_writer_t  *writer = nullptr;

  /* Search for requested mesh */

  for (id = 0; id < _cs_post_n_writers; id++) {
    writer = _cs_post_writers + id;
    if (writer->id == writer_id)
      return true;
  }

  return false;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Check for the existence of a post-processing mesh of the given id.
 *
 * \param[in]  mesh_id  mesh id to check
 *
 * \return  true if mesh with this id exists, false otherwise
 */
/*----------------------------------------------------------------------------*/

bool
cs_post_mesh_exists(int  mesh_id)
{
  int id;
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Search for requested mesh */

  for (id = 0; id < _cs_post_n_meshes; id++) {
    post_mesh = _cs_post_meshes + id;
    if (post_mesh->id == mesh_id)
      return true;
  }

  return false;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Find next mesh with a given category id
 *
 * \param[in]  cat_id         mesh category id filter
 * \param[in]  start_mesh_id  O at start, then previously returned mesh id
 *
 * \return  id of next mesh matching catogory, or 0 if none is found
 */
/*----------------------------------------------------------------------------*/

int
cs_post_mesh_find_next_with_cat_id(int  cat_id,
                                   int  start_mesh_id)
{
  int retval = 0;

  int s_id = 0;
  if (start_mesh_id != 0) {
    s_id = _cs_post_mesh_id_try(start_mesh_id);
    if (s_id < 0)
      s_id = _cs_post_n_meshes;
    else
      s_id += 1;
  }

  for (int id = s_id; id < _cs_post_n_meshes; id++) {
    cs_post_mesh_t  *post_mesh = _cs_post_meshes + id;
    if (post_mesh->cat_id == cat_id) {
      retval = post_mesh->id;
      break;
    }
  }

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return the default writer format name
 *
 * \return  name of the default writer format
 */
/*----------------------------------------------------------------------------*/

const char *
cs_post_get_default_format(void)
{
  return (fvm_writer_format_name(_cs_post_default_format_id));
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return the default writer format options
 *
 * \return  default writer format options string
 */
/*----------------------------------------------------------------------------*/

const char *
cs_post_get_default_format_options(void)
{
  return (_cs_post_default_format_options);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return the next "reservable" (i.e. non-user) writer id available.
 *
 * \return  the smallest negative integer present, -1
 */
/*----------------------------------------------------------------------------*/

int
cs_post_get_free_writer_id(void)
{
  return (_cs_post_min_writer_id - 1);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return the next "reservable" (i.e. non-user) mesh id available.
 *
 * \return  the smallest negative integer present, -1
 */
/*----------------------------------------------------------------------------*/

int
cs_post_get_free_mesh_id(void)
{
  return (_cs_post_min_mesh_id - 1);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update "active" or "inactive" flag of writers based on the time step.
 *
 * Writers are activated if their output interval is a divisor of the
 * current time step, or if their optional time step and value output lists
 * contain matches for the current time step.
 *
 * \param[in]  ts  time step status structure
 */
/*----------------------------------------------------------------------------*/

void
cs_post_activate_by_time_step(const cs_time_step_t  *ts)
{
  assert(ts != nullptr);

  /* Activation based on time-control (interval) */

  for (int i = 0; i < _cs_post_n_writers; i++) {

    cs_post_writer_t  *writer = _cs_post_writers + i;

    if (writer->active < 0)
      continue;

    /* In case of previous calls for a given time step,
       a writer's status may not be changed */

    if (writer->tc.last_nt == ts->nt_cur) {
      writer->active = 1;
      continue;
    }

    writer->active = cs_time_control_is_active(&(writer->tc), ts);

  }

  /* Activation by formula;

     TODO: in the future, use `cs_time_control_init_by_func` in
     the matching time control function, so as to unify the
     behavior. */

  cs_meg_post_activate();
  cs_user_postprocess_activate(ts->nt_max, ts->nt_cur, ts->t_cur);

  /* Activation at first or last time step is given priority over
     MEG-based or user-defined functions, as it can be set independently */

  for (int i = 0; i < _cs_post_n_writers; i++) {
    cs_post_writer_t  *writer = _cs_post_writers + i;
    if (writer->active == 0) {
      if (   (ts->nt_cur == ts->nt_prev && writer->tc.at_start)
          || (ts->nt_cur == ts->nt_max && writer->tc.at_end)) {
        writer->active = 1;
      }
    }
  }

  /* Ensure consistency and priority of controls by list */

  for (int i = 0; i < _cs_post_n_writers; i++) {

    cs_post_writer_t  *writer = _cs_post_writers + i;

    if (writer->active < 0)
      continue;

    /* Activation based on time step lists */

    _activate_if_listed(writer, ts);

    /* Deactivate transient writers for time-independent stages */

    if (ts->nt_cur < 0) {
      fvm_writer_time_dep_t  time_dep;
      if (writer->writer)
        time_dep = fvm_writer_get_time_dep(writer->writer);
      else
        time_dep = writer->wd->time_dep;
      if (time_dep != FVM_WRITER_FIXED_MESH)
        writer->active = 0;
    }

  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Query if a given writer is currently active.
 *
 * \param[in]  writer_id  writer id
 *
 * \return  true if writer is active at this time step, false otherwise
 */
/*----------------------------------------------------------------------------*/

bool
cs_post_writer_is_active(int  writer_id)
{
  int i = _cs_post_writer_id(writer_id);
  const cs_post_writer_t  *writer = _cs_post_writers + i;

  return writer->active;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Force the "active" or "inactive" flag for a specific writer or for all
 * writers for the current time step.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 * \param[in]  activate   false to deactivate, true to activate
 */
/*----------------------------------------------------------------------------*/

void
cs_post_activate_writer(int   writer_id,
                        bool  activate)
{
  if (writer_id != 0) {
    int i = _cs_post_writer_id(writer_id);
    cs_post_writer_t  *writer = _cs_post_writers + i;
    writer->active = (activate) ? 1 : 0;
  }
  else {
    for (int i = 0; i < _cs_post_n_writers; i++) {
      cs_post_writer_t  *writer = _cs_post_writers + i;
      writer->active = (activate) ? 1 : 0;
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Force the "active" or "inactive" flag for a specific writer or for all
 * writers for the current time step.
 *
 * This is ignored for writers which are currently disabled.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 * \param[in]  activate   false to deactivate, true to activate
 */
/*----------------------------------------------------------------------------*/

void
cs_post_activate_writer_if_enabled(int   writer_id,
                                   bool  activate)
{
  if (writer_id != 0) {
    int i = _cs_post_writer_id(writer_id);
    cs_post_writer_t  *writer = _cs_post_writers + i;
    if (writer->active > -1) {
      writer->active = (activate) ? 1 : 0;
    }
  }
  else {
    for (int i = 0; i < _cs_post_n_writers; i++) {
      cs_post_writer_t  *writer = _cs_post_writers + i;
      if (writer->active > -1) {
        writer->active = (activate) ? 1 : 0;
      }
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Disable specific writer or all writers not currently active until
 *        \ref cs_post_enable_writer or \ref cs_post_activate_writer
 *        is called for those writers.
 *
 * For each call to this function for a given writer, the same number
 * of calls to \ref cs_post_enable_writer or a single call to
 * \ref cs_post_activate_writer is required to re-enable the writer.
 *
 * This is useful to disable output even of fixed meshes in preprocessing
 * stages.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_disable_writer(int   writer_id)
{
  int i;
  cs_post_writer_t  *writer;

  if (writer_id != 0) {
    i = _cs_post_writer_id(writer_id);
    writer = _cs_post_writers + i;
    if (writer->active < 1)
      writer->active -= 1;
  }
  else {
    for (i = 0; i < _cs_post_n_writers; i++) {
      writer = _cs_post_writers + i;
      if (writer->active < 1)
        writer->active -= 1;
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Enable a specific writer or all writers currently disabled by
 *        previous calls to \ref cs_post_disable_writer.
 *
 * For each previous call to \ref cs_post_disable_writer for a given writer,
 * a call to this function (or a single call to \ref cs_post_activate_writer)
 * is required to re-enable the writer.
 *
 * This is useful to disable output even of fixed meshes in preprocessing
 * stages.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_enable_writer(int   writer_id)
{
  int i;
  cs_post_writer_t  *writer;

  if (writer_id != 0) {
    i = _cs_post_writer_id(writer_id);
    writer = _cs_post_writers + i;
    if (writer->active < 0)
      writer->active += 1;
  }
  else {
    for (i = 0; i < _cs_post_n_writers; i++) {
      writer = _cs_post_writers + i;
      if (writer->active < 0)
        writer->active += 1;
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return a pointer to the FVM writer associated to a writer_id.
 *
 * \param[in]  writer_id  associated writer id
 *
 * \return  a pointer to a fvm_writer_t structure
 */
/*----------------------------------------------------------------------------*/

fvm_writer_t *
cs_post_get_writer(int  writer_id)
{
  int  id;
  cs_post_writer_t  *writer = nullptr;

  id = _cs_post_writer_id(writer_id);
  writer = _cs_post_writers + id;

  if (writer->writer == nullptr)
    _init_writer(writer);

  return writer->writer;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return a pointer to the time control associated to a writer_id.
 *
 * \param[in]  writer_id  associated writer id
 *
 * \return  a pointer to a cs_time_control_t structure
 */
/*----------------------------------------------------------------------------*/

cs_time_control_t *
cs_post_get_time_control(int  writer_id)
{
  int id = _cs_post_writer_id(writer_id);
  cs_post_writer_t  *writer = _cs_post_writers + id;

  return &(writer->tc);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return time dependency associated to a writer_id.
 *
 * \param[in]  writer_id  associated writer id
 *
 * \return  associated writer's time dependency
 */
/*----------------------------------------------------------------------------*/

fvm_writer_time_dep_t
cs_post_get_writer_time_dep(int  writer_id)
{
  int  id;
  cs_post_writer_t  *writer = nullptr;

  fvm_writer_time_dep_t   time_dep = FVM_WRITER_FIXED_MESH;

  id = _cs_post_writer_id(writer_id);
  writer = _cs_post_writers + id;

  if (writer->wd != nullptr)
    time_dep = writer->wd->time_dep;
  else if (writer->writer != nullptr)
    time_dep = fvm_writer_get_time_dep(writer->writer);

  return time_dep;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add an activation time step for a specific writer or for all writers.
 *
 * If a negative value is provided, a previously added activation time
 * step matching that absolute value will be removed, if present.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 * \param[in]  nt         time step value to add (or remove)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_add_writer_t_step(int  writer_id,
                          int  nt)
{
  int i;

  if (writer_id != 0) {
    i = _cs_post_writer_id(writer_id);
    _add_writer_ts(_cs_post_writers + i, nt);
  }
  else {
    for (i = 0; i < _cs_post_n_writers; i++)
      _add_writer_ts(_cs_post_writers + i, nt);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add an activation time value for a specific writer or for all writers.
 *
 * If a negative value is provided, a previously added activation time
 * step matching that absolute value will be removed, if present.
 *
 * \param[in]  writer_id  writer id, or 0 for all writers
 * \param[in]  t          time value to add (or remove)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_add_writer_t_value(int     writer_id,
                           double  t)
{
  int i;

  if (writer_id != 0) {
    i = _cs_post_writer_id(writer_id);
    _add_writer_tv(_cs_post_writers + i, t);
  }
  else {
    for (i = 0; i < _cs_post_n_writers; i++)
      _add_writer_tv(_cs_post_writers + i, t);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output post-processing meshes using associated writers.
 *
 * If the time step structure argument passed is null, a time-independent
 * output will be assumed.
 *
 * \param[in]  ts  time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_meshes(const cs_time_step_t  *ts)
{
  int  i;
  cs_post_mesh_t  *post_mesh;

  int t_top_id = cs_timer_stats_switch(_post_out_stat_id);

  /* First loop on meshes, for probes and profiles (which must not be
     "reduced" afer first output, as coordinates may be required for
     interpolation, and also and share volume or surface location meshes) */

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->ent_flag[4] != 0)
      _cs_post_write_mesh(post_mesh, ts);
  }

  /* Main Loops on meshes and writers for output */

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->ent_flag[4] != 0)
      continue;
    _cs_post_write_mesh(post_mesh, ts);
    /* reduce mesh definitions if not required anymore */
    if (   post_mesh->mod_flag_max == FVM_WRITER_FIXED_MESH
        && post_mesh->_exp_mesh != nullptr)
      fvm_nodal_reduce(post_mesh->_exp_mesh, 0);
  }

  cs_timer_stats_switch(t_top_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output a variable defined at cells or faces of a post-processing mesh
 *        using associated writers.
 *
 * \param[in]  mesh_id      id of associated mesh
 * \param[in]  writer_id    id of specified associated writer,
 *                          or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  var_name     name of variable to output
 * \param[in]  var_dim      1 for scalar, 3 for vector, 6 for symmetric tensor,
 *                          9 for non-symmetric tensor
 * \param[in]  interlace    if a vector, true for interlaced values,
 *                          false otherwise
 * \param[in]  use_parent   true if values are defined on "parent" mesh,
 *                          false if values are defined on post-processing mesh
 * \param[in]  datatype     variable's data type
 * \param[in]  cel_vals     cell values
 * \param[in]  i_face_vals  interior face values
 * \param[in]  b_face_vals  boundary face values
 * \param[in]  ts           time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_var(int                    mesh_id,
                  int                    writer_id,
                  const char            *var_name,
                  int                    var_dim,
                  bool                   interlace,
                  bool                   use_parent,
                  cs_datatype_t          datatype,
                  const void            *cel_vals,
                  const void            *i_face_vals,
                  const void            *b_face_vals,
                  const cs_time_step_t  *ts)
{
  cs_lnum_t  i;
  int        _mesh_id;

  cs_interlace_t  _interlace;

  size_t       dec_ptr = 0;
  int          n_parent_lists = 0;
  cs_lnum_t    parent_num_shift[2]  = {0, 0};
  cs_real_t   *var_tmp = nullptr;
  cs_post_mesh_t  *post_mesh = nullptr;
  cs_post_writer_t    *writer = nullptr;

  const void  *var_ptr[2*9] = {nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr};

  /* Avoid a SIGSEV in lower-level functions */

  if (var_name == nullptr)
    bft_error(__FILE__, __LINE__, 0, "%s: var_name is not set.\n", __func__);

  /* Initializations */

  _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  post_mesh = _cs_post_meshes + _mesh_id;

  if (interlace)
    _interlace = CS_INTERLACE;
  else
    _interlace = CS_NO_INTERLACE;

  /* Assign appropriate array to FVM for output */

  /* Case of cells */
  /*---------------*/

  if (post_mesh->ent_flag[CS_POST_LOCATION_CELL] == 1) {

    if (use_parent) {
      n_parent_lists = 1;
      parent_num_shift[0] = 0;
    }
    else
      n_parent_lists = 0;

    var_ptr[0] = cel_vals;
    if (interlace == false) {
      if (use_parent)
        dec_ptr = cs_glob_mesh->n_cells_with_ghosts;
      else
        dec_ptr = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 3);
      dec_ptr *= cs_datatype_size[datatype];
      for (i = 1; i < var_dim; i++)
        var_ptr[i] = ((const char *)cel_vals) + i*dec_ptr;
    }
  }

  /* Case of faces */
  /*---------------*/

  else if (   post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1
           || post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {

    /* In case of indirection, all that is necessary is to set pointers */

    if (use_parent) {

      n_parent_lists = 2;
      parent_num_shift[0] = 0;
      parent_num_shift[1] = cs_glob_mesh->n_b_faces;

      if (post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {
        if (interlace == false) {
          dec_ptr = cs_glob_mesh->n_b_faces * cs_datatype_size[datatype];
          for (i = 0; i < var_dim; i++)
            var_ptr[i] = ((const char *)b_face_vals) + i*dec_ptr;
        }
        else
          var_ptr[0] = b_face_vals;
      }

      if (post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1) {
        /* For the specific case with cell centers only, faces types cannot
           currently be mixed, and data is output at the first parent list */
        int p_flag = (post_mesh->centers_only) ? 0 : 1;

        if (interlace == false) {
          dec_ptr = cs_glob_mesh->n_i_faces * cs_datatype_size[datatype];
          for (i = 0; i < var_dim; i++)
            var_ptr[p_flag*var_dim + i]
              = ((const char *)i_face_vals) + i*dec_ptr;
        }
        else
          var_ptr[p_flag] = i_face_vals;
      }

    }

    /* With no indirection, we must switch to a variable defined on two
       lists of faces to a variable defined on one list */

    else {

      n_parent_lists = 0;

      if (post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {

        /* Case where a variable is defined both on boundary and
           interior faces: we must switch to a single list, as
           indirection is not used */

        if (post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1) {

          CS_MALLOC(var_tmp,
                    (   post_mesh->n_i_faces
                     +  post_mesh->n_b_faces) * var_dim,
                    cs_real_t);

          _cs_post_assmb_var_faces
            (post_mesh->exp_mesh,
             post_mesh->n_i_faces,
             post_mesh->n_b_faces,
             var_dim,
             _interlace,
             reinterpret_cast<const cs_real_t *>(i_face_vals),
             reinterpret_cast<const cs_real_t *>(b_face_vals),
             var_tmp);

          _interlace = CS_NO_INTERLACE;

          dec_ptr = cs_datatype_size[datatype] * (  post_mesh->n_i_faces
                                                  + post_mesh->n_b_faces);

          for (i = 0; i < var_dim; i++)
            var_ptr[i] = ((char *)var_tmp) + i*dec_ptr;

        }

        /* Case where we only have boundary faces */

        else {

          if (interlace == false) {
            dec_ptr = cs_datatype_size[datatype] * post_mesh->n_b_faces;
            for (i = 0; i < var_dim; i++)
              var_ptr[i] = ((const char *)b_face_vals) + i*dec_ptr;
          }
          else
            var_ptr[0] = b_face_vals;
        }

      }

      /* Case where we only have interior faces */

      else if (post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1) {

        if (interlace == false) {
          dec_ptr = cs_datatype_size[datatype] * post_mesh->n_i_faces;
          for (i = 0; i < var_dim; i++)
            var_ptr[i] = ((const char *)i_face_vals) + i*dec_ptr;
        }
        else
          var_ptr[0] = i_face_vals;
      }

    }

  }

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (i = 0; i < post_mesh->n_writers; i++) {

    writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
      double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

      _check_non_transient(writer, &nt_cur, &t_cur);

      if (nt_cur < 0 && writer->tc.last_nt > 0)
        continue;

      if (post_mesh->centers_only == false)
        fvm_writer_export_field(writer->writer,
                                post_mesh->exp_mesh,
                                var_name,
                                FVM_WRITER_PER_ELEMENT,
                                var_dim,
                                _interlace,
                                n_parent_lists,
                                parent_num_shift,
                                datatype,
                                nt_cur,
                                t_cur,
                                (const void * *)var_ptr);

      else {
        cs_lnum_t  parent_num_shift_n[1] = {0};
        fvm_writer_export_field(writer->writer,
                                post_mesh->exp_mesh,
                                var_name,
                                FVM_WRITER_PER_NODE,
                                var_dim,
                                _interlace,
                                0, /* n_parent_lists */
                                parent_num_shift_n,
                                datatype,
                                nt_cur,
                                t_cur,
                                (const void * *)var_ptr);
      }
    }

  }

  /* Free memory (if both interior and boundary faces present) */

  if (var_tmp != nullptr)
    CS_FREE(var_tmp);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output a function evaluation at cells or faces of a
 *        post-processing mesh using associated writers.
 *
 * The associated mesh and function locations must have compatible types.
 *
 * Note that while providing functions for multiple mesh locations
 * (such as interior and boundary faces when a postprocessing mesh contains
 * both) is possible, it is not handled yet, so such cases will be ignored.
 *
 * \param[in]  mesh_id    id of associated mesh
 * \param[in]  writer_id  id of specified associated writer,
 *                        or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  cell_f     pointer to function object at cells
 * \param[in]  i_face_f   pointer to function object at interior faces
 * \param[in]  b_face_f   pointer to function object at boundary faces
 * \param[in]  ts         time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_function(int                    mesh_id,
                       int                    writer_id,
                       const cs_function_t   *cell_f,
                       const cs_function_t   *i_face_f,
                       const cs_function_t   *b_face_f,
                       const cs_time_step_t  *ts)
{
  cs_post_mesh_t  *post_mesh = nullptr;

  const void  *var_ptr[2*9] = {nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr};

  int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
  double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

  /* Initializations */

  const int _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  post_mesh = _cs_post_meshes + _mesh_id;

  cs_mesh_location_type_t loc_type = CS_MESH_LOCATION_NONE;
  cs_lnum_t elt_id_shift = 0;
  const cs_function_t *f = nullptr;

  int ent_dim = -1;

  /* Case of cells */
  /*---------------*/

  if (post_mesh->ent_flag[CS_POST_LOCATION_CELL] == 1) {

    loc_type = CS_MESH_LOCATION_CELLS;
    f = cell_f;
    ent_dim = 3;

  }

  /* Case of faces */
  /*---------------*/

  else if (   post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1
           || post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {

    ent_dim = 2;

    /* FIXME: case where a mesh includes both boundary and interior faces
       is silently ignored for now. This is rarely used, and does not occur
       in the default, "automatic" postprocessing meshes where function
       objects are most relevant.
       Handling this case would require either:
       - Filtering parent face ids to call the appropriate function
         for each subset, then reassembling (merging) interleaved values
         based on parent ids (where parent ids are grouped by element
         type, so ids of boundary and interior faces may be interleaved).
       - Or in the future, perhaps appending boundary and interior faces
         in a more systematic manner, removing the need for manipulation
         of parent ids, but requiring functions which can work on both
         types of faces. */

    if (   post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1
        && post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] == 1) {

      const char *m_name = fvm_nodal_get_name(post_mesh->exp_mesh);
      if (i_face_f == nullptr || b_face_f == nullptr) {
        bft_error(__FILE__, __LINE__, 0,
                  _("%s: For postprocessing mesh \"%s\", both\n"
                    "interior and boundary face function objects must be given\n"
                    "\n"
                    "In addition, this combination is not yet handled, so will\n"
                    "be ignored with a warning."),
                  __func__, m_name);
      }
      else {
        static bool warned = false;
        if (warned == false) {
          bft_printf
            (_("\nWarning: in %s, handling of combined\n"
               "interior and boundary face postprocessing mesh and function\n"
               "objects is not handled yet, so output of function objects\n"
               "\"%s\" and \"%s\" is ignored for mesh \"%s\".\n"
               "\n"
               "This warning applies to all similar potprocessing meshes."),
             __func__, i_face_f->name, b_face_f->name, m_name);
          warned = true;
        }
      }

    }

    if (post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] == 1) {
      loc_type = CS_MESH_LOCATION_INTERIOR_FACES;
      f = i_face_f;
      elt_id_shift = cs_glob_mesh->n_b_faces;
    }
    else {
      loc_type = CS_MESH_LOCATION_BOUNDARY_FACES;
      f = b_face_f;
    }

  }

  if (f == nullptr)
    return;

  if (loc_type != cs_mesh_location_get_type(f->location_id)) {
    const char *m_name = fvm_nodal_get_name(post_mesh->exp_mesh);
    bft_error(__FILE__, __LINE__, 0,
              _("%s: postprocessing mesh \"%s\" and function \"%s\"\n"
                "are not based on compatible mesh locations."),
              __func__, m_name, f->name);
  }

  cs_lnum_t n_elts = fvm_nodal_get_n_entities(post_mesh->exp_mesh, ent_dim);

  cs_lnum_t *elt_ids;
  CS_MALLOC(elt_ids, n_elts, cs_lnum_t);
  fvm_nodal_get_parent_id(post_mesh->exp_mesh, ent_dim, elt_ids);

  if (elt_id_shift > 0) {
    for (cs_lnum_t i = 0; i < n_elts; i++)
      elt_ids[i] -= elt_id_shift;
  }

  unsigned char *_vals = nullptr;
  size_t elt_size = cs_datatype_size[f->datatype] * f->dim;
  CS_MALLOC(_vals, ((size_t)n_elts) * elt_size,  unsigned char);

  cs_function_evaluate(f,
                       ts,
                       loc_type,
                       n_elts,
                       elt_ids,
                       _vals);

  CS_FREE(elt_ids);

  var_ptr[0] = _vals;

  const char *var_name = f->label;
  if (var_name == nullptr)
    var_name = f->name;

  cs_lnum_t   parent_num_shift[1]  = {0};

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (int i = 0; i < post_mesh->n_writers; i++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      _check_non_transient(writer, &nt_cur, &t_cur);

      if (nt_cur < 0 && writer->tc.last_nt > 0)
        continue;

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_ELEMENT,
                              f->dim,
                              CS_INTERLACE,
                              0, /* n_parent_lists, */
                              parent_num_shift,
                              f->datatype,
                              nt_cur,
                              t_cur,
                              (const void * *)var_ptr);

    }

  }

  /* Free memory */

  CS_FREE(_vals);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output a variable defined at vertices of a post-processing mesh using
 *        associated writers.
 *
 * \param[in]  mesh_id     id of associated mesh
 * \param[in]  writer_id   id of specified associated writer,
 *                         or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  var_name    name of variable to output
 * \param[in]  var_dim     1 for scalar, 3 for vector, 6 for symmetric tensor,
 *                         9 for non-symmetric tensor
 * \param[in]  interlace   if a vector, true for interlaced values,
 *                         false otherwise
 * \param[in]  use_parent  true if values are defined on "parent" mesh,
 *                         false if values are defined on post-processing mesh
 * \param[in]  datatype    variable's data type
 * \param[in]  vtx_vals    vertex values
 * \param[in]  ts          time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_vertex_var(int                    mesh_id,
                         int                    writer_id,
                         const char            *var_name,
                         int                    var_dim,
                         bool                   interlace,
                         bool                   use_parent,
                         cs_datatype_t          datatype,
                         const void            *vtx_vals,
                         const cs_time_step_t  *ts)
{
  cs_lnum_t  i;
  int        _mesh_id;

  cs_post_mesh_t  *post_mesh;
  cs_post_writer_t  *writer;
  cs_interlace_t  _interlace;

  size_t       dec_ptr = 0;
  int          n_parent_lists = 0;
  cs_lnum_t    parent_num_shift[1]  = {0};

  const void  *var_ptr[9] = {nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr};

  /* Avoid a SIGSEV in lower-level functions */

  if (var_name == nullptr)
    bft_error(__FILE__, __LINE__, 0, "%s: var_name is not set.\n", __func__);

  /* Initializations */

  _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  post_mesh = _cs_post_meshes + _mesh_id;

  if (interlace)
    _interlace = CS_INTERLACE;
  else
    _interlace = CS_NO_INTERLACE;

  assert(   sizeof(cs_real_t) == sizeof(double)
         || sizeof(cs_real_t) == sizeof(float));

  /* Assign appropriate array to FVM for output */

  if (use_parent)
    n_parent_lists = 1;
  else
    n_parent_lists = 0;

  var_ptr[0] = vtx_vals;
  if (interlace == false) {
    if (use_parent)
      dec_ptr = cs_glob_mesh->n_vertices;
    else
      dec_ptr =   fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0)
                * cs_datatype_size[datatype];
    for (i = 1; i < var_dim; i++)
      var_ptr[i] = ((const char *)vtx_vals) + i*dec_ptr;
  }

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (i = 0; i < post_mesh->n_writers; i++) {

    writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
      double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

      _check_non_transient(writer, &nt_cur, &t_cur);

      if (nt_cur < 0 && writer->tc.last_nt > 0)
        continue;

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_NODE,
                              var_dim,
                              _interlace,
                              n_parent_lists,
                              parent_num_shift,
                              datatype,
                              nt_cur,
                              t_cur,
                              (const void * *)var_ptr);

    }

  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output a function evaluation at cells or faces of a
 *        post-processing mesh using associated writers.
 *
 * The associated mesh and function locations must have compatible types.
 *
 * \param[in]  mesh_id    id of associated mesh
 * \param[in]  writer_id  id of specified associated writer,
 *                        or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  f          pointer to function object at vertices
 * \param[in]  ts         time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_vertex_function(int                    mesh_id,
                              int                    writer_id,
                              const cs_function_t   *f,
                              const cs_time_step_t  *ts)
{
  cs_lnum_t    parent_num_shift[1]  = {0};

  const void  *var_ptr[9] = {nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr};

  /* Initializations */

  const int _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;

  if (   CS_MESH_LOCATION_VERTICES
      != cs_mesh_location_get_type(f->location_id)) {
    const char *m_name = fvm_nodal_get_name(post_mesh->exp_mesh);
    bft_error(__FILE__, __LINE__, 0,
              _("%s: postprocessing mesh \"%s\" and function \"%s\"\n"
                "are not based on compatible mesh locations."),
              __func__, m_name, f->name);
  }

  cs_lnum_t n_elts = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0);

  cs_lnum_t *elt_ids;
  CS_MALLOC(elt_ids, n_elts, cs_lnum_t);
  fvm_nodal_get_parent_id(post_mesh->exp_mesh, 0, elt_ids);

  unsigned char *_vals = nullptr;
  size_t elt_size = cs_datatype_size[f->datatype] * f->dim;
  CS_MALLOC(_vals, ((size_t)n_elts) * elt_size,  unsigned char);

  cs_function_evaluate(f,
                       ts,
                       CS_MESH_LOCATION_VERTICES,
                       n_elts,
                       elt_ids,
                       _vals);

  CS_FREE(elt_ids);

  var_ptr[0] = _vals;

  const char *var_name = f->label;
  if (var_name == nullptr)
    var_name = f->name;

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (int i = 0; i < post_mesh->n_writers; i++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
      double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

      _check_non_transient(writer, &nt_cur, &t_cur);

      if (nt_cur < 0 && writer->tc.last_nt > 0)
        continue;

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_NODE,
                              f->dim,
                              CS_INTERLACE,
                              0, /* n_parent_lists, */
                              parent_num_shift,
                              f->datatype,
                              nt_cur,
                              t_cur,
                              (const void * *)var_ptr);

    }

  }

  /* Free memory */

  CS_FREE(_vals);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output an existing lagrangian particle attribute at particle
 *        positions or trajectory endpoints of a particle mesh using
 *        associated writers.
 *
 * \param[in]  mesh_id       id of associated mesh
 * \param[in]  writer_id     id of specified associated writer,
 *                           or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  attr_id       associated particle attribute id
 * \param[in]  var_name      name of variable to output
 * \param[in]  component_id  if -1 : extract the whole attribute
 *                           if >0 : id of the component to extract
 * \param[in]  ts            time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_particle_values(int                    mesh_id,
                              int                    writer_id,
                              int                    attr_id,
                              const char            *var_name,
                              int                    component_id,
                              const cs_time_step_t  *ts)
{
  int  _mesh_id, i, _length;
  int _stride_export_field = 1;
  cs_post_mesh_t  *post_mesh;
  cs_post_writer_t  *writer;

  cs_lagr_attribute_t  attr = (cs_lagr_attribute_t)attr_id;

  cs_lnum_t    n_particles = 0, n_pts = 0;
  cs_lnum_t    parent_num_shift[1]  = {0};
  cs_lnum_t   *particle_list = nullptr;

  size_t  extents, size;
  ptrdiff_t  displ;
  cs_datatype_t datatype;
  int  stride;
  unsigned char *vals = nullptr;

  int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
  double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

  const void  *var_ptr[1] = {nullptr};

  /* Initializations */

  _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  post_mesh = _cs_post_meshes + _mesh_id;

  if (post_mesh->ent_flag[3] == 0 || post_mesh->exp_mesh == nullptr)
    return;

  n_particles = cs_lagr_get_n_particles();

  const cs_lagr_particle_set_t  *p_set = cs_lagr_get_particle_set();

  assert(p_set != nullptr);

  /* Get attribute values info, returning if not present */

  cs_lagr_get_attr_info(p_set, 0, attr,
                        &extents, &size, &displ, &datatype, &stride);

  if (stride == 0)
    return;
  else {
    if (component_id == -1) {
      _length = size;
      _stride_export_field = stride;
    }
    else {
      _length = size/stride;
      _stride_export_field = 1;
     }
  }

  assert(ts->nt_cur > -1);

  /* Allocate work arrays */

  n_pts = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0);

  CS_MALLOC(vals, n_pts*_length, unsigned char);

  var_ptr[0] = vals;

  if (n_pts != n_particles) {
    int parent_dim = (post_mesh->ent_flag[3] == 2) ? 1 : 0;
    CS_MALLOC(particle_list, n_particles, cs_lnum_t);
    fvm_nodal_get_parent_num(post_mesh->exp_mesh, parent_dim, particle_list);
  }

  /* Particle values */

  if (post_mesh->ent_flag[3] == 1)
    cs_lagr_get_particle_values(p_set,
                                attr,
                                datatype,
                                stride,
                                component_id,
                                n_pts,
                                particle_list,
                                vals);

  else if (post_mesh->ent_flag[3] == 2) {
    nt_cur = -1; t_cur = 0.;
    cs_lagr_get_trajectory_values(p_set,
                                  attr,
                                  datatype,
                                  stride,
                                  component_id,
                                  n_pts/2,
                                  particle_list,
                                  vals);
  }

  CS_FREE(particle_list);

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (i = 0; i < post_mesh->n_writers; i++) {

    writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_NODE,
                              _stride_export_field,
                              CS_INTERLACE,
                              0, /* n_parent_lists, */
                              parent_num_shift,
                              datatype,
                              nt_cur,
                              t_cur,
                              (const void * *)var_ptr);

    }

  }

  CS_FREE(vals);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output a variable defined at cells or faces of a post-processing mesh
 *        using associated writers.
 *
 * \param[in]  mesh_id              id of associated mesh
 * \param[in]  writer_id            id of specified associated writer,
 *                                  or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  var_name             name of variable to output
 * \param[in]  var_dim              1 for scalar, 3 for vector, 6 for symmetric
 *                                  tensor, 9 for non-symmetric tensor
 * \param[in]  datatype             variable's data type
 * \param[in]  parent_location_id   asociated values mesh location, or 0
 *                                  if values are passed directly
 * \param[in]  interpolate_func     pointer to interpolation function,
 *                                  or null for default
 * \param[in]  interpolate_input    pointer to optional interpolation input
 *                                  data, or null for default
 * \param[in]  vals                 variable's values
 * \param[in]  ts                   time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_probe_values(int                              mesh_id,
                           int                              writer_id,
                           const char                      *var_name,
                           int                              var_dim,
                           cs_datatype_t                    datatype,
                           int                              parent_location_id,
                           cs_interpolate_from_location_t  *interpolate_func,
                           void                            *interpolate_input,
                           const void                      *vals,
                           const cs_time_step_t            *ts)
{
  int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
  double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

  /* Initializations */

  int _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;
  cs_probe_set_t *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  const void *var_ptr[1] = {vals};
  unsigned char *_vals = nullptr;

  /* Extract or interpolate values */

  if (parent_location_id > 0) {

    const cs_lnum_t n_points = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0);
    const cs_lnum_t *elt_ids = cs_probe_set_get_elt_ids(pset,
                                                        parent_location_id);

    cs_interpolate_from_location_t *_interpolate_func = interpolate_func;

    cs_coord_t *point_coords = nullptr;

    if (_interpolate_func == nullptr)
      _interpolate_func = cs_interpolate_from_location_p0;

    CS_MALLOC(_vals,
              n_points*cs_datatype_size[datatype]*var_dim,
              unsigned char);

    if (_interpolate_func != cs_interpolate_from_location_p0) {
      CS_MALLOC(point_coords, n_points*3, cs_coord_t);
      fvm_nodal_get_vertex_coords(post_mesh->exp_mesh,
                                  CS_INTERLACE,
                                  point_coords);
    }

    _interpolate_func(interpolate_input,
                      datatype,
                      var_dim,
                      n_points,
                      elt_ids,
                      (const cs_real_3_t *)point_coords,
                      vals,
                      _vals);
    var_ptr[0] = _vals;

    CS_FREE(point_coords);
  }

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (int i = 0; i < post_mesh->n_writers; i++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      cs_lnum_t  parent_num_shift[1] = {0};

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_NODE,
                              var_dim,
                              CS_INTERLACE,
                              0, /* n_parent_lists */
                              parent_num_shift,
                              datatype,
                              nt_cur,
                              t_cur,
                              (const void **)var_ptr);

    }

  }

  CS_FREE(_vals);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Output function-evaluated values at cells or faces of a
 *        post-processing probe set using associated writers.
 *
 * - For real-valued data with interpolation, the function is evaluated on the
 *   whole parent domain so as to be able to compute gradients, then
 *    interpolate on the probe set.
 *    * For vertex-based values, a lighter, cell-based interpolation would
 *      be feasible, but is not available yet.
 *    * For cell or face-based values, likewise, only the immediate neighborhood
 *      of elements containing probes are needed, but such filtering would
 *      require building the matching element list and adding an indirection
 *      to the gradients computation.
 *
 * - In the specific case where the function evaluation uses an analytic
 *   function, to which the exact coordinates are provides, such interpolation
 *   is not deemed necessary, as the anaytic function may handle this more
 *   efficiently.
 *
 * - For non-real-based values, or real-based values other than cs_real_t,
 *   no interpolation is performed
 *
 * \param[in]  mesh_id              id of associated mesh
 * \param[in]  writer_id            id of specified associated writer,
 *                                  or \ref CS_POST_WRITER_ALL_ASSOCIATED for all
 * \param[in]  f                    pointer to associated function object
 * \param[in]  parent_location_id   associated values at mesh location, or 0
 *                                  if values are passed directly
 * \param[in]  interpolate_func     pointer to interpolation function,
 *                                  or null for default
 * \param[in]  interpolate_input    pointer to optional interpolation input
 *                                  data, or null for default
 * \param[in]  ts                   time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_probe_function(int                              mesh_id,
                             int                              writer_id,
                             const cs_function_t             *f,
                             int                              parent_location_id,
                             cs_interpolate_from_location_t  *interpolate_func,
                             void                            *interpolate_input,
                             const cs_time_step_t            *ts)
{
  int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
  double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

  /* Initializations */

  int _mesh_id = _cs_post_mesh_id_try(mesh_id);

  if (_mesh_id < 0)
    return;

  cs_post_mesh_t *post_mesh = _cs_post_meshes + _mesh_id;
  cs_probe_set_t *pset = (cs_probe_set_t *)post_mesh->sel_input[4];

  cs_coord_t *point_coords = nullptr;

  const void *var_ptr[1] = {nullptr};
  unsigned char *_vals = nullptr;

  cs_lnum_t _dim = f->dim;

  const char *var_name = f->label;
  if (var_name == nullptr)
    var_name = f->name;

  /* Extract or interpolate values (see rules in function description above) */

  if (parent_location_id > 0) {

    const cs_lnum_t n_points = fvm_nodal_get_n_entities(post_mesh->exp_mesh, 0);
    const cs_lnum_t *elt_ids = cs_probe_set_get_elt_ids(pset,
                                                        parent_location_id);

    cs_interpolate_from_location_t *_interpolate_func = interpolate_func;

    if (   _interpolate_func == cs_interpolate_from_location_p0
        || f->analytic_func != nullptr
        || f->datatype != CS_REAL_TYPE)
      _interpolate_func = nullptr;

    CS_MALLOC(_vals,
              (size_t)n_points*cs_datatype_size[f->datatype]*((size_t)f->dim),
              unsigned char);

    if (   _interpolate_func != cs_interpolate_from_location_p0
        || f->analytic_func != nullptr) {
      CS_MALLOC(point_coords, n_points*3, cs_coord_t);
      fvm_nodal_get_vertex_coords(post_mesh->exp_mesh,
                                  CS_INTERLACE,
                                  point_coords);
    }

    if (_interpolate_func != nullptr) {
      const cs_lnum_t *n_p_elts
        = cs_mesh_location_get_n_elts(parent_location_id);
      cs_real_t *_p_vals;
      CS_MALLOC(_p_vals, n_p_elts[2]*_dim, cs_real_t);

      cs_function_evaluate(f,
                           ts,
                           parent_location_id,
                           n_p_elts[0],
                           nullptr,
                           _p_vals);

      _interpolate_func(interpolate_input,
                        f->datatype,
                        f->dim,
                        n_points,
                        elt_ids,
                        (const cs_real_3_t *)point_coords,
                        _p_vals,
                        _vals);

      CS_FREE(_p_vals);
    }

    else if (f->analytic_func != nullptr)
      f->analytic_func(ts->t_cur,
                       n_points,
                       elt_ids,
                       (cs_real_t *)point_coords,
                       true,
                       f->func_input,
                       (cs_real_t *)_vals);
    else
      cs_function_evaluate(f,
                           ts,
                           parent_location_id,
                           n_points,
                           elt_ids,
                           _vals);

    var_ptr[0] = _vals;

    CS_FREE(point_coords);
  }

  /* Effective output: loop on writers */
  /*-----------------------------------*/

  for (int i = 0; i < post_mesh->n_writers; i++) {

    cs_post_writer_t *writer = _cs_post_writers + post_mesh->writer_id[i];

    if (writer->id != writer_id && writer_id != CS_POST_WRITER_ALL_ASSOCIATED)
      continue;

    if (writer->active == 1) {

      int nt_cur_w = nt_cur;
      double t_cur_w = t_cur;

      _check_non_transient(writer, &nt_cur, &t_cur);

      if (nt_cur < 0 && writer->tc.last_nt > 0)
        continue;

      cs_lnum_t  parent_num_shift[1] = {0};

      fvm_writer_export_field(writer->writer,
                              post_mesh->exp_mesh,
                              var_name,
                              FVM_WRITER_PER_NODE,
                              f->dim,
                              CS_INTERLACE,
                              0, /* n_parent_lists */
                              parent_num_shift,
                              f->datatype,
                              nt_cur_w,
                              t_cur_w,
                              (const void **)var_ptr);

    }

  }

  CS_FREE(_vals);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update references to parent mesh of post-processing meshes in case of
 * computational mesh cell renumbering.
 *
 * This function may be called only once, after possible renumbering of cells,
 * to update existing post-processing meshes. Post-processing meshes defined
 * after renumbering will automatically be based upon the new numbering,
 * so this function will not need to be called again.
 *
 * \param[in]  init_cell_num  initial cell numbering (new -> old)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_renum_cells(const cs_lnum_t  init_cell_num[])
{
  int        i;
  cs_lnum_t  icel;
  cs_lnum_t  n_elts;

  cs_lnum_t  *renum_ent_parent = nullptr;

  bool  need_doing = false;

  cs_post_mesh_t   *post_mesh;
  const cs_mesh_t  *mesh = cs_glob_mesh;

  if (init_cell_num == nullptr)
    return;

  /* Loop on meshes */

  for (i = 0; i < _cs_post_n_meshes; i++) {

    post_mesh = _cs_post_meshes + i;

    if (post_mesh->ent_flag[CS_POST_LOCATION_CELL] > 0)
      need_doing = true;
  }

  if (need_doing == true) {

    /* Prepare renumbering */

    n_elts = mesh->n_cells;

    CS_MALLOC(renum_ent_parent, n_elts, cs_lnum_t);

    for (icel = 0; icel < mesh->n_cells; icel++)
      renum_ent_parent[init_cell_num[icel]] = icel;

    /* Effective modification */

    for (i = 0; i < _cs_post_n_meshes; i++) {

      post_mesh = _cs_post_meshes + i;

      if (   post_mesh->_exp_mesh != nullptr
          && post_mesh->ent_flag[CS_POST_LOCATION_CELL] > 0) {

        fvm_nodal_change_parent_id(post_mesh->_exp_mesh,
                                   renum_ent_parent,
                                   3);

      }

    }

    CS_FREE(renum_ent_parent);

  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update references to parent mesh of post-processing meshes in case of
 * computational mesh interior and/or boundary faces renumbering.
 *
 * This function may be called only once, after possible renumbering of faces,
 * to update existing post-processing meshes. Post-processing meshes defined
 * after renumbering will automatically be based upon the new numbering,
 * so this function will not need to be called again.
 *
 * \param[in]  init_i_face_num  initial interior numbering (new -> old)
 * \param[in]  init_b_face_num  initial boundary numbering (new -> old)
 */
/*----------------------------------------------------------------------------*/

void
cs_post_renum_faces(const cs_lnum_t  init_i_face_num[],
                    const cs_lnum_t  init_b_face_num[])
{
  int       i;
  cs_lnum_t  ifac;
  cs_lnum_t  n_elts;

  cs_lnum_t  *renum_ent_parent = nullptr;

  bool  need_doing = false;

  cs_post_mesh_t   *post_mesh;
  const cs_mesh_t  *mesh = cs_glob_mesh;

  /* Loop on meshes */

  for (i = 0; i < _cs_post_n_meshes; i++) {

    post_mesh = _cs_post_meshes + i;

    if (   post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] > 0
        || post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] > 0) {
      need_doing = true;
    }

  }

  if (need_doing == true) {

    /* Prepare renumbering */

    n_elts = mesh->n_i_faces + mesh->n_b_faces;

    CS_MALLOC(renum_ent_parent, n_elts, cs_lnum_t);

    if (init_b_face_num == nullptr) {
      for (ifac = 0; ifac < mesh->n_b_faces; ifac++)
        renum_ent_parent[ifac] = ifac;
    }
    else {
      for (ifac = 0; ifac < mesh->n_b_faces; ifac++)
        renum_ent_parent[init_b_face_num[ifac]] = ifac;
    }

    if (init_i_face_num == nullptr) {
      for (ifac = 0, i = mesh->n_b_faces;
           ifac < mesh->n_i_faces;
           ifac++, i++)
        renum_ent_parent[mesh->n_b_faces + ifac]
          = mesh->n_b_faces + ifac;
    }
    else {
      for (ifac = 0, i = mesh->n_b_faces;
           ifac < mesh->n_i_faces;
           ifac++, i++)
        renum_ent_parent[mesh->n_b_faces + init_i_face_num[ifac]]
          = mesh->n_b_faces + ifac;
    }

    /* Effective modification */

    for (i = 0; i < _cs_post_n_meshes; i++) {

      post_mesh = _cs_post_meshes + i;

      if (   post_mesh->_exp_mesh != nullptr
          && (   post_mesh->ent_flag[CS_POST_LOCATION_I_FACE] > 0
              || post_mesh->ent_flag[CS_POST_LOCATION_B_FACE] > 0)) {

        fvm_nodal_change_parent_id(post_mesh->_exp_mesh,
                                   renum_ent_parent,
                                   2);

      }

    }

    CS_FREE(renum_ent_parent);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Configure the post-processing output so that mesh connectivity
 * may be automatically updated.
 *
 * This is done for meshes defined using selection criteria or functions.
 * The behavior of Lagrangian meshes is unchanged.
 *
 * To be effective, this function should be called before defining
 * postprocessing meshes.
 */
/*----------------------------------------------------------------------------*/

void
cs_post_set_changing_connectivity(void)
{
  _cs_post_mod_flag_min = FVM_WRITER_TRANSIENT_CONNECT;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize post-processing writers
 */
/*----------------------------------------------------------------------------*/

void
cs_post_init_writers(void)
{
  /* Ensure default is defined */

  if (!cs_post_writer_exists(CS_POST_WRITER_DEFAULT))
    cs_post_define_writer(CS_POST_WRITER_DEFAULT,   /* writer_id */
                          "results",                /* writer name */
                          _cs_post_dirname,
                          "EnSight Gold",           /* format name */
                          "separate_meshes",        /* format options */
                          FVM_WRITER_FIXED_MESH,
                          false,                    /* output_at_start */
                          true,                     /* output at end */
                          -1,                       /* time step interval */
                          -1.0);                    /* time value interval */

  /* Additional writers for Lagrangian output */

  if (_lagrangian_needed(nullptr)) {

    /* Particles */

    if (!cs_post_writer_exists(CS_POST_WRITER_PARTICLES))
      cs_post_define_writer(CS_POST_WRITER_PARTICLES,  /* writer_id */
                            "particles",            /* writer name */
                            _cs_post_dirname,
                            "EnSight Gold",         /* format name */
                            "",                     /* format options */
                            FVM_WRITER_TRANSIENT_CONNECT,
                            false,                  /* output_at_start */
                            true,                   /* output at end */
                            -1,                     /* time step interval */
                            -1.0);                  /* time value interval */

    if (!cs_post_writer_exists(CS_POST_WRITER_TRAJECTORIES))
      cs_post_define_writer(CS_POST_WRITER_TRAJECTORIES, /* writer_id */
                            "trajectories",         /* writer name */
                            _cs_post_dirname,
                            "EnSight Gold",         /* format name */
                            "",                     /* format options */
                            FVM_WRITER_FIXED_MESH,
                            false,                  /* output_at_start */
                            true,                   /* output at end */
                            1,                      /* time step interval */
                            -1.0);                  /* time value interval */

  }

  /* Additional writers for probe monitoring, profiles */

  if (!cs_post_writer_exists(CS_POST_WRITER_PROBES))
    cs_post_define_writer(CS_POST_WRITER_PROBES,    /* writer_id */
                          "",                       /* writer name */
                          "monitoring",
                          "time_plot",              /* format name */
                          "",                       /* format options */
                          FVM_WRITER_FIXED_MESH,
                          false,                    /* output_at_start */
                          false,                    /* output at end */
                          1,                        /* time step interval */
                          -1.0);                    /* time value interval */

  if (!cs_post_writer_exists(CS_POST_WRITER_PROFILES))
    cs_post_define_writer(CS_POST_WRITER_PROFILES,  /* writer_id */
                          "",                       /* writer name */
                          "profiles",
                          "plot",                   /* format name */
                          "",                       /* format options */
                          FVM_WRITER_FIXED_MESH,
                          false,                    /* output_at_start */
                          true,                     /* output at end */
                          -1,                       /* time step interval */
                          -1.0);                    /* time value interval */

  /* Additional writers for histograms */

  if (!cs_post_writer_exists(CS_POST_WRITER_HISTOGRAMS))
    cs_post_define_writer(CS_POST_WRITER_HISTOGRAMS,  /* writer_id */
                          "histograms",               /* writer name */
                          "histograms",
                          "histogram",                /* format name */
                          "txt",                      /* format options */
                          FVM_WRITER_FIXED_MESH,
                          false,                      /* output_at_start */
                          true,                       /* output at end */
                          -1,                         /* time step interval */
                          -1.0);                      /* time value interval */

  /* Print info on writers */

  _writer_info();
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize main post-processing meshes
 *
 * The check_flag variable is a mask, used for additionnal post-processing:
 *
 *  - If (check_flag & 1), volume submeshes are output by groups if more
 *    than one group is present and the default writer uses the EnSight format.
 *
 *  - If (check_flag & 2), boundary submeshes are output by groups if more
 *    than one group is present and the default writer uses the EnSight format.
 *
 * It is recommended that post-processing meshes be defined before calling
 * this function, though specific "automatic" meshes (for example those
 * related to couplings) may be defined between this call and a time loop.
 *
 * \param[in]  check_mask  mask used for additional output
 */
/*----------------------------------------------------------------------------*/

void
cs_post_init_meshes(int check_mask)
{
  { /* Definition of default post-processing meshes if this has not been
       done yet */

    const int  writer_ids[] = {CS_POST_WRITER_DEFAULT};

    if (!cs_post_mesh_exists(CS_POST_MESH_VOLUME))
      cs_post_define_volume_mesh(CS_POST_MESH_VOLUME,
                                 _("Fluid domain"),
                                 "all[]",
                                 true,
                                 true,
                                 1,
                                 writer_ids);

    if (!cs_post_mesh_exists(CS_POST_MESH_BOUNDARY))
      cs_post_define_surface_mesh(CS_POST_MESH_BOUNDARY,
                                  _("Boundary"),
                                  nullptr,
                                  "all[]",
                                  true,
                                  true,
                                  1,
                                  writer_ids);

  }

  /* Additional writers for Lagrangian output */

  if (_lagrangian_needed(nullptr)) {
    if (!cs_post_mesh_exists(CS_POST_MESH_PARTICLES)) {
      const int writer_ids[] = {CS_POST_WRITER_PARTICLES};
      cs_post_define_particles_mesh(CS_POST_MESH_PARTICLES,
                                    _("Particles"),
                                    "all[]",
                                    1.0,    /* density */
                                    false,  /* trajectory */
                                    true,   /* auto_variables */
                                    1,
                                    writer_ids);
    }
  }

  /* Define probe meshes if needed */

  int n_probe_sets = cs_probe_get_n_sets();

  for (int pset_id = 0; pset_id < n_probe_sets; pset_id++) {

    bool  time_varying, is_profile, on_boundary, auto_variables;

    int  n_writers = 0;
    int  *writer_ids = nullptr;
    cs_probe_set_t  *pset = cs_probe_set_get_by_id(pset_id);
    int  post_mesh_id = cs_post_get_free_mesh_id();

    cs_probe_set_get_post_info(pset,
                               &time_varying,
                               &on_boundary,
                               &is_profile,
                               &auto_variables,
                               nullptr,
                               nullptr,
                               &n_writers,
                               &writer_ids);

    if (is_profile) { /* User has to define an associated writer */

      _cs_post_define_probe_mesh(post_mesh_id,
                                 pset,
                                 time_varying,
                                 is_profile,
                                 on_boundary,
                                 auto_variables,
                                 n_writers,
                                 writer_ids);

    }
    else { /* Monitoring points */

      if (pset_id == 0) // Use reserved mesh id rather than next free one.
        post_mesh_id = CS_POST_MESH_PROBES;

      /* Handle default writer assignment */

      if (n_writers < 0) {

        const int  default_writer_ids[] = {CS_POST_WRITER_PROBES};
        cs_probe_set_associate_writers(pset, 1, default_writer_ids);

        cs_probe_set_get_post_info(pset, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr,
                                   &n_writers, &writer_ids);

      }

      /* Now define associated mesh if active */

      if (n_writers > 0)
        _cs_post_define_probe_mesh(post_mesh_id,
                                   pset,
                                   time_varying,
                                   is_profile,
                                   on_boundary,
                                   auto_variables,
                                   n_writers,
                                   writer_ids);

    }

  } /* Loop on sets of probes */

  /* Remove meshes which are associated with no writer */

  _clear_unused_meshes();

  /* Add group parts if necessary (EnSight format) */

  if (check_mask & 1) {
    const char *fmt_name = fvm_writer_format_name(_cs_post_default_format_id);
    if (!strcmp(fmt_name, "EnSight Gold")) {
      for (int id = 0; id < _cs_post_n_meshes; id++) {
        if ((_cs_post_meshes + id)->id == CS_POST_MESH_VOLUME)
          _vol_submeshes_by_group(cs_glob_mesh,
                                  fmt_name,
                                  _cs_post_default_format_options);
        if ((_cs_post_meshes + id)->id == CS_POST_MESH_BOUNDARY)
          _boundary_submeshes_by_group(cs_glob_mesh,
                                       fmt_name,
                                       _cs_post_default_format_options);
      }
    }
  }

#if 0
  /* Compute connectivity if not already done for delayed definitions */

  for (i = 0; i < _cs_post_n_meshes; i++) {
    cs_post_mesh_t  *post_mesh = _cs_post_meshes + i;
    if (post_mesh->exp_mesh == nullptr)
      _define_mesh(post_mesh, nullptr);
  }
#endif

  /* Initial output */

  cs_post_write_meshes(nullptr);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Check if post-processing is activated and then update post-processing
 *        of meshes if there is a need to update time-dependent meshes
 *
 * \param[in]  ts  time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_time_step_begin(const cs_time_step_t  *ts)
{
  assert(ts != nullptr); /* Sanity check */

  /* Activation or not of each writer according to the time step */

  cs_post_activate_by_time_step(ts);

  /* User-defined activation of writers for a fine-grained control */

  cs_user_postprocess_activate(ts->nt_max,
                               ts->nt_cur,
                               ts->t_cur);

  /* Possible modification of post-processing meshes */
  /*-------------------------------------------------*/

  _update_meshes(ts);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Loop on post-processing meshes to output variables.
 *
 * This handles all default fields output, as well as all
 * registered output functions and outputs defined in
 * \ref cs_user_postprocess_values
 *
 * \param[in]  ts  time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_time_step_output(const cs_time_step_t  *ts)
{
  int  j;

  /* Loop on writers to check if something must be done */
  /*----------------------------------------------------*/

  for (j = 0; j < _cs_post_n_writers; j++) {
    cs_post_writer_t  *writer = _cs_post_writers + j;
    if (writer->active == 1)
      break;
  }
  if (j == _cs_post_n_writers)
    return;

  int t_top_id = cs_timer_stats_switch(_post_out_stat_id);

  /* Update time control of this writer
     (before any actual output, to allow filtering such
     as avoiding multiple writes of time-independent values) */

  {
    int nt_cur = (ts != nullptr) ? ts->nt_cur : -1;
    double t_cur = (ts != nullptr) ? ts->t_cur : 0.;

    for (j = 0; j < _cs_post_n_writers; j++) {
      cs_post_writer_t  *writer = _cs_post_writers + j;
      if (writer->active == 1 && nt_cur > writer->tc.last_nt) {
        writer->tc.last_nt = nt_cur;
        writer->tc.last_t = t_cur;
      }
    }
  }

  /* Prepare flag to avoid multiple field synchronizations */

  const int n_fields = cs_field_n_fields();
  CS_MALLOC(_field_sync, n_fields, char);
  for (int f_id = 0; f_id < n_fields; f_id++)
    _field_sync[f_id] = 0;

  /* Output of variables by registered function instances */
  /*------------------------------------------------------*/

  for (int i = 0; i < _cs_post_n_output_tp; i++)
    _cs_post_f_output_tp[i](_cs_post_i_output_tp[i], ts);

  /* Output of variables associated with post-processing meshes */
  /*------------------------------------------------------------*/

  /* n_elts_max already initialized before and during the
     eventual modification of post-processing mesh definitions,
     and parent_ids allocated if n_elts_max > 0 */

  cs_lnum_t  *parent_ids = nullptr;
  cs_lnum_t  n_elts_max = 0;

  /* Main loop on post-processing meshes */

  for (int i = 0; i < _cs_post_n_meshes; i++) {

    cs_post_mesh_t  *post_mesh = _cs_post_meshes + i;

    bool active = false;

    for (j = 0; j < post_mesh->n_writers; j++) {
      cs_post_writer_t  *writer = _cs_post_writers + post_mesh->writer_id[j];
      if (writer->active == 1)
        active = true;
    }

    /* If the mesh is active at this time step */
    /*-----------------------------------------*/

    if (active == true) {

      const fvm_nodal_t  *exp_mesh = post_mesh->exp_mesh;

      if (exp_mesh == nullptr)
        continue;

      int  dim_ent = fvm_nodal_get_max_entity_dim(exp_mesh);
      cs_lnum_t  n_elts = fvm_nodal_get_n_entities(exp_mesh, dim_ent);

      if (n_elts > n_elts_max) {
        n_elts_max = n_elts;
        CS_REALLOC(parent_ids, n_elts_max, cs_lnum_t);
      }

      /* Get corresponding element ids */

      fvm_nodal_get_parent_num(exp_mesh, dim_ent, parent_ids);

      for (cs_lnum_t k = 0; k < n_elts; k++)
        parent_ids[k] -= 1;

      /* We can output variables for this time step */
      /*--------------------------------------------*/

      cs_lnum_t  n_cells = 0, n_i_faces = 0, n_b_faces = 0;
      cs_lnum_t  *cell_ids = nullptr;
      cs_lnum_t  *i_face_ids = nullptr, *b_face_ids = nullptr;

      /* Here list sizes are adjusted, and we point to the array filled
         by fvm_nodal_get_parent_num() if possible. */

      if (dim_ent == 3) {
        n_cells = n_elts;
        cell_ids = parent_ids;
      }

      /* The numbers of "parent" interior faces known by FVM
         are shifted by the total number of boundary faces */

      else if (dim_ent == 2 && n_elts > 0) {

        cs_lnum_t  b_f_num_shift = cs_glob_mesh->n_b_faces;

        for (cs_lnum_t ind_fac = 0; ind_fac < n_elts; ind_fac++) {
          if (parent_ids[ind_fac] >= b_f_num_shift)
            n_i_faces++;
          else
            n_b_faces++;
        }

        /* boundary faces only: parent FVM face numbers unchanged */
        if (n_i_faces == 0) {
          b_face_ids = parent_ids;
        }

        /* interior faces only: parents FVM face numbers shifted */
        else if (n_b_faces == 0) {
          for (cs_lnum_t ind_fac = 0; ind_fac < n_elts; ind_fac++)
            parent_ids[ind_fac] -= b_f_num_shift;
          i_face_ids = parent_ids;
        }

        /* interior and boundary faces: numbers must be separated */

        else {

          CS_MALLOC(i_face_ids, n_i_faces, cs_lnum_t);
          CS_MALLOC(b_face_ids, n_b_faces, cs_lnum_t);

          n_i_faces = 0, n_b_faces = 0;

          for (cs_lnum_t ind_fac = 0; ind_fac < n_elts; ind_fac++) {
            if (parent_ids[ind_fac] >= b_f_num_shift)
              i_face_ids[n_i_faces++] = parent_ids[ind_fac] - b_f_num_shift;
            else
              b_face_ids[n_b_faces++] = parent_ids[ind_fac];
          }

        }

        /* In all cases, update the number of interior and boundary faces
           (useful in case of splitting of FVM mesh elements) for functions
           called by this one */

        post_mesh->n_i_faces = n_i_faces;
        post_mesh->n_b_faces = n_b_faces;

      }

      /* Output of zone information if necessary */

      _cs_post_write_transient_zone_info(post_mesh, ts);

      /* Standard post-processing */

      if (post_mesh->sel_input[4] != nullptr)
        _cs_post_output_profile_coords(post_mesh, ts);

      if (post_mesh->cat_id < 0)
        _cs_post_output_fields(post_mesh, ts);

      if (post_mesh->n_a_fields > 0)
        _cs_post_output_attached_fields(post_mesh, ts);

      if (post_mesh->cat_id < 0)
        _output_function_data(post_mesh, ts);

      /* Output of variables by registered function instances */

      for (j = 0; j < _cs_post_n_output_mtp; j++)
        _cs_post_f_output_mtp[j](_cs_post_i_output_mtp[j],
                                 post_mesh->id,
                                 post_mesh->cat_id,
                                 post_mesh->ent_flag,
                                 n_cells,
                                 n_i_faces,
                                 n_b_faces,
                                 cell_ids,
                                 i_face_ids,
                                 b_face_ids,
                                 ts);

      /* User-defined output */

      cs_lnum_t  n_vertices = cs_post_mesh_get_n_vertices(post_mesh->id);

      if (post_mesh->sel_input[4] == nullptr) {

        cs_lnum_t *vertex_ids;
        CS_MALLOC(vertex_ids, n_vertices, cs_lnum_t);
        cs_post_mesh_get_vertex_ids(post_mesh->id, vertex_ids);

        cs_user_postprocess_values(post_mesh->name,
                                   post_mesh->id,
                                   post_mesh->cat_id,
                                   nullptr,
                                   n_cells,
                                   n_i_faces,
                                   n_b_faces,
                                   n_vertices,
                                   cell_ids,
                                   i_face_ids,
                                   b_face_ids,
                                   vertex_ids,
                                   ts);

        CS_FREE(vertex_ids);

        /* In case of mixed interior and boundary faces, free
           additional arrays */

        if (i_face_ids != nullptr && b_face_ids != nullptr) {
          CS_FREE(i_face_ids);
          CS_FREE(b_face_ids);
        }

      }

      else { /* if (post_mesh->sel_input[4] != nullptr) */

        bool on_boundary = false;
        const cs_lnum_t *_cell_ids = nullptr, *_b_face_ids = nullptr;
        const cs_lnum_t *vertex_ids = nullptr;
        cs_probe_set_t  *pset = (cs_probe_set_t *)post_mesh->sel_input[4];
        const char *mesh_name = cs_probe_set_get_name(pset);

        cs_probe_set_get_post_info(pset,
                                   nullptr,
                                   &on_boundary,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr);

        if (on_boundary) {
          n_b_faces = n_vertices; /* n_probes */
          _b_face_ids = cs_probe_set_get_elt_ids(pset,
                                                 CS_MESH_LOCATION_BOUNDARY_FACES);
        }
        else {
          n_cells = n_vertices; /* n_probes */
          _cell_ids = cs_probe_set_get_elt_ids(pset,
                                               CS_MESH_LOCATION_CELLS);
        }
        vertex_ids = cs_probe_set_get_elt_ids(pset,
                                              CS_MESH_LOCATION_VERTICES);

        cs_user_postprocess_values(mesh_name,
                                   post_mesh->id,
                                   post_mesh->cat_id,
                                   pset,
                                   n_cells,
                                   0,
                                   n_b_faces,
                                   n_vertices,
                                   _cell_ids,
                                   nullptr,
                                   _b_face_ids,
                                   vertex_ids,
                                   ts);

      }

    }

  }

  /* Free memory */

  CS_FREE(_field_sync);
  CS_FREE(parent_ids);

  cs_timer_stats_switch(t_top_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Flush writers and free time-varying and Lagragian mesh if needed
 *        of meshes if there is a time-dependent mesh
 */
/*----------------------------------------------------------------------------*/

void
cs_post_time_step_end(void)
{
  int t_top_id = cs_timer_stats_switch(_post_out_stat_id);

  /* Flush writers if necessary */

  for (int i = 0; i < _cs_post_n_writers; i++) {
    cs_post_writer_t  *writer = _cs_post_writers + i;
    if (writer->active == 1) {
      if (writer->writer != nullptr)
        fvm_writer_flush(writer->writer);
    }
  }

  /* Free time-varying and Lagrangian meshes unless they
     are mapped to an existing mesh */

  for (int i = 0; i < _cs_post_n_meshes; i++) {
    cs_post_mesh_t  *post_mesh = _cs_post_meshes + i;
    if (post_mesh->_exp_mesh != nullptr) {
      if (   post_mesh->ent_flag[3]
          || post_mesh->mod_flag_min == FVM_WRITER_TRANSIENT_CONNECT) {
        post_mesh->exp_mesh = nullptr;
        post_mesh->_exp_mesh = fvm_nodal_destroy(post_mesh->_exp_mesh);
      }
    }
  }

  cs_timer_stats_switch(t_top_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Loop on post-processing meshes to output variables.
 *
 * This handles all default fields output, as well as all
 * registered output functions and outputs defined in
 * \ref cs_user_postprocess_values
 *
 * \param[in]  ts  time step status structure, or null
 */
/*----------------------------------------------------------------------------*/

void
cs_post_write_vars(const cs_time_step_t  *ts)
{
  /* Output meshes if needed */

  _update_meshes(ts);

  /* Loop on post-processing meshes to output variables */

  cs_post_time_step_output(ts);

  /* Flush writers and free time-varying and Lagragian mesh if needed */

  cs_post_time_step_end();
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Destroy all structures associated with post-processing
 */
/*----------------------------------------------------------------------------*/

void
cs_post_finalize(void)
{
  int i, j;
  cs_post_mesh_t  *post_mesh = nullptr;

  /* Timings */

  for (i = 0; i < _cs_post_n_writers; i++) {
    cs_timer_counter_t m_time, f_time, a_time;
    fvm_writer_t *writer = (_cs_post_writers + i)->writer;
    CS_TIMER_COUNTER_INIT(m_time);
    CS_TIMER_COUNTER_INIT(f_time);
    CS_TIMER_COUNTER_INIT(a_time);
    if (writer != nullptr) {
      fvm_writer_get_times(writer,
                           &m_time, &f_time, &a_time);
      cs_log_printf(CS_LOG_PERFORMANCE,
                    _("\n"
                      "Writing of \"%s\" (%s) summary:\n"
                      "\n"
                      "  Elapsed time for meshes:          %12.3f\n"
                      "  Elapsed time for variables:       %12.3f\n"
                      "  Elapsed time forcing output:      %12.3f\n"),
                    fvm_writer_get_name(writer),
                    fvm_writer_get_format(writer),
                    m_time.nsec*1e-9,
                    f_time.nsec*1e-9,
                    a_time.nsec*1e-9);
    }
  }

  cs_log_printf(CS_LOG_PERFORMANCE, "\n");
  cs_log_separator(CS_LOG_PERFORMANCE);

  /* Exportable meshes */

  for (i = 0; i < _cs_post_n_meshes; i++) {
    post_mesh = _cs_post_meshes + i;
    if (post_mesh->_exp_mesh != nullptr)
      fvm_nodal_destroy(post_mesh->_exp_mesh);
    CS_FREE(post_mesh->name);
    for (j = 0; j < 4; j++)
      CS_FREE(post_mesh->criteria[j]);
    CS_FREE(post_mesh->writer_id);
    CS_FREE(post_mesh->nt_last);
    CS_FREE(post_mesh->a_field_info);
  }

  CS_FREE(_cs_post_meshes);

  _cs_post_min_mesh_id = _MIN_RESERVED_MESH_ID;
  _cs_post_n_meshes = 0;
  _cs_post_n_meshes_max = 0;

  /* Writers */

  for (i = 0; i < _cs_post_n_writers; i++) {
    cs_post_writer_t  *writer = _cs_post_writers + i;
    if (writer->ot != nullptr)
      _free_writer_times(writer);
    if (writer->wd != nullptr)
      _destroy_writer_def(writer);
    if (writer->writer != nullptr)
      fvm_writer_finalize((_cs_post_writers + i)->writer);
  }

  CS_FREE(_cs_post_writers);

  _cs_post_n_writers = 0;
  _cs_post_n_writers_max = 0;

  /* Registered processings if necessary */

  if (_cs_post_n_output_tp_max > 0) {
    CS_FREE(_cs_post_f_output_tp);
    CS_FREE(_cs_post_i_output_tp);
  }

  if (_cs_post_n_output_mtp_max > 0) {
    CS_FREE(_cs_post_f_output_mtp);
    CS_FREE(_cs_post_i_output_mtp);
  }

  /* Options */

  CS_FREE(_cs_post_default_format_options);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Postprocess free (isolated) faces of the current global mesh
 */
/*----------------------------------------------------------------------------*/

void
cs_post_add_free_faces(void)
{
  cs_lnum_t i, j;
  cs_lnum_t n_f_faces = 0;
  cs_gnum_t n_no_group = 0;
  int max_null_family = 0;
  cs_lnum_t *f_face_list = nullptr;

  fvm_writer_t *writer = nullptr;
  fvm_nodal_t *exp_mesh = nullptr;

  bool  generate_submeshes = false;
  cs_mesh_t *mesh = cs_glob_mesh;
  const char *fmt_name = fvm_writer_format_name(_cs_post_default_format_id);

  if (mesh->n_g_free_faces == 0)
    return;

  /* Create default writer */

  writer = fvm_writer_init("isolated_faces",
                           _cs_post_dirname,
                           fmt_name,
                           _cs_post_default_format_options,
                           FVM_WRITER_FIXED_MESH);

  /* Build list of faces to extract */

  CS_MALLOC(f_face_list, mesh->n_b_faces, cs_lnum_t);

  for (i = 0; i < mesh->n_b_faces; i++) {
    if (mesh->b_face_cells[i] < 0)
      f_face_list[n_f_faces++] = i + 1;
  }

  /* Extract and output mesh of isolated faces */

  exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                            "isolated faces",
                                            true,
                                            0,
                                            n_f_faces,
                                            nullptr,
                                            f_face_list);

  if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
    fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

  fvm_writer_set_mesh_time(writer, -1, 0);
  fvm_writer_export_nodal(writer, exp_mesh);

  exp_mesh = fvm_nodal_destroy(exp_mesh);

  /* Now check if we should generate additional meshes (EnSight Gold format) */

  if (!strcmp(fmt_name, "EnSight Gold") && mesh->n_families > 0) {

    generate_submeshes = true;

    /* Families should be sorted, so if a nonzero family is empty,
       it is family 1 */
    if (mesh->family_item[0] == 0)
      max_null_family = 1;
    if (mesh->n_families <= max_null_family)
      generate_submeshes = false;

    /* Check how many boundary faces belong to no group */

    if (mesh->b_face_family != nullptr) {
      for (j = 0; j < n_f_faces; j++) {
        if (mesh->b_face_family[f_face_list[j] - 1] <= max_null_family)
          n_no_group += 1;
      }
    }
    else
      n_no_group = n_f_faces;

    cs_parall_counter(&n_no_group, 1);

    if (n_no_group == mesh->n_g_free_faces)
      generate_submeshes = false;
  }

  /* Generate submeshes if necessary */

  if (generate_submeshes) {

    cs_lnum_t n_b_faces;
    int *fam_flag = nullptr;
    char *group_flag = nullptr;
    cs_lnum_t *b_face_list = nullptr;
    char part_name[81];

    /* Now detect which groups may be referenced */

    CS_MALLOC(fam_flag, mesh->n_families + 1, int);
    memset(fam_flag, 0, (mesh->n_families + 1)*sizeof(int));

    if (mesh->b_face_family != nullptr) {
      for (i = 0; i < n_f_faces; i++)
        fam_flag[mesh->b_face_family[f_face_list[i] - 1]] = 1;
    }

    group_flag = _build_group_flag(mesh, fam_flag);

    /* Now extract isolated faces by groups.
       Selector structures may not have been initialized yet,
       so we use a direct selection here. */

    CS_REALLOC(fam_flag, mesh->n_families, int);

    CS_MALLOC(b_face_list, mesh->n_b_faces, cs_lnum_t);

    for (i = 0; i < mesh->n_groups; i++) {

      if (group_flag[i] != 0) {

        const char *g_name = mesh->group + mesh->group_idx[i];

        _set_fam_flags(mesh, i, fam_flag);

        n_b_faces = 0;
        if (mesh->b_face_family != nullptr) {
          for (j = 0; j < n_f_faces; j++) {
            cs_lnum_t face_id = f_face_list[j] - 1;
            int fam_id = mesh->b_face_family[face_id];
            if (fam_id > 0 && fam_flag[fam_id - 1])
              b_face_list[n_b_faces++] = face_id + 1;
          }
        }

        strcpy(part_name, "isolated: ");
        strncat(part_name, g_name, 80 - strlen(part_name));

        exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                  part_name,
                                                  false,
                                                  0,
                                                  n_b_faces,
                                                  nullptr,
                                                  b_face_list);

        if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
          fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

        fvm_writer_set_mesh_time(writer, -1, 0);
        fvm_writer_export_nodal(writer, exp_mesh);

        exp_mesh = fvm_nodal_destroy(exp_mesh);
      }

    }

    /* Output boundary faces belonging to no group */

    if (n_no_group > 0) {

      if (mesh->b_face_family != nullptr) {
        for (j = 0, n_b_faces = 0; j < n_f_faces; j++) {
          cs_lnum_t face_id = f_face_list[j] - 1;
          if (mesh->b_face_family[face_id] <= max_null_family)
            b_face_list[n_b_faces++] = face_id + 1;
        }
      }
      else {
        for (j = 0, n_b_faces = 0; j < n_f_faces; j++) {
          cs_lnum_t face_id = f_face_list[j] - 1;
          b_face_list[n_b_faces++] = face_id + 1;
        }
      }

      exp_mesh = cs_mesh_connect_faces_to_nodal(cs_glob_mesh,
                                                "isolated: no_group",
                                                false,
                                                0,
                                                n_b_faces,
                                                nullptr,
                                                b_face_list);

      if (fvm_writer_needs_tesselation(writer, exp_mesh, FVM_FACE_POLY) > 0)
        fvm_nodal_tesselate(exp_mesh, FVM_FACE_POLY, nullptr);

      fvm_writer_set_mesh_time(writer, -1, 0);
      fvm_writer_export_nodal(writer, exp_mesh);

      exp_mesh = fvm_nodal_destroy(exp_mesh);
    }

    CS_FREE(b_face_list);

    CS_FREE(fam_flag);
    CS_FREE(group_flag);

  } /* End of submeshes generation */

  /* Free memory */

  writer = fvm_writer_finalize(writer);

  CS_FREE(f_face_list);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize post-processing writer with same format and associated
 * options as default writer, but no time dependency, intended to
 * troubleshoot errors.
 */
/*----------------------------------------------------------------------------*/

void
cs_post_init_error_writer(void)
{
  /* Default values */

  int writer_id = CS_POST_WRITER_ERRORS;
  if (cs_post_writer_exists(writer_id))
    return;

  /* Create default writer */

  int default_format_id = _cs_post_default_format_id;
  const char *default_format_options = _cs_post_default_format_options;
  const char null_str[] = "";

  /* Special case for Catalyst: if matching co-processing script is
     not available, revert to EnSight Gold format */

  if (default_format_id == fvm_writer_get_format_id("Catalyst")) {
    if (! cs_file_isreg("error.py")) {
      default_format_id = fvm_writer_get_format_id("EnSight Gold");
      default_format_options = null_str;
    }
  }

  cs_post_define_writer(writer_id,
                        "error",
                        _cs_post_dirname,
                        fvm_writer_format_name(default_format_id),
                        default_format_options,
                        FVM_WRITER_FIXED_MESH, /* No time dependency here */
                        false,
                        true,
                        -1,
                        -1.0);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize post-processing writer with same format and associated
 * options as default writer, but no time dependency, and associate
 * and output global volume mesh.
 *
 * This is intended to help troubleshoot errors using fields based
 * on cells.
 *
 * \return  id of error output mesh (< 0), or 0 if all writers are deactivated
 */
/*----------------------------------------------------------------------------*/

int
cs_post_init_error_writer_cells(void)
{
  int mesh_id = 0;

  const int writer_id = CS_POST_WRITER_ERRORS;
  const char *mesh_name = N_("Calculation domain");

  cs_post_init_error_writer();
  cs_post_activate_writer(writer_id, 1);

  mesh_id = cs_post_get_free_mesh_id();

  cs_post_define_volume_mesh(mesh_id,
                             _(mesh_name),
                             "all[]",
                             false,
                             false,
                             1,
                             &writer_id);

  cs_post_write_meshes(nullptr);

  return mesh_id;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Register a processing of time-dependent variables to the call to
 * cs_post_write_vars().
 *
 * Note: if the input pointer is non-null, it must point to valid data
 * when the output function is called, so either:
 * - that value or structure should not be temporary (i.e. local);
 * - post-processing output must be ensured using cs_post_write_var()
 *   or similar before the data pointed to goes out of scope.
 *
 * \param[in]       function  function to register
 * \param[in, out]  input     pointer to optional (untyped) value or structure
 */
/*----------------------------------------------------------------------------*/

void
cs_post_add_time_dep_output(cs_post_time_dep_output_t  *function,
                            void                       *input)
{
  /* Resize array of registered post-processings if necessary */

  if (_cs_post_n_output_tp >= _cs_post_n_output_tp_max) {
    if (_cs_post_n_output_tp_max == 0)
      _cs_post_n_output_tp_max = 8;
    else
      _cs_post_n_output_tp_max *= 2;
    CS_REALLOC(_cs_post_f_output_tp,
               _cs_post_n_output_tp_max,
               cs_post_time_dep_output_t *);
    CS_REALLOC(_cs_post_i_output_tp, _cs_post_n_output_tp_max, void *);
  }

  /* Add a post-processing */

  _cs_post_f_output_tp[_cs_post_n_output_tp] = function;
  _cs_post_i_output_tp[_cs_post_n_output_tp] = input;

  _cs_post_n_output_tp += 1;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Register a processing of time-dependent variables than can be output
 * on different meshes to the call to cs_post_write_vars().
 *
 * Note: if the input pointer is non-null, it must point to valid data
 * when the output function is called, so either:
 * - that value or structure should not be temporary (i.e. local);
 * - post-processing output must be ensured using cs_post_write_var()
 *   or similar before the data pointed to goes out of scope.
 *
 * \param[in]       function  function to register
 * \param[in, out]  input     pointer to optional (untyped) value or structure
 */
/*----------------------------------------------------------------------------*/

void
cs_post_add_time_mesh_dep_output(cs_post_time_mesh_dep_output_t  *function,
                                 void                            *input)
{
  /* Resize array of registered post-processings if necessary */

  if (_cs_post_n_output_mtp >= _cs_post_n_output_mtp_max) {
    if (_cs_post_n_output_mtp_max == 0)
      _cs_post_n_output_mtp_max = 8;
    else
      _cs_post_n_output_mtp_max *= 2;
    CS_REALLOC(_cs_post_f_output_mtp,
               _cs_post_n_output_mtp_max,
               cs_post_time_mesh_dep_output_t *);
    CS_REALLOC(_cs_post_i_output_mtp, _cs_post_n_output_mtp_max, void *);
  }

  /* Add a post-processing */

  _cs_post_f_output_mtp[_cs_post_n_output_mtp] = function;
  _cs_post_i_output_mtp[_cs_post_n_output_mtp] = input;

  _cs_post_n_output_mtp += 1;
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
