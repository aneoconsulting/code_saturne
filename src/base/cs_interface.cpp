/*============================================================================
 * \file Handling of interfaces associating mesh elements (such as
 * inter-processor or periodic connectivity between cells, faces,
 * or vertices).
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
#include <stdio.h>
#include <string.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_error.h"
#include "bft/bft_printf.h"

#include "base/cs_all_to_all.h"
#include "base/cs_base.h"
#include "base/cs_block_dist.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "base/cs_order.h"

#include "fvm/fvm_periodicity.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "base/cs_interface.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Local structure definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Structure defining an interface
 *----------------------------------------------------------------------------*/

struct _cs_interface_t {

  int          rank;           /* Associated rank */

  cs_lnum_t    size;           /* Number of equivalent elements */

  int          tr_index_size;  /* Size of perio_index */
  cs_lnum_t   *tr_index;       /* Index of sub-sections in elt_id, distant_id,
                                  and match_id for different transformations;
                                  purely parallel equivalences appear at
                                  position 0, equivalences through periodic
                                  transform i appear at position i+1;
                                  note that send_order crosses subsection
                                  boundaries, and is not indexed by this array.
                                  nullptr in absence of transformations */

  cs_lnum_t   *elt_id;         /* Local element ids (ordered, always present) */
  cs_lnum_t   *match_id;       /* Matching element ids for same-rank interface,
                                  distant element ids such that match_id[i]
                                  on distant rank matches local_id[i]
                                  (temporary life cycle even in parallel) */
  cs_lnum_t   *send_order;     /* Local element ids ordered so that
                                  receive matches elt_id for other-rank
                                  interfaces, and match_id[send_order[i]]
                                  equals elt_id[i] on same-rank interface */

};

/*----------------------------------------------------------------------------
 * Structure defining a set of interfaces
 *----------------------------------------------------------------------------*/

struct _cs_interface_set_t {

  int                       size;          /* Number of interfaces */

  cs_interface_t          **interfaces;    /* Interface structures array */

  const fvm_periodicity_t  *periodicity;   /* Optional periodicity structure */

  int                       match_id_rc;   /* Match_id reference count */

#if defined(HAVE_MPI)
  MPI_Comm                  comm;          /* Associated communicator */
#endif
};

/*----------------------------------------------------------------------------
 * Local structure defining a temporary list of interfaces
 *----------------------------------------------------------------------------*/

typedef struct {

  int          count;    /* Number of equivalences */
  cs_lnum_t   *shift;    /* Index of per-equivalence data in rank[] and num[] */
  int         *rank;     /* Rank associated with each element */
  int         *tr_id;    /* Transformation id associated with each element,
                            + 1, with 0 indicating no transformation
                            (nullptr in absence of periodicity) */
  cs_lnum_t   *num;      /* Local number associated with each element */

} _per_block_equiv_t;

/*----------------------------------------------------------------------------
 * Local structure defining a temporary list of periodic interfaces
 *----------------------------------------------------------------------------*/

typedef struct {

  int          count;     /* Number of periodic couples */
  cs_lnum_t   *block_id;  /* local id in block */
  int         *tr_id;     /* Transform id associated with each couple */
  int         *shift;     /* Index of per-couple data in rank[] and num[] */
  int         *rank;      /* Ranks associated with periodic elements */
  cs_lnum_t   *num;       /* Local numbers associated with periodic elements */

} _per_block_period_t;

/*=============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Creation of an empty interface between elements of a same type.
 *
 * This interface may be used to identify equivalent vertices or faces using
 * domain splitting, as well as periodic elements (on the same or on
 * distant ranks).
 *
 * returns:
 *  pointer to allocated interface structure
 *----------------------------------------------------------------------------*/

static cs_interface_t *
_cs_interface_create(void)
{
  cs_interface_t  *_interface;

  CS_MALLOC(_interface, 1, cs_interface_t);

  _interface->rank = -1;
  _interface->size = 0;

  _interface->tr_index_size = 0;
  _interface->tr_index = nullptr;

  _interface->elt_id = nullptr;
  _interface->match_id = nullptr;
  _interface->send_order = nullptr;

  return _interface;
}

/*----------------------------------------------------------------------------
 * Destruction of an interface.
 *
 * parameters:
 *   itf <-> pointer to pointer to structure to destroy
 *
 * returns:
 *   null pointer
 *----------------------------------------------------------------------------*/

static void
_cs_interface_destroy(cs_interface_t  **itf)
{
  cs_interface_t  *_itf = *itf;

  if (_itf != nullptr) {
    CS_FREE(_itf->tr_index);
    CS_FREE(_itf->elt_id);
    CS_FREE(_itf->match_id);
    CS_FREE(_itf->send_order);
    CS_FREE(_itf);
  }

  *itf = _itf;
}

/*----------------------------------------------------------------------------
 * Dump printout of an interface.
 *
 * parameters:
 *   itf <-- pointer to structure that should be dumped
 *----------------------------------------------------------------------------*/

static void
_cs_interface_dump(const cs_interface_t  *itf)
{
  int i, section_id;
  cs_lnum_t start_id, end_id;

  int  tr_index_size = 2;
  cs_lnum_t   _tr_index[2] = {0, 0};
  const cs_lnum_t   *tr_index = _tr_index;

  if (itf == nullptr) {
    bft_printf("  interface: nil\n");
    return;
  }

  bft_printf("  interface:             %p\n"
             "  associated rank:       %d\n"
             "  size:                  %llu\n"
             "  transform index size:  %d\n",
             (const void *)itf,
             itf->rank,
             (unsigned long long)(itf->size),
             itf->tr_index_size);

  if (itf->tr_index_size > 0) {
    bft_printf("  transform index:\n");
    for (i = 0; i < itf->tr_index_size; i++)
      bft_printf("    %5d %lu\n", i, (unsigned long)itf->tr_index[i]);
  }


  _tr_index[1] = itf->size;

  if (itf->tr_index_size > 0) {
    tr_index_size = itf->tr_index_size;
    tr_index = itf->tr_index;
  }

  /* Case for other-rank interface */

  if (itf->match_id != nullptr) {

    for (section_id = 0; section_id < tr_index_size - 1; section_id++) {

      if (section_id == 0)
        bft_printf("\n"
                   "            id      elt_id   match_id (parallel)\n");
      else
        bft_printf("\n"
                   "            id      elt_id   match_id (transform %d)\n",
                   section_id - 1);

      start_id = tr_index[section_id];
      end_id = tr_index[section_id + 1];

      for (i = start_id; i < end_id; i++)
        bft_printf("    %10ld %10ld %10ld\n",
                   (long)i, (long)itf->elt_id[i], (long)itf->match_id[i]);

    }

  }

  else {

    for (section_id = 0; section_id < tr_index_size - 1; section_id++) {

      if (section_id == 0)
        bft_printf("\n"
                   "            id      elt_id (parallel)\n");
      else
        bft_printf("\n"
                   "            id      elt_id (transform %d)\n",
                   section_id - 1);

      start_id = tr_index[section_id];
      end_id = tr_index[section_id + 1];

      for (i = start_id; i < end_id; i++)
        bft_printf("    %10ld %10ld\n", (long)i, (long)itf->elt_id[i]);

    }

  }

  /* Print send order separately, as it is section-independant */

  if (itf->send_order != nullptr) {

    bft_printf("\n"
               "            id      send_order\n");

    for (i = 0; i < itf->size; i++)
      bft_printf("    %10ld %10ld\n", (long)i, (long)itf->send_order[i]);

  }

  bft_printf("\n");
}

/*----------------------------------------------------------------------------
 * Sort and remove duplicates from periodic tuple information.
 *
 * parameters:
 *   n_block_tuples  <-> number of tuples in current block
 *   block_tuples    <-> tuple information for this rank: for each tuple,
 *                       {global number of local element,
 *                        global number of periodic element,
 *                        transform id}
 *----------------------------------------------------------------------------*/

static void
_sort_periodic_tuples(cs_lnum_t    *n_block_tuples,
                      cs_gnum_t   **block_tuples)
{
  cs_lnum_t   i, j, k;

  cs_lnum_t    n_tuples = *n_block_tuples;
  cs_lnum_t   *order = nullptr;
  cs_gnum_t   *tuples = *block_tuples;
  cs_gnum_t   *tuples_tmp = nullptr;

  if (n_tuples < 1)
    return;

  /* Sort periodic tuples by local correspondant */

  CS_MALLOC(order, n_tuples, cs_lnum_t);
  CS_MALLOC(tuples_tmp, n_tuples*3, cs_gnum_t);

  cs_order_gnum_allocated_s(nullptr,
                            tuples,
                            3,
                            order,
                            n_tuples);

  /* Copy to temporary array, ignoring duplicates */

  k = order[0]*3;
  tuples_tmp[0] = tuples[k];
  tuples_tmp[1] = tuples[k + 1];
  tuples_tmp[2] = tuples[k + 2];
  j = 3;

  for (i = 1; i < n_tuples; i++) {
    k = order[i] * 3;
    if (   (tuples[k]   != tuples_tmp[j-3])
        || (tuples[k+1] != tuples_tmp[j-2])
        || (tuples[k+2] != tuples_tmp[j-1])) {
      tuples_tmp[j]     = tuples[k];
      tuples_tmp[j + 1] = tuples[k + 1];
      tuples_tmp[j + 2] = tuples[k + 2];
      j += 3;
    }
  }
  n_tuples = j / 3;

  CS_FREE(order);

  /* Resize input/outpout array if duplicates were removed */

  if (n_tuples <= *n_block_tuples) {
    CS_REALLOC(tuples, n_tuples*3, cs_gnum_t);
    *n_block_tuples = n_tuples;
    *block_tuples = tuples;
  }

  /* Copy sorted data to input/output array and free temporary storage */

  memcpy(tuples, tuples_tmp, sizeof(cs_gnum_t)*n_tuples*3);

  CS_FREE(tuples_tmp);
}

/*----------------------------------------------------------------------------
 * Extract periodicity transform data necessary for periodic combinations.
 *
 * Builds a square transformation combination matrix associating a combination
 * transform id with transform ids of levels lower than that given by
 * the level argument; the number of rows and columns is thus equal to the
 * number of transformations of level lower than the argument.
 * Entries corresponding to impossible combinations are set to -1;
 *
 * The caller is responsible for freeing the tr_combine array returned by
 * this function.
 *
 * parameters:
 *   periodicity  <-- periodicity information
 *   level        --> transform combination level (1 or 2)
 *   n_rows       --> number of rows of combination matrix
 *                    (equals transform_level_index[level])
 *   tr_combine   --> combination id matrix (size n_rows * n_rows)
 *----------------------------------------------------------------------------*/

static void
_transform_combine_info(const fvm_periodicity_t   *periodicity,
                        int                        level,
                        int                       *n_rows,
                        int                      **tr_combine)
{
  int  i;
  int  tr_level_idx[4], parent_id[2];

  int n_vals_1 = 0, n_vals_2 = 0;
  int n_rows_1 = 0, n_rows_2 = 0;
  int *tr_combine_1, *tr_combine_2 = nullptr;

  assert(periodicity != nullptr);
  assert(level == 1 || level == 2);

  /* Extract base periodicity dimension info */

  fvm_periodicity_get_tr_level_idx(periodicity, tr_level_idx);

  /* We always need the level0 x level0 -> level1 array */

  n_rows_1 = tr_level_idx[1];
  n_vals_1 = n_rows_1*n_rows_1;

  CS_MALLOC(tr_combine_1, n_vals_1, int);

  for (i = 0; i < n_vals_1; i++)
    tr_combine_1[i] = -1;

  for (i = tr_level_idx[1]; i < tr_level_idx[2]; i++) {

    fvm_periodicity_get_parent_ids(periodicity, i, parent_id);

    assert(parent_id[0] > -1 && parent_id[1] > -1);
    assert(parent_id[0] < n_rows_1 && parent_id[1] < n_rows_1);

    tr_combine_1[parent_id[0]*n_rows_1 + parent_id[1]] = i;
    tr_combine_1[parent_id[1]*n_rows_1 + parent_id[0]] = i;

  }

  /* For level 1 transforms, we are done once return values are set */

  if (level == 1) {

    *n_rows = n_rows_1;
    *tr_combine = tr_combine_1;

  }

  /* Handle level 2 transforms */

  else { /* if (level == 2) */

    int  tr_01, tr_02, tr_12;
    int  comp_id[3];

    n_rows_2 = tr_level_idx[2];
    n_vals_2 = n_rows_2*n_rows_2;

    /* Allocate and populate array */

    CS_MALLOC(tr_combine_2, n_vals_2, int);

    for (i = 0; i < n_vals_2; i++)
      tr_combine_2[i] = -1;

    for (i = tr_level_idx[2]; i < tr_level_idx[3]; i++) {

      fvm_periodicity_get_components(periodicity, i, comp_id);

      assert(comp_id[0] > -1 && comp_id[1] > -1 && comp_id[2] > -1);

      tr_01 = tr_combine_1[comp_id[0]*n_rows_1 + comp_id[1]];
      tr_02 = tr_combine_1[comp_id[0]*n_rows_1 + comp_id[2]];
      tr_12 = tr_combine_1[comp_id[1]*n_rows_1 + comp_id[2]];

      assert(tr_01 > -1 && tr_02 > -1 && tr_12 > -1);

      tr_combine_2[tr_01*n_rows_2 + comp_id[2]] = i;
      tr_combine_2[comp_id[2]*n_rows_2 + tr_01] = i;

      tr_combine_2[tr_02*n_rows_2 + comp_id[1]] = i;
      tr_combine_2[comp_id[1]*n_rows_2 + tr_02] = i;

      tr_combine_2[tr_12*n_rows_2 + comp_id[0]] = i;
      tr_combine_2[comp_id[0]*n_rows_2 + tr_12] = i;

    }

    CS_FREE(tr_combine_1);

    /* Set return values */

    *n_rows = n_rows_2;
    *tr_combine = tr_combine_2;
  }

}

#if defined(HAVE_MPI)

/*----------------------------------------------------------------------------
 * Maximum global number associated with an I/O numbering structure
 *
 * parameters:
 *   n_elts     <-- local number of elements
 *   global_num <-- global number (id) associated with each element
 *   comm       <-- associated MPI communicator
 *
 * returns:
 *   maximum global number associated with the I/O numbering
 *----------------------------------------------------------------------------*/

static cs_gnum_t
_global_num_max(cs_lnum_t         n_elts,
                const cs_gnum_t   global_num[],
                MPI_Comm          comm)
{
  cs_gnum_t  global_max;
  cs_gnum_t  local_max = 0;

  /* Get maximum global number value */

  for (cs_lnum_t i = 0; i < n_elts; i++) {
    if (global_num[i] > local_max)
      local_max = global_num[i];
  }

  MPI_Allreduce(&local_max, &global_max, 1, CS_MPI_GNUM, MPI_MAX, comm);

  return global_max;
}

/*----------------------------------------------------------------------------
 * Build temporary equivalence structure for data in a given block,
 * and associate an equivalence id to received elements (-1 for elements
 * with no correponding elements)
 *
 * parameters:
 *   n_elts_recv     <-- number of elements received
 *   recv_rank       <-- source ranks received
 *   recv_global_num <-- global numbering received
 *   recv_num        <-- local numbering received
 *   equiv_id        --> equivalence id for each element (-1 if none)
 *
 * returns:
 *   temporary equivalence structure for block
 *----------------------------------------------------------------------------*/

#if defined(__INTEL_COMPILER)
#pragma optimization_level 2 /* Crash with O3 on IA64 with icc 9.1 20070320 */
#endif

static _per_block_equiv_t
_block_global_num_to_equiv(cs_lnum_t          n_elts_recv,
                           const int          recv_rank[],
                           const cs_gnum_t    recv_global_num[],
                           const cs_lnum_t    recv_num[],
                           cs_lnum_t          equiv_id[])
{
  /* Initialize return structure */

  _per_block_equiv_t  e = {.count = 0,
                           .shift = nullptr,
                           .rank  = nullptr,
                           .tr_id = nullptr,
                           .num   = nullptr};

  if (n_elts_recv == 0)
    return e;

  /* Determine equivalent elements; requires ordering to loop through buffer
     by increasing number. */

  cs_lnum_t   *recv_order = nullptr;
  CS_MALLOC(recv_order, n_elts_recv, cs_lnum_t);

  cs_order_gnum_allocated(nullptr,
                          recv_global_num,
                          recv_order,
                          n_elts_recv);

  /* Loop by increasing number: if two elements have the same global
     number, they are equivalent. We do not increment equivalence counts
     as soon as two elements are equivalent, as three equivalent elements
     should have the same equivalence id. Rather, we increment the
     equivalence counter when the previous element was part of an
     equivalence and the current element is not part of this same
     equivalence. */

  equiv_id[recv_order[0]] = -1;
  cs_gnum_t prev_num = recv_global_num[recv_order[0]];

  for (cs_lnum_t i = 1; i < n_elts_recv; i++) {
    cs_gnum_t cur_num = recv_global_num[recv_order[i]];
    if (cur_num == prev_num) {
      equiv_id[recv_order[i-1]] = e.count;
      equiv_id[recv_order[i]]   = e.count;
    }
    else {
      if (equiv_id[recv_order[i-1]] > -1)
        e.count++;
      equiv_id[recv_order[i]] = -1;
    }
    prev_num = cur_num;
  }
  if (equiv_id[recv_order[n_elts_recv-1]] > -1)
    e.count++;

  CS_FREE(recv_order);

  /* Count number of elements associated with each equivalence */

  int  *multiple;
  CS_MALLOC(multiple, e.count, int);
  for (cs_lnum_t i = 0; i < e.count; multiple[i++] = 0);
  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    if (equiv_id[i] > -1)
      multiple[equiv_id[i]] += 1;
  }

  CS_MALLOC(e.shift, e.count+1, cs_lnum_t);
  e.shift[0] = 0;
  for (cs_lnum_t i = 0; i < e.count; i++) {
    e.shift[i+1] = e.shift[i] + multiple[i];
    multiple[i] = 0;
  }

  /* Build equivalence data */

  CS_MALLOC(e.rank, e.shift[e.count], int);
  CS_MALLOC(e.num, e.shift[e.count], cs_lnum_t);

  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    if (equiv_id[i] > -1) {
      cs_lnum_t j = e.shift[equiv_id[i]] + multiple[equiv_id[i]];
      e.rank[j] = recv_rank[i];
      e.num[j] = recv_num[i];
      multiple[equiv_id[i]] += 1;
    }
  }

  CS_FREE(multiple);

  return e;
}

/*----------------------------------------------------------------------------
 * Build global interface data from flat equivalence data
 * (usually prepared and received from distant ranks).
 *
 * parameters:
 *   ifs  <-> pointer to structure that should be updated
 *   tr_index_size       <-- size of transform index (number of transforms
 *                           + 1 for idelement + 1 for past-the-end);
 *                           0 or 1 if no transforms are present
 *   n_elts_recv         <-- size of received data
 *   equiv_recv          <-- flat (received) equivalence data; for each
 *                           equivalence, we have:
 *                           {local_number, n_equivalents,
 *                            {distant_number, distant_rank}*n_equivalents}
 *----------------------------------------------------------------------------*/

static void
_interfaces_from_flat_equiv(cs_interface_set_t  *ifs,
                            int                  tr_index_size,
                            cs_lnum_t            n_elts_recv,
                            const cs_lnum_t      equiv_recv[])
{
  cs_lnum_t i, j, k, l;
  cs_lnum_t local_num, distant_num, n_sub, n_elts_rank_tr_size;
  int rank, tr_id;

  int _tr_index_size = tr_index_size;
  int _tr_stride = tr_index_size > 1 ? tr_index_size - 1 : 1;
  int max_rank = 0, n_ranks = 0, start_id = 0;
  int recv_step = 1;

  cs_lnum_t   *n_elts_rank = nullptr;
  cs_lnum_t   *n_elts_rank_tr = nullptr;
  int *interface_id = nullptr;

  cs_interface_t *_interface = nullptr;

  if (_tr_index_size == 0) {
    _tr_index_size = 1;
    _tr_stride = 1;
  }
  else if (_tr_index_size > 1)
    recv_step = 2;

  /* Compute size of subsections for each rank */

  i = 0;
  while (i < n_elts_recv) {
    i++;
    n_sub = equiv_recv[i++];
    for (j = 0; j < n_sub; j++) {
      i += recv_step;
      rank = equiv_recv[i++];
      if (rank > max_rank)
        max_rank = rank;
    }
  }

  CS_MALLOC(n_elts_rank, max_rank + 1, cs_lnum_t);

  for (i = 0; i < max_rank + 1; n_elts_rank[i++] = 0);

  i = 0;
  while (i < n_elts_recv) {
    i++;
    n_sub = equiv_recv[i++];
    for (j = 0; j < n_sub; j++) {
      i += recv_step;
      rank = equiv_recv[i++];
      n_elts_rank[rank] += 1;
    }
  }

  /* Build final data structures */

  n_ranks = 0;
  for (i = 0; i < max_rank + 1; i++) {
    if (n_elts_rank[i] > 0)
      n_ranks++;
  }

  /* (Re-)Allocate structures */

  start_id = ifs->size;

  ifs->size += n_ranks;

  CS_REALLOC(ifs->interfaces,
             ifs->size,
             cs_interface_t *);

  for (i = start_id; i < ifs->size; i++)
    ifs->interfaces[i] = _cs_interface_create();

  /* Initialize rank info and interface id */

  n_ranks = 0;
  CS_MALLOC(interface_id, max_rank + 1, int);
  for (i = 0; i < max_rank + 1; i++) {
    if (n_elts_rank[i] > 0) {
      interface_id[i] = start_id + n_ranks++;
      (ifs->interfaces[interface_id[i]])->rank = i;
      (ifs->interfaces[interface_id[i]])->size = n_elts_rank[i];
    }
    else
      interface_id[i] = -1;
  }

  CS_FREE(n_elts_rank);

  /* n_elts_rank_tr will be used as a position counter for new interfaces */

  n_elts_rank_tr_size = (ifs->size - start_id)*_tr_stride;
  CS_MALLOC(n_elts_rank_tr, n_elts_rank_tr_size, cs_lnum_t);
  for (i = 0; i < n_elts_rank_tr_size; i++)
    n_elts_rank_tr[i] = 0;

  for (i = start_id; i < ifs->size; i++) {

    _interface = ifs->interfaces[i];

    CS_MALLOC(_interface->elt_id, _interface->size, cs_lnum_t);
    CS_MALLOC(_interface->match_id, _interface->size, cs_lnum_t);

    if (_tr_index_size > 1) {
      _interface->tr_index_size = _tr_index_size;
      CS_MALLOC(_interface->tr_index, _interface->tr_index_size, cs_lnum_t);
      for (k = 0; k < _interface->tr_index_size; k++)
        _interface->tr_index[k] = 0;
    }
    else {
      _interface->tr_index_size = 0;
      _interface->tr_index = nullptr;
    }

  }

  /* In absence of transforms, we may build the interface in one pass */

  if (_tr_index_size < 2)  {

    i = 0;
    while (i < n_elts_recv) {

      local_num = equiv_recv[i++];
      n_sub = equiv_recv[i++];

      for (j = 0; j < n_sub; j++) {

        distant_num = equiv_recv[i++];
        rank = equiv_recv[i++];

        _interface = ifs->interfaces[interface_id[rank]];
        k = interface_id[rank] - start_id;

        _interface->elt_id[n_elts_rank_tr[k]] = local_num - 1;
        _interface->match_id[n_elts_rank_tr[k]] = distant_num - 1;
        n_elts_rank_tr[k] += 1;

      }
    }

  }

  /* If we have transforms, build the transform index first */

  else { /* if (_tr_index_size > 1) */

    /* Initial count */

    i = 0;
    while (i < n_elts_recv) {

      i++;
      n_sub = equiv_recv[i++];

      for (j = 0; j < n_sub; j++) {

        i++;
        tr_id = equiv_recv[i++];
        rank = equiv_recv[i++];

        _interface = ifs->interfaces[interface_id[rank]];

        _interface->tr_index[tr_id + 1] += 1;

      }
    }

    /* Build index from initial count */

    for (i = start_id; i < ifs->size; i++) {
      _interface = ifs->interfaces[i];
      _interface->tr_index[0] = 0;
      for (j = 1; j < _tr_index_size; j++)
        _interface->tr_index[j] += _interface->tr_index[j-1];
    }

    /* Now populate the arrays */

    i = 0;
    while (i < n_elts_recv) {

      local_num = equiv_recv[i++];
      n_sub = equiv_recv[i++];

      for (j = 0; j < n_sub; j++) {

        distant_num = equiv_recv[i++];
        tr_id = equiv_recv[i++];
        rank = equiv_recv[i++];

        _interface = ifs->interfaces[interface_id[rank]];
        k = (interface_id[rank] - start_id)*_tr_stride + tr_id;

        l = _interface->tr_index[tr_id] + n_elts_rank_tr[k];

        _interface->elt_id[l] = local_num - 1;
        _interface->match_id[l] = distant_num - 1;

        n_elts_rank_tr[k] += 1;

      }
    }

  }

  /* n_elts_rank will now be used as a position counter for new interfaces */

  CS_FREE(n_elts_rank_tr);
  CS_FREE(interface_id);
}

/*----------------------------------------------------------------------------
 * Creation of a list of interfaces between elements of a same type.
 *
 * The global_num values need not be ordered or contiguous.
 *
 * parameters:
 *   ifs  <-> pointer to structure that should be updated
 *   n_elts              <-- local number of elements
 *   global_num          <-- global number (id) associated with each element
 *   comm                <-- associated MPI communicator
 *----------------------------------------------------------------------------*/

static void
_add_global_equiv(cs_interface_set_t  *ifs,
                  cs_lnum_t            n_elts,
                  const cs_gnum_t      global_num[],
                  MPI_Comm             comm)
{
  /* Initialization */

  int  size, local_rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &local_rank);

  /* Get temporary maximum global number value */

  cs_gnum_t global_max = _global_num_max(n_elts,
                                         global_num,
                                         comm);

  cs_block_dist_info_t
    bi = cs_block_dist_compute_sizes(local_rank,
                                     size,
                                     1,
                                     0,
                                     global_max);

  int flags = CS_ALL_TO_ALL_ORDER_BY_SRC_RANK;

  cs_all_to_all_t
    *d = cs_all_to_all_create_from_block(n_elts,
                                         flags,
                                         global_num,
                                         bi,
                                         comm);

  cs_gnum_t *recv_global_num = cs_all_to_all_copy_array(d,
                                                        1,
                                                        false, /* reverse */
                                                        global_num);

  cs_lnum_t  *send_num = nullptr;
  CS_MALLOC(send_num, n_elts, cs_lnum_t);
  for (cs_lnum_t i = 0; i < n_elts; i++)
    send_num[i] = i+1;

  cs_lnum_t *recv_num = cs_all_to_all_copy_array(d,
                                                 1,
                                                 false, /* reverse */
                                                 send_num);

  CS_FREE(send_num);

  cs_lnum_t n_elts_recv = cs_all_to_all_n_elts_dest(d);

  int *src_rank = cs_all_to_all_get_src_rank(d);

  /* Build equivalence data */

  cs_lnum_t  *equiv_id = nullptr;
  CS_MALLOC(equiv_id, n_elts_recv, cs_lnum_t);

  _per_block_equiv_t  e = _block_global_num_to_equiv(n_elts_recv,
                                                     src_rank,
                                                     recv_global_num,
                                                     recv_num,
                                                     equiv_id);

  /* Now free Memory */

  CS_FREE(recv_num);
  CS_FREE(recv_global_num);

  /* Now that equivalences are marked, count for each rank; for each
     equivalence, we will need to send the corresponding element
     numbers and ranks, for a total of 2*(e.multiple[...] - 1) values. */

  cs_lnum_t  *block_index;
  CS_MALLOC(block_index, n_elts_recv+1, cs_lnum_t);

  block_index[0] = 0;
  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    cs_lnum_t n_eq = 0;
    if (equiv_id[i] > -1) {
      size_t e_id = equiv_id[i];
      n_eq = 2*(e.shift[e_id+1] - e.shift[e_id]);
    }
    block_index[i+1] = block_index[i] + n_eq;
  }

  cs_lnum_t *part_index = cs_all_to_all_copy_index(d,
                                                   true, /* reverse */
                                                   block_index,
                                                   nullptr);

  /* Now prepare new send buffer */

  cs_lnum_t  *block_equiv;
  CS_MALLOC(block_equiv, block_index[n_elts_recv], cs_lnum_t);

  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    if (equiv_id[i] > -1) {

      cs_lnum_t *block_equiv_p = block_equiv + block_index[i];

      cs_lnum_t        e_id = equiv_id[i];
      cs_lnum_t        k = 2;
      const int        multiple = e.shift[e_id+1] - e.shift[e_id];
      const int       *rank_p = e.rank + e.shift[e_id];
      const cs_lnum_t *num_p  = e.num  + e.shift[e_id];

      for (cs_lnum_t j = 0; j < multiple; j++) {
        if (rank_p[j] == src_rank[i]) {
          block_equiv_p[0] = num_p[j];
          block_equiv_p[1] = multiple - 1;
        }
        else {
          block_equiv_p[k++] = num_p[j];
          block_equiv_p[k++] = rank_p[j];
        }
      }

    }
  }

  /* Free temporary (block) equivalence info */

  e.count = 0;
  CS_FREE(e.shift);
  CS_FREE(e.rank);
  if (e.tr_id != nullptr)
    CS_FREE(e.tr_id);
  CS_FREE(e.num);

  CS_FREE(src_rank);
  CS_FREE(equiv_id);

  /* Send prepared block data to destination rank */

  cs_lnum_t *part_equiv = cs_all_to_all_copy_indexed(d,
                                                     true, /* reverse */
                                                     block_index,
                                                     block_equiv,
                                                     part_index);

  cs_lnum_t n_vals_part = part_index[n_elts];

  /* At this stage, MPI operations are finished; we may release
     the corresponding counts and indexes */

  CS_FREE(block_equiv);
  CS_FREE(part_index);
  CS_FREE(block_index);

  cs_all_to_all_destroy(&d);

  /* Add interface */

  _interfaces_from_flat_equiv(ifs,
                              1,
                              n_vals_part,
                              part_equiv);

  CS_FREE(part_equiv);
}

/*----------------------------------------------------------------------------
 * Build eventual tuples belonging to combined periodicities.
 *
 * parameters:
 *   block_size       <-- size of the block handled by each processor
 *   periodicit       <-- periodicity information (nullptr if none)
 *   n_block_tuples   <-> number of tuples in current block
 *   block_tuples     <-> tuple information for this rank: for each tuple,
 *                        {global number of local element,
 *                         global number of periodic element,
 *                         transform id}
 *   comm             <-- associated MPI communicator
 *----------------------------------------------------------------------------*/

static void
_combine_periodic_tuples(size_t                     block_size,
                         const fvm_periodicity_t   *periodicity,
                         cs_lnum_t                 *n_block_tuples,
                         cs_gnum_t                **block_tuples,
                         MPI_Comm                   comm)
{
  int  *tr_reverse_id = nullptr;

  int         *send_rank = nullptr;
  cs_gnum_t   *send_tuples = nullptr;

  cs_lnum_t   _n_block_tuples = *n_block_tuples;
  cs_gnum_t   *_block_tuples = *block_tuples;

  assert (periodicity != nullptr);

  /* Initialization */

  /* Build periodicity related arrays for quick access */

  int n_tr = fvm_periodicity_get_n_transforms(periodicity);

  CS_MALLOC(tr_reverse_id, n_tr, int);

  for (int i = 0; i < n_tr; i++)
    tr_reverse_id[i] = fvm_periodicity_get_reverse_id(periodicity, i);

  /* Loop on combination levels */

  for (int level = 1;
       level < fvm_periodicity_get_n_levels(periodicity);
       level++) {

    int  n_rows = 0;
    int  *tr_combine = nullptr;

    _transform_combine_info(periodicity,
                            level,
                            &n_rows,
                            &tr_combine);

    /* Count values to exchange */

    size_t n_send = 0;

    cs_lnum_t start_id = 0;
    cs_lnum_t end_id = 1;

    while (end_id < _n_block_tuples) {

      if (_block_tuples[start_id*3] == _block_tuples[end_id*3]) {

        end_id++;
        while (end_id < _n_block_tuples) {
          if (_block_tuples[end_id*3] != _block_tuples[start_id*3])
            break;
          end_id++;
        }

        for (cs_lnum_t j = start_id; j < end_id; j++) {
          for (cs_lnum_t k = j+1; k < end_id; k++) {

            int tr_1 = tr_reverse_id[_block_tuples[j*3 + 2]];
            int tr_2 = _block_tuples[k*3 + 2];

            if (tr_combine[tr_1 *n_rows + tr_2] > -1)
              n_send += 2;

          }
        }

      }

      start_id = end_id;
      end_id += 1;
    }

    CS_MALLOC(send_rank, n_send, int);
    CS_MALLOC(send_tuples, n_send*3, cs_gnum_t);

    /* Now exchange combined tuples */

    start_id = 0;
    end_id = 1;

    cs_lnum_t l = 0;

    while (end_id < _n_block_tuples) {

      if (_block_tuples[start_id*3] == _block_tuples[end_id*3]) {

        end_id++;
        while (end_id < _n_block_tuples) {
          if (_block_tuples[end_id*3] != _block_tuples[start_id*3])
            break;
          end_id++;
        }

        for (cs_lnum_t j = start_id; j < end_id; j++) {
          for (cs_lnum_t k = j+1; k < end_id; k++) {

            int tr_1 = tr_reverse_id[_block_tuples[j*3 + 2]];
            int tr_2 = _block_tuples[k*3 + 2];
            int tr_c = tr_combine[tr_1 *n_rows + tr_2];

            if (tr_c > -1) {

              cs_gnum_t num_1 = _block_tuples[j*3 + 1];
              cs_gnum_t num_2 = _block_tuples[k*3 + 1];

              send_rank[l*2]   = (num_1 - 1) / block_size;
              send_rank[l*2+1] = (num_2 - 1) / block_size;

              send_tuples[l*6]   = num_1;
              send_tuples[l*6+1] = num_2;
              send_tuples[l*6+2] = tr_c;

              send_tuples[l*6+3] = num_2;
              send_tuples[l*6+4] = num_1;
              send_tuples[l*6+5] = tr_reverse_id[tr_c];

              l += 1;

            }

          }
        }

      }

      start_id = end_id;
      end_id += 1;
    }

    assert((cs_gnum_t)(l*2) == n_send);

    CS_FREE(tr_combine);

    cs_all_to_all_t  *d = cs_all_to_all_create(n_send,
                                               0,
                                               nullptr,
                                               send_rank,
                                               comm);

    cs_all_to_all_transfer_dest_rank(d, &send_rank);

    cs_gnum_t *recv_tuples = cs_all_to_all_copy_array(d,
                                                      3,
                                                      false, /* reverse */
                                                      send_tuples);

    CS_FREE(send_tuples);

    cs_lnum_t n_recv = cs_all_to_all_n_elts_dest(d);

    cs_all_to_all_destroy(&d);

    if (n_recv > 0) {
      CS_REALLOC(_block_tuples, (_n_block_tuples + n_recv)*3, cs_gnum_t);
      memcpy(_block_tuples + _n_block_tuples*3,
             recv_tuples,
             n_recv*3*sizeof(cs_gnum_t));
    }

    CS_FREE(recv_tuples);

    /* Finally, merge additional tuples received for block with
       existing periodicity information */

    if (n_recv > 0) {

      _n_block_tuples += n_recv;

      /* Sort and remove duplicates to update block periodicity info */

      _sort_periodic_tuples(&_n_block_tuples, &_block_tuples);

      *n_block_tuples = _n_block_tuples;
      *block_tuples = _block_tuples;

    }

  }

  CS_FREE(tr_reverse_id);
}

/*----------------------------------------------------------------------------
 * Exchange periodic couple info between processors providing the data
 * and processors handling the related global numbering interval blocks.
 *
 * Note that the array pointed to by block_couples is allocated here,
 * and must be freed by the calling code.
 *
 * parameters:
 *   block_size          <-- size of the block handled by each processor
 *   periodicity         <-- periodicity information (nullptr if none)
 *   n_periodic_lists    <-- number of periodic lists (may be local)
 *   periodicity_num     <-- periodicity number (1 to n) associated with
 *                           each periodic list (primary periodicities only)
 *   n_periodic_couples  <-- number of periodic couples associated with
 *                           each periodic list
 *   periodic_couples    <-- array indicating periodic couples (using
 *                           global numberings) for each list
 *   n_block_tuples      --> number of tuples in current block
 *   block_tuples       --> tuple information for block: for each tuple,
 *                           {global number of local element,
 *                            global number of periodic element,
 *                            transform id}
 *   comm                <-- associated MPI communicator
 *----------------------------------------------------------------------------*/

static void
_exchange_periodic_tuples(size_t                    block_size,
                          const fvm_periodicity_t  *periodicity,
                          int                       n_periodic_lists,
                          const int                 periodicity_num[],
                          const cs_lnum_t           n_periodic_couples[],
                          const cs_gnum_t    *const periodic_couples[],
                          cs_lnum_t                *n_block_tuples,
                          cs_gnum_t               **block_tuples,
                          MPI_Comm                  comm)
{
  int        *periodic_block_rank = nullptr;
  cs_gnum_t  *send_tuples = nullptr;

  cs_gnum_t n_g_periodic_tuples = 0;

  /* Initialization */

  *n_block_tuples = 0;
  *block_tuples = nullptr;

  for (int list_id = 0; list_id < n_periodic_lists; list_id++)
    n_g_periodic_tuples += 2 * n_periodic_couples[list_id];

  CS_MALLOC(periodic_block_rank, n_g_periodic_tuples, int);
  CS_MALLOC(send_tuples, n_g_periodic_tuples*3, cs_gnum_t);

  /* Prepare lists to send to distant processors */

  cs_lnum_t k = 0;

  for (int list_id = 0; list_id < n_periodic_lists; list_id++) {

    const int external_num = periodicity_num[list_id];
    const int direct_id = fvm_periodicity_get_transform_id(periodicity,
                                                           external_num,
                                                           1);
    const int reverse_id = fvm_periodicity_get_transform_id(periodicity,
                                                            external_num,
                                                            -1);

    const cs_lnum_t   _n_periodic_couples = n_periodic_couples[list_id];
    const cs_gnum_t   *_periodic_couples = periodic_couples[list_id];

    assert(direct_id >= 0 && reverse_id >= 0);

    for (cs_lnum_t couple_id = 0; couple_id < _n_periodic_couples; couple_id++) {

      cs_gnum_t num_1 = _periodic_couples[couple_id*2];
      cs_gnum_t num_2 = _periodic_couples[couple_id*2 + 1];

      periodic_block_rank[k*2]   = (num_1 - 1) / block_size;
      periodic_block_rank[k*2+1] = (num_2 - 1) / block_size;

      send_tuples[k*6]   = num_1;
      send_tuples[k*6+1] = num_2;
      send_tuples[k*6+2] = direct_id;

      send_tuples[k*6+3] = num_2;
      send_tuples[k*6+4] = num_1;
      send_tuples[k*6+5] = reverse_id;

      k += 1;

    }

  }

  assert((cs_gnum_t)(k*2) == n_g_periodic_tuples);

  /* Exchange data */

  cs_all_to_all_t
    *d_periodic = cs_all_to_all_create(n_g_periodic_tuples,
                                       0,
                                       nullptr,
                                       periodic_block_rank,
                                       comm);

  cs_gnum_t *recv_tuples = cs_all_to_all_copy_array(d_periodic,
                                                    3,
                                                    false, /* reverse */
                                                    send_tuples);

  cs_lnum_t _n_block_tuples = cs_all_to_all_n_elts_dest(d_periodic);

  CS_FREE(send_tuples);

  cs_all_to_all_destroy(&d_periodic);

  CS_FREE(periodic_block_rank);

  /* Sort periodic couples by local correspondant, remove duplicates */

  _sort_periodic_tuples(&_n_block_tuples,
                        &recv_tuples);

  /* Set return values */

  *n_block_tuples = _n_block_tuples;
  *block_tuples = recv_tuples;
}

/*----------------------------------------------------------------------------
 * Associate block ids for periodic couples.
 *
 * If a global number appears multiple times in a block, the lowest
 * occurence id is returned.
 *
 * parameters:
 *   n_block_elements <-- number of elements in block
 *   order            <-- block ordering by global number received
 *   block_global_num <-- global numbering received
 *   n_block_couples  <-- number of couples in current block
 *   stride           <-- stride for global number of local element
 *                        in block_couples[]
 *   block_couples    <-- couple information for block: for each couple,
 *                        {global number of local element,
 *                         global number of periodic element,
 *                         transform id} if stride = 3,
 *                        {global number of local element} if stride = 1
 *   couple_block_id  --> id in block of local couple element
 *   comm             <-- associated MPI communicator
 *----------------------------------------------------------------------------*/

static void
_periodic_couples_block_id(cs_lnum_t          n_block_elements,
                           const cs_lnum_t    order[],
                           const cs_gnum_t    block_global_num[],
                           cs_lnum_t          n_block_couples,
                           int                stride,
                           const cs_gnum_t    block_couples[],
                           cs_lnum_t          couple_block_id[])
{
  cs_lnum_t    couple_id;

  /* Initialization */

  assert(stride == 3 || stride == 1);

  if (n_block_couples == 0)
    return;

  /* Use binary search */

  for (couple_id = 0; couple_id < n_block_couples; couple_id++) {

    cs_gnum_t num_cmp;
    cs_lnum_t start_id = 0;
    cs_lnum_t end_id = n_block_elements - 1;
    cs_lnum_t mid_id = (end_id -start_id) / 2;

    const cs_gnum_t num_1 = block_couples[couple_id*stride];

    /* use binary search */

    while (start_id <= end_id) {
      num_cmp = block_global_num[order[mid_id]];
      if (num_cmp < num_1)
        start_id = mid_id + 1;
      else if (num_cmp > num_1)
        end_id = mid_id - 1;
      else
        break;
      mid_id = start_id + ((end_id -start_id) / 2);
    }

    /* In case of multiple occurences, find lowest one */

    while (mid_id > 0 &&  block_global_num[order[mid_id-1]] == num_1)
      mid_id--;

    assert(block_global_num[order[mid_id]] == num_1);

    couple_block_id[couple_id] = order[mid_id];
  }

}

/*----------------------------------------------------------------------------
 * Exchange periodic couple info between processors providing the data
 * and processors handling the related global numbering interval blocks.
 *
 * _periodic_couples_block_id() should have been called first.
 *
 * parameters:
 *   block_size       <-- size of the block handled by each processor
 *   equiv_id         <-- equivalence id for each block element (-1 if none)
 *   equiv            <-- temporary equivalence structure for block
 *   n_block_couples  <-- number of couples in current block
 *   block_couples    <-- couple information for block: for each couple,
 *                        {global number of local element,
 *                         global number of periodic element,
 *                         transform id}
 *   couple_block_id  <-- local id in block
 *   dest_rank        --> local number of values to send to each rank
 *   src index        --> local number of values to receive from each rank
 *----------------------------------------------------------------------------*/

static void
_count_periodic_equiv_exchange(size_t                     block_size,
                               const cs_lnum_t            equiv_id[],
                               const _per_block_equiv_t  *equiv,
                               cs_lnum_t                  n_block_couples,
                               const cs_gnum_t            block_couples[],
                               const cs_lnum_t            couple_block_id[],
                               int                        dest_rank[],
                               cs_lnum_t                  src_index[])
{
  src_index[0] = 0;

  if (equiv != nullptr && equiv_id != nullptr) {

    /* Compute list sizes to send to distant processors */

    for (cs_lnum_t couple_id = 0; couple_id < n_block_couples; couple_id++) {

      int e_mult;
      cs_gnum_t num_2 = block_couples[couple_id*3 + 1];
      cs_lnum_t e_id = equiv_id[couple_block_id[couple_id]];

      if (e_id > -1)
        e_mult = equiv->shift[e_id +1] - equiv->shift[e_id];
      else
        e_mult = 1;

      dest_rank[couple_id] = (num_2 - 1) / block_size;
      src_index[couple_id+1] = src_index[couple_id] + 3 + 2*e_mult;

    }

  }
  else { /* if (equiv == nullptr || equiv_id == nullptr) */

    for (cs_lnum_t couple_id = 0; couple_id < n_block_couples; couple_id++) {

      cs_gnum_t num_2 = block_couples[couple_id*3 + 1];

      dest_rank[couple_id] = (num_2 - 1) / block_size;
      src_index[couple_id+1] = src_index[couple_id] + 5;

    }

  }
}

/*----------------------------------------------------------------------------
 * Exchange periodic couple info between processors providing the data
 * and processors handling the related global numbering interval blocks.
 *
 * parameters:
 *   block_size       <-- size of the block handled by each processor
 *   n_block_elements <-- number of elements in local block
 *   src_rank         <-- source rank (size: block_size)
 *   block_global_num <-- global numbering received
 *   block_num        <-- local numbering received
 *   equiv_id         <-- equivalence id for each block element (-1 if none)
 *   equiv            <-- temporary equivalence structure for block
 *   periodicity      <-- periodicity information (nullptr if none)
 *   n_block_couples  <-- number of couples in current block
 *   block_couples    <-- couple information for block: for each couple,
 *                        {global number of local element,
 *                         global number of periodic element,
 *                         transform id}
 *   comm             <-- associated MPI communicator
 *
 * returns:
 *   structure defining a temporary list of periodic interfaces
 *----------------------------------------------------------------------------*/

static _per_block_period_t
_exchange_periodic_equiv(size_t                     block_size,
                         cs_lnum_t                  n_block_elements,
                         const int                  src_rank[],
                         const cs_gnum_t            block_global_num[],
                         const cs_lnum_t            block_num[],
                         const cs_lnum_t            equiv_id[],
                         const _per_block_equiv_t  *equiv,
                         const fvm_periodicity_t   *periodicity,
                         cs_lnum_t                  n_block_couples,
                         const cs_gnum_t            block_couples[],
                         MPI_Comm                   comm)
{
  cs_lnum_t *couple_block_id = nullptr;
  int *reverse_tr_id = nullptr;
  cs_lnum_t *order = nullptr;
  cs_gnum_t *block_recv_num = nullptr;

  _per_block_period_t pe;

  /* Initialize return structure */

  pe.count = 0;
  pe.block_id = nullptr;
  pe.tr_id = nullptr;
  pe.shift = nullptr;
  pe.rank  = nullptr;
  pe.num   = nullptr;

  if (periodicity == nullptr)
    return pe;

  /* Build ordering array for binary search */

  order = cs_order_gnum(nullptr, block_global_num, n_block_elements);

  /* Associate id in block for periodic couples prior to sending */

  CS_MALLOC(couple_block_id, n_block_couples, cs_lnum_t);

  _periodic_couples_block_id(n_block_elements,
                             order,
                             block_global_num,
                             n_block_couples,
                             3,
                             block_couples,
                             couple_block_id);

  /* build count and shift arrays for parallel exchange */

  int  *send_rank;
  cs_lnum_t  *src_index;
  CS_MALLOC(send_rank, n_block_couples, int);
  CS_MALLOC(src_index, n_block_couples+1, cs_lnum_t);

  _count_periodic_equiv_exchange(block_size,
                                 equiv_id,
                                 equiv,
                                 n_block_couples,
                                 block_couples,
                                 couple_block_id,
                                 send_rank,
                                 src_index);

  cs_all_to_all_t  *d = cs_all_to_all_create(n_block_couples,
                                             0,
                                             nullptr,
                                             send_rank,
                                             comm);

  cs_all_to_all_transfer_dest_rank(d, &send_rank);

  cs_lnum_t *dest_index = cs_all_to_all_copy_index(d,
                                                   false, /* reverse */
                                                   src_index,
                                                   nullptr);

  /* arrays to exchange; most of the exchanged data is of type int or
     cs_lnum_t, using only positive values. For each periodic couple,
     one value of type cs_gnum_t is exchanged, so to group all communication
     in one MPI call, all data is exchanged as cs_gnum_t, even if this
     means larger messages if cs_gnum_t is larger than cs_lnum_t.
     As the number of elements per couple is variable (depending on prior
     equivalence info), using an MPI datatype to mix int and cs_gnum_t
     types rather than casting all to cs_gnum_t is not feasible */

  cs_gnum_t *equiv_send;
  CS_MALLOC(equiv_send, src_index[n_block_couples], cs_gnum_t);

  /* temporary array to find reverse transforms */

  int n_tr = fvm_periodicity_get_n_transforms(periodicity);

  CS_MALLOC(reverse_tr_id, n_tr, int);

  for (int tr_id = 0; tr_id < n_tr; tr_id++)
    reverse_tr_id[tr_id] = fvm_periodicity_get_reverse_id(periodicity, tr_id);

  if (equiv != nullptr && equiv_id != nullptr) {

    /* Compute list sizes to send to distant processors */

    for (cs_lnum_t couple_id = 0; couple_id < n_block_couples; couple_id++) {

      const cs_gnum_t num_2 = block_couples[couple_id*3 + 1];
      const cs_lnum_t local_id = couple_block_id[couple_id];
      const cs_lnum_t e_id = equiv_id[local_id];

      cs_lnum_t i = src_index[couple_id];

      if (e_id > -1) {

        int j_start = equiv->shift[e_id];
        int j_end = equiv->shift[e_id + 1];

        equiv_send[i++] = j_end - j_start;
        equiv_send[i++] = num_2;
        equiv_send[i++] = reverse_tr_id[block_couples[couple_id*3 + 2]];

        for (int j = j_start; j < j_end; j++) {
          equiv_send[i++] = equiv->rank[j];
          equiv_send[i++] = equiv->num[j];
        }

      }
      else {

        equiv_send[i++] = 1;
        equiv_send[i++] = num_2;
        equiv_send[i++] = reverse_tr_id[block_couples[couple_id*3 + 2]];
        equiv_send[i++] = src_rank[local_id];
        equiv_send[i++] = block_num[local_id];

      }

    }

  }
  else { /* if (equiv == nullptr || equiv_id == nullptr) */

    for (cs_lnum_t couple_id = 0; couple_id < n_block_couples; couple_id++) {

      const cs_gnum_t num_2 = block_couples[couple_id*3 + 1];
      const cs_lnum_t local_id = couple_block_id[couple_id];

      cs_lnum_t i = src_index[couple_id];

      equiv_send[i++] = 1;
      equiv_send[i++] = num_2;
      equiv_send[i++] = reverse_tr_id[block_couples[couple_id*3 + 2]];
      equiv_send[i++] = src_rank[local_id];
      equiv_send[i++] = block_num[local_id];

    }

  }

  CS_FREE(couple_block_id);
  CS_FREE(reverse_tr_id);

  /* Parallel exchange */

  cs_gnum_t *equiv_recv = cs_all_to_all_copy_indexed(d,
                                                     false, /* reverse */
                                                     src_index,
                                                     equiv_send,
                                                     dest_index);

  cs_lnum_t  n_elts_recv = cs_all_to_all_n_elts_dest(d);
  size_t  recv_size = dest_index[n_elts_recv];

  /* Free memory */

  CS_FREE(src_index);
  CS_FREE(dest_index);
  CS_FREE(equiv_send);

  cs_all_to_all_destroy(&d);

  /* Build return structure */

  {
    size_t k, l, e_mult;

    pe.count = 0;
    size_t i = 0;
    size_t j = 0;

    while (i < recv_size) {
      pe.count += 1;
      j += equiv_recv[i];
      i += 3 + 2*equiv_recv[i];
    }

    CS_MALLOC(block_recv_num, pe.count, cs_gnum_t);

    CS_MALLOC(pe.tr_id, pe.count, int);

    CS_MALLOC(pe.shift, pe.count + 1, int);

    CS_MALLOC(pe.rank, j, int);
    CS_MALLOC(pe.num, j, cs_lnum_t);

    pe.shift[0] = 0;

    for (i = 0, j = 0, k = 0, l = 0; i < (size_t)(pe.count); i++) {

      e_mult = equiv_recv[k++];
      block_recv_num[i] = equiv_recv[k++];
      pe.tr_id[i] = equiv_recv[k++];

      for (l = 0; l < e_mult; l++) {
        pe.rank[j] = equiv_recv[k++];
        pe.num[j] = equiv_recv[k++];
        pe.shift[i+1] = ++j;
      }

    }

  }

  CS_FREE(equiv_recv);

  /* Associate id in block for received periodic equivalences */

  CS_MALLOC(pe.block_id, pe.count, cs_lnum_t);

  _periodic_couples_block_id(n_block_elements,
                             order,
                             block_global_num,
                             pe.count,
                             1,
                             block_recv_num,
                             pe.block_id);

  /* Free remaining working arrays */

  CS_FREE(block_recv_num);
  CS_FREE(couple_block_id);
  CS_FREE(order);

  return pe;
}

/*----------------------------------------------------------------------------
 * Merge periodic equivalent interface info with block equivalence info.
 *
 * Expands block equivalence info, and frees temporary list of periodic
 * interfaces.
 *
 * parameters:
 *   n_block_elts <-- number of block elements
 *   src_rank     <-- source rank
 *   block_num    <-- local numbering received
 *   equiv_id     <-> equivalence id for each block element (-1 if none)
 *   equiv        <-> temporary equivalence structure for block
 *   perio_equiv  <-> temporary list of periodic interfaces
 *
 * returns:
 *   structure defining a temporary equivalence structure
 *----------------------------------------------------------------------------*/

#if defined(__INTEL_COMPILER)
#pragma optimization_level 2 /* Crash with O3 on IA64 with icc 9.1 20070320 */
#endif

static void
_merge_periodic_equiv(cs_lnum_t             n_block_elts,
                      const int             src_rank[],
                      const cs_lnum_t       block_num[],
                      cs_lnum_t             equiv_id[],
                      _per_block_equiv_t   *equiv,
                      _per_block_period_t  *perio_equiv)
{
  cs_lnum_t i;
  size_t j;

  cs_lnum_t  old_count, new_count;
  size_t new_size;

  cs_lnum_t  *eq_mult = nullptr;
  cs_lnum_t  *new_shift = nullptr;

  _per_block_period_t *pe = perio_equiv;

  assert(equiv != nullptr);

  if (perio_equiv == nullptr)
    return;

  old_count = equiv->count;

  /* Note that by construction, the global numbers of elements appearing
     in the original (parallel) equivalence must appear multiple times
     in the block (the original equivalences are built using this), and
     the global numbers of elements appearing only in the periodic
     equivalence must appear only once; thus only one equiv_id[] value
     needs to be updated when appending purely periodic equivalences */

  for (i = 0, new_count = old_count; i < pe->count; i++) {
    if (equiv_id[pe->block_id[i]] == -1)
      equiv_id[pe->block_id[i]] = new_count++;
  }

  CS_MALLOC(eq_mult, new_count, cs_lnum_t);

  for (i = 0; i < old_count; i++)
    eq_mult[i] = equiv->shift[i+1] - equiv->shift[i];

  for (i = old_count; i < new_count; i++)
    eq_mult[i] = 0;

  for (i = 0; i < pe->count; i++) {
    if (eq_mult[equiv_id[pe->block_id[i]]] == 0)
      eq_mult[equiv_id[pe->block_id[i]]] += pe->shift[i+1] - pe->shift[i] + 1;
    else
      eq_mult[equiv_id[pe->block_id[i]]] += pe->shift[i+1] - pe->shift[i];
  }

  /* Build new (merged) index, resetting eq_mult to use as a counter */

  CS_MALLOC(new_shift, new_count+1, cs_lnum_t);

  new_shift[0] = 0;

  for (i = 0; i < new_count; i++) {
    assert(eq_mult[i] > 0);
    new_shift[i+1] = new_shift[i] + eq_mult[i];
    eq_mult[i] = 0;
  }

  new_size = new_shift[new_count];

  /* Expand previous periodicity info */
  /*----------------------------------*/

  equiv->count = new_count;

  if (old_count > 0) {

    int k;
    int *new_rank = nullptr;
    cs_lnum_t *new_num = nullptr;

    CS_MALLOC(new_rank, new_size, int);

    for (i = 0; i < old_count; i++) {
      eq_mult[i] = equiv->shift[i+1] - equiv->shift[i];
      for (k = 0; k < eq_mult[i]; k++)
        new_rank[new_shift[i] + k] = equiv->rank[equiv->shift[i] + k];
    }

    CS_FREE(equiv->rank);
    equiv->rank = new_rank;

    CS_MALLOC(new_num, new_size, cs_lnum_t);

    for (i = 0; i < old_count; i++) {
      for (k = 0; k < eq_mult[i]; k++)
        new_num[new_shift[i] + k] = equiv->num[equiv->shift[i] + k];
    }
    CS_FREE(equiv->num);
    equiv->num = new_num;

    if (equiv->tr_id != nullptr) {

      int *new_tr_id = nullptr;
      CS_MALLOC(new_tr_id, new_size, int);
      for (i = 0; i < old_count; i++) {
        for (k = 0; k < eq_mult[i]; k++)
          new_tr_id[new_shift[i] + k] = equiv->tr_id[equiv->shift[i] + k];
      }
      CS_FREE(equiv->tr_id);
      equiv->tr_id = new_tr_id;

    }

    /* All is expanded at this stage, so old index may be replaced */

    CS_FREE(equiv->shift);
    equiv->shift = new_shift;

  }
  else {

    CS_FREE(equiv->shift);
    equiv->shift = new_shift;
    CS_MALLOC(equiv->rank, new_size, int);
    CS_MALLOC(equiv->num, new_size, cs_lnum_t);

  }

  if (equiv->tr_id == nullptr) {
    CS_MALLOC(equiv->tr_id, new_size, int);
    for (j = 0; j < new_size; j++)
      equiv->tr_id[j] = 0;
  }

  /* Now insert periodic equivalence info */

  for (cs_lnum_t k = 0; k < n_block_elts; k++) {

    if (equiv_id[k] >= old_count) {

      const cs_lnum_t eq_id = equiv_id[k];
      const cs_lnum_t l = equiv->shift[eq_id];

      assert(eq_mult[eq_id] == 0);
      equiv->rank[l] = src_rank[k];
      equiv->num[l] = block_num[k];
      equiv->tr_id[l] = 0;
      eq_mult[eq_id] = 1;

    }
  }

  for (i = 0; i < pe->count; i++) {

    int k, l;
    const cs_lnum_t _block_id = pe->block_id[i];
    const int eq_id = equiv_id[_block_id];

    for (k = pe->shift[i]; k < pe->shift[i+1]; k++) {

      l = equiv->shift[eq_id] + eq_mult[eq_id];

      equiv->rank[l] = pe->rank[k];
      equiv->num[l] = pe->num[k];

      equiv->tr_id[l] = pe->tr_id[i] + 1;

      eq_mult[eq_id] += 1;

    }

  }

  CS_FREE(eq_mult);

  /* Free temporary periodic equivalence structure elements */

  CS_FREE(pe->block_id);
  CS_FREE(pe->tr_id);
  CS_FREE(pe->shift);
  CS_FREE(pe->rank);
  CS_FREE(pe->num);
}

/*----------------------------------------------------------------------------
 * Creation of a list of interfaces between elements of a same type.
 *
 * parameters:
 *   ifs                 <-> pointer to structure that should be updated
 *   n_elts              <-- local number of elements
 *   global_num          <-- global number (id) associated with each element
 *   periodicity         <-- periodicity information (nullptr if none)
 *   n_periodic_lists    <-- number of periodic lists (may be local)
 *   periodicity_num     <-- periodicity number (1 to n) associated with
 *                           each periodic list (primary periodicities only)
 *   n_periodic_couples  <-- number of periodic couples associated with
 *                           each periodic list
 *   periodic_couples    <-- array indicating periodic couples (using
 *                           global numberings) for each list
 *   comm                <-- associated MPI communicator
 *----------------------------------------------------------------------------*/

static void
_add_global_equiv_periodic(cs_interface_set_t       *ifs,
                           cs_lnum_t                 n_elts,
                           const cs_gnum_t           global_num[],
                           const fvm_periodicity_t  *periodicity,
                           int                       n_periodic_lists,
                           const int                 periodicity_num[],
                           const cs_lnum_t           n_periodic_couples[],
                           const cs_gnum_t    *const periodic_couples[],
                           MPI_Comm                  comm)
{
  cs_lnum_t   n_block_couples = 0;
  cs_lnum_t   *equiv_id = nullptr, *couple_equiv_id = nullptr;
  cs_gnum_t   *block_couples = nullptr;

  /* Initialization */

  int  size, local_rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &local_rank);

  /* Get temporary maximum global number value */

  cs_gnum_t global_max = _global_num_max(n_elts,
                                         global_num,
                                         comm);

  cs_block_dist_info_t
    bi = cs_block_dist_compute_sizes(local_rank,
                                     size,
                                     1,
                                     0,
                                     global_max);

  int flags = CS_ALL_TO_ALL_NEED_SRC_RANK;

  cs_all_to_all_t
    *d = cs_all_to_all_create_from_block(n_elts,
                                         flags,
                                         global_num,
                                         bi,
                                         comm);

  assert(sizeof(cs_gnum_t) >= sizeof(cs_lnum_t));

  cs_gnum_t *recv_global_num = cs_all_to_all_copy_array(d,
                                                        1,
                                                        false, /* reverse */
                                                        global_num);

  cs_lnum_t  *send_num = nullptr;
  CS_MALLOC(send_num, n_elts, cs_lnum_t);
  for (cs_lnum_t i = 0; i < n_elts; i++)
    send_num[i] = i+1;

  cs_lnum_t *recv_num = cs_all_to_all_copy_array(d,
                                                 1,
                                                 false, /* reverse */
                                                 send_num);

  CS_FREE(send_num);

  cs_lnum_t n_elts_recv = cs_all_to_all_n_elts_dest(d);

  int *src_rank = cs_all_to_all_get_src_rank(d);

  /* Exchange periodicity information */

  _exchange_periodic_tuples(bi.block_size,
                            periodicity,
                            n_periodic_lists,
                            periodicity_num,
                            n_periodic_couples,
                            periodic_couples,
                            &n_block_couples,
                            &block_couples,
                            comm);

  /* Combine periodic couples if necessary */

  if (fvm_periodicity_get_n_levels(periodicity) > 1)
    _combine_periodic_tuples(bi.block_size,
                             periodicity,
                             &n_block_couples,
                             &block_couples,
                             comm);

  /* Build purely parallel equivalence data first */

  if (n_elts_recv > 0)
    CS_MALLOC(equiv_id, n_elts_recv, cs_lnum_t);

  _per_block_equiv_t
    e = _block_global_num_to_equiv(n_elts_recv,
                                   src_rank,
                                   recv_global_num,
                                   recv_num,
                                   equiv_id);

  /* Now combine periodic and parallel equivalences */

  _per_block_period_t
    pe = _exchange_periodic_equiv(bi.block_size,
                                  n_elts_recv,
                                  src_rank,
                                  recv_global_num,
                                  recv_num,
                                  equiv_id,
                                  &e,
                                  periodicity,
                                  n_block_couples,
                                  block_couples,
                                  comm);

  CS_FREE(recv_global_num);

  _merge_periodic_equiv(n_elts_recv,
                        src_rank,
                        recv_num,
                        equiv_id,
                        &e,
                        &pe);

  /* Free all arrays not needed anymore */

  CS_FREE(recv_num);

  CS_FREE(couple_equiv_id);
  CS_FREE(block_couples);
  n_block_couples = 0;

  /* Now that equivalences are marked, count for each rank; for each
     equivalence, we will need to send the corresponding element
     numbers, ranks, and transform ids,
     for a total of 2 + 3*(e.multiple[...] - 1)
     = (3*e.multiple[...]) - 1 values. */

  cs_lnum_t  *block_index;
  CS_MALLOC(block_index, n_elts_recv+1, cs_lnum_t);

  block_index[0] = 0;
  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    cs_lnum_t n_eq = 0;
    if (equiv_id[i] > -1) {
      size_t e_id = equiv_id[i];
      n_eq = 3*(e.shift[e_id+1] - e.shift[e_id]) - 1;
    }
    block_index[i+1] = block_index[i] + n_eq;
  }

  cs_lnum_t *part_index = cs_all_to_all_copy_index(d,
                                                   true, /* reverse */
                                                   block_index,
                                                   nullptr);

  /* Now prepare new send buffer */

  cs_lnum_t  *block_equiv;
  CS_MALLOC(block_equiv, block_index[n_elts_recv], cs_lnum_t);

  for (cs_lnum_t i = 0; i < n_elts_recv; i++) {
    if (equiv_id[i] > -1) {

      cs_lnum_t *block_equiv_p = block_equiv + block_index[i];

      cs_lnum_t        e_id = equiv_id[i];
      cs_lnum_t        k = 2;
      const int        multiple = e.shift[e_id+1] - e.shift[e_id];
      const int       *rank_p = e.rank + e.shift[e_id];
      const int       *tr_id_p = e.tr_id + e.shift[e_id];
      const cs_lnum_t *num_p  = e.num  + e.shift[e_id];

      for (cs_lnum_t j = 0; j < multiple; j++) {
        if (rank_p[j] == src_rank[i] && tr_id_p[j] == 0) {
          block_equiv_p[0] = num_p[j];
          block_equiv_p[1] = multiple - 1;
        }
        else {
          block_equiv_p[k++] = num_p[j];
          block_equiv_p[k++] = tr_id_p[j];
          block_equiv_p[k++] = rank_p[j];
        }
      }

    }
  }

  /* Free temporary (block) equivalence info */

  e.count = 0;
  CS_FREE(e.shift);
  CS_FREE(e.rank);
  CS_FREE(e.tr_id);
  CS_FREE(e.num);

  CS_FREE(src_rank);
  CS_FREE(equiv_id);

  cs_lnum_t *part_equiv = cs_all_to_all_copy_indexed(d,
                                                     true, /* reverse */
                                                     block_index,
                                                     block_equiv,
                                                     part_index);

  cs_lnum_t n_vals_part = part_index[n_elts];

  /* At this stage, MPI operations are finished; we may release
     the corresponding counts and indexes */

  CS_FREE(block_equiv);
  CS_FREE(part_index);
  CS_FREE(block_index);

  cs_all_to_all_destroy(&d);

  /* Add interface */

  int tr_index_size = fvm_periodicity_get_n_transforms(periodicity) + 2;

  _interfaces_from_flat_equiv(ifs,
                              tr_index_size,
                              n_vals_part,
                              part_equiv);

  CS_FREE(part_equiv);
}

#endif /* defined(HAVE_MPI) */

/*----------------------------------------------------------------------------
 * Prepare periodic couple info in single process mode.
 *
 * Note that the array pointed to by block_couples is allocated here,
 * and must be freed by the calling code.
 *
 * parameters:
 *   periodicity        <-- periodicity information (nullptr if none)
 *   n_periodic_lists   <-- number of periodic lists (may be local)
 *   periodicity_num    <-- periodicity number (1 to n) associated with
 *                          each periodic list (primary periodicities only)
 *   n_periodic_couples <-- number of periodic couples associated with
 *                          each periodic list
 *   periodic_couples   <-- array indicating periodic couples (using
 *                          global numberings) for each list
 *   n_couples          --> number of couples
 *   couples            --> couple information: for each couple,
 *                          {global number of local element,
 *                           global number of periodic element,
 *                           transform id}
 *----------------------------------------------------------------------------*/

static void
_define_periodic_couples_sp(const fvm_periodicity_t  *periodicity,
                            int                       n_periodic_lists,
                            const int                 periodicity_num[],
                            const cs_lnum_t           n_periodic_couples[],
                            const cs_gnum_t    *const periodic_couples[],
                            cs_lnum_t                *n_couples,
                            cs_gnum_t               **couples)
{
  int         list_id;

  cs_lnum_t   count = 0;
  cs_lnum_t    _n_couples = 0;
  cs_gnum_t   *_couples = nullptr;

  /* Initialization */

  *n_couples = 0;
  *couples = nullptr;

  for (list_id = 0, _n_couples = 0; list_id < n_periodic_lists; list_id++)
    _n_couples += n_periodic_couples[list_id] * 2;

  CS_MALLOC(_couples, _n_couples*3, cs_gnum_t);

  /* Prepare lists */

  for (list_id = 0; list_id < n_periodic_lists; list_id++) {

    cs_lnum_t couple_id;
    cs_gnum_t num_1, num_2;

    const int external_num = periodicity_num[list_id];
    const int direct_id = fvm_periodicity_get_transform_id(periodicity,
                                                           external_num,
                                                           1);
    const int reverse_id = fvm_periodicity_get_transform_id(periodicity,
                                                            external_num,
                                                            -1);

    const cs_lnum_t   _n_periodic_couples = n_periodic_couples[list_id];
    const cs_gnum_t   *_periodic_couples = periodic_couples[list_id];

    assert(direct_id >= 0 && reverse_id >= 0);

    for (couple_id = 0; couple_id < _n_periodic_couples; couple_id++) {

      num_1 = _periodic_couples[couple_id*2];
      num_2 = _periodic_couples[couple_id*2 + 1];

      _couples[count] = num_1;
      _couples[count+1] = num_2;
      _couples[count+2] = direct_id;

      _couples[count+3] = num_2;
      _couples[count+4] = num_1;
      _couples[count+5] = reverse_id;

      count += 6;

    }

  }

  /* Sort periodic couples by local match, remove duplicates */

  _sort_periodic_tuples(&_n_couples, &_couples);

  /* Set return values */

  *n_couples = _n_couples;
  *couples = _couples;
}

/*----------------------------------------------------------------------------
 * Build eventual couples belonging to combined periodicities in single
 * process mode.
 *
 * parameters:
 *   periodicity <-- periodicity information (nullptr if none)
 *   n_couples   <-> number of couples in current block
 *   couples     <-> couple information for this rank: for each couple,
 *                   {global number of local element,
 *                    global number of periodic element,
 *                    transform id}
 *----------------------------------------------------------------------------*/

static void
_combine_periodic_couples_sp(const fvm_periodicity_t   *periodicity,
                             cs_lnum_t                 *n_couples,
                             cs_gnum_t                **couples)
{
  int  i, n_tr, level;

  cs_lnum_t   start_id, end_id, j, k;

  int  add_count = 0;
  int  *tr_reverse_id = nullptr;
  cs_gnum_t *_add_couples = nullptr;

  cs_lnum_t   _n_couples = *n_couples;
  cs_gnum_t   *_couples = *couples;

  assert (periodicity != nullptr);

  /* Build periodicity related arrays for quick access */

  n_tr = fvm_periodicity_get_n_transforms(periodicity);

  CS_MALLOC(tr_reverse_id, n_tr, int);

  for (i = 0; i < n_tr; i++)
    tr_reverse_id[i] = fvm_periodicity_get_reverse_id(periodicity, i);

  /* Loop on combination levels */

  for (level = 1;
       level < fvm_periodicity_get_n_levels(periodicity);
       level++) {

    int  n_rows = 0;
    int  *tr_combine = nullptr;

    _transform_combine_info(periodicity,
                            level,
                            &n_rows,
                            &tr_combine);

    /* Count values to add */

    add_count = 0;

    start_id = 0;
    end_id = 1;

    while (end_id < _n_couples) {

      if (_couples[start_id*3] == _couples[end_id*3]) {

        end_id++;
        while (end_id < _n_couples) {
          if (_couples[end_id*3] != _couples[start_id*3])
            break;
          end_id++;
        }

        for (j = start_id; j < end_id; j++) {
          for (k = j+1; k < end_id; k++) {

            int tr_1 = tr_reverse_id[_couples[j*3 + 2]];
            int tr_2 = _couples[k*3 + 2];

            if (tr_combine[tr_1 *n_rows + tr_2] > -1)
              add_count += 6;

          }
        }

      }

      start_id = end_id;
      end_id += 1;
    }

    /* Nothing to do for this combination level if add_count = 0 */

    if (add_count == 0) {
      CS_FREE(tr_combine);
      continue;
    }

    CS_REALLOC(_couples, _n_couples*3 + add_count, cs_gnum_t);

    _add_couples = _couples + _n_couples*3;

    /* Now add combined couples */

    start_id = 0;
    end_id = 1;

    while (end_id < _n_couples) {

      if (_couples[start_id*3] == _couples[end_id*3]) {

        end_id++;
        while (end_id < _n_couples) {
          if (_couples[end_id*3] != _couples[start_id*3])
            break;
          end_id++;
        }

        /* Loop on couple combinations */

        for (j = start_id; j < end_id; j++) {
          for (k = j+1; k < end_id; k++) {

            cs_gnum_t num_1 = _couples[j*3 + 1];
            cs_gnum_t num_2 = _couples[k*3 + 1];
            int tr_1 = tr_reverse_id[_couples[j*3 + 2]];
            int tr_2 = _couples[k*3 + 2];
            int tr_c = tr_combine[tr_1 *n_rows + tr_2];

            if (tr_c > -1) {

              _add_couples[0] = num_1;
              _add_couples[1] = num_2;
              _add_couples[2] = tr_c;

              _add_couples[3] = num_2;
              _add_couples[4] = num_1;
              _add_couples[5] = tr_reverse_id[tr_c];

              _add_couples += 6;

            }

          }
        } /* End of loop on couple combinations */

      }

      start_id = end_id;
      end_id += 1;
    }

    CS_FREE(tr_combine);

    /* Finally, merge additional couples received for block
       existing periodicity information */

    assert(add_count % 3 == 0);

    _n_couples += add_count / 3;

    /* Sort and remove duplicates to update periodicity info */

    _sort_periodic_tuples(&_n_couples, &_couples);

    *n_couples = _n_couples;
    *couples = _couples;

  }

  CS_FREE(tr_reverse_id);
}

/*----------------------------------------------------------------------------
 * Creation of a list of interfaces between elements of a same type.
 *
 * This simplified algorithm is intended for single-process mode.
 *
 * parameters:
 *   ifs                <-> pointer to structure that should be updated
 *   n_elts              <-- local number of elements
 *   global_num          <-- global number (id) associated with each element
 *   periodicity        <-- periodicity information (nullptr if none)
 *   n_periodic_lists   <-- number of periodic lists (may be local)
 *   periodicity_num    <-- periodicity number (1 to n) associated with
 *                          each periodic list (primary periodicities only)
 *   n_periodic_couples <-- number of periodic couples associated with
 *                          each periodic list
 *   periodic_couples   <-- array indicating periodic couples (using
 *                          global numberings) for each list
 *----------------------------------------------------------------------------*/

static void
_add_global_equiv_periodic_sp(cs_interface_set_t       *ifs,
                              cs_lnum_t                 n_elts,
                              const cs_gnum_t           global_num[],
                              const fvm_periodicity_t  *periodicity,
                              int                       n_periodic_lists,
                              const int                 periodicity_num[],
                              const cs_lnum_t           n_periodic_couples[],
                              const cs_gnum_t    *const periodic_couples[])
{
  cs_lnum_t   i, couple_id;
  cs_lnum_t   n_couples = 0;
  cs_lnum_t   *n_elts_tr = nullptr;
  cs_gnum_t   *couples = nullptr;

  cs_interface_t  *_interface = nullptr;

  assert(sizeof(cs_gnum_t) >= sizeof(cs_lnum_t));

  /* Organize periodicity information */

  _define_periodic_couples_sp(periodicity,
                              n_periodic_lists,
                              periodicity_num,
                              n_periodic_couples,
                              periodic_couples,
                              &n_couples,
                              &couples);

  /* Combine periodic couples if necessary */

  if (fvm_periodicity_get_n_levels(periodicity) > 1)
    _combine_periodic_couples_sp(periodicity,
                                 &n_couples,
                                 &couples);

  /* Add interface to set */

  ifs->size += 1;

  CS_REALLOC(ifs->interfaces,
             ifs->size,
             cs_interface_t *);

  _interface = _cs_interface_create();

  ifs->interfaces[ifs->size - 1] = _interface;

  /* Build interface */

  _interface->rank = 0;
  _interface->size = n_couples;

  _interface->tr_index_size
    = fvm_periodicity_get_n_transforms(periodicity) + 2;

  CS_MALLOC(_interface->tr_index, _interface->tr_index_size, cs_lnum_t);
  CS_MALLOC(_interface->elt_id, _interface->size, cs_lnum_t);
  CS_MALLOC(_interface->match_id, _interface->size, cs_lnum_t);

  /* Count couples for each transform */

  CS_MALLOC(n_elts_tr, _interface->tr_index_size - 2, cs_lnum_t);

  for (i = 0; i < _interface->tr_index_size - 2; i++)
    n_elts_tr[i] = 0;

  for (couple_id = 0; couple_id < n_couples; couple_id++)
    n_elts_tr[couples[couple_id*3 + 2]] += 1;

  /* Build index */

  _interface->tr_index[0] = 0;
  _interface->tr_index[1] = 0;

  for (i = 2; i < _interface->tr_index_size; i++) {
    _interface->tr_index[i] = _interface->tr_index[i-1] + n_elts_tr[i-2];
    n_elts_tr[i-2] = 0;
  }

  /* Build local and distant correspondants */

  if (global_num == nullptr) {

    for (couple_id = 0; couple_id < n_couples; couple_id++) {

      const int tr_id = couples[couple_id*3 + 2];
      const cs_lnum_t j = _interface->tr_index[tr_id + 1] + n_elts_tr[tr_id];

      _interface->elt_id[j] = couples[couple_id*3] - 1;
      _interface->match_id[j] = couples[couple_id*3 + 1] - 1;

      n_elts_tr[tr_id] += 1;

    }

  }
  else {

    cs_lnum_t *renum;
    CS_MALLOC(renum, n_elts, cs_lnum_t);
    for (i = 0; i < n_elts; i++) {
      cs_lnum_t j = global_num[i] - 1;
      assert(j >= 0 && j < n_elts);
      renum[j] = i;
    }

    for (couple_id = 0; couple_id < n_couples; couple_id++) {

      const int tr_id = couples[couple_id*3 + 2];
      const cs_lnum_t j = _interface->tr_index[tr_id + 1] + n_elts_tr[tr_id];

      _interface->elt_id[j] = renum[couples[couple_id*3] - 1];
      _interface->match_id[j] = renum[couples[couple_id*3 + 1] - 1];

      n_elts_tr[tr_id] += 1;

    }

    CS_FREE(renum);
  }

  /* Free temporary arrays */

  CS_FREE(n_elts_tr);
  CS_FREE(couples);
}

/*----------------------------------------------------------------------------
 * Order element id lists and update matching id lists for each
 * section of each interface of a set.
 *
 * This must be called as part of a renumbering algorithm, before
 * matching element ids may be replaced by send orderings.
 *
 * parameters:
 *   ifs <-> pointer to interface set
 *----------------------------------------------------------------------------*/

static void
_order_elt_id(cs_interface_set_t  *ifs)
{
  int i;

  for (i = 0; i < ifs->size; i++) {

    int section_id;
    cs_lnum_t j;

    int  tr_index_size = 2;
    cs_lnum_t  _tr_index[2] = {0, 0};
    cs_lnum_t  *order = nullptr;
    cs_lnum_t  *tmp = nullptr;
    const cs_lnum_t  *tr_index = _tr_index;

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf == nullptr)
      return;

    /* When this function is called, a distant-rank interface should have
       a match_id array, but not a send_order array */

    assert(itf->send_order == nullptr);

    _tr_index[1] = itf->size;

    if (itf->tr_index_size > 0) {
      tr_index_size = itf->tr_index_size;
      tr_index = itf->tr_index;
    }

    CS_MALLOC(order, tr_index[tr_index_size - 1], cs_lnum_t);
    CS_MALLOC(tmp, tr_index[tr_index_size - 1], cs_lnum_t);

    for (section_id = 0; section_id < tr_index_size - 1; section_id++) {

      cs_lnum_t start_id = tr_index[section_id];
      cs_lnum_t l = tr_index[section_id + 1] - start_id;

      /* Now compute distant ordering to build send_order */

      cs_order_lnum_allocated(nullptr,
                              itf->elt_id + start_id,
                              order,
                              l);

      /* Update elt_id */

      for (j = 0; j < l; j++)
        tmp[j] = itf->elt_id[order[j] + start_id];

      memcpy(itf->elt_id + start_id, tmp, l*sizeof(cs_lnum_t));

      /* Update match_id */

      for (j = 0; j < l; j++)
        tmp[j] = itf->match_id[order[j] + start_id];

      memcpy(itf->match_id + start_id, tmp, l*sizeof(cs_lnum_t));

    }

    CS_FREE(tmp);
    CS_FREE(order);

  }
}

/*----------------------------------------------------------------------------
 * Order interfaces by increasing element id.
 *
 * parameters:
 *   ifs <-> pointer to interface set
 *----------------------------------------------------------------------------*/

static void
_order_by_elt_id(cs_interface_set_t  *ifs)
{
  int i;

  for (i = 0; i < ifs->size; i++) {

    int section_id;

    int  tr_index_size = 2;
    cs_lnum_t  _tr_index[2] = {0, 0};
    cs_lnum_t  *order = nullptr, *buffer = nullptr;
    const cs_lnum_t  *tr_index = _tr_index;

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf == nullptr)
      return;

    /* When this function is called, a distant-rank interface should have
       a match_id array, but not a send_order array */

    assert(itf->send_order == nullptr);

    _tr_index[1] = itf->size;

    if (itf->tr_index_size > 0) {
      tr_index_size = itf->tr_index_size;
      tr_index = itf->tr_index;
    }

    CS_MALLOC(order, tr_index[tr_index_size - 1], cs_lnum_t);
    CS_MALLOC(buffer, tr_index[tr_index_size - 1]*2, cs_lnum_t);

    for (section_id = 0; section_id < tr_index_size - 1; section_id++) {

      cs_lnum_t start_id = tr_index[section_id];
      cs_lnum_t end_id = tr_index[section_id + 1];

      /* Now compute distant ordering to build send_order */

      cs_order_lnum_allocated(nullptr,
                              itf->elt_id + start_id,
                              order + start_id,
                              end_id - start_id);

      for (cs_lnum_t j = start_id; j < end_id; j++) {
        buffer[j*2] = itf->elt_id[j];
        buffer[j*2+1] = itf->match_id[j];
      }

      for (cs_lnum_t j = start_id; j < end_id; j++) {
        cs_lnum_t k = order[j] + start_id;
        itf->elt_id[j] = buffer[k*2];
        itf->match_id[j] = buffer[k*2+1];
      }

    }

    CS_FREE(buffer);
    CS_FREE(order);

  }
}

/*----------------------------------------------------------------------------
 * Replace array of distant element ids with ordering of list of ids to send,
 * so that sends will match receives.
 *
 * parameters:
 *   ifs <-> pointer to interface set
 *----------------------------------------------------------------------------*/

static void
_match_id_to_send_order(cs_interface_set_t  *ifs)
{
  int i;

  for (i = 0; i < ifs->size; i++) {

    int section_id;
    cs_lnum_t j, s_id, e_id;

    int  tr_index_size = 2;
    cs_lnum_t  _tr_index[2] = {0, 0};
    cs_lnum_t  *order = nullptr;
    const cs_lnum_t  *tr_index = _tr_index;

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf == nullptr)
      return;

    /* When this function is called, a distant-rank interface should have
       a match_id array, but not a send_order array */

    assert(itf->send_order == nullptr);

    _tr_index[1] = itf->size;

    if (itf->tr_index_size > 0) {
      tr_index_size = itf->tr_index_size;
      tr_index = itf->tr_index;
    }

    CS_MALLOC(order, tr_index[tr_index_size - 1], cs_lnum_t);

    for (section_id = 0; section_id < tr_index_size - 1; section_id++) {

      cs_lnum_t start_id = tr_index[section_id];
      cs_lnum_t l = tr_index[section_id + 1] - start_id;

      /* Now compute distant ordering to build send_order */

      cs_order_lnum_allocated(nullptr,
                              itf->match_id + start_id,
                              order + start_id,
                              l);

    }

    /* Swap match_id and send_order arrays */

    itf->send_order = itf->match_id;
    itf->match_id = nullptr;

    /* Parallel-only elements */

    s_id = tr_index[0]; e_id = tr_index[1];
    for (j = s_id; j < e_id; j++)
      itf->send_order[j] = order[j] + s_id;

    /* Periodic elements */

    if (itf->tr_index_size > 0) {

      int tr_id;
      cs_lnum_t k = tr_index[1];
      const int n_tr = tr_index_size - 2;

      for (tr_id = 0; tr_id < n_tr; tr_id++) {
        int r_tr_id = fvm_periodicity_get_reverse_id(ifs->periodicity,
                                                     tr_id);
        s_id = tr_index[r_tr_id+1];
        e_id = tr_index[r_tr_id+2];
        for (j = s_id; j < e_id; j++)
          itf->send_order[k++] = order[j] + s_id;
      }

      assert(k == itf->size);
    }

    CS_FREE(order);

  }
}

/*----------------------------------------------------------------------------
 * Prepare renumbering of elements referenced by an interface set.
 *
 * This requires replacing the send ordering of interfaces from a set
 * with the matching (distant or periodic) element id, to which renumbering
 * is applied. The send ordering will be rebuilt later.
 *
 * For any given element i, a negative old_to_new[i] value means that that
 * element does not appear anymore in the new numbering, but the filtering
 * is not applied at this stage.
 *
 * parameters:
 *   ifs        <-> pointer to interface set structure
 *   old_to_new <-- renumbering array (0 to n-1 numbering)
 *----------------------------------------------------------------------------*/

static void
_set_renumber_update_ids(cs_interface_set_t  *ifs,
                         const cs_lnum_t      old_to_new[])
{
  int i;
  cs_lnum_t j;
  int local_rank = 0;
  cs_lnum_t *send_buf = nullptr;

#if defined(HAVE_MPI)

  int n_ranks = 1;
  int request_count = 0;
  MPI_Request *request = nullptr;
  MPI_Status *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

#endif

  assert(ifs != nullptr);

#if defined(HAVE_MPI)

  if (n_ranks > 1)
    CS_MALLOC(send_buf, cs_interface_set_n_elts(ifs), cs_lnum_t);

#endif /* HAVE_MPI */

  /* Prepare send buffer first (for same rank, send_order is swapped
     with match_id directly) */

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_lnum_t k;
    cs_interface_t *itf = ifs->interfaces[i];

    /* When this function is called, a distant-rank interface should have
       a send_order array, but not a match_id array */

    assert(itf->match_id == nullptr);

    for (k = 0; k < itf->size; k++)
      itf->elt_id[k] = old_to_new[itf->elt_id[k]];

    itf->match_id = itf->send_order; /* Pre swap of send_order with match_id */

    if (itf->rank != local_rank)
      for (k = 0; k < itf->size; k++) {
        send_buf[j + k] = itf->elt_id[itf->send_order[k]];
    }
    else {
      for (k = 0; k < itf->size; k++)
        itf->match_id[k] = itf->elt_id[itf->send_order[k]];
    }

    itf->send_order = nullptr; /* Finish swap of send_order with match_id */

    j += itf->size;

  }

#if defined(HAVE_MPI)

  /* Now exchange data using MPI */

  if (n_ranks > 1) {

    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Irecv(itf->match_id,
                  itf->size,
                  CS_MPI_LNUM,
                  itf->rank,
                  itf->rank,
                  ifs->comm,
                  &(request[request_count++]));
      j += itf->size;
    }

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Isend(send_buf + j,
                  itf->size,
                  CS_MPI_LNUM,
                  itf->rank,
                  local_rank,
                  ifs->comm,
                  &(request[request_count++]));
      j += itf->size;
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);
    CS_FREE(send_buf);

  }

#endif /* defined(HAVE_MPI) */
}

/*----------------------------------------------------------------------------
 * Copy array from distant or matching interface elements to local elements,
 * for strided and non-interleaved elements.
 *
 * The destination arrays defines values for all elements in the
 * interface set (i.e. all elements listed by cs_interface_get_elt_ids()
 * when looping over interfaces of a set, while the source array
 * defines values for all elements of the type associated with the interfaces.
 *
 * parameters:
 *   ifs           <-> pointer to interface set structure
 *   datatype      <-- type of data considered
 *   n_elts        <-- number of elements associated with interface
 *   stride        <-- number of values per entity (interlaced)
 *   src           <-- source array (size: cs_interface_set_n_elts(ifs)*stride
 *                     or parent array size * stride)
 *   dest          <-- destination array
 *                     (size: cs_interface_set_n_elts(ifs)*stride)
 *----------------------------------------------------------------------------*/

static void
_interface_set_copy_array_ni(const cs_interface_set_t  *ifs,
                             cs_datatype_t              datatype,
                             cs_lnum_t                  n_elts,
                             int                        stride,
                             const void                *src,
                             void                      *dest)
{
  int i;
  cs_lnum_t j;
  int local_rank = 0;
  int type_size = cs_datatype_size[datatype];
  int stride_size = cs_datatype_size[datatype]*stride;
  cs_lnum_t shift_size = cs_datatype_size[datatype]*n_elts;

  unsigned char       *send_buf = nullptr;
  unsigned char       *_dest    = static_cast<unsigned char *>(dest);
  const unsigned char *_src     = static_cast<const unsigned char *>(src);

#if defined(HAVE_MPI)

  int n_ranks = 1;
  int request_count = 0;
  MPI_Datatype mpi_type = cs_datatype_to_mpi[datatype];
  MPI_Request  *request = nullptr;
  MPI_Status  *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

#endif

  assert(ifs != nullptr);

  CS_MALLOC(send_buf,
            cs_interface_set_n_elts(ifs)*stride_size,
            unsigned char);

  /* Prepare send buffer first (so that src is not used
     anymore after this, and may overlap with dest if desired);
     for same-rank interface, assign values directly to dest instead */

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_lnum_t k, l, m;
    cs_interface_t *itf = ifs->interfaces[i];
    unsigned char *p = send_buf;

    p += j*stride_size;

    for (k = 0; k < itf->size; k++) {
      cs_lnum_t send_id = itf->elt_id[itf->send_order[k]];
      for (m = 0; m < stride; m++) {
        for (l = 0; l < type_size; l++)
          p[(k*stride + m) * type_size + l]
            = _src[send_id*type_size + m*shift_size + l];
      }
    }

    j += itf->size;

  }

  /* Now exchange data */

#if defined(HAVE_MPI)

  if (n_ranks > 1) {
    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);
  }

#endif

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf->rank == local_rank)
      memcpy(_dest + j*stride_size,
             send_buf + j*stride_size,
             itf->size*stride_size);

#if defined(HAVE_MPI)

    else /* if (itf->rank != local_rank) */
      MPI_Irecv(_dest + j*stride_size,
                itf->size*stride,
                mpi_type,
                itf->rank,
                itf->rank,
                ifs->comm,
                &(request[request_count++]));

#endif

    j += itf->size;

  }

#if defined(HAVE_MPI)

  if (n_ranks > 1) {

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Isend(send_buf + j*stride_size,
                  itf->size*stride,
                  mpi_type,
                  itf->rank,
                  local_rank,
                  ifs->comm,
                  &(request[request_count++]));
      j += itf->size;
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);

  }

#endif /* defined(HAVE_MPI) */

  CS_FREE(send_buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Apply strided subdivision of elements to id array;
 *
 * \param[in]  size     original array size
 * \param[in]  stride   number of sub-elements per element
 * \param[in]  array_o  original array
 *
 * \return  subdivided array
 */
/*----------------------------------------------------------------------------*/

static cs_lnum_t *
_copy_sub_strided(cs_lnum_t   size_old,
                  cs_lnum_t   stride,
                  cs_lnum_t  *array_o)
{
  cs_lnum_t  size_new = size_old*stride;

  cs_lnum_t *array_n = nullptr;

  if (array_o != nullptr) {
    CS_MALLOC(array_n, size_new, cs_lnum_t);
    for (cs_lnum_t i = 0; i < size_new; i++)
      array_n[i] = array_o[i/stride]*stride + i%stride;
  }

  return array_n;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Apply bloc subdivision of elements to interface;
 *
 * The match ids of interfaces should be available for this operation.
 *
 * \param[in]  o             reference interface
 * \param[in]  l_block_size  local block size
 * \param[in]  d_block_size  distant block size
 * \param[in]  n_blocks      number of blocks in new interface
 *
 * \return  blocked interface
 */
/*----------------------------------------------------------------------------*/

static cs_interface_t  *
_copy_sub_blocked(cs_interface_t  *o,
                  cs_lnum_t        l_block_size,
                  cs_lnum_t        d_block_size,
                  cs_lnum_t        n_blocks)
{
  cs_interface_t *n = _cs_interface_create();

  n->rank = o->rank;
  n->size = o->size * n_blocks;

  n->tr_index_size = o->tr_index_size;
  if (o->tr_index != nullptr) {
    CS_MALLOC(n->tr_index, n->tr_index_size, cs_lnum_t);
    for (int j = 0; j < n->tr_index_size; j++)
      n->tr_index[j] = o->tr_index[j] * n_blocks;
  }

  const int _tr_index[2] = {0, o->size};
  const int *tr_index = (o->tr_index != nullptr) ? o->tr_index : _tr_index;
  const int n_tr = (o->tr_index != nullptr) ? o->tr_index_size - 1: 1;

  cs_lnum_t  size_new = o->size * n_blocks;

  if (o->elt_id != nullptr) {

    CS_MALLOC(n->elt_id, size_new, cs_lnum_t);

    cs_lnum_t j = 0;
    for (int tr_id = 0; tr_id < n_tr; tr_id++) {
      cs_lnum_t s_id = tr_index[tr_id];
      cs_lnum_t e_id = tr_index[tr_id+1];
      for (cs_lnum_t b_id = 0; b_id < n_blocks; b_id++) {
        for (cs_lnum_t i = s_id; i < e_id; i++) {
          n->elt_id[j++] = o->elt_id[i] + l_block_size*b_id;
        }
      }
    }


    CS_MALLOC(n->match_id, size_new, cs_lnum_t);

    j = 0;
    for (int tr_id = 0; tr_id < n_tr; tr_id++) {
      cs_lnum_t s_id = tr_index[tr_id];
      cs_lnum_t e_id = tr_index[tr_id+1];
      for (cs_lnum_t b_id = 0; b_id < n_blocks; b_id++) {
        for (cs_lnum_t i = s_id; i < e_id; i++) {
          n->match_id[j++] = o->match_id[i] + d_block_size*b_id;
        }
      }
    }
  }

  return n;
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*=============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return process rank associated with an interface's distant elements.
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  process rank associated with the interface's distant elements
 */
/*----------------------------------------------------------------------------*/

int
cs_interface_rank(const cs_interface_t  *itf)
{
  int retval = -1;

  if (itf != nullptr)
    retval = itf->rank;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return number of local and distant elements defining an interface.
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  number of local and distant elements defining the interface
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_interface_size(const cs_interface_t  *itf)
{
  cs_lnum_t retval = 0;

  if (itf != nullptr)
    retval = itf->size;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return pointer to array of local element ids defining an interface.
 *
 * The size of the array may be obtained by cs_interface_size().
 * The array is owned by the interface structure, and is not copied
 * (hence the constant qualifier for the return value).
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  pointer to array of local element ids (0 to n-1) defining
 * the interface
 */
/*----------------------------------------------------------------------------*/

const cs_lnum_t *
cs_interface_get_elt_ids(const cs_interface_t  *itf)
{
  const cs_lnum_t *retval = nullptr;

  if (itf != nullptr)
    retval = itf->elt_id;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return pointer to array of matching element ids defining an interface.
 *
 * This array is only available if cs_interface_set_add_match_ids() has
 * been called for the containing interface set.
 *
 * The size of the array may be obtained by cs_interface_size().
 * The array is owned by the interface structure, and is not copied
 * (hence the constant qualifier for the return value).
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  pointer to array of local element ids (0 to n-1) defining
 * the interface
 */
/*----------------------------------------------------------------------------*/

const cs_lnum_t *
cs_interface_get_match_ids(const cs_interface_t  *itf)
{
  const cs_lnum_t *retval = nullptr;

  if (itf != nullptr)
    retval = itf->match_id;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return size of index of sub-sections for different transformations.
 *
 * The index is applicable to both local_num and distant_num arrays,
 * with purely parallel equivalences appearing at position 0, and
 * equivalences through periodic transform i at position i+1;
 * Its size should thus be equal to 1 + number of periodic transforms + 1,
 * In absence of periodicity, it may be 0, as the index is not needed.
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  transform index size for the interface
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_interface_get_tr_index_size(const cs_interface_t  *itf)
{
  cs_lnum_t retval = 0;

  if (itf != nullptr)
    retval = itf->tr_index_size;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return pointer to index of sub-sections for different transformations.
 *
 * The index is applicable to both local_num and distant_num arrays,
 * with purely parallel equivalences appearing at position 0, and
 * equivalences through periodic transform i at position i+1;
 * In absence of periodicity, it may be nullptr, as it is not needed.
 *
 * \param[in]  itf  pointer to interface structure
 *
 * \return  pointer to transform index for the interface
 */
/*----------------------------------------------------------------------------*/

const cs_lnum_t *
cs_interface_get_tr_index(const cs_interface_t  *itf)
{
  const cs_lnum_t *tr_index = 0;

  if (itf != nullptr)
    tr_index = itf->tr_index;

  return tr_index;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Creation of a list of interfaces between elements of a same type.
 *
 * These interfaces may be used to identify equivalent vertices or faces using
 * domain splitting, as well as periodic elements (on the same or on
 * distant ranks).
 *
 * Note that periodicity information will be completed and made consistent
 * based on the input, so that if a periodic couple is defined on a given rank,
 * the reverse couple wil be defined, whether it is also defined on the same
 * or a different rank.
 *
 * In addition, multiple periodicity interfaces will be built automatically
 * if the periodicity structure provides for composed periodicities, so they
 * need not be defined prior to this function being called.
 *
 * \param[in]  n_elts                 number of local elements considered
 *                                    (size of parent_element_id[])
 * \param[in]  parent_element_id      pointer to list of selected elements
 *                                    local ids (0 to n-1), or nullptr if all
 *                                    first n_elts elements are used
 * \param[in]  global_number          pointer to list of global (i.e. domain
 *                                    splitting independent) element numbers
 * \param[in]  periodicity            periodicity information (nullptr if none)
 * \param[in]  n_periodic_lists       number of periodic lists (may be local)
 * \param[in]  periodicity_num        periodicity number (1 to n) associated
 *                                    with each periodic list (primary
 *                                    periodicities only)
 * \param[in]  n_periodic_couples     number of periodic couples associated
 *                                    with each periodic list
 * \param[in]  periodic_couples       array indicating periodic couples
 *                                    (interlaced, using global numberings)
 *                                    for each list
 *
 * \return  pointer to list of interfaces (possibly null in serial mode)
 */
/*----------------------------------------------------------------------------*/

cs_interface_set_t *
cs_interface_set_create(cs_lnum_t                 n_elts,
                        const cs_lnum_t           parent_element_id[],
                        const cs_gnum_t           global_number[],
                        const fvm_periodicity_t  *periodicity,
                        int                       n_periodic_lists,
                        const int                 periodicity_num[],
                        const cs_lnum_t           n_periodic_couples[],
                        const cs_gnum_t    *const periodic_couples[])
{
  cs_interface_set_t  *ifs;

  /* Initial checks */

  if (   (cs_glob_n_ranks < 2)
      && (periodicity == nullptr || n_periodic_lists == 0))
    return nullptr;

  /* Create structure */

  CS_MALLOC(ifs, 1, cs_interface_set_t);
  ifs->size = 0;
  ifs->interfaces = nullptr;
  ifs->periodicity = periodicity;
  ifs->match_id_rc = 0;

  const cs_gnum_t  *global_num = global_number;

  cs_gnum_t  *_global_num = nullptr;

  if (global_number != nullptr && parent_element_id != nullptr) {

    CS_MALLOC(_global_num, n_elts, cs_gnum_t);

    for (size_t i = 0 ; i < (size_t)n_elts ; i++)
      _global_num[i] = global_number[parent_element_id[i]];

    global_num = _global_num;

  }

  /* Build interfaces */

#if defined(HAVE_MPI)

  ifs->comm = cs_glob_mpi_comm;

  if (cs_glob_n_ranks > 1) {

    /* Build interfaces */

    if (periodicity == nullptr)
      _add_global_equiv(ifs,
                        n_elts,
                        global_num,
                        cs_glob_mpi_comm);

    else
      _add_global_equiv_periodic(ifs,
                                 n_elts,
                                 global_num,
                                 periodicity,
                                 n_periodic_lists,
                                 periodicity_num,
                                 n_periodic_couples,
                                 periodic_couples,
                                 cs_glob_mpi_comm);

  }

#endif /* defined(HAVE_MPI) */

  if (cs_glob_n_ranks == 1 && (periodicity != nullptr && n_periodic_lists > 0)) {

    _add_global_equiv_periodic_sp(ifs,
                                  n_elts,
                                  global_num,
                                  periodicity,
                                  n_periodic_lists,
                                  periodicity_num,
                                  n_periodic_couples,
                                  periodic_couples);


  }

  CS_FREE(_global_num);

  /* Finish preparation of interface set and return */

  _order_by_elt_id(ifs);
  _match_id_to_send_order(ifs);

  return ifs;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Destruction of an interface set.
 *
 * \param[in, out]  ifs  pointer to pointer to structure to destroy
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_destroy(cs_interface_set_t  **ifs)
{
  int i;
  cs_interface_set_t  *itfs = *ifs;

  if (itfs != nullptr) {
    for (i = 0; i < itfs->size; i++) {
      _cs_interface_destroy(&(itfs->interfaces[i]));
    }
    CS_FREE(itfs->interfaces);
    CS_FREE(itfs);
    *ifs = itfs;
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Duplicate an interface set, applying an optional constant stride.
 *
 * \param[in, out]  ifs     pointer to interface set structure
 * \param[in]       stride  if > 1, each element subdivided in stride elements
 *
 * \return  pointer to new interface set
 */
/*----------------------------------------------------------------------------*/

cs_interface_set_t  *
cs_interface_set_dup(const cs_interface_set_t  *ifs,
                     cs_lnum_t                  stride)
{
  cs_interface_set_t  *ifs_new = nullptr;

  if (ifs == nullptr)
    return ifs_new;

  if (stride < 1)
    stride = 1;

  CS_MALLOC(ifs_new, 1, cs_interface_set_t);

  ifs_new->size = ifs->size;
  ifs_new->periodicity = ifs->periodicity;
  ifs_new->match_id_rc = 0;

#if defined(HAVE_MPI)
  ifs_new->comm = ifs->comm;
#endif

  CS_MALLOC(ifs_new->interfaces,
            ifs->size,
            cs_interface_t *);

  /* Loop on interfaces */

  for (int i = 0; i < ifs->size; i++) {

    cs_interface_t *o = ifs->interfaces[i];
    cs_interface_t *n = _cs_interface_create();

    n->rank = o->rank;
    n->size = o->size * stride;

    n->tr_index_size = o->tr_index_size;
    if (o->tr_index != nullptr) {
      CS_MALLOC(n->tr_index, n->tr_index_size, cs_lnum_t);
      for (int j = 0; j < n->tr_index_size; j++)
        n->tr_index[j] = o->tr_index[j] * stride;
    }

    n->elt_id = _copy_sub_strided(o->size, stride, o->elt_id);
    n->send_order = _copy_sub_strided(o->size, stride, o->send_order);

    n->match_id = nullptr;

    ifs_new->interfaces[i] = n;
  }

  return ifs_new;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Duplicate an interface set for coupled variable blocks.
 *
 * \param[in, out]  ifs         pointer to interface set structure
 * \param[in]       block_size  local block size (number of elements)
 * \param[in]       n_blocks    number of associated blocks
 *
 * \return  pointer to new interface set
 */
/*----------------------------------------------------------------------------*/

cs_interface_set_t  *
cs_interface_set_dup_blocks(cs_interface_set_t  *ifs,
                            cs_lnum_t            block_size,
                            cs_lnum_t            n_blocks)
{
  cs_interface_set_t  *ifs_new = nullptr;

  if (ifs == nullptr)
    return ifs_new;

  if (n_blocks < 1)
    n_blocks = 1;

  CS_MALLOC(ifs_new, 1, cs_interface_set_t);
  ifs->match_id_rc = 0;

  ifs_new->size = ifs->size;
  ifs_new->periodicity = ifs->periodicity;

  cs_lnum_t *d_block_size;
  CS_MALLOC(d_block_size, ifs->size, cs_lnum_t);

  int n_ranks = 1;

#if defined(HAVE_MPI)

  ifs_new->comm = ifs->comm;

  /* Exchange block sizes */

  int request_count = 0, local_rank = -1;
  MPI_Request *request = nullptr;
  MPI_Status *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

  if (n_ranks > 1) {

    cs_lnum_t send_buf[1] = {block_size};

    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);

    for (int i = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Irecv(d_block_size + i,
                  1,
                  CS_MPI_LNUM,
                  itf->rank,
                  itf->rank,
                  ifs->comm,
                  &(request[request_count++]));
      else
        d_block_size[i] = block_size;
    }

    for (int i = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Isend(send_buf,
                  1,
                  CS_MPI_LNUM,
                  itf->rank,
                  local_rank,
                  ifs->comm,
                  &(request[request_count++]));
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);

  }

#endif /* defined(HAVE_MPI) */

  if (n_ranks <= 1 && ifs->size > 0) {
    assert(ifs->size <= 1);
    assert(ifs->interfaces[0]->rank == 0);

    d_block_size[0] = block_size;
  }

  /* Ensure match ids are available on reference interface */

  cs_interface_set_add_match_ids(ifs);

  /* Build new interface
     ------------------- */

  CS_MALLOC(ifs_new->interfaces,
            ifs->size,
            cs_interface_t *);

  /* Loop on interfaces */

  for (int i = 0; i < ifs->size; i++) {
    cs_interface_t *o = ifs->interfaces[i];
    ifs_new->interfaces[i]
      = _copy_sub_blocked(o, block_size, d_block_size[i], n_blocks);
  }

  /* Free memory */

  cs_interface_set_free_match_ids(ifs);

  CS_FREE(d_block_size);

  /* Finish preparation of interface set and return */

  _match_id_to_send_order(ifs_new);

  return ifs_new;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return number of interfaces associated with an interface set.
 *
 * \param[in]  ifs  pointer to interface set structure
 *
 * \return  number of interfaces in set
 */
/*----------------------------------------------------------------------------*/

int
cs_interface_set_size(const cs_interface_set_t  *ifs)
{
  int retval = 0;

  if (ifs != nullptr)
    retval = ifs->size;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return total number of elements in interface set.
 *
 * This is equal to the sum of cs_interface_size() on the cs_interface_size()
 * interfaces of a set.
 *
 * \param[in]  ifs  pointer to interface set structure
 *
 * \return  number of interfaces in set
 */
/*----------------------------------------------------------------------------*/

cs_lnum_t
cs_interface_set_n_elts(const cs_interface_set_t  *ifs)
{
  int i;

  cs_lnum_t retval = 0;

  for (i = 0; i < ifs->size; i++)
    retval += (ifs->interfaces[i])->size;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return pointer to a given interface in an interface set.
 *
 * \param[in]  ifs           pointer to interface set structure
 * \param[in]  interface_id  index of interface in set (0 to n-1)
 *
 * \return  pointer to interface structure
 */
/*----------------------------------------------------------------------------*/

const cs_interface_t *
cs_interface_set_get(const cs_interface_set_t  *ifs,
                     int                        interface_id)
{
  const cs_interface_t  *retval = nullptr;

  if (ifs != nullptr) {
    if (interface_id > -1 && interface_id < ifs->size)
      retval = ifs->interfaces[interface_id];
  }

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return pointer to the periocicity structure associated of an
 * interface set.
 *
 * \param[in]  ifs  pointer to interface set structure
 *
 * \return  pointer to periodicity structure, or nullptr
 */
/*----------------------------------------------------------------------------*/

const fvm_periodicity_t *
cs_interface_set_periodicity(const cs_interface_set_t  *ifs)
{
  const fvm_periodicity_t  *retval = nullptr;

  if (ifs != nullptr)
    retval = ifs->periodicity;

  return retval;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Apply renumbering of elements referenced by an interface set.
 *
 * For any given element i, a negative old_to_new[i] value means that that
 * element does not appear anymore in the new numbering.
 *
 * \param[in, out]  ifs         pointer to interface set structure
 * \param[in]       old_to_new  renumbering array (0 to n-1 numbering)
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_renumber(cs_interface_set_t  *ifs,
                          const cs_lnum_t      old_to_new[])
{
  int i;
  int n_interfaces = 0;

  assert(ifs != nullptr);
  assert(old_to_new != nullptr);

  /* Compute new element and match ids */

  _set_renumber_update_ids(ifs, old_to_new);

  _order_elt_id(ifs);

  /* Remove references to elements not appearing anymore */

  for (i = 0; i < ifs->size; i++) {

    cs_lnum_t j, k;
    cs_interface_t *itf = ifs->interfaces[i];

    cs_lnum_t *loc_id = itf->elt_id;
    cs_lnum_t *dist_id = itf->match_id;

    if (itf->tr_index_size == 0) {
      for (j = 0, k = 0; j < itf->size; j++) {
        if (loc_id[j] > -1 && dist_id[j] > -1) {
          loc_id[k] = loc_id[j];
          dist_id[k] = dist_id[j];
          k += 1;
        }
      }
    }
    else {
      int tr_id;
      cs_lnum_t end_id = itf->tr_index[0];
      k = 0;
      for (tr_id = 0; tr_id < itf->tr_index_size - 1; tr_id++) {
        cs_lnum_t start_id = end_id;
        end_id = itf->tr_index[tr_id + 1];
        for (j = start_id; j < end_id; j++) {
          if (loc_id[j] > -1 && dist_id[j] > -1) {
            loc_id[k] = loc_id[j];
            dist_id[k] = dist_id[j];
            k += 1;
          }
        }
        itf->tr_index[tr_id + 1] = k;
      }
    }

    if (k < itf->size) {
      if (k > 0) {
        itf->size = k;
        CS_REALLOC(itf->elt_id, k, cs_lnum_t);
        CS_REALLOC(itf->match_id, k, cs_lnum_t);
      }
      else {
        CS_FREE(itf->elt_id);
        CS_FREE(itf->match_id);
        CS_FREE(ifs->interfaces[i]);
      }
    }
  }

  for (i = 0, n_interfaces = 0; i < ifs->size; i++) {
    if (ifs->interfaces[i] != nullptr)
      ifs->interfaces[n_interfaces++] = ifs->interfaces[i];
  }

  if (n_interfaces < ifs->size) {
    CS_REALLOC(ifs->interfaces,
               n_interfaces,
               cs_interface_t *);
    ifs->size = n_interfaces;
  }

  _match_id_to_send_order(ifs);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Copy array from distant or matching interface elements to
 * local elements.
 *
 * Source and destination arrays define values for all elements in the
 * interface set (i.e. all elements listed by cs_interface_get_elt_ids()
 * when looping over interfaces of a set.
 *
 * \param[in]   ifs            pointer to interface set structure
 * \param[in]   datatype       type of data considered
 * \param[in]   stride         number of values per entity (interlaced)
 * \param[in]   src_on_parent  true if source array is defined on the elements
 *                             defined by ifs->elt_ids, false if source array
 *                             defined directly on cs_interface_set_n_elts(ifs)
 * \param[in]   src            source array (size:
 *                             cs_interface_set_n_elts(ifs)*stride
 *                             or parent array size * stride)
 * \param[out]  dest           destination array (size:
 *                             cs_interface_set_n_elts(ifs)*stride)
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_copy_array(const cs_interface_set_t  *ifs,
                            cs_datatype_t              datatype,
                            int                        stride,
                            bool                       src_on_parent,
                            const void                *src,
                            void                      *dest)
{
  int i;
  cs_lnum_t j;
  int local_rank = 0;
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *send_buf = nullptr;
  unsigned char       *_dest       = static_cast<unsigned char *>(dest);
  const unsigned char *_src        = static_cast<const unsigned char *>(src);

#if defined(HAVE_MPI)

  int n_ranks = 1;
  int request_count = 0;
  MPI_Datatype mpi_type = cs_datatype_to_mpi[datatype];
  MPI_Request  *request = nullptr;
  MPI_Status  *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

#endif

  assert(ifs != nullptr);

  CS_MALLOC(send_buf,
            cs_interface_set_n_elts(ifs)*stride_size,
            unsigned char);

  /* Prepare send buffer first (so that src is not used
     anymore after this, and may overlap with dest if desired); */

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_lnum_t k, l;
    cs_interface_t *itf = ifs->interfaces[i];
    unsigned char *p = send_buf;

    p += j*stride_size;

    if (src_on_parent) {
      for (k = 0; k < itf->size; k++) {
        cs_lnum_t send_id = itf->elt_id[itf->send_order[k]];
        for (l = 0; l < stride_size; l++)
          p[k*stride_size + l] = _src[send_id*stride_size + l];
      }
    }
    else {
      for (k = 0; k < itf->size; k++) {
        cs_lnum_t send_id = itf->send_order[k] + j;
        for (l = 0; l < stride_size; l++)
          p[k*stride_size + l] = _src[send_id*stride_size + l];
      }
    }

    j += itf->size;

  }

  /* Now exchange data */

#if defined(HAVE_MPI)
  if (n_ranks > 1) {
    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);
  }
#endif

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf->rank == local_rank)
      memcpy(_dest + j*stride_size,
             send_buf + j*stride_size,
             itf->size*stride_size);

#if defined(HAVE_MPI)

    else /* if (itf->rank != local_rank) */
        MPI_Irecv(_dest + j*stride_size,
                  itf->size*stride,
                  mpi_type,
                  itf->rank,
                  itf->rank,
                  ifs->comm,
                  &(request[request_count++]));

#endif

    j += itf->size;

  }

#if defined(HAVE_MPI)

  if (n_ranks > 1) {

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Isend(send_buf + j*stride_size,
                  itf->size*stride,
                  mpi_type,
                  itf->rank,
                  local_rank,
                  ifs->comm,
                  &(request[request_count++]));
      j += itf->size;
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);

  }

#endif /* defined(HAVE_MPI) */

  CS_FREE(send_buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Copy indexed array from distant or matching interface elements to
 * local elements.
 *
 * Source and destination arrays define values for all elements in the
 * interface set (i.e. all elements listed by cs_interface_get_elt_ids()
 * when looping over interfaces of a set.
 *
 * Note that when copying the same type of data to all matching elements,
 * the source and destination index may be the same, if src_on_parent is true.
 * To avoid requiring a separate destination index, the dest_index argument
 * may be set to nullptr, in which case it is assumed that source and destination
 * are symmetric, and src_index is sufficient to determine sizes (whether
 * src_on_parent is true or not).
 *
 * In some use cases, for example when copying values only in one direction,
 * the copying is not symmetric, so both a source and destination buffer must
 * be provided.
 *
 * \param[in]   ifs            pointer to interface set structure
 * \param[in]   datatype       type of data considered
 * \param[in]   src_on_parent  true if source array is defined on the elements
 *                             defined by ifs->elt_ids, false if source array
 *                             defined directly on cs_interface_set_n_elts(ifs)
 * \param[in]   src_index      index for source array
 * \param[in]   dest_index     index for destination array, or nullptr
 * \param[in]   src            source array (size:
 *                             src_index[cs_interface_set_n_elts(ifs)]
 *                             or parent array size)
 * \param[out]  dest           destination array (size:
 *                             src_index[cs_interface_set_n_elts(ifs)] or
 *                             dest_index[cs_interface_set_n_elts(ifs)])
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_copy_indexed(const cs_interface_set_t  *ifs,
                              cs_datatype_t              datatype,
                              bool                       src_on_parent,
                              const cs_lnum_t            src_index[],
                              const cs_lnum_t            dest_index[],
                              const void                *src,
                              void                      *dest)
{
  int i;
  cs_lnum_t j;
  int local_rank = 0;
  int type_size = cs_datatype_size[datatype];
  cs_lnum_t send_size = 0;
  cs_lnum_t *itf_index = nullptr, *itf_s_index = nullptr, *itf_r_index = nullptr;
  unsigned char *send_buf = nullptr;
  unsigned char *_dest      = static_cast<unsigned char *>(dest);
  const unsigned char *_src = static_cast<const unsigned char *>(src);

#if defined(HAVE_MPI)

  int n_ranks = 1;
  int request_count = 0;
  MPI_Datatype mpi_type = cs_datatype_to_mpi[datatype];
  MPI_Request  *request = nullptr;
  MPI_Status  *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

#endif

  assert(ifs != nullptr);

  /* Count number of elements to send or receive */

  CS_MALLOC(itf_index, 2*(ifs->size + 1), cs_lnum_t);
  itf_s_index = itf_index;
  itf_s_index[0] = 0;

  if (src_on_parent) {
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_lnum_t k;
      cs_interface_t *itf = ifs->interfaces[i];
      for (k = 0; k < itf->size; k++) {
        cs_lnum_t send_id = itf->elt_id[itf->send_order[k]];
        send_size += src_index[send_id+1] - src_index[send_id];
      }
      itf_s_index[i+1] = send_size;
    }
  }
  else {
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      j += itf->size;
      itf_s_index[i+1] = src_index[j];
    }
    send_size = itf_s_index[ifs->size];
  }

  if (dest_index != nullptr) {
    itf_r_index = itf_index + ifs->size + 1;
    itf_r_index[0] = 0;
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      j += itf->size;
      itf_r_index[i+1] = dest_index[j];
    }
  }
  else
    itf_r_index = itf_s_index;

  CS_MALLOC(send_buf, send_size*type_size, unsigned char);

  /* Prepare send buffer first (so that src is not used
     anymore after this, and may overlap with dest if desired); */

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_lnum_t k, l, m;
    cs_interface_t *itf = ifs->interfaces[i];
    unsigned char *p = send_buf + (itf_s_index[i]*type_size);

    if (src_on_parent) {
      for (k = 0, l = 0; k < itf->size; k++) {
        cs_lnum_t send_id = itf->elt_id[itf->send_order[k]];
        cs_lnum_t start_id = src_index[send_id]*type_size;
        cs_lnum_t end_id = src_index[send_id+1]*type_size;
        for (m = start_id; m < end_id; m++)
          p[l++] = _src[m];
      }
    }
    else {
      for (k = 0, l = 0; k < itf->size; k++) {
        cs_lnum_t send_id = itf->send_order[k] + j;
        cs_lnum_t start_id = src_index[send_id]*type_size;
        cs_lnum_t end_id = src_index[send_id+1]*type_size;
        for (m = start_id; m < end_id; m++)
          p[l++] = _src[m];
      }
      j += itf->size;
    }
  }

  /* Now exchange data */

#if defined(HAVE_MPI)
  if (n_ranks > 1) {
    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);
  }
#endif

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_interface_t *itf = ifs->interfaces[i];
    cs_lnum_t r_buf_shift = itf_r_index[i]*type_size;

    if (itf->rank == local_rank) {
      cs_lnum_t s_buf_shift = itf_s_index[i]*type_size;
      int msg_size = (itf_s_index[i+1]-itf_s_index[i])*type_size;
      memcpy(_dest + r_buf_shift, send_buf + s_buf_shift, msg_size);
    }

#if defined(HAVE_MPI)

    else /* if (itf->rank != local_rank) */
      MPI_Irecv(_dest + r_buf_shift,
                (itf_r_index[i+1]-itf_r_index[i]),
                mpi_type,
                itf->rank,
                itf->rank,
                ifs->comm,
                &(request[request_count++]));

#endif

    j += itf->size;

  }

#if defined(HAVE_MPI)

  if (n_ranks > 1) {

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      cs_lnum_t s_buf_shift = itf_s_index[i]*type_size;
        if (itf->rank != local_rank)
          MPI_Isend(send_buf + s_buf_shift,
                    (itf_s_index[i+1]-itf_s_index[i]),
                    mpi_type,
                    itf->rank,
                    local_rank,
                    ifs->comm,
                    &(request[request_count++]));
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);

  }

#endif /* defined(HAVE_MPI) */

  CS_FREE(send_buf);
  CS_FREE(itf_index);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update values using the bitwise inclusive or operation for elements
 *        associated with an interface set.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been processed.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]       ifs        pointer to a fvm_interface_set_t structure
 * \param[in]       n_elts     number of elements in var buffer
 * \param[in]       stride     number of values (non interlaced) by entity
 * \param[in]       interlace  true if variable is interlaced (for stride > 1)
 * \param[in]       datatype   type of data considered
 * \param[in, out]  var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_inclusive_or(const cs_interface_set_t  *ifs,
                              cs_lnum_t                  n_elts,
                              cs_lnum_t                  stride,
                              bool                       interlace,
                              cs_datatype_t              datatype,
                              void                      *var)
{
  int i;
  cs_lnum_t j, k, l;
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now update values */

  switch (datatype) {
    case CS_CHAR: {
      char *v = static_cast<char *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const char     *p   = (const char *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_INT32: {
      int32_t *v = static_cast<int32_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const int32_t  *p   = (const int32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_INT64: {
      int64_t *v = static_cast<int64_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const int64_t  *p   = (const int64_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_UINT16: {
      uint16_t *v = static_cast<uint16_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const uint16_t *p   = (const uint16_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_UINT32: {
      uint32_t *v = static_cast<uint32_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const uint32_t *p   = (const uint32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_UINT64: {
      uint64_t *v = static_cast<uint64_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const uint64_t *p   = (const uint64_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] |= p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] |= p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

  default:
    bft_error(__FILE__, __LINE__, 0,
              _("Called %s with unhandled datatype (%d)."), __func__,
              (int)datatype);
  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the sum of values for elements associated with an
 * interface set.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been added.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]       ifs        pointer to a fvm_interface_set_t structure
 * \param[in]       n_elts     number of elements in var buffer
 * \param[in]       stride     number of values (non interlaced) by entity
 * \param[in]       interlace  true if variable is interlaced (for stride > 1)
 * \param[in]       datatype   type of data considered
 * \param[in, out]  var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_sum(const cs_interface_set_t  *ifs,
                     cs_lnum_t                  n_elts,
                     cs_lnum_t                  stride,
                     bool                       interlace,
                     cs_datatype_t              datatype,
                     void                      *var)
{
  int i;
  cs_lnum_t j, k, l;
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now increment values */

  switch (datatype) {
    case CS_CHAR: {
      char *v = static_cast<char *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const char     *p   = (const char *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_FLOAT: {
      float *v = static_cast<float *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const float    *p   = (const float *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_DOUBLE: {
      double *v = static_cast<double *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const double   *p   = (const double *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_INT32: {
      int32_t *v = static_cast<int32_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const int32_t  *p   = (const int32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_INT64: {
      int64_t *v = static_cast<int64_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const int64_t  *p   = (const int64_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_UINT16: {
      uint16_t *v = static_cast<uint16_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const uint16_t *p   = (const uint16_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_UINT32: {
      uint32_t *v = static_cast<uint32_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const uint32_t *p   = (const uint32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] += p[k * stride + l];
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] += p[k * stride + l];
          }
        }
        j += itf->size;
      }
      break;
    }

  case CS_UINT64: {
    uint64_t *v = static_cast<uint64_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint64_t *p   = (const uint64_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] += p[k * stride + l];
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] += p[k * stride + l];
        }
      }
      j += itf->size;
    }
    break;
  }

  default:
    bft_error(__FILE__, __LINE__, 0,
              _("Called cs_interface_set_sum with unhandled datatype (%d)."),
              (int)datatype);
  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the sum of values for elements associated with an
 * interface set, allowing control over periodicity.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been added.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]       ifs        pointer to a fvm_interface_set_t structure
 * \param[in]       n_elts     number of elements in var buffer
 * \param[in]       stride     number of values (non interlaced) by entity
 * \param[in]       interlace  true if variable is interlaced (for stride > 1)
 * \param[in]       datatype   type of data considered
 * \param[in]       tr_ignore  if > 0, ignore periodicity with rotation;
 *                             if > 1, ignore all periodic transforms
 * \param[in, out]  var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_sum_tr(const cs_interface_set_t  *ifs,
                        cs_lnum_t                  n_elts,
                        cs_lnum_t                  stride,
                        bool                       interlace,
                        cs_datatype_t              datatype,
                        int                        tr_ignore,
                        void                      *var)
{
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  int n_tr = 0;

  if (tr_ignore > 0 && ifs->periodicity != nullptr) {
    if (tr_ignore < 2) {
      int n_tr_max = fvm_periodicity_get_n_transforms(ifs->periodicity);
      for (int tr_id = 0; tr_id < n_tr_max; tr_id++) {
        if (fvm_periodicity_get_type(ifs->periodicity, tr_id)
            < FVM_PERIODICITY_ROTATION)
          n_tr = tr_id + 1;
      }
    }
    n_tr += 1; /* add base "identity" transform_id */
  }

  if (n_tr < 1) {
    cs_interface_set_sum(ifs,
                         n_elts,
                         stride,
                         interlace,
                         datatype,
                         var);
    return;
  }

  /* We can use a fixed max periodicity type here based on translation,
     because when even translation is ignored, n_tr has been set to 1 above,
     and the first transform is "no transform", so tests should always
     return a correct value */

  fvm_periodicity_type_t tr_threshold = FVM_PERIODICITY_ROTATION;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now increment values */

  cs_lnum_t j = 0;

  switch (datatype) {
    case CS_FLOAT: {
      float *v = static_cast<float *>(var);
      for (int i = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        for (int tr_id = 0; tr_id < n_tr; tr_id++) {
          cs_lnum_t s_id = itf->tr_index[tr_id];
          cs_lnum_t e_id = itf->tr_index[tr_id + 1];
          if (e_id > s_id && tr_id > 0) {
            if (fvm_periodicity_get_type(ifs->periodicity, tr_id - 1) >=
                tr_threshold)
              continue;
          }
          const float *p = (const float *)buf + j * stride;
          if (stride < 2 || interlace) {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id * stride + l] += p[k * stride + l];
            }
          }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l * n_elts] += p[k * stride + l];
            }
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_DOUBLE: {
      double *v = static_cast<double *>(var);
      for (int i = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        for (int tr_id = 0; tr_id < n_tr; tr_id++) {
          cs_lnum_t s_id = itf->tr_index[tr_id];
          cs_lnum_t e_id = itf->tr_index[tr_id + 1];
          if (e_id > s_id && tr_id > 0) {
            if (fvm_periodicity_get_type(ifs->periodicity, tr_id - 1) >=
                tr_threshold)
              continue;
          }
          const double *p = (const double *)buf + j * stride;
          if (stride < 2 || interlace) {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id * stride + l] += p[k * stride + l];
            }
          }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l * n_elts] += p[k * stride + l];
            }
          }
        }
        j += itf->size;
      }
      break;
    }

    default: {
      char     *v       = static_cast<char *>(var);
      cs_lnum_t _stride = stride * cs_datatype_size[datatype];
      for (int i = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        for (int tr_id = 0; tr_id < n_tr; tr_id++) {
          cs_lnum_t s_id = itf->tr_index[tr_id];
          cs_lnum_t e_id = itf->tr_index[tr_id + 1];
          if (e_id > s_id && tr_id > 0) {
            if (fvm_periodicity_get_type(ifs->periodicity, tr_id - 1) >=
                tr_threshold)
              continue;
          }

          const char *p = (const char *)buf + j * _stride;
          if (stride < 2 || interlace) {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < _stride; l++)
                v[elt_id * _stride + l] += p[k * _stride + l];
            }
          }
          else {
            cs_lnum_t elt_size = cs_datatype_size[datatype];
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++) {
                for (cs_lnum_t q = 0; q < elt_size; q++)
                  v[elt_id * elt_size + l * n_elts + q] +=
                    p[k * _stride + l * elt_size + q];
              }
            }
          }
        }
        j += itf->size;
      }
    }
  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the minimum value of elements associated with an interface
 *        set.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been updated.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]      ifs        pointer to a cs_interface_set_t structure
 * \param[in]      n_elts     number of elements in var buffer
 * \param[in]      stride     number of values (non interlaced) by entity
 * \param[in]      interlace  true if variable is interlaced (for stride > 1)
 * \param[in]      datatype   type of data considered
 * \param[in, out] var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_min(const cs_interface_set_t  *ifs,
                     cs_lnum_t                  n_elts,
                     cs_lnum_t                  stride,
                     bool                       interlace,
                     cs_datatype_t              datatype,
                     void                      *var)
{
  int i;
  cs_lnum_t j, k, l;
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now increment values */

  switch (datatype) {
    case CS_CHAR: {
      char *v = static_cast<char *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const char     *p   = (const char *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::min(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_FLOAT: {
      float *v = static_cast<float *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const float    *p   = (const float *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::min(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::min(p[k * stride + l], v[elt_id + l * n_elts]);
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_DOUBLE: {
      double *v = static_cast<double *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const double   *p   = (const double *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::min(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_INT32: {
      int32_t *v = static_cast<int32_t *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const int32_t  *p   = (const int32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::min(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
          }
        }
        j += itf->size;
      }
      break;
    }

  case CS_INT64: {
    int64_t *v = static_cast<int64_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const int64_t  *p   = (const int64_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::min(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT16: {
    uint16_t *v = static_cast<uint16_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint16_t *p   = (const uint16_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::min(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT32: {
    uint32_t *v = static_cast<uint32_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint32_t *p   = (const uint32_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::min(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT64: {
    uint64_t *v = static_cast<uint64_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint64_t *p   = (const uint64_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::min(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::min(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  default:
    bft_error(__FILE__,
              __LINE__,
              0,
              _("Called cs_interface_set_min with unhandled datatype (%d)."),
              (int)datatype);
  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the maximum value of elements associated with an interface
 *        set.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been updated.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]      ifs        pointer to a cs_interface_set_t structure
 * \param[in]      n_elts     number of elements in var buffer
 * \param[in]      stride     number of values (non interlaced) by entity
 * \param[in]      interlace  true if variable is interlaced (for stride > 1)
 * \param[in]      datatype   type of data considered
 * \param[in, out] var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_max(const cs_interface_set_t  *ifs,
                     cs_lnum_t                  n_elts,
                     cs_lnum_t                  stride,
                     bool                       interlace,
                     cs_datatype_t              datatype,
                     void                      *var)
{
  int i;
  cs_lnum_t j, k, l;
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now increment values */

  switch (datatype) {
    case CS_CHAR: {
      char *v = static_cast<char *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const char     *p   = (const char *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_FLOAT: {
      float *v = static_cast<float *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const float    *p   = (const float *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::max(p[k * stride + l], v[elt_id + l * n_elts]);
          }
        }
        j += itf->size;
      }
      break;
    }

    case CS_DOUBLE: {
      double *v = static_cast<double *>(var);
      for (i = 0, j = 0; i < ifs->size; i++) {
        cs_interface_t *itf = ifs->interfaces[i];
        const double   *p   = (const double *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
        else {
          for (k = 0; k < itf->size; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (l = 0; l < stride; l++)
              v[elt_id + l * n_elts] =
                cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
          }
        }
        j += itf->size;
      }
      break;
    }

  case CS_INT32: {
    int32_t *v = static_cast<int32_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const int32_t  *p   = (const int32_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::max(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_INT64: {
    int64_t *v = static_cast<int64_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const int64_t  *p   = (const int64_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::max(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT16: {
    uint16_t *v = static_cast<uint16_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint16_t *p   = (const uint16_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::max(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT32: {
    uint32_t *v = static_cast<uint32_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint32_t *p   = (const uint32_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::max(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  case CS_UINT64: {
    uint64_t *v = static_cast<uint64_t *>(var);
    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      const uint64_t *p   = (const uint64_t *)buf + j * stride;
      if (stride < 2 || interlace) {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id * stride + l] =
              cs::max(v[elt_id * stride + l], p[k * stride + l]);
        }
      }
      else {
        for (k = 0; k < itf->size; k++) {
          cs_lnum_t elt_id = itf->elt_id[k];
          for (l = 0; l < stride; l++)
            v[elt_id + l * n_elts] =
              cs::max(v[elt_id + l * n_elts], p[k * stride + l]);
        }
      }
      j += itf->size;
    }
    break;
  }

  default:
    bft_error(__FILE__, __LINE__, 0,
              _("Called cs_interface_set_max with unhandled datatype (%d)."),
              (int)datatype);
  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the maximum of values for elements associated with an
 * interface set, allowing control over periodicity.
 *
 * On input, the variable array should contain local contributions. On output,
 * contributions from matching elements on parallel or periodic boundaries
 * have been added.
 *
 * Only the values of elements belonging to the interfaces are modified.
 *
 * \param[in]       ifs        pointer to a fvm_interface_set_t structure
 * \param[in]       n_elts     number of elements in var buffer
 * \param[in]       stride     number of values (non interlaced) by entity
 * \param[in]       interlace  true if variable is interlaced (for stride > 1)
 * \param[in]       datatype   type of data considered
 * \param[in]       tr_ignore  if > 0, ignore periodicity with rotation;
 *                             if > 1, ignore all periodic transforms
 * \param[in, out]  var        variable buffer
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_max_tr(const cs_interface_set_t  *ifs,
                        cs_lnum_t                  n_elts,
                        cs_lnum_t                  stride,
                        bool                       interlace,
                        cs_datatype_t              datatype,
                        int                        tr_ignore,
                        void                      *var)
{
  cs_lnum_t stride_size = cs_datatype_size[datatype]*stride;
  unsigned char *buf = nullptr;

  int n_tr = 0;

  if (tr_ignore > 0 && ifs->periodicity != nullptr) {
    if (tr_ignore < 2) {
      int n_tr_max = fvm_periodicity_get_n_transforms(ifs->periodicity);
      for (int tr_id = 0; tr_id < n_tr_max; tr_id++) {
        if (fvm_periodicity_get_type(ifs->periodicity, tr_id)
            < FVM_PERIODICITY_ROTATION)
          n_tr = tr_id + 1;
      }
    }
    n_tr += 1; /* add base "identity" transform_id */
  }

  if (n_tr < 1) {
    cs_interface_set_max(ifs,
                         n_elts,
                         stride,
                         interlace,
                         datatype,
                         var);
    return;
  }

  /* We can use a fixed max periodicity type here based on translation,
     because when even translation is ignored, n_tr has been set to 1 above,
     and the first transform is "no transform", so tests should always
     return a correct value */

  fvm_periodicity_type_t tr_threshold = FVM_PERIODICITY_ROTATION;

  CS_MALLOC(buf, cs_interface_set_n_elts(ifs)*stride_size, unsigned char);

  if (stride < 2 || interlace)
    cs_interface_set_copy_array(ifs,
                                datatype,
                                stride,
                                true, /* src_on_parent */
                                var,
                                buf);

  else
    _interface_set_copy_array_ni(ifs,
                                 datatype,
                                 n_elts,
                                 stride,
                                 var,
                                 buf);

  /* Now increment values */

  cs_lnum_t j = 0;

  for (int i = 0; i < ifs->size; i++) {

    cs_interface_t *itf = ifs->interfaces[i];

    for (int tr_id = 0; tr_id < n_tr; tr_id++) {

      cs_lnum_t s_id = itf->tr_index[tr_id];
      cs_lnum_t e_id = itf->tr_index[tr_id+1];
      if (e_id > s_id && tr_id > 0) {
        if (  fvm_periodicity_get_type(ifs->periodicity, tr_id-1)
            >= tr_threshold)
          continue;
      }

      switch (datatype) {

      case CS_CHAR:
        {
        char       *v = static_cast<char *>(var);
        const char *p = (const char *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_FLOAT:
        {
        float       *v = static_cast<float *>(var);
        const float *p = (const float *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_DOUBLE:
        {
        double       *v = static_cast<double *>(var);
        const double *p = (const double *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_INT32:
        {
        int32_t       *v = static_cast<int32_t *>(var);
        const int32_t *p = (const int32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_INT64:
        {
        int64_t       *v = static_cast<int64_t *>(var);
        const int64_t *p = (const int64_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_UINT16:
        {
        uint16_t       *v = static_cast<uint16_t *>(var);
        const uint16_t *p = (const uint16_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_UINT32:
        {
        uint32_t       *v = static_cast<uint32_t *>(var);
        const uint32_t *p = (const uint32_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      case CS_UINT64:
        {
        uint64_t       *v = static_cast<uint64_t *>(var);
        const uint64_t *p = (const uint64_t *)buf + j * stride;
        if (stride < 2 || interlace) {
          for (cs_lnum_t k = s_id; k < e_id; k++) {
            cs_lnum_t elt_id = itf->elt_id[k];
            for (cs_lnum_t l = 0; l < stride; l++)
              v[elt_id * stride + l] =
                cs::max(v[elt_id * stride + l], p[k * stride + l]);
          }
        }
          else {
            for (cs_lnum_t k = s_id; k < e_id; k++) {
              cs_lnum_t elt_id = itf->elt_id[k];
              for (cs_lnum_t l = 0; l < stride; l++)
                v[elt_id + l*n_elts] = cs::max(v[elt_id + l*n_elts],
                                               p[k*stride + l]);
            }
          }
        }
        break;

      default:
        bft_error(__FILE__, __LINE__, 0,
                  _("Called %s with unhandled datatype (%d)."),
                  __func__, (int)datatype);
      }

    } /* End of loop on transformations */

    j += itf->size;

  }

  CS_FREE(buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add matching element id information to an interface set.
 *
 * This information is required by calls to cs_interface_get_match_ids(),
 * and may be freed using cs_interface_set_free_match_ids().
 *
 * \param[in]  ifs  pointer to interface set structure
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_add_match_ids(cs_interface_set_t  *ifs)
{
  ifs->match_id_rc += 1;

  if (ifs->match_id_rc > 1)
    return;

  int i;
  cs_lnum_t j;
  int local_rank = 0;
  cs_lnum_t *send_buf = nullptr;

#if defined(HAVE_MPI)

  int n_ranks = 1;
  int request_count = 0;
  MPI_Request *request = nullptr;
  MPI_Status *status  = nullptr;

  if (ifs->comm != MPI_COMM_NULL) {
    MPI_Comm_rank(ifs->comm, &local_rank);
    MPI_Comm_size(ifs->comm, &n_ranks);
  }

#endif

  assert(ifs != nullptr);

  CS_MALLOC(send_buf, cs_interface_set_n_elts(ifs), cs_lnum_t);

  /* Prepare send buffer first (for same rank, send_order is swapped
     with match_id directly) */

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_lnum_t k;
    cs_interface_t *itf = ifs->interfaces[i];

    /* When this function is called, a distant-rank interface should have
       a send_order array, but not a match_id array */

    assert(itf->match_id == nullptr);
    CS_MALLOC(itf->match_id, itf->size, cs_lnum_t);

    for (k = 0; k < itf->size; k++)
      send_buf[j + k] = itf->elt_id[itf->send_order[k]];

    j += itf->size;

  }

  /* Now exchange data */

#if defined(HAVE_MPI)
  if (n_ranks > 1) {
    CS_MALLOC(request, ifs->size*2, MPI_Request);
    CS_MALLOC(status, ifs->size*2, MPI_Status);
  }
#endif

  for (i = 0, j = 0; i < ifs->size; i++) {

    cs_interface_t *itf = ifs->interfaces[i];

    if (itf->rank == local_rank)
      memcpy(itf->match_id, send_buf + j, itf->size*sizeof(cs_lnum_t));

#if defined(HAVE_MPI)

    else /* if (itf->rank != local_rank) */
      MPI_Irecv(itf->match_id,
                itf->size,
                CS_MPI_LNUM,
                itf->rank,
                itf->rank,
                ifs->comm,
                &(request[request_count++]));

#endif

      j += itf->size;

    }

#if defined(HAVE_MPI)

  if (n_ranks > 1) {

    for (i = 0, j = 0; i < ifs->size; i++) {
      cs_interface_t *itf = ifs->interfaces[i];
      if (itf->rank != local_rank)
        MPI_Isend(send_buf + j,
                  itf->size,
                  CS_MPI_LNUM,
                  itf->rank,
                  local_rank,
                  ifs->comm,
                  &(request[request_count++]));
      j += itf->size;
    }

    MPI_Waitall(request_count, request, status);

    CS_FREE(request);
    CS_FREE(status);

  }

#endif /* defined(HAVE_MPI) */

  CS_FREE(send_buf);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Free matching element id information of an interface set.
 *
 * This information is used by calls to cs_interface_get_match_ids(),
 * and may be defined using cs_interface_set_add_match_ids().
 *
 * \param[in]  ifs  pointer to interface set structure
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_free_match_ids(cs_interface_set_t  *ifs)
{
  assert(ifs != nullptr);

  if (ifs->match_id_rc > 0)
    ifs->match_id_rc -= 1;

  if (ifs->match_id_rc > 0)
    return;

  for (int i = 0; i < ifs->size; i++) {
    cs_interface_t *itf = ifs->interfaces[i];
    assert(itf->send_order != nullptr || itf->size == 0);
    CS_FREE(itf->match_id);
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Tag mutiple elements of local interface with a given values.
 *
 * This is effective only on an interface matching the current rank,
 * and when multiple (periodic) instances of a given element appear on that
 * rank, al instances except the first are tagged with the chosen value.
 *
 * \param[in]       itf          pointer to interface structure
 * \param[in]       periodicity  periodicity information (nullptr if none)
 * \param[in]       tr_ignore   if > 0, ignore periodicity with rotation;
 *                              if > 1, ignore all periodic transforms
 * \param[in]       tag_value   tag to assign
 * \param[in, out]  tag         global tag array for elements
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_tag_local_matches(const cs_interface_t     *itf,
                               const fvm_periodicity_t  *periodicity,
                               int                       tr_ignore,
                               cs_gnum_t                 tag_value,
                               cs_gnum_t                *tag)
{
  int l_rank = cs::max(cs_glob_rank_id, 0);

  if (itf->rank != l_rank)
    return;

  /* Build temporary local match id */

  cs_lnum_t  *match_id;
  CS_MALLOC(match_id, itf->size, cs_lnum_t);

  for (cs_lnum_t i = 0; i < itf->size; i++)
    match_id[i] = itf->elt_id[itf->send_order[i]];

  /* Also filter by transform type if needed */

  fvm_periodicity_type_t p_type_max = FVM_PERIODICITY_MIXED;
  int n_tr_max = itf->tr_index_size - 2;

  assert(n_tr_max == fvm_periodicity_get_n_transforms(periodicity));

  if (tr_ignore == 1)
    p_type_max = FVM_PERIODICITY_TRANSLATION;
  else if (tr_ignore  == 2)
    p_type_max = FVM_PERIODICITY_NULL;

  const cs_lnum_t *tr_index = itf->tr_index;

  for (int tr_id = 0; tr_id < n_tr_max; tr_id++) {

    if (fvm_periodicity_get_type(periodicity, tr_id) > p_type_max)
      continue;

    cs_lnum_t s_id = tr_index[tr_id+1];
    cs_lnum_t e_id = tr_index[tr_id+2];

    for (cs_lnum_t j = s_id; j < e_id; j++) {
      cs_lnum_t k = cs::max(itf->elt_id[j], match_id[j]);
      tag[k] = tag_value;
    }

  }

  CS_FREE(match_id);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Dump printout of an interface list.
 *
 * \param[in]  ifs  pointer to structure that should be dumped
 */
/*----------------------------------------------------------------------------*/

void
cs_interface_set_dump(const cs_interface_set_t  *ifs)
{
  int i;
  cs_lnum_t j;

  /* Dump local info */

  if (ifs == nullptr) {
    bft_printf("  interface list: nil\n");
    return;
  }

  bft_printf("  interface list: %p\n"
             "  n interfaces:   %d\n",
             (const void *)ifs, ifs->size);

  for (i = 0, j = 0; i < ifs->size; i++) {
    bft_printf("\n  interface %d:\n", i);
    _cs_interface_dump(ifs->interfaces[i]);
    j += (ifs->interfaces[i])->size;
  }

  if (ifs->periodicity != nullptr)
    bft_printf("\n  periodicity %p:\n", (const void *)(ifs->periodicity));
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
