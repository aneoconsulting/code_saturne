/*============================================================================
 * Manipulation of global indexed list
 *===========================================================================*/

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
 *---------------------------------------------------------------------------*/

#include <assert.h>
#include <stdio.h>
#include <string.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *---------------------------------------------------------------------------*/

#include "fvm/fvm_io_num.h"

#include "base/cs_all_to_all.h"
#include "base/cs_block_dist.h"
#include "mesh/cs_join_util.h"
#include "base/cs_mem.h"
#include "base/cs_order.h"
#include "base/cs_search.h"
#include "base/cs_sort.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *---------------------------------------------------------------------------*/

#include "mesh/cs_join_set.h"

/*---------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Macro and type definitions
 *===========================================================================*/

/*============================================================================
 * Private function definitions
 *===========================================================================*/

/*----------------------------------------------------------------------------
 * Sort an array "a" and apply the sort to its associated array "b"
 * (global numbering).
 *
 * Sort is realized using a shell sort (Knuth algorithm).
 *
 * parameters:
 *   l <-- left bound
 *   r <-- right bound
 *   a <-> array to sort
 *   b <-> associated array
 *---------------------------------------------------------------------------*/

static void
_coupled_adapted_gnum_shellsort(cs_lnum_t  l,
                                cs_lnum_t  r,
                                cs_gnum_t  a[],
                                cs_gnum_t  b[])
{
  cs_lnum_t  i, start;
  cs_gnum_t  ref;

  cs_lnum_t  size = r - l;

  if (size == 0)
    return;

  cs_sort_coupled_gnum_shell(l, r, a, b);

  /* Order b array for each sub-list where a[] is constant */

  start = l;
  while (start < r) {

    ref = a[start];
    for (i = start; i < r && ref == a[i]; i++);

    cs_sort_gnum_shell(start, i, b);
    start = i;

  }

}

/*----------------------------------------------------------------------------
 * Test if an array of local numbers with stride 2 is lexicographically
 * ordered.
 *
 * parameters:
 *   number   <-- array of all entity numbers (number of entity i
 *                given by number[i] or number[list[i] - 1])
 *   nb_ent   <-- number of entities considered
 *
 * returns:
 *   1 if ordered, 0 otherwise.
 *----------------------------------------------------------------------------*/

static int
_order_local_test_s2(const cs_lnum_t  number[],
                     size_t           n_elts)
{
  size_t i = 0;

  assert (number != nullptr || n_elts == 0);

  for (i = 1 ; i < n_elts ; i++) {
    size_t i_prev, k;
    bool unordered = false;
    i_prev = i-1;
    for (k = 0; k < 2; k++) {
      if (number[i_prev*2 + k] < number[i*2 + k])
        break;
      else if (number[i_prev*2 + k] > number[i*2 + k])
        unordered = true;
    }
    if (unordered == true)
      break;
  }

  if (i == n_elts || n_elts == 0)
    return 1;
  else
    return 0;
}

/*----------------------------------------------------------------------------
 * Descend binary tree for the lexicographical ordering of an local numbering
 * array of stride 2.
 *
 * parameters:
 *   number    <-- pointer to numbers of entities that should be ordered.
 *   level     <-- level of the binary tree to descend
 *   n_elts    <-- number of entities in the binary tree to descend
 *   order     <-> ordering array
 *----------------------------------------------------------------------------*/

inline static void
_order_descend_tree_s2(const cs_lnum_t  number[],
                       size_t           level,
                       const size_t     n_elts,
                       cs_lnum_t        order[])
{
  size_t i_save, i1, i2, j, lv_cur;

  i_save = (size_t)(order[level]);

  while (level <= (n_elts/2)) {

    lv_cur = (2*level) + 1;

    if (lv_cur < n_elts - 1) {

      i1 = (size_t)(order[lv_cur+1]);
      i2 = (size_t)(order[lv_cur]);

      for (j = 0; j < 2; j++) {
        if (number[i1*2 + j] != number[i2*2 + j])
          break;
      }

      if (j < 2) {
        if (number[i1*2 + j] > number[i2*2 + j])
          lv_cur++;
      }

    }

    if (lv_cur >= n_elts) break;

    i1 = i_save;
    i2 = (size_t)(order[lv_cur]);

    for (j = 0; j < 2; j++) {
      if (number[i1*2 + j] != number[i2*2 + j])
        break;
    }

    if (j == 2) break;
    if (number[i1*2 + j] >= number[i2*2 + j]) break;

    order[level] = order[lv_cur];
    level = lv_cur;

  }

  order[level] = i_save;
}

/*----------------------------------------------------------------------------
 * Order array of local numbers with stride 2 lexicographically.
 *
 * parameters:
 *   number   <-- array of entity numbers
 *   order    --> pre-allocated ordering table
 *   n_elts   <-- number of entities considered
 *----------------------------------------------------------------------------*/

static void
_order_local_s2(const cs_lnum_t  number[],
                cs_lnum_t        order[],
                const size_t     n_elts)
{
  size_t i;
  cs_lnum_t o_save;

  assert (number != nullptr || n_elts == 0);

  /* Initialize ordering array */

  for (i = 0 ; i < n_elts ; i++)
    order[i] = i;

  if (n_elts < 2)
    return;

  /* Create binary tree */

  i = (n_elts / 2) ;
  do {
    i--;
    _order_descend_tree_s2(number, i, n_elts, order);
  } while (i > 0);

  /* Sort binary tree */

  for (i = n_elts - 1 ; i > 0 ; i--) {
    o_save   = order[0];
    order[0] = order[i];
    order[i] = o_save;
    _order_descend_tree_s2(number, 0, i, order);
  }
}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *===========================================================================*/

/*----------------------------------------------------------------------------
 * Allocate a resizable array.
 *
 * parameters:
 *   max_size <-- initial number of elements to allocate
 *
 * returns:
 *   pointer to a new alloacted resizable array
 *---------------------------------------------------------------------------*/

cs_join_rset_t *
cs_join_rset_create(cs_lnum_t  max_size)
{
  cs_join_rset_t  *new_set = nullptr;

  if (max_size > 0) {

    CS_MALLOC(new_set, 1, cs_join_rset_t);

    new_set->n_max_elts = max_size;
    new_set->n_elts = 0;

    CS_MALLOC(new_set->array, max_size, cs_lnum_t);

  }

  return new_set;
}

/*----------------------------------------------------------------------------
 * Destroy a cs_join_rset_t structure.
 *
 * parameters:
 *   set <-- pointer to pointer to the cs_join_rset_t structure to destroy
 *---------------------------------------------------------------------------*/

void
cs_join_rset_destroy(cs_join_rset_t  **set)
{
  if (*set != nullptr) {
    CS_FREE((*set)->array);
    CS_FREE(*set);
  }
}

/*----------------------------------------------------------------------------
 * Check if we need to resize the current cs_join_rset_t structure and do
 * it if necessary.
 *
 * parameters:
 *   set       <-- pointer to pointer to the cs_join_rset_t structure to test
 *   test_size <-- target size
 *---------------------------------------------------------------------------*/

void
cs_join_rset_resize(cs_join_rset_t  **set,
                    cs_lnum_t         test_size)
{
  if (*set != nullptr) {

    if (test_size > 0) {

      cs_join_rset_t  *_set = *set;

      if (test_size < _set->n_max_elts)
        return;

      if (_set->n_max_elts == 0)
        _set->n_max_elts = test_size;
      else if (test_size >= _set->n_max_elts) {
        while (test_size >= _set->n_max_elts)
          _set->n_max_elts *= 2; /* Double the list size */
      }

      CS_REALLOC(_set->array, _set->n_max_elts, cs_lnum_t);
      assert(test_size <= _set->n_max_elts);

    }

  }
  else
    *set = cs_join_rset_create(test_size);
}

/*----------------------------------------------------------------------------
 * Create a new cs_join_eset_t structure.
 *
 * parameters:
 *   init_size <-- number of initial equivalences to allocate
 *
 * returns:
 *   a pointer to a new cs_join_eset_t structure
 *---------------------------------------------------------------------------*/

cs_join_eset_t *
cs_join_eset_create(cs_lnum_t  init_size)
{
  cs_join_eset_t  *new_set = nullptr;

  CS_MALLOC(new_set, 1, cs_join_eset_t);

  new_set->n_max_equiv = init_size; /* default value */
  new_set->n_equiv = 0;

  CS_MALLOC(new_set->equiv_couple, 2*new_set->n_max_equiv, cs_lnum_t);

  return new_set;
}

/*----------------------------------------------------------------------------
 * Check if the requested size if allocated in the structure.
 *
 * Reallocate cs_join_eset_t structure if necessary.
 *
 * parameters:
 *   request_size <-- necessary size
 *   equiv_set    <-> pointer to pointer to the cs_join_eset_t struct.
 *---------------------------------------------------------------------------*/

void
cs_join_eset_check_size(cs_lnum_t         request_size,
                        cs_join_eset_t  **equiv_set)
{
  assert(equiv_set != nullptr);

  cs_join_eset_t  *eset = *equiv_set;

  if (eset == nullptr)
    eset = cs_join_eset_create(request_size);

  if (request_size + 1 > eset->n_max_equiv) {

    if (eset->n_max_equiv == 0)
      eset->n_max_equiv = 2;

    eset->n_max_equiv *= 2;

    CS_REALLOC(eset->equiv_couple, 2*eset->n_max_equiv, cs_lnum_t);

  }

  /* Ensure return value is set */

  *equiv_set = eset;
}

/*----------------------------------------------------------------------------
 * Destroy a cs_join_eset_t structure.
 *
 * parameters:
 *   equiv_set <-- pointer to pointer to the structure to destroy
 *---------------------------------------------------------------------------*/

void
cs_join_eset_destroy(cs_join_eset_t  **equiv_set)
{
  if (*equiv_set != nullptr) {
    CS_FREE((*equiv_set)->equiv_couple);
    CS_FREE(*equiv_set);
  }
}

/*----------------------------------------------------------------------------
 * Clean a cs_join_eset_t structure.
 *
 * If necessary, create a new cs_join_eset_t structure with no redundancy.
 *
 * parameters:
 *   eset <-- pointer to pointer to the cs_join_eset_t structure to clean
 *---------------------------------------------------------------------------*/

void
cs_join_eset_clean(cs_join_eset_t  **eset)
{
  assert(eset != nullptr);

  int  i;
  cs_lnum_t  prev, current;

  cs_lnum_t  count = 0;
  cs_lnum_t  *order = nullptr;
  cs_join_eset_t  *new_eset = nullptr;
  cs_join_eset_t  *_eset = *eset;

  if (_eset == nullptr)
    return;

  if (_eset->n_equiv == 1)
    return;

  CS_MALLOC(order, _eset->n_equiv, cs_lnum_t);

  if (_order_local_test_s2(_eset->equiv_couple,
                           _eset->n_equiv) == false) {

    /* Order equiv_lst */

    _order_local_s2(_eset->equiv_couple,
                    order,
                    _eset->n_equiv);

  }
  else
    for (i = 0; i < _eset->n_equiv; i++)
      order[i] = i;

  /* Count the number of redundancies */

  count = 0;

  for (i = 1; i < _eset->n_equiv; i++) {

    prev = order[i-1];
    current = order[i];

    if (_eset->equiv_couple[2*prev] == _eset->equiv_couple[2*current])
      if (_eset->equiv_couple[2*prev+1] == _eset->equiv_couple[2*current+1])
        count++;

  } /* End of loop on equivalences */

  new_eset = cs_join_eset_create(_eset->n_equiv - count);

  new_eset->n_equiv =  _eset->n_equiv - count;

  if (new_eset->n_equiv > new_eset->n_max_equiv) {
    new_eset->n_max_equiv = new_eset->n_equiv;
    CS_REALLOC(new_eset->equiv_couple, 2*new_eset->n_max_equiv, cs_lnum_t);
  }

  if (new_eset->n_equiv > 0) {

    new_eset->equiv_couple[0] = _eset->equiv_couple[2*order[0]];
    new_eset->equiv_couple[1] = _eset->equiv_couple[2*order[0]+1];
    count = 1;

    for (i = 1; i < _eset->n_equiv; i++) {

      prev = order[i-1];
      current = order[i];

      if (_eset->equiv_couple[2*prev] != _eset->equiv_couple[2*current]) {
        new_eset->equiv_couple[2*count] = _eset->equiv_couple[2*current];
        new_eset->equiv_couple[2*count+1] = _eset->equiv_couple[2*current+1];
        count++;
      }
      else {

        if (_eset->equiv_couple[2*prev+1] != _eset->equiv_couple[2*current+1]) {
          new_eset->equiv_couple[2*count] = _eset->equiv_couple[2*current];
          new_eset->equiv_couple[2*count+1] = _eset->equiv_couple[2*current+1];
          count++;
        }

      }

    } /* End of loop on equivalences */

    assert(count == new_eset->n_equiv);

  }

  *eset = new_eset;

  /* Free memory */

  cs_join_eset_destroy(&_eset);

  CS_FREE(order);
}

/*----------------------------------------------------------------------------
 * Create a cs_join_gset_t structure (indexed list on global numbering)
 *
 * parameters:
 *   n_elts <-- number of elements composing the list
 *
 * returns:
 *   a new allocated pointer to a cs_join_gset_t structure.
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_create(cs_lnum_t  n_elts)
{
  cs_lnum_t  i;

  cs_join_gset_t  *new_set = nullptr;

  CS_MALLOC(new_set, 1, cs_join_gset_t);

  new_set->n_elts = n_elts;
  new_set->n_g_elts = 0;

  CS_MALLOC(new_set->g_elts, n_elts, cs_gnum_t);
  for (i = 0; i < n_elts; i++)
    new_set->g_elts[i] = 0;

  CS_MALLOC(new_set->index, n_elts + 1, cs_lnum_t);
  for (i = 0; i < n_elts + 1; i++)
    new_set->index[i] = 0;

  new_set->g_list = nullptr;

  return new_set;
}

/*----------------------------------------------------------------------------
 * Build a cs_join_gset_t structure to store all the potential groups
 * between elements.
 *
 * Values in g_elts are the tag values and values in g_list
 * are position in tag array.
 *
 * parameters:
 *   n_elts <-- number of elements in tag array
 *   tag    <-- tag array used to define a new cs_join_gset_t
 *
 * returns:
 *   a new allocated cs_join_gset_t structure
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_create_from_tag(cs_lnum_t        n_elts,
                             const cs_gnum_t  tag[])
{
  cs_lnum_t  i, n_list_elts;
  cs_gnum_t  prev;

  cs_lnum_t  *order = nullptr;
  cs_join_gset_t  *set = nullptr;

  if (n_elts == 0) {
    set = cs_join_gset_create(n_elts);
    return  set;
  }

  /* Order tag */

  assert(tag != nullptr);

  CS_MALLOC(order, n_elts, cs_lnum_t);

  cs_order_gnum_allocated(nullptr, tag, order, n_elts);

  /* Create a cs_join_gset_t structure to store the initial position of equiv.
     element in tag */

  prev = tag[order[0]];
  n_list_elts = 1;

  /* Count the number of elements which will compose the set->g_elts */

  for (i = 1; i < n_elts; i++) {

    cs_gnum_t  cur = tag[order[i]];

    if (prev != cur) {
      n_list_elts++;
      prev = cur;
    }

  }

  set = cs_join_gset_create(n_list_elts);

  if (n_list_elts > 0) {

    cs_lnum_t  shift;
    cs_lnum_t  count = 0;

    /* Define the list of elements in set->g_elts and count the number of
       associated entities */

    prev = tag[order[0]];
    set->g_elts[0] = prev;
    set->index[1] += 1;
    n_list_elts = 1;

    for (i = 1; i < n_elts; i++) {

      cs_gnum_t  cur = tag[order[i]];

      if (prev != cur) {
        prev = cur;
        set->g_elts[n_list_elts] = cur;
        n_list_elts++;
        set->index[n_list_elts] += 1;
      }
      else
        set->index[n_list_elts] += 1;

    }

    /* Build index for the set */

    for (i = 0; i < set->n_elts; i++)
      set->index[i+1] += set->index[i];

    /* Fill list */

    CS_MALLOC(set->g_list, set->index[set->n_elts], cs_gnum_t);

    n_list_elts = 0;
    prev = tag[order[0]];
    set->g_list[0] = order[0];

    for (i = 1; i < n_elts; i++) {

      cs_lnum_t  o_id = order[i];
      cs_gnum_t  cur = tag[o_id];

      if (prev != cur) {
        prev = cur;
        count = 0;
        n_list_elts++;
        shift = set->index[n_list_elts];
        set->g_list[shift] = o_id;
      }
      else {
        count++;
        shift = count + set->index[n_list_elts];
        set->g_list[shift] = o_id;
      }

    }

  } /* End if n_elts > 0 */

  /* Free memory */

  CS_FREE(order);

  /* Returns pointers */

  return  set;
}

/*----------------------------------------------------------------------------
 * Create a new cs_join_gset_t which holds equivalences between elements of
 * g_list in cs_join_gset_t.
 *
 * For a subset of equivalences, we store their initial value in the return
 * cs_join_gset_t structure. A subset is defined if at least two elements
 * are equivalent.
 *
 * The behavior of this function is near from cs_join_gset_create_from_tag
 * but we don't store the position in init_array but its value in init_array.
 *
 * parameters:
 *   set        <-- pointer to a cs_join_gset_t structure
 *   init_array <-- initial values of set->g_list
 *
 * returns:
 *   a new allocated cs_join_gset_t structure, or nullptr if it would be empty
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_create_by_equiv(const cs_join_gset_t  *set,
                             const cs_gnum_t        init_array[])
{
  cs_lnum_t  i, list_size, n_equiv_grp, count, shift, o_id;
  cs_gnum_t  prev, cur;

  cs_lnum_t  save_i = -1;
  cs_lnum_t  *order = nullptr;
  cs_gnum_t  *couple_list = nullptr;
  cs_join_gset_t  *equiv = nullptr;

  if (init_array == nullptr)
    return nullptr;

  assert(set != nullptr);

  list_size = set->index[set->n_elts];

  /* Order set->g_list */

  CS_MALLOC(order, list_size, cs_lnum_t);
  CS_MALLOC(couple_list, 2*list_size, cs_gnum_t);

  for (i = 0; i < list_size; i++) {
    couple_list[2*i] = set->g_list[i];
    couple_list[2*i+1] = init_array[i];
  }

  cs_order_gnum_allocated_s(nullptr, couple_list, 2, order, list_size);

  /* Create a cs_join_gset_t structure to store the initial value of equiv.
     element in set->g_list */

  /* Count the number of elements which will compose the equiv->g_elts */

  n_equiv_grp = 0;
  prev = set->g_list[order[0]];
  count = 0;

  for (i = 1; i < list_size; i++) {

    cur = set->g_list[order[i]];

    if (prev != cur) {
      count = 0;
      prev = cur;
    }
    else {
      count++;
      if (count == 1)
        n_equiv_grp++;
    }

  }

  equiv = cs_join_gset_create(n_equiv_grp);

  if (n_equiv_grp > 0) {

    /* Define the list of elements in equiv->g_list and count the number of
       associated elements */

    n_equiv_grp = 0;
    prev = set->g_list[order[0]];
    count = 0;

    for (i = 1; i < list_size; i++) {

      cur = set->g_list[order[i]];

      if (prev != cur) {
        count = 0;
        prev = cur;
      }
      else {
        count++;
        if (count == 1) { /* Add this group */
          equiv->g_elts[n_equiv_grp] = cur;
          n_equiv_grp++;
          equiv->index[n_equiv_grp] = 1;
        }
        else {
          assert(count > 1);
          equiv->index[n_equiv_grp] += 1;
        }

      } /* cur == prev */

    } /* End of loop on list_size */

    /* Build index for the set */

    for (i = 0; i < equiv->n_elts; i++)
      equiv->index[i+1] += equiv->index[i];

    /* Fill list */

    CS_MALLOC(equiv->g_list, equiv->index[equiv->n_elts], cs_gnum_t);

    n_equiv_grp = 0;
    prev = set->g_list[order[0]] + 1;

    for (i = 0; i < list_size; i++) {

      o_id = order[i];
      cur = set->g_list[o_id];

      if (prev != cur) {
        count = 0;
        prev = cur;
        save_i = o_id;
      }
      else {

        if (count == 0)
          n_equiv_grp++;

        shift = count + equiv->index[n_equiv_grp-1];
        if (cur != init_array[o_id])
          equiv->g_list[shift] = init_array[o_id];
        else
          equiv->g_list[shift] = init_array[save_i];
        count++;

      } /* cur == prev */

    } /* End of loop on list_size */

  } /* End if n_elts > 0 */

  /* Free memory */

  CS_FREE(couple_list);
  CS_FREE(order);

  /* Returns pointer */

  return  equiv;
}

/*----------------------------------------------------------------------------
 * Copy a cs_join_gset_t structure.
 *
 * parameters:
 *   src <-- pointer to the cs_join_gset_t structure to copy
 *
 * returns:
 *   a new allocated cs_join_gset_t structure.
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_copy(const cs_join_gset_t  *src)
{
  cs_lnum_t  i;

  cs_join_gset_t  *copy = nullptr;

  if (src == nullptr)
    return copy;

  copy = cs_join_gset_create(src->n_elts);

  for (i = 0; i < src->n_elts; i++)
    copy->g_elts[i] = src->g_elts[i];

  for (i = 0; i < src->n_elts + 1; i++)
    copy->index[i] = src->index[i];

  CS_MALLOC(copy->g_list, copy->index[copy->n_elts], cs_gnum_t);

  for (i = 0; i < src->index[src->n_elts]; i++)
    copy->g_list[i] = src->g_list[i];

  return copy;
}

/*----------------------------------------------------------------------------
 * Destroy a cs_join_gset_t structure.
 *
 * parameters:
 *   set <-- pointer to pointer to the cs_join_gset_t structure to destroy
 *---------------------------------------------------------------------------*/

void
cs_join_gset_destroy(cs_join_gset_t  **set)
{
  if (*set != nullptr) {
    CS_FREE((*set)->index);
    CS_FREE((*set)->g_elts);
    CS_FREE((*set)->g_list);
    CS_FREE(*set);
  }
}

/*----------------------------------------------------------------------------
 * Sort a cs_join_gset_t structure according to the global numbering of
 * the g_elts in cs_join_gset_t structure.
 *
 * parameters:
 *   set <-> pointer to the structure to order
 *---------------------------------------------------------------------------*/

void
cs_join_gset_sort_elts(cs_join_gset_t  *set)
{
  int  i, j, k, o_id, shift;
  cs_lnum_t  n_elts;

  cs_lnum_t  *new_index = nullptr;
  cs_lnum_t  *order = nullptr;
  cs_gnum_t  *tmp = nullptr, *g_elts = nullptr, *g_list = nullptr;

  if (set == nullptr)
    return;

  g_elts = set->g_elts;
  g_list = set->g_list;
  n_elts = set->n_elts;

  CS_MALLOC(order, n_elts, cs_lnum_t);
  CS_MALLOC(tmp, n_elts, cs_gnum_t);
  CS_MALLOC(new_index, n_elts + 1, cs_lnum_t);

  for (i = 0; i < n_elts; i++)
    tmp[i] = g_elts[i];

  /* Sort g_elts */

  cs_order_gnum_allocated(nullptr, g_elts, order, n_elts);

  /* Reshape cs_join_gset_t according to the new ordering */

  new_index[0] = 0;

  for (i = 0; i < n_elts; i++) {

    o_id = order[i];
    g_elts[i] = tmp[o_id];
    new_index[i+1] =  new_index[i] + set->index[o_id+1] - set->index[o_id];

  } /* End of loop on elements */

  assert(new_index[n_elts] == set->index[n_elts]);

  /* Define new g_list */

  CS_REALLOC(tmp, set->index[n_elts], cs_gnum_t);

  for (i = 0; i < set->index[n_elts]; i++)
    tmp[i] = g_list[i];

  for (i = 0; i < n_elts; i++) {

    o_id = order[i];
    shift = new_index[i];

    for (k = 0, j = set->index[o_id]; j < set->index[o_id+1]; j++, k++)
      g_list[shift + k] = tmp[j];

  } /* End of loop on elements */

  /* Free memory */

  CS_FREE(set->index);
  CS_FREE(order);
  CS_FREE(tmp);

  /* Return structure */

  set->index = new_index;
  set->g_elts = g_elts;
  set->g_list = g_list;
}

/*----------------------------------------------------------------------------
 * Sort each sub-list of the g_list array in a cs_join_gset_t structure.
 *
 * parameters:
 *   p_set <-> pointer to the structure to sort
 *---------------------------------------------------------------------------*/

void
cs_join_gset_sort_sublist(cs_join_gset_t  *set)
{
  int  i;

  if (set == nullptr)
    return;

  /* Sort each sub-list */

  for (i = 0; i < set->n_elts; i++)
    cs_sort_gnum_shell(set->index[i], set->index[i+1], set->g_list);
}

/*----------------------------------------------------------------------------
 * Invert a cs_join_gset_t structure.
 *
 * parameters:
 *   set <-- pointer to the cs_join_gset_t structure to work with
 *
 * returns:
 *   the new allocated and inverted set structure
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_invert(const cs_join_gset_t  *set)
{
  int  i, j, o_id, shift, elt_id;
  cs_gnum_t  prev, cur;

  cs_lnum_t  list_size = 0, n_elts = 0;
  cs_lnum_t  *count = nullptr;
  cs_lnum_t  *order = nullptr;
  cs_join_gset_t  *invert_set = nullptr;

  if (set == nullptr)
    return invert_set;

  /* Order g_list to count the number of different entities */

  list_size = set->index[set->n_elts];

  if (list_size == 0)
    return cs_join_gset_create(list_size);

  CS_MALLOC(order, list_size, cs_lnum_t);

  cs_order_gnum_allocated(nullptr, set->g_list, order, list_size);

  /* Count the number of elements */

  prev = set->g_list[order[0]] + 1;

  for (i = 0; i < list_size; i++) {

    o_id = order[i];
    cur = set->g_list[o_id];

    if (prev != cur) {
      prev = cur;
      n_elts++;
    }

  }

  invert_set = cs_join_gset_create(n_elts);

  /* Fill g_elts for the inverted set */

  n_elts = 0;
  prev = set->g_list[order[0]] + 1;

  for (i = 0; i < list_size; i++) {

    o_id = order[i];
    cur = set->g_list[o_id];

    if (prev != cur) {
      prev = cur;
      invert_set->g_elts[n_elts] = cur;
      n_elts++;
    }

  }

  CS_FREE(order);

  /* Define an index for the inverted set */

  for (i = 0; i < set->n_elts; i++) {
    for (j = set->index[i]; j < set->index[i+1]; j++) {

      elt_id = cs_search_g_binary(invert_set->n_elts,
                                  set->g_list[j],
                                  invert_set->g_elts);

      if (elt_id == -1)
        bft_error(__FILE__, __LINE__, 0,
                  _("  Fail to build an inverted cs_join_gset_t structure.\n"
                    "  Cannot find %llu in element list.\n"),
                  (unsigned long long)(set->g_list[j]));

      invert_set->index[elt_id+1] += 1;

    }
  } /* End of loop on elements of list */

  for (i = 0; i < invert_set->n_elts; i++)
    invert_set->index[i+1] += invert_set->index[i];

  CS_MALLOC(invert_set->g_list,
            invert_set->index[invert_set->n_elts],
            cs_gnum_t);

  /* Define invert_set->g_list */

  CS_MALLOC(count, invert_set->n_elts, cs_lnum_t);

  for (i = 0; i < invert_set->n_elts; i++)
    count[i] = 0;

  for (i = 0; i < set->n_elts; i++) {
    for (j = set->index[i]; j < set->index[i+1]; j++) {

      elt_id = cs_search_g_binary(invert_set->n_elts,
                                  set->g_list[j],
                                  invert_set->g_elts);

      shift = count[elt_id] + invert_set->index[elt_id];
      invert_set->g_list[shift] = set->g_elts[i];
      count[elt_id] += 1;

    }

  } /* End of loop on elements of list */

  CS_FREE(count);

  return invert_set;
}

/*----------------------------------------------------------------------------
 * Delete redudancies in a cs_join_gset_t structure.
 *
 * Output set has an ordered sub-list for each element in set.
 *
 * parameters:
 *   set <-> pointer to the structure to clean
 *---------------------------------------------------------------------------*/

void
cs_join_gset_clean(cs_join_gset_t  *set)
{
  int  i, j, l, r, save, n_elts;

  int  shift = 0;
  cs_gnum_t  *g_list = nullptr;

  if (set == nullptr)
    return;

  g_list = set->g_list;
  n_elts = set->n_elts;

  /* Sort g_list for each element in index */

  cs_join_gset_sort_sublist(set);

  /* Define a new index without redundant elements */

  save = set->index[0];

  for (i = 0; i < n_elts; i++) {

    l = save;
    r = set->index[i+1];

    if (r - l > 0) {

      g_list[shift++] = g_list[l];

      for (j = l + 1; j < r; j++)
        if (g_list[j] != g_list[j-1])
          g_list[shift++] = g_list[j];

    }

    save = r;
    set->index[i+1] = shift;

  } /* End of loop on elements */
}

/*----------------------------------------------------------------------------
 * Delete redudancies in g_list array of a cs_join_gset_t structure.
 *
 * parameters:
 *   set          <-> pointer to the structure to clean
 *   linked_array <-> array for which redundancies are scanned
 *---------------------------------------------------------------------------*/

void
cs_join_gset_clean_from_array(cs_join_gset_t  *set,
                              cs_gnum_t        linked_array[])
{
  int  i, j, l, r;
  cs_lnum_t  n_elts;

  int  shift = 0;
  cs_lnum_t  *new_index = nullptr;
  cs_gnum_t  *g_list = nullptr;

  if (set == nullptr)
    return;

  if (linked_array == nullptr)
    return;

  g_list = set->g_list;
  n_elts = set->n_elts;

  /* Sort linked_array and apply change to g_list for each element in index */

  for (i = 0; i < n_elts; i++)
    _coupled_adapted_gnum_shellsort(set->index[i],
                                    set->index[i+1],
                                    linked_array,
                                    g_list);

  /* Define a new index without redundant elements */

  CS_MALLOC(new_index, n_elts + 1, cs_lnum_t);
  new_index[0] = 0;

  for (i = 0; i < n_elts; i++) {

    l = set->index[i];
    r = set->index[i+1];

    if (r - l > 0) {

      g_list[shift] = g_list[l];
      shift++;

      for (j = l + 1; j < r; j++) {

        if (linked_array[j] != linked_array[j-1]) {
          g_list[shift] = g_list[j];
          shift++;
        }
        assert(g_list[shift-1] <= g_list[j]);

      }
      new_index[i+1] = shift;

    }
    else { /* No match for this element */

      new_index[i+1] = new_index[i];

    }

  } /* End of loop on elements */

  /* Reshape cs_join_gset_t structure */

  CS_REALLOC(g_list, new_index[n_elts], cs_gnum_t);
  CS_FREE(set->index);

  set->index = new_index;
  set->g_list = g_list;
}

/*----------------------------------------------------------------------------
 * Concatenate the two g_elts and g_list arrays.
 *
 * Order the new concatenated array and delete redundant elements.
 * We get a single ordered array.
 *
 * parameters:
 *   set       <-- pointer to the structure to work with
 *   n_elts    --> number of elements in the new set
 *   new_array --> pointer to the new created array
 *---------------------------------------------------------------------------*/

void
cs_join_gset_single_order(const cs_join_gset_t  *set,
                          cs_lnum_t             *n_elts,
                          cs_gnum_t             *new_array[])
{
  cs_lnum_t  _n_elts = 0;
  cs_gnum_t  *_new_array = nullptr;

  *n_elts = _n_elts;
  *new_array = _new_array;

  if (set == nullptr) /* Nothing to do */
    return;

  _n_elts = set->n_elts;

  if (_n_elts > 0) {

    cs_lnum_t  i, shift;
    cs_gnum_t  prev;

    cs_lnum_t  *order = nullptr;
    cs_gnum_t  *elt_list = nullptr;

    _n_elts += set->index[_n_elts]; /* Add number of elements in g_list */

    CS_MALLOC(elt_list, _n_elts, cs_gnum_t);

    for (i = 0; i < set->n_elts; i++)
      elt_list[i] = set->g_elts[i];

    shift = set->n_elts;
    for (i = 0; i < set->index[set->n_elts]; i++)
      elt_list[shift + i] = set->g_list[i];

    /* Define an ordered list of elements */

    CS_MALLOC(_new_array, _n_elts, cs_gnum_t);
    CS_MALLOC(order, _n_elts, cs_lnum_t);

    cs_order_gnum_allocated(nullptr, elt_list, order, _n_elts);

    for (i = 0; i < _n_elts; i++)
      _new_array[i] = elt_list[order[i]];

    /* Delete redundant elements */

    shift = 0;
    prev = _new_array[0] + 1;

    for (i = 0; i < _n_elts; i++) {

      if (prev != _new_array[i]) {

        _new_array[shift] = _new_array[i];
        prev = _new_array[i];
        shift++;

      }

    }
    _n_elts = shift; /* Real number of elements without redundancy */

    /* Memory management */

    CS_FREE(order);
    CS_FREE(elt_list);
    CS_REALLOC(_new_array, _n_elts, cs_gnum_t);

  } /* End if n_elts > 0 */

  /* Set return pointers */

  *n_elts = _n_elts;
  *new_array = _new_array;
}

/*----------------------------------------------------------------------------
 * Compress a g_list such as for each element "e" in g_elts:
 *  - there is no redundancy for the linked elements of set->g_list
 *  - there is no element in set->g_list < e except if this element is not
 *    present in g_elts
 *
 * g_list and g_elts must be ordered before calling this function
 *
 * parameters:
 *   set <-> pointer to the structure to work with
 *---------------------------------------------------------------------------*/

void
cs_join_gset_compress(cs_join_gset_t  *set)
{
  cs_lnum_t  i, j, start, end, save, shift;
  cs_gnum_t  cur;

  if (set == nullptr)
    return;

  if (set->n_elts == 0)
    return;

  shift = 0;
  save = set->index[0];

  for (i = 0; i < set->n_elts; i++) {

    cur = set->g_elts[i];
    start = save;
    end = set->index[i+1];

    if (end - start > 0) {

      /* Sub-lists must be ordered */

      if (cur < set->g_list[start])
        set->g_list[shift++] = set->g_list[start];
      else if (cur > set->g_list[start]) {

        int  id = cs_search_g_binary(i+1,
                                     set->g_list[start],
                                     set->g_elts);

        if (id == -1) /* Not found. Keep it. */
          set->g_list[shift++] = set->g_list[start];

      }

      for (j = start + 1; j < end; j++) {

        if (cur < set->g_list[j]) {
          if (set->g_list[j-1] != set->g_list[j])
            set->g_list[shift++] = set->g_list[j];
        }
        else if (cur > set->g_list[j]) {

          int  id = cs_search_g_binary(i+1,
                                       set->g_list[j],
                                       set->g_elts);

          if (id == -1) /* Not found. Keep it. */
            set->g_list[shift++] = set->g_list[j];

        }

      } /* End of loop on sub-elements */

    } /* end - start > 0 */

    save = end;
    set->index[i+1] = shift;

  } /* End of loop on elements in g_elts */

  /* Reshape cs_join_gset_t structure if necessary */

  if (save != set->index[set->n_elts]) {
    assert(save > set->index[set->n_elts]);
    CS_REALLOC(set->g_list, set->index[set->n_elts], cs_gnum_t);
  }

#if 0 && defined(DEBUG) && !defined(NDEBUG)
  cs_join_gset_dump(nullptr, set);
#endif
}

/*----------------------------------------------------------------------------
 * Delete redundancies in set->g_elts.
 *
 * Merge sub-arrays associated to a common set->g_elts[i].
 *
 * parameters:
 *   set       <-- pointer to the structure to work with
 *   order_tag <-- 0: set->g_elts is not ordered, 1: ordered
 *---------------------------------------------------------------------------*/

void
cs_join_gset_merge_elts(cs_join_gset_t  *set,
                        int              order_tag)
{
  cs_lnum_t  i, save, start, end, n_init_elts, n_sub_elts;
  cs_gnum_t  prev, cur;

  if (set == nullptr)
    return;

  n_init_elts = set->n_elts;

  if (n_init_elts < 2)
    return;

  assert(order_tag == 0 || order_tag == 1);

  /* Delete redundancies in g_elts. Merge sub-lists associated to common
     g_elts */

  if (order_tag == 0)
    cs_join_gset_sort_elts(set);

  set->n_elts = 0;            /* Reset and will be redefinied */
  prev = set->g_elts[0] + 1;  /* Force prev to be different from g_elts[0] */
  save = set->index[0];

  for (i = 0; i < n_init_elts; i++) {

    cur = set->g_elts[i];
    start = save;
    end = set->index[i+1];
    save = end;
    n_sub_elts = end - start;

    if (prev != cur) {

      set->g_elts[set->n_elts] = cur;
      set->n_elts += 1;
      set->index[set->n_elts] = n_sub_elts;
      prev = cur;

    }
    else {

      set->index[set->n_elts] += n_sub_elts;

    } /* prev != next */

  } /* Loop on elements of set->g_elts */

  /* Get the new index */

  for (i = 0; i < set->n_elts; i++)
    set->index[i+1] += set->index[i];

  /* Reshape cs_join_gset_t structure if necessary */

  if (n_init_elts != set->n_elts) {

    assert(n_init_elts > set->n_elts);

    CS_REALLOC(set->g_elts, set->n_elts, cs_gnum_t);
    CS_REALLOC(set->index, set->n_elts + 1, cs_lnum_t);
    CS_REALLOC(set->g_list, set->index[set->n_elts], cs_gnum_t);

  }

#if 0 && defined(DEBUG) && !defined(NDEBUG)
  cs_join_gset_dump(nullptr, set);
#endif
}

#if defined(HAVE_MPI)

/*----------------------------------------------------------------------------
 * Synchronize a cs_join_gset_t structure and distribute the resulting set
 * over the rank by block
 *
 * parameters:
 *   max_gnum <-- max global number in global element numbering
 *   loc_set  <-- pointer to the local structure to work with
 *   comm     <-- mpi_comm on which synchro. and distribution take place
 *
 * returns:
 *   a synchronized and distributed cs_join_gset_t structure.
 *---------------------------------------------------------------------------*/

cs_join_gset_t *
cs_join_gset_block_sync(cs_gnum_t        max_gnum,
                        cs_join_gset_t  *loc_set,
                        MPI_Comm         comm)
{
  cs_join_gset_t  *sync_set = nullptr;

  int  local_rank, n_ranks;

  assert(loc_set != nullptr);

  if (max_gnum == 0)
    return  sync_set;

  MPI_Comm_rank(comm, &local_rank);
  MPI_Comm_size(comm, &n_ranks);

  cs_block_dist_info_t  bi
    = cs_block_dist_compute_sizes(local_rank,
                                  n_ranks,
                                  1,
                                  0,
                                  max_gnum);

  cs_lnum_t block_size = 0;
  if (bi.gnum_range[1] > bi.gnum_range[0])
    block_size = bi.gnum_range[1] - bi.gnum_range[0];

  cs_all_to_all_t *
    d = cs_all_to_all_create_from_block(loc_set->n_elts,
                                        0,  /* flags */
                                        loc_set->g_elts,
                                        bi,
                                        comm);

  /* Synchronize list definition for each global element */

  cs_lnum_t *p_index;
  CS_MALLOC(p_index, loc_set->n_elts+1, cs_lnum_t);
  cs_gnum_t *p_buffer;
  CS_MALLOC(p_buffer,
            loc_set->index[loc_set->n_elts] + loc_set->n_elts,
            cs_gnum_t);

  p_index[0] = 0;
  for (cs_lnum_t i = 0; i < loc_set->n_elts; i++) {

    cs_lnum_t shift = p_index[i];
    p_buffer[shift++] = loc_set->g_elts[i];
    cs_lnum_t s_id = loc_set->index[i], e_id = loc_set->index[i+1];
    for (cs_lnum_t j = s_id; j < e_id; j++)
      p_buffer[shift++] = loc_set->g_list[j];

    p_index[i+1] = shift;

  }

  cs_lnum_t *r_index
    = cs_all_to_all_copy_index(d,
                               false,  /* reverse */
                               p_index,
                               nullptr);

  cs_gnum_t *r_buffer
    = cs_all_to_all_copy_indexed(d,
                                 false,  /* reverse */
                                 p_index,
                                 p_buffer,
                                 r_index);

  CS_FREE(p_index);
  CS_FREE(p_buffer);

  cs_lnum_t n_r_elts = cs_all_to_all_n_elts_dest(d);

  cs_all_to_all_destroy(&d);

  /* Define sync_set: a distributed cs_join_gset_t structure which
     synchronizes data over the ranks */

  sync_set = cs_join_gset_create(block_size);

  for (cs_lnum_t i = 0; i < sync_set->n_elts; i++) {
    sync_set->g_elts[i] = bi.gnum_range[0] + (cs_gnum_t)i;
  }

  /* Build index */

  for (cs_lnum_t i = 0; i < n_r_elts; i++) {
    cs_lnum_t j = r_buffer[r_index[i]] - bi.gnum_range[0];
    sync_set->index[j+1] += r_index[i+1] - r_index[i] - 1;
  }

  for (cs_lnum_t i = 0; i < sync_set->n_elts; i++) {
    sync_set->index[i+1] += sync_set->index[i];
  }

  CS_MALLOC(sync_set->g_list,
            sync_set->index[sync_set->n_elts],
            cs_gnum_t);

  /* Now build set */

  cs_lnum_t *count;
  CS_MALLOC(count, sync_set->n_elts, cs_lnum_t);
  for (cs_lnum_t i = 0; i < sync_set->n_elts; i++)
    count[i] = 0;

  for (cs_lnum_t i = 0; i < n_r_elts; i++) {
    cs_lnum_t r_shift = r_index[i];
    cs_lnum_t j = r_buffer[r_shift] - bi.gnum_range[0];
    cs_lnum_t w_shift = sync_set->index[j] + count[j];

    cs_lnum_t n_sub_elts = r_index[i+1] - r_index[i] - 1;
    for (cs_lnum_t k = 0; k < n_sub_elts; k++)
      sync_set->g_list[w_shift + k] = r_buffer[r_shift + 1 + k];
    count[j] += n_sub_elts;
  }

  CS_FREE(count);
  CS_FREE(r_buffer);
  CS_FREE(r_index);

  /* Return pointer */

  cs_join_gset_clean(sync_set);

  return  sync_set;
}

/*----------------------------------------------------------------------------
 * Update a local cs_join_gset_t structure from a distributed and
 * synchronized cs_join_gset_t structure.
 *
 * Loc_set should not have redundant elements.
 *
 * parameters:
 *   max_gnum <-- max global number in global element numbering
 *   sync_set <-- pointer to the structure which holds a synchronized block
 *   loc_set  <-> pointer to a local structure holding elements to update
 *   comm     <-- comm on which synchronization and distribution take place
 *---------------------------------------------------------------------------*/

void
cs_join_gset_block_update(cs_gnum_t              max_gnum,
                          const cs_join_gset_t  *sync_set,
                          cs_join_gset_t        *loc_set,
                          MPI_Comm               comm)
{
  int  local_rank, n_ranks;

  if (max_gnum == 0)
    return;

  /* Sanity checks */

  assert(loc_set != nullptr);
  assert(sync_set != nullptr);

  /* Build a cs_join_block_info_t structure */

  MPI_Comm_rank(comm, &local_rank);
  MPI_Comm_size(comm, &n_ranks);

  cs_block_dist_info_t  bi = cs_block_dist_compute_sizes(local_rank,
                                                         n_ranks,
                                                         1,
                                                         0,
                                                         max_gnum);

  cs_all_to_all_t
    *d = cs_all_to_all_create_from_block(loc_set->n_elts,
                                         0, /* flags */
                                         loc_set->g_elts,
                                         bi,
                                         comm);

  cs_gnum_t *wanted_elts = cs_all_to_all_copy_array(d,
                                                    1,
                                                    false, /* reverse */
                                                    loc_set->g_elts);

  cs_lnum_t n_recv_elts = cs_all_to_all_n_elts_dest(d);

  /* Get a synchronized list definition for each global element */

  /* Send new list definition holding by sync_set to ranks which have
     request it */

  cs_lnum_t *block_index;
  CS_MALLOC(block_index, n_recv_elts+1, cs_lnum_t);

  block_index[0] = 0;
  for (cs_lnum_t i = 0; i < n_recv_elts; i++) {
    cs_lnum_t block_id = wanted_elts[i] - bi.gnum_range[0];
    cs_lnum_t n_sub_elts =   sync_set->index[block_id+1]
                           - sync_set->index[block_id];
    block_index[i+1] = block_index[i] + n_sub_elts;
  }

  cs_all_to_all_copy_index(d,
                           true, /* reverse */
                           block_index,
                           loc_set->index);

  cs_gnum_t *block_tuples;
  CS_MALLOC(block_tuples, block_index[n_recv_elts], cs_gnum_t);

  for (cs_lnum_t i = 0, shift = 0; i < n_recv_elts; i++) {

    cs_lnum_t block_id = wanted_elts[i] - bi.gnum_range[0];

    cs_lnum_t s_id = sync_set->index[block_id];
    cs_lnum_t e_id = sync_set->index[block_id+1];
    cs_lnum_t n_sub_elts = e_id - s_id;

    for (cs_lnum_t j = s_id, k = 0; j < e_id; j++, k++)
      block_tuples[shift+k] = sync_set->g_list[j];

    shift += n_sub_elts;

  }

  CS_FREE(wanted_elts);

  /* Re-initialize loc_set */

  CS_FREE(loc_set->g_list);

  loc_set->g_list = cs_all_to_all_copy_indexed(d,
                                               true, /* reverse */
                                               block_index,
                                               block_tuples,
                                               loc_set->index);

  /* Free some memory */

  cs_all_to_all_destroy(&d);

  CS_FREE(block_index);
  CS_FREE(block_tuples);
}

#endif /* HAVE_MPI */

/*----------------------------------------------------------------------------
 * Dump an array (int or double).
 *
 * This function is called according to the verbosity.
 *
 * parameters:
 *   f       <-- handle to output file
 *   type    <-- type of the array to display
 *   header  <-- header to display in front of the array
 *   n_elts  <-- number of elements to display
 *   array   <-- array to display
 *---------------------------------------------------------------------------*/

void
cs_join_dump_array(FILE        *f,
                   const char  *type,
                   const char  *header,
                   int          n_elts,
                   const void  *array)
{
  int  i;

  fprintf(f, "  %s: ", header);

  if (!strncmp(type, "int", strlen("int"))) { /* "int" array  */

    const int *i_array = static_cast<const int *>(array);

    for (i = 0; i < n_elts; i++)
      fprintf(f, " %8d", i_array[i]);

  }
  else if (!strncmp(type, "bool", strlen("bool"))) { /* "boolean" array  */

    const bool *b_array = static_cast<const bool *>(array);

    for (i = 0; i < n_elts; i++)
      if (b_array[i] == true)
        fprintf(f, " T");
      else {
        assert(b_array[i] == false);
        fprintf(f, " F");
      }
  }
  else if (!strncmp(type, "double", strlen("double"))) { /* "double" array */

    const double *d_array = static_cast<const double *>(array);

    for (i = 0; i < n_elts; i++)
      fprintf(f, " %10.8e", d_array[i]);

  }
  else if (!strncmp(type, "gnum", strlen("gnum"))) { /* "gnum" array */

    const cs_gnum_t *u_array = static_cast<const cs_gnum_t *>(array);

    for (i = 0; i < n_elts; i++)
      fprintf(f, " %9llu", (unsigned long long)u_array[i]);

  }
  else
    bft_error(__FILE__, __LINE__, 0,
              " Unexpected type (%s) to display in the current dump.\n",
              type);

  fprintf(f, "\n");
}

/*----------------------------------------------------------------------------
 * Dump a cs_join_gset_t structure.
 *
 * parameters:
 *   f    <-- handle to output file
 *   set  <-- pointer to the cs_join_gset_t structure to dump
 *---------------------------------------------------------------------------*/

void
cs_join_gset_dump(FILE                  *f,
                  const cs_join_gset_t  *set)
{
  int  i, j;

  if (set == nullptr)
    return;

  if (f == nullptr)
    f = stdout;

  fprintf(f, "\nDump cs_join_gset_t structure: %p\n", (const void *)set);
  fprintf(f, "number of elements: %10ld\n", (long)set->n_elts);
  fprintf(f, "size of the list  : %10ld\n\n", (long)set->index[set->n_elts]);

  for (i = 0; i < set->n_elts; i++) {

    int  s = set->index[i], e = set->index[i+1];
    int  n_matches = e-s;
    int  n_loops = n_matches/10;

    fprintf(f, "Global num: %8llu | subsize: %3d |",
            (unsigned long long)set->g_elts[i], n_matches);

    for (j = 0; j < n_loops; j++) {
      if (j == 0)
        fprintf(f,
                "%8llu %8llu %8llu %8llu %8llu "
                "%8llu %8llu %8llu %8llu %8llu\n",
                (unsigned long long)set->g_list[s+ 10*j],
                (unsigned long long)set->g_list[s+ 10*j + 1],
                (unsigned long long)set->g_list[s+ 10*j + 2],
                (unsigned long long)set->g_list[s+ 10*j + 3],
                (unsigned long long)set->g_list[s+ 10*j + 4],
                (unsigned long long)set->g_list[s+ 10*j + 5],
                (unsigned long long)set->g_list[s+ 10*j + 6],
                (unsigned long long)set->g_list[s+ 10*j + 7],
                (unsigned long long)set->g_list[s+ 10*j + 8],
                (unsigned long long)set->g_list[s+ 10*j + 9]);
      else
        fprintf(f, "                                     "
                "%8llu %8llu %8llu %8llu %8llu "
                "%8llu %8llu %8llu %8llu %8llu\n",
                (unsigned long long)set->g_list[s+ 10*j],
                (unsigned long long)set->g_list[s+ 10*j + 1],
                (unsigned long long)set->g_list[s+ 10*j + 2],
                (unsigned long long)set->g_list[s+ 10*j + 3],
                (unsigned long long)set->g_list[s+ 10*j + 4],
                (unsigned long long)set->g_list[s+ 10*j + 5],
                (unsigned long long)set->g_list[s+ 10*j + 6],
                (unsigned long long)set->g_list[s+ 10*j + 7],
                (unsigned long long)set->g_list[s+ 10*j + 8],
                (unsigned long long)set->g_list[s+ 10*j + 9]);
    }

    if (e - s+10*n_loops > 0) {
      for (j = s + 10*n_loops; j < e; j++) {
        if (j == s + 10*n_loops && n_loops > 0)
          fprintf(f, "                                     ");
        fprintf(f, "%8llu ", (unsigned long long)set->g_list[j]);
      }
      fprintf(f, "\n");
    }

    if (n_matches == 0)
      fprintf(f, "\n");

  } /* End of loop on boxes */

  fflush(f);
}

/*---------------------------------------------------------------------------*/

END_C_DECLS
