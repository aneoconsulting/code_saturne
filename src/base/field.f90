!-------------------------------------------------------------------------------

! This file is part of code_saturne, a general-purpose CFD tool.
!
! Copyright (C) 1998-2024 EDF S.A.
!
! This program is free software; you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation; either version 2 of the License, or (at your option) any later
! version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT
! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
! FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
! Street, Fifth Floor, Boston, MA 02110-1301, USA.

!-------------------------------------------------------------------------------


!> \file field.f90
!> Module for field-related operations

module field

  !=============================================================================

  implicit none

  !=============================================================================

  integer :: FIELD_INTENSIVE, FIELD_EXTENSIVE
  integer :: FIELD_VARIABLE, FIELD_PROPERTY

  parameter (FIELD_INTENSIVE=1)
  parameter (FIELD_EXTENSIVE=2)
  parameter (FIELD_VARIABLE=4)
  parameter (FIELD_PROPERTY=8)

  !=============================================================================

  interface

    ! Interface to C function assigning integer value to a key

    !> \brief Assign a floating point value for a given key to a field.

    !> If the key id is not valid, or the value type or field category is not
    !> compatible, a fatal error is provoked.

    !> \param[in]   f_id     field id
    !> \param[in]   k_id     id of associated key
    !> \param[in]   k_value  value associated with key

    subroutine field_set_key_int(f_id, k_id, k_value)  &
      bind(C, name='cs_f_field_set_key_int')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value :: f_id, k_id, k_value
    end subroutine field_set_key_int

    !---------------------------------------------------------------------------

    ! Interface to C function assigning floating-point value to a key

    !> \brief Assign a floating point value for a given key to a field.

    !> If the key id is not valid, or the value type or field category is not
    !> compatible, a fatal error is provoked.

    !> \param[in]   f_id     field id
    !> \param[in]   k_id     id of associated key
    !> \param[in]   k_value  value associated with key

    subroutine field_set_key_double(f_id, k_id, k_value)  &
      bind(C, name='cs_f_field_set_key_double')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value :: f_id, k_id
      real(c_double), value :: k_value
    end subroutine field_set_key_double

    !---------------------------------------------------------------------------

    !> \cond DOXYGEN_SHOULD_SKIP_THIS

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining the number of fields

    function cs_f_field_n_fields() result(id) &
      bind(C, name='cs_f_field_n_fields')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int)                                           :: id
    end function cs_f_field_n_fields

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining a field's id by its name

    function cs_f_field_id_by_name(name) result(id) &
      bind(C, name='cs_f_field_id_by_name')
      use, intrinsic :: iso_c_binding
      implicit none
      character(kind=c_char, len=1), dimension(*), intent(in)  :: name
      integer(c_int)                                           :: id
    end function cs_f_field_id_by_name

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining a field's location

    function cs_f_field_location(f) result(f_loc) &
      bind(C, name='cs_f_field_location')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value    :: f
      integer(c_int)        :: f_loc
    end function cs_f_field_location

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining a field's id by its name

    function cs_f_field_id_by_name_try(name) result(id) &
      bind(C, name='cs_f_field_id_by_name_try')
      use, intrinsic :: iso_c_binding
      implicit none
      character(kind=c_char, len=1), dimension(*), intent(in)  :: name
      integer(c_int)                                           :: id
    end function cs_f_field_id_by_name_try

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining field's pointer by its id

    function cs_field_by_id(id) result(f) &
      bind(C, name='cs_field_by_id')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value :: id
      type(c_ptr)           :: f
    end function cs_field_by_id

    !---------------------------------------------------------------------------

    ! Interface to C function returning a given field name pointer and length.

    subroutine cs_f_field_get_name(f_id, f_name_max, f_name, f_name_len)  &
      bind(C, name='cs_f_field_get_name')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value       :: f_id
      integer(c_int), value       :: f_name_max
      type(c_ptr), intent(out)    :: f_name
      integer(c_int), intent(out) :: f_name_len
    end subroutine cs_f_field_get_name

    !---------------------------------------------------------------------------

    ! Interface to C function returning a given field's dimension info

    subroutine cs_f_field_get_dimension(f_id, f_dim)  &
      bind(C, name='cs_f_field_get_dimension')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value :: f_id
      integer(c_int), dimension(1), intent(out) :: f_dim
    end subroutine cs_f_field_get_dimension

    !---------------------------------------------------------------------------

    ! Interface to C function returning a given field's type info

    subroutine cs_f_field_get_type(f_id, f_type)  &
      bind(C, name='cs_f_field_get_type')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value :: f_id
      integer(c_int), intent(out) :: f_type
    end subroutine cs_f_field_get_type

    !---------------------------------------------------------------------------

    ! Interface to C function returning field's value pointer and dimensions.

    ! If the field id is not valid, a fatal error is provoked.

    subroutine cs_f_field_var_ptr_by_id(id, p_type, p_rank, f_dim, c_p)  &
      bind(C, name='cs_f_field_var_ptr_by_id')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value        :: id
      integer(c_int), value        :: p_type
      integer(c_int), value        :: p_rank
      integer(c_int), dimension(2) :: f_dim
      type(c_ptr), intent(out)     :: c_p
    end subroutine cs_f_field_var_ptr_by_id

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining a field key id by its name

    function cs_f_field_key_id(name) result(id) &
      bind(C, name='cs_field_key_id')
      use, intrinsic :: iso_c_binding
      implicit none
      character(kind=c_char, len=1), dimension(*), intent(in)  :: name
      integer(c_int)                                           :: id
    end function cs_f_field_key_id

    !---------------------------------------------------------------------------

    ! Interface to C function obtaining a field key id by its name

    function cs_f_field_key_id_try(name) result(id) &
      bind(C, name='cs_field_key_id_try')
      use, intrinsic :: iso_c_binding
      implicit none
      character(kind=c_char, len=1), dimension(*), intent(in)  :: name
      integer(c_int)                                           :: id
    end function cs_f_field_key_id_try

    !---------------------------------------------------------------------------

    ! Interface to C function returning an integer for a given key associated
    ! with a field

    function cs_field_get_key_int(f, k_id) result(k_value) &
      bind(C, name='cs_field_get_key_int')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value    :: f
      integer(c_int), value :: k_id
      integer(c_int)        :: k_value
    end function cs_field_get_key_int

    !---------------------------------------------------------------------------

    ! Interface to C function returning an floating-point valuer for a given
    ! key associated with a field

    function cs_field_get_key_double(f, k_id) result(k_value) &
      bind(C, name='cs_field_get_key_double')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value    :: f
      integer(c_int), value :: k_id
      real(c_double)        :: k_value
    end function cs_field_get_key_double

    !---------------------------------------------------------------------------

    ! Interface to C function returning a label associated with a field.

    subroutine cs_f_field_get_label(f_id, str_max, str, str_len) &
      bind(C, name='cs_f_field_get_label')
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), value       :: f_id, str_max
      type(c_ptr), intent(out)    :: str
      integer(c_int), intent(out) :: str_len
    end subroutine cs_f_field_get_label

    !---------------------------------------------------------------------------

    !> (DOXYGEN_SHOULD_SKIP_THIS) \endcond

    !---------------------------------------------------------------------------

  end interface

  !=============================================================================

contains

  !=============================================================================

  !> \brief  Return the number of defined fields.

  !> \param[out] nfld           number of field

  subroutine field_get_n_fields(nfld)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(out)         :: nfld

    nfld = cs_f_field_n_fields()

    return

  end subroutine field_get_n_fields

  !=============================================================================

  !> \brief  Return an id associated with a given field name.

  !> \param[in]  name           field name
  !> \param[out] id             id of field

  subroutine field_get_id(name, id)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    character(len=*), intent(in) :: name
    integer, intent(out)         :: id

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name

    c_name = trim(name)//c_null_char

    id = cs_f_field_id_by_name(c_name)

    return

  end subroutine field_get_id

  !=============================================================================

  !> \brief  Return  the location of a given field.

  !> \param[in]  f_id           field id
  !> \param[out] f_loc          location of the field

  subroutine field_get_location(f_id, f_loc)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)         :: f_id
    integer, intent(out)        :: f_loc

    ! Local variables

    integer(c_int) :: cf_id
    type(c_ptr)    :: f

    cf_id = f_id
    f = cs_field_by_id(cf_id)

    f_loc = cs_f_field_location(f)

    return

  end subroutine field_get_location

  !=============================================================================

  !> \brief  Return an id associated with a given field name if present.

  !> If the field has not been defined previously, -1 is returned.

  !> \param[in]  name           field name
  !> \param[out] id             id of field

  subroutine field_get_id_try(name, id)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    character(len=*), intent(in) :: name
    integer, intent(out)         :: id

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name

    c_name = trim(name)//c_null_char

    id = cs_f_field_id_by_name_try(c_name)

    return

  end subroutine field_get_id_try

  !=============================================================================

  !> \brief Return a given field's name.

  !> \param[in]   f_id  field id
  !> \param[out]  name  field's name

  subroutine field_get_name(f_id, name)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)           :: f_id
    character(len=*), intent(out) :: name

    ! Local variables

    integer :: i
    integer(c_int) :: c_f_id, name_max, c_name_len
    type(c_ptr) :: c_name_p
    character(kind=c_char, len=1), dimension(:), pointer :: c_name

    c_f_id = f_id
    name_max = len(name)

    call cs_f_field_get_name(f_id, name_max, c_name_p, c_name_len)
    call c_f_pointer(c_name_p, c_name, [c_name_len])

    do i = 1, c_name_len
      name(i:i) = c_name(i)
    enddo
    do i = c_name_len + 1, name_max
      name(i:i) = ' '
    enddo

    return

  end subroutine field_get_name

  !=============================================================================

  !> \brief Return a given field's dimension.

  !> \param[in]   f_id   field id
  !> \param[out]  f_dim  number of field components (dimension)

  subroutine field_get_dim(f_id, f_dim)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)  :: f_id
    integer, intent(out) :: f_dim

    ! Local variables

    integer(c_int) :: c_f_id
    integer(c_int), dimension(1) :: c_dim

    c_f_id = f_id

    call cs_f_field_get_dimension(c_f_id, c_dim)

    f_dim = c_dim(1)

    return

  end subroutine field_get_dim

  !=============================================================================

  !> \brief Return a given field's type.

  !> \param[in]   f_id         field id
  !> \param[out]  f_type       field type flag

  subroutine field_get_type(f_id, f_type)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)  :: f_id
    integer, intent(out) :: f_type

    ! Local variables

    integer(c_int) :: c_f_id
    integer(c_int) :: c_type

    c_f_id = f_id

    call cs_f_field_get_type(c_f_id, c_type)

    f_type = c_type

    return

  end subroutine field_get_type

  !=============================================================================

  !> \brief  Return an id associated with a given key name if present.

  !> If the key has not been defined previously, -1 is returned.

  !> \param[in]   name  key name
  !> \param[out]  id    associated key id

  subroutine field_get_key_id(name, id)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    character(len=*), intent(in) :: name
    integer, intent(out)         :: id

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name
    integer(c_int)                               :: c_id

    c_name = trim(name)//c_null_char

    c_id = cs_f_field_key_id_try(c_name)
    id = c_id

    return

  end subroutine field_get_key_id

  !=============================================================================

  !> \brief Return an integer value for a given key associated with a field.

  !> If the key id is not valid, or the value type or field category is not
  !> compatible, a fatal error is provoked.

  !> \param[in]   f_id     field id
  !> \param[in]   k_id     id of associated key
  !> \param[out]  k_value  integer value associated with key id for this field

  subroutine field_get_key_int(f_id, k_id, k_value)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)   :: f_id, k_id
    integer, intent(out)  :: k_value

    ! Local variables

    integer(c_int) :: c_f_id, c_k_id, c_k_value
    type(c_ptr) :: f

    c_f_id = f_id
    c_k_id = k_id
    f = cs_field_by_id(c_f_id)
    c_k_value = cs_field_get_key_int(f, c_k_id)
    k_value = c_k_value

    return

  end subroutine field_get_key_int

  !=============================================================================

  !> \brief Return an integer value for a given key associated with a field.

  !> If the key id is not valid, or the value type or field category is not
  !> compatible, a fatal error is provoked.

  !> \param[in]   f_id     field id
  !> \param[in]   k_name   key name
  !> \param[out]  k_value  integer value associated with key id for this field

  subroutine field_get_key_int_by_name(f_id, k_name, k_value)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)   :: f_id
    character(len=*), intent(in) :: k_name
    integer, intent(out)  :: k_value

    ! Local variables

    integer(c_int) :: c_f_id, c_k_id, c_k_value
    character(len=len_trim(k_name)+1, kind=c_char) :: c_k_name
    type(c_ptr) :: f

    c_k_name = trim(k_name)//c_null_char

    c_k_id = cs_f_field_key_id_try(c_k_name)
    c_f_id = f_id
    f = cs_field_by_id(c_f_id)
    c_k_value = cs_field_get_key_int(f, c_k_id)
    k_value = c_k_value

    return

  end subroutine field_get_key_int_by_name

  !=============================================================================

  !> \brief Return a floating-point value for a given key associated with a
  !> field.

  !> If the key id is not valid, or the value type or field category is not
  !> compatible, a fatal error is provoked.

  !> \param[in]   f_id     field id
  !> \param[in]   k_id     id of associated key
  !> \param[out]  k_value  integer value associated with key id for this field

  subroutine field_get_key_double(f_id, k_id, k_value)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)            :: f_id, k_id
    double precision, intent(out)  :: k_value

    ! Local variables

    integer(c_int) :: c_f_id, c_k_id
    real(c_double) :: c_k_value
    type(c_ptr) :: f

    c_f_id = f_id
    c_k_id = k_id
    f = cs_field_by_id(c_f_id)
    c_k_value = cs_field_get_key_double(f, k_id)
    k_value = c_k_value

    return

  end subroutine field_get_key_double

  !=============================================================================

  !> \brief Return a label associated with a field.

  !> If the "label" key has been set for this field, its associated string
  !> is returned. Otherwise, the field's name is returned.

  !> \param[in]   f_id  field id
  !> \param[out]  str   string associated with key

  subroutine field_get_label(f_id, str)

    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments

    integer, intent(in)           :: f_id
    character(len=*), intent(out) :: str

    ! Local variables

    integer :: i
    integer(c_int) :: c_f_id, c_str_max, c_str_len
    type(c_ptr) :: c_str_p
    character(kind=c_char, len=1), dimension(:), pointer :: c_str

    c_f_id = f_id
    c_str_max = len(str)

    call cs_f_field_get_label(c_f_id, c_str_max, c_str_p, c_str_len)
    call c_f_pointer(c_str_p, c_str, [c_str_len])

    do i = 1, c_str_len
      str(i:i) = c_str(i)
    enddo
    do i = c_str_len + 1, c_str_max
      str(i:i) = ' '
    enddo

    return

  end subroutine field_get_label

  !=============================================================================

  !> \brief Return pointer to the values array of a given scalar field

  !> \param[in]     field_id  id of given field (which must be scalar)
  !> \param[out]    p         pointer to scalar field values

  subroutine field_get_val_s(field_id, p)

    use, intrinsic :: iso_c_binding
    implicit none

    integer, intent(in)                                    :: field_id
    double precision, dimension(:), pointer, intent(inout) :: p

    ! Local variables

    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(2) :: f_dim
    type(c_ptr) :: c_p

    f_id = field_id
    p_type = 1
    p_rank = 1

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1)])

  end subroutine field_get_val_s

  !=============================================================================

  !> \brief Return pointer to the values array of a given scalar field

  !> \param[in]     name      name of given field (which must be scalar)
  !> \param[out]    p         pointer to scalar field values

  subroutine field_get_val_s_by_name(name, p)

    use, intrinsic :: iso_c_binding
    implicit none

    character(len=*), intent(in)                           :: name
    double precision, dimension(:), pointer, intent(inout) :: p

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name
    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(2) :: f_dim
    type(c_ptr) :: c_p

    c_name = trim(name)//c_null_char

    f_id = cs_f_field_id_by_name(c_name)
    p_type = 1
    p_rank = 1

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1)])

  end subroutine field_get_val_s_by_name

  !=============================================================================

  !> \brief Return pointer to the array's previous values of a given scalar
  !> field

  !> \param[in]     name      name of given field (which must be scalar)
  !> \param[out]    p         pointer to scalar field values at the previous
  !>                          iteration

  subroutine field_get_val_prev_s_by_name(name, p)

    use, intrinsic :: iso_c_binding
    implicit none

    character(len=*), intent(in)                           :: name
    double precision, dimension(:), pointer, intent(inout) :: p

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name
    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(2) :: f_dim
    type(c_ptr) :: c_p

    c_name = trim(name)//c_null_char

    f_id = cs_f_field_id_by_name(c_name)

    p_type = 2
    p_rank = 1

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1)])

  end subroutine field_get_val_prev_s_by_name

  !=============================================================================

  !> \brief Return pointer to the values array of a given vector field

  !> \param[in]     field_id  id of given field (which must be vectorial)
  !> \param[out]    p         pointer to vector field values

  subroutine field_get_val_v(field_id, p)

    use, intrinsic :: iso_c_binding
    implicit none

    integer, intent(in)                                      :: field_id
    double precision, dimension(:,:), pointer, intent(inout) :: p

    ! Local variables

    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(2) :: f_dim
    type(c_ptr) :: c_p

    f_id = field_id
    p_type = 1
    p_rank = 2

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1), f_dim(2)])

  end subroutine field_get_val_v

  !=============================================================================

  !> \brief Return pointer to the values array of a given vector field

  !> \param[in]     name      name of given field (which must be vectorial)
  !> \param[out]    p         pointer to scalar field values

  subroutine field_get_val_v_by_name(name, p)

    use, intrinsic :: iso_c_binding
    implicit none

    character(len=*), intent(in)                             :: name
    double precision, dimension(:,:), pointer, intent(inout) :: p

    ! Local variables

    character(len=len_trim(name)+1, kind=c_char) :: c_name
    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(2) :: f_dim
    type(c_ptr) :: c_p

    c_name = trim(name)//c_null_char

    f_id = cs_f_field_id_by_name(c_name)
    p_type = 1
    p_rank = 2

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1), f_dim(2)])

  end subroutine field_get_val_v_by_name

  !=============================================================================

  !> \brief Return pointer to the previous values array of a given scalar field

  !> \param[in]     field_id  id of given field (which must be scalar)
  !> \param[out]    p         pointer to previous scalar field values

  subroutine field_get_val_prev_s(field_id, p)

    use, intrinsic :: iso_c_binding
    implicit none

    integer, intent(in)                                    :: field_id
    double precision, dimension(:), pointer, intent(inout) :: p

    ! Local variables

    integer(c_int) :: f_id, p_type, p_rank
    integer(c_int), dimension(3) :: f_dim
    type(c_ptr) :: c_p

    f_id = field_id
    p_type = 2
    p_rank = 1

    call cs_f_field_var_ptr_by_id(f_id, p_type, p_rank, f_dim, c_p)
    call c_f_pointer(c_p, p, [f_dim(1)])

  end subroutine field_get_val_prev_s

  !=============================================================================

end module field
