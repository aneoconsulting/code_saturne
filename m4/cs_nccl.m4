dnl--------------------------------------------------------------------------------
dnl
dnl This file is part of code_saturne, a general-purpose CFD tool.
dnl
dnl Copyright (C) 1998-2024 EDF S.A.
dnl
dnl This program is free software; you can redistribute it and/or modify it under
dnl the terms of the GNU General Public License as published by the Free Software
dnl Foundation; either version 2 of the License, or (at your option) any later
dnl version.
dnl
dnl This program is distributed in the hope that it will be useful, but WITHOUT
dnl ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
dnl FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
dnl details.
dnl
dnl You should have received a copy of the GNU General Public License along with
dnl this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
dnl Street, Fifth Floor, Boston, MA 02110-1301, USA.
dnl
dnl--------------------------------------------------------------------------------

# CS_AC_TEST_NCCL
#----------------
# modifies or sets cs_have_nccl, NCCL_CPPFLAGS, NCCL_LDFLAGS, and NCCL_LIBS
# depending on libraries found

AC_DEFUN([CS_AC_TEST_NCCL], [

cs_have_nccl=no
cs_abs_srcdir=`cd $srcdir && pwd`

AC_ARG_WITH(nccl,
            [AS_HELP_STRING([--with-nccl=PATH],
                            [specify prefix directory for NCCL])],
            [if test "x$withval" = "x"; then
               with_nccl=no
             fi],
            [with_nccl=no])



if test "x$with_nccl" != "xno" ; then
  NCCL_CPPFLAGS="-I$with_nccl/include"
  NCCL_LDLAGS="-I$with_nccl/lib"
  NCCL_LIBS="-lnccl"

  # If CUDA is used and NCCL unspecified, check
  # if the associated flags include NCCL

  saved_CPPFLAGS="$CPPFLAGS"
  saved_LDFLAGS="$LDFLAGS"
  saved_LIBS="$LIBS"
  
  CPPFLAGS="$CPPFLAGS $NCCL_CPPFLAGS $CUDA_CPPFLAGS"
  LDFLAGS="$LDFLAGS  $NCCL_LDFLAGS $CUDA_LDFLAGS"
  LIBS="$LIBS $NCCL_LIBS $CUDA_LIBS"

  AC_CHECK_HEADER([nccl.h], [have_nccl_h=yes], [have_nccl_h=no])
  AC_CHECK_LIB([nccl], [ncclCommInitRank], [cs_have_nccl=yes], [cs_have_nccl=no])

  if test "x$cs_have_nccl" = "xno"; then
    NCCL_CPPFLAGS=""
    NCCL_LDFLAGS=""
    NCCL_LIBS=""
    if test "x$with_nccl" != "xcheck" ; then
      AC_MSG_FAILURE([NCCL support is requested, but test for NCCL failed!])
    else
      AC_MSG_WARN([no NCCL file support])
    fi
  fi

  CPPFLAGS="$saved_CPPFLAGS"
  LDFLAGS="$saved_LDFLAGS"
  LIBS="$saved_LIBS"

  unset saved_CPPFLAGS
  unset saved_LDFLAGS
  unset saved_LIBS

fi


AC_DEFINE([HAVE_NCCL], 1, [NCCL support])
AC_SUBST(cs_have_nccl)
AC_SUBST(NCCL_CPPFLAGS)
AC_SUBST(NCCL_LDFLAGS)
NCCL_LIBS="-lnccl"
AC_SUBST(NCCL_LIBS)

LIBS="$NCCL_LIBS $LIBS"


])dnl
