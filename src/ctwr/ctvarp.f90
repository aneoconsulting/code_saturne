!-------------------------------------------------------------------------------

! This file is part of Code_Saturne, a general-purpose CFD tool.
!
! Copyright (C) 1998-2016 EDF S.A.
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

subroutine ctvarp

!===============================================================================
!  FONCTION  :
!  ---------

!      INIT DES POSITIONS DES VARIABLES POUR LE MODULE AEROS
! REMPLISSAGE DES PARAMETRES (DEJA DEFINIS) POUR LES SCALAIRES PP
! (EQUIVALENT TO SUBROUTINE cs_fuel_varpos for COMBUSTION)
!-------------------------------------------------------------------------------
! Arguments
!__________________.____._____.________________________________________________.
! name             !type!mode ! role                                           !
!__________________!____!_____!________________________________________________!
!__________________!____!_____!________________________________________________!

!===============================================================================

!===============================================================================
! Module files
!===============================================================================

use paramx
use dimens
use numvar
use optcal
use cstphy
use entsor
use cstnum
use ppppar
use ppthch
use ppincl
use ctincl
use field

!===============================================================================

implicit none

! Local variables

integer          keyccl, keydri
integer          icla, ifcvsl, iscdri, f_id

!===============================================================================

! Key id of the scalar class
call field_get_key_id("scalar_class", keyccl)

! Key id for drift scalar
call field_get_key_id("drift_scalar_model", keydri)

!===============================================================================
! 1. Definition of fields
!===============================================================================

! Bulk definition - For cooling towers, the bulk is the humid air
! By definition, humid air is composed of two species: dry air and
! water vapour (whether in gas or condensate form)
! ---------------------------------------------------------------

! Thermal model - Set parameters of calculations (module optcal)

itherm = 1  ! Solve for temperature of the bulk (humid air)

itpscl = 2  ! Temperature in Celsius

icp = 0     ! Cp is variable (>=0 means variable, -1 means constant)
            ! It has to vary with humidity
            ! Needs to be specified here because the automated creation and initialisation
            ! of the cell array for Cp in 'iniva0' depends on its value
            ! (unlike the cell arrays for density and viscosity which are initialised
            ! irrespective of the values of irovar and ivivar)

call add_model_scalar_field('temperature', 'Temperature humid air', itempm)

! The thermal transported scalar is the temperature.
iscalt = itempm

ifcvsl = 0 ! Set variable diffusivity for the humid air enthalpy
           ! The diffusivity used in the transport equation will be
           ! the cell value of the viscls array for ivarfl(isca(itempm)).
           ! This value is updated at the top of each time step in 'ctphyv'
           ! along with the other variable properties
call field_set_key_int(ivarfl(isca(iscalt)), kivisl, ifcvsl)

! Mass fraction of dry air in the bulk, humid air
call add_model_scalar_field('ym_dry_air', 'Ym dry air', iyma)

ifcvsl = -1 ! Set constant diffusivity for the dry air mass fraction
            ! The diffusivity used in the transport equation will be
            ! the value of visls0(iyma)
call field_set_key_int(ivarfl(isca(iyma)), kivisl, ifcvsl)

! Injected liquid water definition - This is the separate phase
! which is injected in the packing zones.  Not to be confused with
! the water content in the humid air.
! ---------------------------------------------------------------

! Activate the drift: 0 (no activation),
!                     1 (transported particle velocity)
!                     2 (limit drop particle velocity)
iscdri = 1

! Associate the injected liquid water with class 1
icla = 1

! Mass fraction of liquid
call add_model_scalar_field('ym_liquid', 'Ym liq', iyml)
f_id = ivarfl(isca(iyml))

call field_set_key_int(f_id, keyccl, icla) ! Set the class index for the field

! Scalar with drift: Create additional mass flux
! This flux will then be reused for all scalars associated with this class

! GNU function to return the value of iscdri
! with the bit value of iscdri at position
! 'DRIFT_SCALAR_ADD_DRIFT_FLUX' set to one
iscdri = ibset(iscdri, DRIFT_SCALAR_ADD_DRIFT_FLUX)

! GNU function to return the value of iscdri
! with the bit value of iscdri at position
! 'DRIFT_SCALAR_IMPOSED_MASS_FLUX' set to one
iscdri = ibset(iscdri, DRIFT_SCALAR_IMPOSED_MASS_FLUX)

call field_set_key_int(f_id, keydri, iscdri)

ifcvsl = -1 ! Set constant diffusivity for the injected liquid mass fraction
            ! The diffusivity used in the transport equation will be
            ! the value of visls0(iyml)
call field_set_key_int(ivarfl(isca(iyml)), kivisl, ifcvsl)

! Transport and solve for the enthalpy of the liquid - with the same drift
! as the mass fraction Y_l
! NB: Enthalpy of the liquidus must be transported after the bulk enthalpy
call add_model_scalar_field('enthalpy_liquid', 'Enthalpy liq', ihml)
f_id = ivarfl(isca(ihml))

call field_set_key_int(f_id, keyccl, icla)

! Scalar with drift, but do not create an additional mass flux (use 'ibclr' instead of 'ibset')
! Incredibly opaque coding!
! for the temperature.  It reuses the mass flux of already identified with the mass fraction
iscdri = ibclr(iscdri, DRIFT_SCALAR_ADD_DRIFT_FLUX)

iscdri = ibset(iscdri, DRIFT_SCALAR_IMPOSED_MASS_FLUX)

call field_set_key_int(f_id, keydri, iscdri)

ifcvsl = 0   ! Set variable diffusivity for the injected liquid enthalpy transport
             ! The diffusivity used in the transport equation will be
             ! the cell value of the viscls array for ivarfl(isca(ihml)
call field_set_key_int(f_id, kivisl, ifcvsl)

end subroutine
