#ifndef __CS_ATMO_H__
#define __CS_ATMO_H__

/*============================================================================
 * Main for atmospheric related functions
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

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Local Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Atmospheric models
 *----------------------------------------------------------------------------*/

typedef enum {

  CS_ATMO_OFF = -1,
  CS_ATMO_CONSTANT_DENSITY = 0,
  CS_ATMO_DRY = 1,
  CS_ATMO_HUMID = 2,

} cs_atmo_model_t;

/*----------------------------------------------------------------------------
 * Atmospheric nucleation models
 *----------------------------------------------------------------------------*/

typedef enum {

  CS_ATMO_NUC_OFF = 0,
  CS_ATMO_NUC_PRUPPACHER_KLETT = 1,
  CS_ATMO_NUC_COHARD = 2,
  CS_ATMO_NUC_ABDUL_RAZZAK = 3

} cs_atmo_nucleation_type_t;

/*----------------------------------------------------------------------------
 * Atmospheric universal functions
 *----------------------------------------------------------------------------*/

typedef enum {

  CS_ATMO_UNIV_FN_CHENG = 0,
  CS_ATMO_UNIV_FN_HOGSTROM = 1,
  CS_ATMO_UNIV_FN_BUSINGER = 2,
  CS_ATMO_UNIV_FN_HARTOGENSIS = 3,
  CS_ATMO_UNIV_FN_CARL = 4,

} cs_atmo_universal_functions_t;

/*----------------------------------------------------------------------------
 * Atmospheric soil model
 *----------------------------------------------------------------------------*/

typedef enum {

  /*! 5 categories: water, forest, diverse, mineral, building */
  CS_ATMO_SOIL_5_CAT = 0,
  /*! 7 categories: water, forest, diverse, mineral, diffuse buildings,
     mixtie buildings, dense buildings  */
  CS_ATMO_SOIL_7_CAT = 1,
  /*! Roughness length classification of Corine land cover classes
      Julieta Silva et al. doi=10.1.1.608.2707 */
  CS_ATMO_SOIL_23_CAT = 2,

} cs_atmo_soil_cat_t;

/*----------------------------------------------------------------------------
 * Atmospheric soil micro-scale options
 *----------------------------------------------------------------------------*/

typedef enum {

  /*! Genuine Force-restore model (bare-soil or equivalent only) */
  CS_ATMO_SOIL_GENUINE = 0,
  /*! Force-restore model including photovoltaic layer */
  CS_ATMO_SOIL_PHOTOVOLTAICS = 1,
  /*! Force-restore model including vegetation layer */
  CS_ATMO_SOIL_VEGETATION = 2,

} cs_atmo_soil_meb_model_t;

/*============================================================================
 * Type definitions
 *============================================================================*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*----------------------------------------------------------------------------
 * Atmospheric model options descriptor
 *----------------------------------------------------------------------------*/

typedef struct {
  /* Space and time reference of the run */
  /*! Starting year */
  int syear;
  /*! Starting quantile */
  int squant;
  /*! Starting hour */
  int shour;
  /*! Starting minute */
  int smin;
  /*! Starting second */
  cs_real_t ssec;
  /*! longitude of the domain origin */
  cs_real_t longitude;
  /*! latitude of the domain origin */
  cs_real_t latitude;
  /*! x coordinate of the domain origin in Lambert-93 */
  cs_real_t x_l93;
  /*! y coordinate of the domain origin in Lambert-93 */
  cs_real_t y_l93;
  /*! numbers of altitudes for the dynamics */
  union {
    int met_1d_nlevels_d;
    int nbmetd; /* deprecated */
  };
  /*! numbers of altitudes for the temperature and specific humidity */
  union {
    int met_1d_nlevels_t;
    int nbmett; /* deprecated */
  };
  /*! numbers of time steps for the meteo profiles */
  union {
    int met_1d_ntimes;
    int nbmetm;
  };

  /*! Number of vertical levels */
  union {
    int met_1d_nlevels_max_t;
    int nbmaxt;
  };
  /*! 1-D radiative model (0 off, 1 on) */
  int radiative_model_1d;
  /*! 1-D radiative model: number of vertical arrays */
  int rad_1d_nvert;
  /*! 1-D radiative model: number of levels (up to the top of the domain) */
  int rad_1d_nlevels;
  /*! 1-D radiative model: number of levels (up to 11000 m)
    (automatically computed) */
  int rad_1d_nlevels_max;
  /*! 1D radiative model pass frequency (1 valu bu default)*/
  int rad_1d_frequency;

  /*! horizontal coordinates of the vertical grid */
  cs_real_t *rad_1d_xy;

  /*! vertical grid for 1D radiative scheme */
  cs_real_t *rad_1d_z;
  /*! absorption for CO2 + 03 */
  cs_real_t *rad_1d_acinfe;
  /*! differential absorption for CO2 + 03 */
  cs_real_t *rad_1d_dacinfe;
  /*! absorption for CO2 only */
  cs_real_t *rad_1d_aco2;
  cs_real_t *rad_1d_aco2s;
  /*! differential absorption for CO2 only */
  cs_real_t *rad_1d_daco2;
  cs_real_t *rad_1d_daco2s;
  /*! as acinfe, downwing flux */
  cs_real_t *rad_1d_acsup;
  cs_real_t *rad_1d_acsups;
  cs_real_t *rad_1d_dacsup;
  cs_real_t *rad_1d_dacsups;
  /*! internal variable for 1D radiative model */
  cs_real_t *rad_1d_tauzq;
  /*! internal variable for 1D radiative model */
  cs_real_t *rad_1d_tauz;
  /*! internal variable for 1D radiative model */
  cs_real_t *rad_1d_zq;
  /*! internal variable for 1D radiative model */
  cs_real_t *rad_1d_zray;
  /*! flux divergence of IR radiation */
  cs_real_t *rad_1d_ir_div;
  /*! flux divergence of solar radiation */
  cs_real_t *rad_1d_sol_div;
  /*! Upward and downward radiative fluxes (infrared, solar)
    along each vertical */
  cs_real_t *rad_1d_iru;
  cs_real_t *rad_1d_ird;
  cs_real_t *rad_1d_solu;
  cs_real_t *rad_1d_sold;

  /*! 1D profiles of total water mass fraction along each vertical */
  cs_real_t *rad_1d_qw;
  /*! 1D profiles of liquid water mass fraction along each vertical */
  cs_real_t *rad_1d_ql;
  /*! 1D profiles of vapor water mass fraction along each vertical */
  cs_real_t *rad_1d_qv;
  /*! 1D profiles of number of droplets along each vertical */
  cs_real_t *rad_1d_nc;
  /*! 1D profiles of nebulosity along each vertical */
  cs_real_t *rad_1d_fn;
  /*! 1D profiles of aerosols along each vertical */
  cs_real_t *rad_1d_aerosols;

  /*! Value of ground albedo for each vertical */
  cs_real_t *rad_1d_albedo0;
  /*! Value of ground emissivity for each vertical */
  cs_real_t *rad_1d_emissi0;
  /*! Value of ground temperature for each vertical */
  cs_real_t *rad_1d_temp0;
  /*! Value of ground potential temperature for each vertical */
  cs_real_t *rad_1d_theta0;
  /*! Value of ground total water mass fraction for each vertical */
  cs_real_t *rad_1d_qw0;
  /*! Value of ground pressure for each vertical */
  cs_real_t *rad_1d_p0;
  /*! Value of ground density for each vertical */
  cs_real_t *rad_1d_rho0;

  /*! Domain orientation (angle in degree between y direction and north),
   * 0 by default */
  cs_real_t domain_orientation;

  /*! Option to compute ground elevation in the domain */
  bool compute_z_ground;

  int open_bcs_treatment;
  int theo_interp;

  /* Model options */
  /*! Sedimentation flag */
  int sedimentation_model;
  /*! Deposition flag */
  int deposition_model;
  /*!
   * Option for nucleation
   *  0: without nucleation
   *  1: Pruppacher and Klett 1997
   *  2: Cohard et al. 1998,1999
   *  3: Abdul-Razzak et al. 1998,2000
   *  logaritmic standard deviation of the log-normal law of the
   *  droplet spectrum
   */
  int nucleation_model;
  /*!  Option for subgrid models
   *   0: the simplest parameterization (for numerical verifications)
   *   1: Bechtold et al. 1995 (Luc Musson-Genon)
   *   2: Bouzereau et al. 2004
   *   3: Cuijpers and Duynkerke 1993, Deardorff 1976, Sommeria and
   *                 Deardorff 1977
   */
  int subgrid_model;
  /*! Option for liquid water content distribution models
   *  1: all or nothing
   *  2: Gaussian distribution
   */
  int distribution_model;
  /*! Use meteo profile:
   *  - 0: not use
   *  - 1: use a meteo file
   *  - 2: directly enter values large scale values
   *  - 3: fill directly meteo_* fields
   *  */
  int meteo_profile;

  /*! Meteo file */
  char *meteo_file_name;

  /*! Meteo Monin obukhov inverse length */
  cs_real_t meteo_dlmo;
  /*! Meteo reference roughness */
  cs_real_t meteo_z0;
  /*! Meteo reference elevation for reference velocity */
  cs_real_t meteo_zref;
  /*! Meteo Boundary layer elevation */
  cs_real_t meteo_zi;
  /*! Meteo reference elevation for reference velocity 1 */
  cs_real_t meteo_zu1;
  /*! Meteo reference elevation for reference velocity 2 */
  cs_real_t meteo_zu2;
  /*! Meteo reference elevation for reference temperature 1 */
  cs_real_t meteo_zt1;
  /*! Meteo reference elevation for reference temperature 2 */
  cs_real_t meteo_zt2;
  /*! Meteo reference velocity */
  cs_real_t meteo_uref;
  /*! Meteo reference velocity 1 */
  cs_real_t meteo_u1;
  /*! Meteo reference velocity 2 */
  cs_real_t meteo_u2;
  /*! Meteo reference ground friction velocity */
  cs_real_t meteo_ustar0;
  /*! Meteo reference convective velocity */
  cs_real_t meteo_wstar0;
  /*! Meteo wind direction */
  cs_real_t meteo_angle;
  /*! Meteo reference temperature (ground temperature) */
  cs_real_t meteo_t0;
  /*! Meteo reference temperature 1 */
  cs_real_t meteo_t1;
  /*! Meteo reference temperature 2 */
  cs_real_t meteo_t2;
  /*! Meteo reference ground friction temperature */
  cs_real_t meteo_tstar;
  /*! Meteo pressure at sea level */
  cs_real_t meteo_psea;

  /*! Meteo reference mass fraction at 2m */
  cs_real_t meteo_qw0;
  /*! Meteo reference ground friction mass fraction */
  cs_real_t meteo_qwstar;
  /*! Meteo reference mass fraction 1 */
  cs_real_t meteo_qw1;
  /*! Meteo reference mass fraction 2 */
  cs_real_t meteo_qw2;
  /*! Meteo reference mass fraction at 2m */
  cs_real_t meteo_ql0;
  /*! Meteo reference evaporation  */
  cs_real_t meteo_evapor;
  /*! Meteo reference sensible heat */
  cs_real_t meteo_sensi;
  /*! Universal function Phi_m for stable condition */
  int meteo_phim_s;
  /*! Universal function Phi_h for stable condition */
  int meteo_phih_s;
  /*! Universal function Phi_m for unstable condition */
  int meteo_phim_u;
  /*! Universal function Phi_h for unstable condition */
  int meteo_phih_u;

  /* 1D meteo profiles */
  /*! meteo x, y, p at sea levels */
  cs_real_t *xyp_met;
  /*! meteo u profiles */
  cs_real_t *u_met;
  /*! meteo v profiles */
  cs_real_t *v_met;
  /*! meteo w profiles */
  cs_real_t *w_met;
  /*! meteo turbulent kinetic energy profile */
  cs_real_t *ek_met;
  /*! meteo turbulent dissipation profile */
  cs_real_t *ep_met;
  /*! meteo temperature profile */
  cs_real_t *temp_met;
  /*! meteo density profile */
  cs_real_t *rho_met;
  /*! meteo water mass fraction profile */
  cs_real_t *qw_met;
  /*! meteo number of droplets profile */
  cs_real_t *ndrop_met;
  /*! flag to compute the hydrostatic pressure by Laplace integration
   *  in the meteo profiles
   *  0: based on P (sea level) value by default
   *  1: based on P computed for the standard atmosphere
   */

  /*! Altitudes of the dynamic profiles */
  cs_real_t *z_dyn_met;
  /*! Altitudes of the temperature profile */
  cs_real_t *z_temp_met;
  /*! Time (in seconds) of the meteo profile */
  cs_real_t *time_met;
  /*! Hydrostatic pressure from Laplace integration */
  cs_real_t *hyd_p_met;
  /*! potential temperature profile */
  cs_real_t *pot_t_met;
  /*! Pressure drop integrated over a time step
     (used for automatic open boundaries) profile */
  cs_real_t *dpdt_met;
  /*! Momentum for each level (used for automatic open boundaries) */
  cs_real_3_t *mom_met;
  /*! code_saturne momentum for each level */
  cs_real_3_t *mom_cs;
  int hydrostatic_pressure_model;
  /*! flag for the standard atmo humidity profile
   *  qv_profile = 0 (default)
   *  qv_profile = 1 decreasing exponential */
  int qv_profile;

  /*! Soil model (1: on, 0: off) */
  int soil_model;
  /*! Soil categories:
   * - CS_ATMO_SOIL_5_CAT
   * - CS_ATMO_SOIL_7_CAT
   * - CS_ATMO_SOIL_23_CAT */
  cs_atmo_soil_cat_t soil_cat;
  /*! Soil zone id (or -1 if inactive) */
  int soil_zone_id;
  /*! Solve supplementary heat budget equation (multi energy budget):
   * - CS_ATMO_SOIL_GENUINE
   * - CS_ATMO_SOIL_PHOTOVOLTAICS
   * - CS_ATMO_SOIL_VEGETATION */
  cs_atmo_soil_meb_model_t soil_meb_model;

  bool rain;
  int cloud_type;
  bool accretion;
  bool autoconversion;
  bool autocollection_cloud;
  bool autocollection_rain;
  bool precipitation;
  bool evaporation;
  bool rupture;

  /*! initial soil surface temperature
   *  for Sea, it is also the surface temperature */
  cs_real_t soil_surf_temp;
  /*! initial deep soil temperature */
  cs_real_t soil_temperature;
  /*! initial soil specific humidity */
  cs_real_t soil_humidity;
  /*! initial water content of the first reservoir */
  cs_real_t soil_w1_ini;
  /*! initial water content of the second reservoir */
  cs_real_t soil_w2_ini;
  /*! Thermal inertia of the soil */
  cs_real_t *soil_cat_thermal_inertia;
  /*! Dynamic roughness length */
  cs_real_t *soil_cat_roughness;
  /*! Thermal roughness length*/
  cs_real_t *soil_cat_thermal_roughness;
  /*! Albedo per soil category */
  cs_real_t *soil_cat_albedo;
  /*! emissivity per soil category */
  cs_real_t *soil_cat_emissi;
  /*! Vegetation index per soil category */
  cs_real_t *soil_cat_vegeta;
  /*! maximum water capacity of shallow reservoir*/
  cs_real_t *soil_cat_w1;
  /*! ratio of the maximum water capacity of the shallow
   *  reservoir to the deep reservoir [0,1]*/
  cs_real_t *soil_cat_w2;
  /*! Rij value for Rij1*/
  cs_real_t *soil_cat_r1;
  /*! Rij value for Rij2*/
  cs_real_t *soil_cat_r2;
 /*! adimensional : sigc=0.53 other referenced values are 0.28, 0.15 */
  cs_real_t sigc;
  /*! 1D infrared profile */
  int infrared_1D_profile;
  /*! 1D solar profile */
  int solar_1D_profile;

  cs_real_t aod_o3_tot;

  cs_real_t aod_h2o_tot;

} cs_atmo_option_t;

/*----------------------------------------------------------------------------
 * Atmospheric model constants descriptor
 *----------------------------------------------------------------------------*/

typedef struct {
  /* Space and time reference of the run */
  /*! Reference pressure (to compute potential temp: 1.0e+5) */
  cs_real_t ps;

} cs_atmo_constants_t;

/*----------------------------------------------------------------------------
 * Atmospheric imbrication option
 *----------------------------------------------------------------------------*/

typedef struct {

  /*! activation flag */
  bool imbrication_flag;
  bool imbrication_verbose;

  /*! Flags for activating the cressman interpolation for the boundary
      conditions */
  bool cressman_u;
  bool cressman_v;
  bool cressman_qw;
  bool cressman_nc;
  bool cressman_tke;
  bool cressman_eps;
  bool cressman_theta;

  /*! numerical parameters for the cressman interpolation formulas */
  cs_real_t vertical_influence_radius;
  cs_real_t horizontal_influence_radius;

  /*! additional variables */
  int id_u;
  int id_v;
  int id_qw;
  int id_nc;
  int id_tke;
  int id_eps;
  int id_theta;

} cs_atmo_imbrication_t;

/*============================================================================
 * Static global variables
 *============================================================================*/

/* Pointer to atmo options structure */
extern cs_atmo_option_t *cs_glob_atmo_option;

/* Pointer to atmo constants structure */
extern cs_atmo_constants_t *cs_glob_atmo_constants;

/* Pointer to atmo imbrication structure */
extern cs_atmo_imbrication_t *cs_glob_atmo_imbrication;

/*============================================================================
 * Public function definitions
 *============================================================================*/

void
cs_atmo_phyvar_update(void);

/*----------------------------------------------------------------------------
 * initialize fields, stage 0
 *----------------------------------------------------------------------------*/

void
cs_atmo_fields_init0(void);

/*----------------------------------------------------------------------------
 * Automatic boundary condition for cooling towers
 *----------------------------------------------------------------------------*/

void
cs_atmo_bcond(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize meteo profiles if no meteo file is given.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_init_meteo_profiles(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute meteo profiles if no meteo file is given.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_compute_meteo_profiles(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief This function computes the ground elevation
 *
 *  This function solves the following transport equation on \f$ \varia \f$:
 *  \f[
 *  \dfrac{\partial \varia}{\partial t} + \divs \left( \varia \vect{g} \right)
 *      - \divs \left( \vect{V} \right) \varia = 0
 *  \f]
 *  where \f$ \vect{g} \f$ is the gravity field
 *
 *  The boundary conditions on \f$ \varia \f$ read:
 *  \f[
 *   \varia = z \textrm{ on walls}
 *  \f]
 *  \f[
 *   \dfrac{\partial \varia}{\partial n} = 0 \textrm{ elsewhere}
 *  \f]
 *
 *  Remarks:
 *  - a steady state is looked for.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_z_ground_compute(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute hydrostatic profiles of density and pressure.
 *
 *  This function solves the following transport equation on \f$ \varia \f$:
 *  \f[
 *  \divs \left( \grad \varia \right)
 *      = \divs \left( \dfrac{\vect{g}}{c_p \theta} \right)
 *  \f]
 *  where \f$ \vect{g} \f$ is the gravity field and \f$ \theta \f$
 *  is the potential temperature.
 *
 *  The boundary conditions on \f$ \varia \f$ read:
 *  \f[
 *   \varia = \left(\dfrac{P_{sea}}{p_s}\right)^{R/C_p} \textrm{on the ground}
 *  \f]
 *  and Neumann elsewhere.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_hydrostatic_profiles_compute(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Universal function phim for neutral, stable and unstable
 *
 * \param[in]  z             altitude
 * \param[in]  dlmo          Inverse Monin Obukhov length
 *
 * \return                   factor
 */
/*----------------------------------------------------------------------------*/

cs_real_t
cs_mo_phim(cs_real_t              z,
           cs_real_t              dlmo);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Universal function phih for neutral, stable and unstable
 *
 * \param[in]  z             altitude
 * \param[in]  dlmo          Inverse Monin Obukhov length
 * \param[in]  prt           Turbulent Prandtl number
 *
 * \return                   factor
 */
/*----------------------------------------------------------------------------*/

cs_real_t
cs_mo_phih(cs_real_t              z,
           cs_real_t              dlmo,
           cs_real_t              prt);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Universal function psim for neutral, stable and unstable
 *
 * \param[in]  z             altitude
 * \param[in]  z0            roughness
 * \param[in]  dlmo          Inverse Monin Obukhov length
 *
 * \return                   factor
 */
/*----------------------------------------------------------------------------*/

cs_real_t
cs_mo_psim(cs_real_t              z,
           cs_real_t              z0,
           cs_real_t              dlmo);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Universal function psih for neutral, stable and unstable
 *
 * \param[in]  z             altitude
 * \param[in]  z0            roughness
 * \param[in]  dlmo          Inverse Monin Obukhov length
 * \param[in]  prt           Turbulent Prandtl number
 *
 * \return                   factor
 */
/*----------------------------------------------------------------------------*/

cs_real_t
cs_mo_psih(cs_real_t              z,
           cs_real_t              z0,
           cs_real_t              dlmo,
           cs_real_t              prt);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute LMO, friction velocity ustar, friction temperature
 *        tstar from a thermal difference using Monin Obukhov
 *
 * \param[in]  z             altitude
 * \param[in]  z0
 * \param[in]  du            velocity difference
 * \param[in]  dt            thermal difference
 * \param[in]  beta          thermal expansion
 * \param[in]  gredu
 * \param[out] dlmo          Inverse Monin Obukhov length
 * \param[out] ustar         friction velocity
 */
/*----------------------------------------------------------------------------*/

void
cs_mo_compute_from_thermal_diff(cs_real_t   z,
                                cs_real_t   z0,
                                cs_real_t   du,
                                cs_real_t   dt,
                                cs_real_t   beta,
                                cs_real_t   gredu,
                                cs_real_t   *dlmo,
                                cs_real_t   *ustar);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute LMO, friction velocity ustar, friction temperature
 *        tstar from a thermal flux using Monin Obukhov
 *
 * \param[in]  z             altitude
 * \param[in]  z0
 * \param[in]  du            velocity difference
 * \param[in]  flux          thermal flux
 * \param[in]  beta          thermal expansion
 * \param[in]  gredu
 * \param[out] dlmo          Inverse Monin Obukhov length
 * \param[out] ustar         friction velocity
 */
/*----------------------------------------------------------------------------*/

void
cs_mo_compute_from_thermal_flux(cs_real_t   z,
                                cs_real_t   z0,
                                cs_real_t   du,
                                cs_real_t   flux,
                                cs_real_t   beta,
                                cs_real_t   gredu,
                                cs_real_t   *dlmo,
                                cs_real_t   *ustar);

/*----------------------------------------------------------------------------*/
/*!
 * \brief This function set the file name of the meteo file.
 *
 * \param[in] file_name  name of the file.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_set_meteo_file_name(const char *file_name);

/*----------------------------------------------------------------------------*/
/*!
 * \brief 1D Radiative scheme - Solar data + zenithal angle)
 *
 * Compute:
 *   - zenithal angle
 *   - solar contant (with correction for earth - solar length)
 *   - albedo if above the sea
 *   (Use analytical formulae of Paltrige and Platt
 *              dev.in atm. science no 5)
 * \param[in]   xlat        latitude
 * \param[in]   xlong       longitude
 * \param[in]   jour        day in the year
 * \param[in]   heurtu      Universal time (hour)
 * \param[in]   imer        sea index
 * \param[out]  albe        albedo
 * \param[out]  za          zenithal angle
 * \param[out]  muzero      cosin of zenithal angle
 *                          (+ correction due to Earth curvature)
 * \param[out]  omega       solar azimut angle
 * \param[out]  fo          solar constant
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_compute_solar_angles(cs_real_t xlat,
                             cs_real_t xlong,
                             cs_real_t jour,
                             cs_real_t heurtu,
                             int       imer,
                             cs_real_t *albe,
                             cs_real_t *za,
                             cs_real_t *muzero,
                             cs_real_t *omega,
                             cs_real_t *fo);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Print the atmospheric module options to setup.log.
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_log_setup(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Deallocate arrays for atmo module
 */
/*----------------------------------------------------------------------------*/

void
cs_atmo_finalize(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Allow call of cs_user fonctions during soil model computation
 */
/*----------------------------------------------------------------------------*/

void
cs_user_soil_model(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute soil and interface values using Deardorff force restore method
 */
/*----------------------------------------------------------------------------*/

void
cs_soil_model(void);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_ATMO_H__ */
