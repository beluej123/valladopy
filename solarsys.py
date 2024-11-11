# -*- coding: utf-8 -*-
"""
Exploring optimal methods to define celestial body parameters that
    may be retrieved or written to by other functions.
2024-11-08, not sure what makes sense for parameter data storage and retrieval.
    dataclass or class or something else (i.e. import txt file with formatted data?)
Solar system parameters/constants:
    dataclass's organized; orbit & body.

References:
----------
    See references.py for references list.

Notes:
----------
    2024-11-07, exploring schemes to capture celestial body data.
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally, angles are saved in [rad], distance [km].

    https://www.dataquest.io/blog/how-to-use-python-data-classes/
    Search, ArjanCodes and dataclass, https://www.youtube.com/watch?v=lX9UQp2NwTk

Orbital elements naming collection, see extensive list in kepler.py.
"""

import math
from dataclasses import dataclass, field, fields

deg2rad = math.pi / 180


class Body:
    x = 0


class Moon(Body):
    # Orbit Params
    sma_ER = 60.27
    sma_km = 384400.0
    ecc = 0.05490
    incl_deg = 5.145396
    raan_deg = None
    long_perihelion = None
    true_long = None
    period_yrs = 0.0748
    tropical_days = 27.321582
    orb_vel = 1.0232  # km/s

    # Body Params
    eq_radius_km = 1738.0
    flat = None
    mu = 4902.799
    mass_norm = 0.01230
    mass_kg = 7.3483e22
    rot_days = 27.32166
    eq_inc = 6.68
    j2 = 0.0002027
    j3 = None
    j4 = None
    density = 3.34


class Earth(Body):
    # Orbit Params
    sma_au = 1.00000100178
    sma_km = 149598023.0
    ecc = 0.016708617
    incl = 0.0
    raan_deg = 0.0
    long_perihelion = 102.93734808
    true_long = 100.46644851
    period_yrs = 0.99997862

    # Body Params
    eq_radius_km = 6378.1363
    flag = 0.0033528131
    mu = 3.986004415e5
    mass_norm = 1.0
    mass_kg = 5.9742e24
    rot_days = 0.99726968
    eq_inc = 23.45
    j2 = 0.0010826269
    j3 = -0.0000025323
    j4 = -0.0000016204
    density = 5.515


class OrbitParameters:

    # Constructor method initializes an instance of the class with the given values.
    def __init__(self, sma, ecc, incl, raan, w_, nu, mu=398600.4418):
        """
        Initialize an OrbitalParameters object.
        Args:
            a (float)   : [km] Semi-major axis
            e (float)   : Eccentricity
            i (float)   : [rad] Inclination
            raan (float): [rad] Right ascension of ascending node
            w_ (float)  : [rad] Argument of perigee, aka argp
            nu (float)  : [rad] True anomaly
            mu (float)  : [km^3/s^2] Gravitational parameter
                            Defaults to Earth's value
        """
        self.sma = sma  # [km] Semi-major axis
        self.ecc = ecc  # Eccentricity
        self.incl = incl  # [rad] Inclination
        self.raan = raan  # [rad] Right Ascension of the Ascending Node (omega)
        self.w_ = w_  # [rad] Argument of Perigee
        self.nu = nu  # [rad] True Anomaly
        self.mu = mu  # [km^3/s^2] Gravitational parameter

    def __str__(self):
        """
        The string representation method prints orbital parameters
            in a human-readable format.
        """
        return (
            f"Orbital Parameters:\n"
            f"Semi-major axis (sma): {self.sma} km\n"
            f"Eccentricity (ecc): {self.ecc}\n"
            f"Inclination (incl): {self.incl} rad\n"
            f"RAAN (omega): {self.raan} rad\n"
            f"Argument of Perigee (w_): {self.w_} rad\n"
            f"True Anomaly (nu): {self.nu} rad\n"
            f"Gravitational Parameter (mu): {self.mu} km^3/s^2"
        )


# orbit = OrbitParameters(sma=7000, ecc=0.05, incl=0.5, raan=1.2, w_=2.5, nu=0.8)
# print(orbit)


@dataclass(frozen=False, kw_only=True, slots=True)
class StarParms:  # includes solar constants
    au_: float  # derrived from earth orbit
    mu: float  # [km^3/s^2] Gravitational parameter
    mass_kg: float  # [kg] star mass


@dataclass(frozen=False, kw_only=True, slots=True)
class OrbitParms:
    ref_plane: str  # data reference; i.e. ecliptic (ecl), equatorial (equ)
    ref_jd: float  # data associated with Julian date
    sma: float  # [km] Semi-major axis
    ecc: float  # Eccentricity
    incl: float  # [rad] Inclination
    raan: float  # [rad] Right Ascension of the Ascending Node (omega)
    # w_bar, longitude of periapsis (aka II), equatorial
    w_bar: float  # [rad] Longitude of Periapsis
    Lt0: float  # [rad] true longitude at epoch


@dataclass(frozen=False, kw_only=True, slots=True)
class BodyParams:
    eq_radius_km: float
    flatten: float  # reciprocal falttening
    mu: float  # [km^3/s^2] Gravitational parameter
    mass_norm: float
    mass_kg: float
    rot_days: float  # [days] day rotation
    eq_inc: float  # equator inclination to orbit
    j2: float
    j3: float
    j4: float
    density: float


# @dataclass(frozen=False, kw_only=True, slots=True)
# class Body_Data:
#     b_name: str  # body name
#     OrbitParms: OrbitParms
#     BodyParams: BodyParams


earth_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or ICRF equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=149598023.0,  # [km], Vallado [4] p.1057, tbl.D-3
    ecc=0.016708617,  # [km], Vallado [4] p.1057, tbl.D-3
    incl=0.0,  # [rad], Vallado [4] p.1057, tbl.D-3
    raan=0.0,  # [rad], Vallado [4] p.1057, tbl.D-3
    # w_bar=longitude of periapsis
    w_bar=102.93734808 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    Lt0=100.46644851 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
earth_b_prms = BodyParams(
    eq_radius_km=6378.1363,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0033528131,  # [], Vallado [4] p.1057, tbl.D-3
    mu=3.986004415e5,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=1.0,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=5.9742e24,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=0.99726968,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=23.45 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.0010826269,  # [], Vallado [4] p.1057, tbl.D-3
    j3=-0.0000025323,  # [], Vallado [4] p.1057, tbl.D-3
    j4=-0.0000016204,  # [], Vallado [4] p.1057, tbl.D-3
    density=5.515,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)
mars_o_prms = OrbitParms(
    ref_plane="equ",  # data reference; ecliptic (ecl) or equatorial (equ)
    ref_jd=2451544.5,  # data associated with Julian date
    sma=149598023.0,  # [km], Vallado [2] p.1041, tbl.D-3
    ecc=0.016708617,
    incl=0.0,
    raan=0.0,
    # w_bar=longitude of periapsis
    w_bar=336.06023398 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    Lt0=355.43327463 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
)
mars_b_prms = BodyParams(
    eq_radius_km=3397.2,  # [km], Vallado [4] p.1057, tbl.D-3
    flatten=0.0064763,  # [], Vallado [4] p.1057, tbl.D-3
    mu=4.305e4,  # [km^3/s^2], Vallado [4] p.1057, tbl.D-3
    mass_norm=0.10744,  # [], Vallado [4] p.1057, tbl.D-3
    mass_kg=6.4191e23,  # [kg], Vallado [4] p.1057, tbl.D-3
    rot_days=1.2595675,  # [days], Vallado [4] p.1057, tbl.D-3
    eq_inc=25.19 * deg2rad,  # [rad], Vallado [4] p.1057, tbl.D-3
    j2=0.001964,  # [], Vallado [4] p.1057, tbl.D-3
    j3=0.00036,  # [], Vallado [4] p.1057, tbl.D-3
    j4=0,  # [], Vallado [4] p.1057, tbl.D-3
    density=3.94,  # [gm/cm^3], Vallado [4] p.1057, tbl.D-3
)
sun_prms = StarParms(
    au_=149598023.0,  # [km], Vallado [4] p.1059, tbl.D-5
    mu=1.32712428e11,  # [km^3/s^2], Vallado [4] p.1059, tbl.D-5
    mass_kg=1.9891e30,  # [kg], Vallado [4] p.1059, tbl.D-5
)

# print(f"{earth_o_prms.sma} [km]")  # [km] earth orbit parameters
# print(f"{earth_b_prms.eq_inc} [rad]")  # [rad] earth body parameters

# add new attribute to class
# earth_params.b_name = "earth" # linter has a problem with this...
# print(f"{earth_params.b_name}")

# field_list = fields(earth_params)
# for field in field_list:
#     print(f"{field.name}, {getattr(earth_params, field.name)}")
