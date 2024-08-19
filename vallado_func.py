# Vallado functions
# Cannot get black formatter to work within VSCode, in terminal type; black vallado_func.py
# Remember VSCode DocString, Keyboard shortcut: ctrl+shift+2
import numpy as np


def cal_a(r1, r2):
    """Calculate semi-major axis, a
    Parameters
    ----------
    r1 : float, 1st orbit radius
    r2 : float, 2nd orbit radius
    Returns
    -------
    float, semi-major axis, a
    """
    return (r1 + r2) / 2


def calc_v1(mu, r1):  # circular orbit velocity
    return np.sqrt(mu / r1)


def calc_v2(mu, r1, a):  # elliptical orbit velocity
    return np.sqrt(mu * ((2 / r1) - (1 / a)))


def calc_tof(mu, a):  # time of flight for complete orbit
    return 2 * np.pi * np.sqrt(a**3 / mu)


def calc_ecc(r_peri: float, r_apo: float) -> float:  # calculate eccentricity
    """Calculate Eccentricity

    Parameters
    ----------
    r_peri : float, radius of periapsis
    r_apo : float, radius of apoapsis

    Returns
    -------
    eccentricity : float
    """
    return (r_apo - r_peri) / (r_apo + r_peri)


def val_hohmann(r_init: float, r_final: float, mu_trans: float):
    """Vallado Hohmann Transfer, algorithm 36
    Assume one central body; inner and outter orbits r circular.
    See Vallado fig 6-5, p324.

    Parameters
    ----------
    r_init : float, initial orbit radius
    r_final: float, final orbit radius
    mu_trans : float, transfer central body gravitational constant

    Returns
    -------

    """
    r1 = r_init
    r2 = r_final
    mu = mu_trans

    print("*** Vallado Hohmann Transfer ***")  # temporary print

    v_init = calc_v1(mu, r1)  # circular orbit velocity
    print(f"v1 initial velocity: {v_init:0.8g} [km/s]")
    v_final = calc_v1(mu, r2)
    print(f"v2 final velocity: {v_final:0.8g} [km/s]")
    a_trans = cal_a(r1, r2)  # transfer semi-major axis
    print(f"transfer semimajor axis (a): {a_trans:0.8g} [km]")

    # transfer ellipse relations: point a and point b
    v_trans_a = calc_v2(mu, r1, a_trans)  # elliptical orbit velocity at a
    print(f"velocity, transfer periapsis: {v_trans_a:0.8g} [km/s]")
    v_trans_b = calc_v2(mu, r2, a_trans)  # elliptical orbit velocity at b
    print(f"velocity, transfer apoapsis: {v_trans_b:0.8g} [km/s]")

    # delta-velocity relations
    dv_total = abs(v_trans_a - v_init) + abs(v_final - v_trans_b)
    print(f"total delta-velocity: {dv_total:0.8g} [km/s]")

    # time of flight (tof) for hohmann transfer
    tof_hoh = calc_tof(mu, a_trans)
    print(f"\nHohmann transfer time: {(tof_hoh/2):0.8g} [s]")
    print(f"Hohmann transfer time: {tof_hoh/(2*60):0.8g} [m]")

    # calculate transfer eccentricity
    if r_init < r_final:
        r_peri = r_init
        v_peri = v_init
        r_apo = r_final
    else:
        r_peri = r_final
        v_peri = v_final
        r_apo = r_init

    ecc_trans = calc_ecc(r_peri, r_apo)
    return (tof_hoh, ecc_trans, dv_total)
    print(f"transfer eccentricity = {ecc_trans:0.8g}")


def val_bielliptic(r_init: float, r_b: float, r_final: float, mu_trans: float):
    """Vallado Bi-Elliptic Transfer, algorithm 37
        Assume one central body.  See Vallado fig 6-5, p324.
    Parameters
    ----------
    r_init : float, initial orbit radius
    r_b : float, point b, beyond r_final orbit
    r_final: float, final orbit radius
    mu_trans : float, transfer central body gravitational constant
    Returns
    -------
    """
    r1 = r_init
    r2 = r_final
    mu = mu_trans

    print("*** Vallado Bi-Elliptic Transfer ***")  # temporary print

    v_init = calc_v1(mu, r1)  # circular orbit velocity
    print(f"v1 initial velocity: {v_init:0.8g} [km/s]")
    v_final = calc_v1(mu, r2)
    print(f"v2 final velocity: {v_final:0.8g} [km/s]")
    a_trans1 = cal_a(r1, r_b)  # transfer 1 semi-major axis
    print(f"semimajor axis, transfer 1: {a_trans1:0.8g} [km]")
    a_trans2 = cal_a(r_b, r2)  # transfer 1 semi-major axis
    print(f"semimajor axis, transfer 2: {a_trans2:0.8g} [km]")

    # transfer ellipse: point a and point c
    v_trans_1a = calc_v2(mu, r1, a_trans1)  # ellipse 1 orbit velocity
    print(f"velocity, transfer 1a: {v_trans_1a:0.8g} [km/s]")
    v_trans_2c = calc_v2(mu, r2, a_trans2)  # ellipse 2 orbit velocity
    print(f"velocity, transfer 2c: {v_trans_2c:0.8g} [km/s]")
    # transfer ellipse: point b and point c
    v_trans_1b = calc_v2(mu, r_b, a_trans1)  # ellipse 1 orbit velocity
    print(f"velocity, transfer 1b: {v_trans_1b:0.8g} [km/s]")
    v_trans_2b = calc_v2(mu, r_b, a_trans2)  # ellipse 2 orbit velocity
    print(f"velocity, transfer 2b: {v_trans_2b:0.8g} [km/s]")

    # delta-velocity relations
    dv_total = (
        abs(v_trans_1a - v_init)
        + abs(v_trans_2b - v_trans_1b)
        + abs(v_final - v_trans_2c)
    )
    print(f"total delta-velocity: {dv_total:0.8g} [km/s]")

    # time of flight (tof) for bi-elliptic transfer
    tof_bielliptic = calc_tof(mu, a_trans1) / 2
    tof_bielliptic = tof_bielliptic + calc_tof(mu, a_trans2) / 2
    print(f"\nBi-Elliptic transfer time: {tof_bielliptic:0.8g} [s]")
    print(f"Bi-Elliptic transfer time: {tof_bielliptic/(60):0.8g} [m]")
    print(f"Bi-Elliptic transfer time: {tof_bielliptic/(60*60):0.8g} [hr]")


def val_one_tan_burn(r_init: float, r_final: float, nu_trans_b: float, mu_trans: float):
    """Vallado One-Tangent Burn Transfer, algorithm 38
        Assume one central body.  See Vallado fig 6-7, p331.
    TODO: account for quadrants; R_ratio, see Vallado p.331-334.
    TODO: manage transfer start; peri vs apoapsis; see Vallado p.335.

    Parameters
    ----------
    r_init : float, initial orbit radius
    r_final: float, final orbit radius
    nu_trans_b : float, true anomaly (angle to point)
    mu_trans : float, transfer central body gravitational constant
    """
    r1 = r_init
    r2 = r_final
    mu = mu_trans
    # convert input degrees to radians
    nu_trans_b1 = nu_trans_b * np.pi / 180

    print("*** Vallado One-Tangent Burn Transfer ***")  # temporary print

    R_ratio = r_init / r_final
    print(f"orbital radius ratio = {R_ratio:0.8g}")

    ecc_trans = (R_ratio - 1) / (np.cos(nu_trans_b1) - R_ratio)
    print(f"transfer eccentricity = {ecc_trans:0.8g}")

    a_trans = r_init / (1 - ecc_trans)  # transfer semi-major axis
    print(f"semimajor axis, transfer: {a_trans:0.8g} [km]")

    v_1 = calc_v1(mu, r1)  # circular orbit velocity
    print(f"v1 initial velocity: {v_1:0.8g} [km/s]")
    v_2 = calc_v1(mu, r2)
    print(f"v2 final velocity: {v_2:0.8g} [km/s]")

    # transfer ellipse
    v_trans_a = calc_v2(mu, r1, a_trans)  # ellipse orbit velocity
    print(f"velocity, transfer a: {v_trans_a:0.8g} [km/s]")
    v_trans_b = calc_v2(mu, r2, a_trans)  # ellipse orbit velocity
    print(f"velocity, transfer b: {v_trans_b:0.8g} [km/s]")

    # delta-velocity relations
    dv_a = v_trans_a - v_1
    print(f"delta-velocity a: {dv_a:0.8g} [km/s]")

    # flight path angle
    fpa_tan = ecc_trans * np.sin(nu_trans_b1) / (1 + ecc_trans * np.cos(nu_trans_b1))
    fpa_trans_b = np.arctan(fpa_tan)  # [rad]
    print(f"flight path angle, transfer = {fpa_trans_b*180/np.pi:0.8g} [deg]")

    # orbit delta velocities
    dv_b = np.sqrt(v_2**2 + v_trans_b**2 - 2 * v_2 * v_trans_b * np.cos(fpa_trans_b))
    print(f"delta velocity b = {dv_b:0.8g} [km/s]")
    dv_orb = abs(dv_a) + abs(dv_b)
    print(f"orbit delta velocity = {dv_orb:0.8g} [km/s]")

    # calculate time-of-flight (tof), see Vallado p335
    tof_cosE = (ecc_trans + np.cos(nu_trans_b1)) / (1 + ecc_trans * np.cos(nu_trans_b1))
    print(f"tof_cosE = {tof_cosE:0.8g}")
    # tof_cosE alway has two allowable solutions
    tof_E = np.arccos(tof_cosE)
    print(f"tof_E = {(tof_E * 180/np.pi):0.8g} [deg]")
    # Notes: 1) transfer starts at periapsis, thus E0=0,
    #   2) transfer does not pass periapsis, thus k=0; see Vallado p335.
    tof_trans = np.sqrt(a_trans**3 / mu) * (
        2 * np.pi * 0
        + (tof_E - ecc_trans * np.sin(tof_E))
        - (0 - ecc_trans * np.sin(0))
    )
    print(f"tof_trans = {tof_trans:0.8g} [sec]")
    print(f"tof_trans = {tof_trans/60:0.8g} [min]")
    print(f"tof_trans = {(tof_trans/(60*60)):0.8g} [hr]")
