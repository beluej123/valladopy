"""
Vallado function collection.
Edits 2024-08-21 +, Jeff Belue.

Notes:
----------
    TODO, This file is organized ...
    Generally, units shown in brackets [km, rad, deg, etc.].
    Generally, angles are saved in [rad], distance [km].
    Reminder to me; cannot get black formatter to work within VSCode,
        so in terminal type; black *.py
    Reminder to me; VSCode DocString, Keyboard shortcut: ctrl+shift+2
    
    Test for the functions below may be found in other files, for example,
        astro_time.py, kepler.py, test_kepler, vallado_ex1.py etc...

References
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
"""

import numpy as np


def cal_a(r1: float, r2: float):
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


def calc_v1(mu: float, r1: float):  # circular orbit velocity
    return np.sqrt(mu / r1)


def calc_v2(mu: float, r1: float, a: float):  # elliptical orbit velocity
    return np.sqrt(mu * ((2 / r1) - (1 / a)))


def calc_tof(mu: float, a: float):  # time of flight for complete orbit
    return 2 * np.pi * np.sqrt(a**3 / mu)


def calc_ecc(r_peri: float, r_apo: float) -> float:  # calculate eccentricity
    """
    Calculate Eccentricity

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
    """
    Vallado Hohmann Transfer, algorithm 36
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
    return None


def val_bielliptic(r_init: float, r_b: float, r_final: float, mu_trans: float):
    """
    Vallado Bi-Elliptic Transfer, algorithm 37
        Assume one central body.  See Vallado fig 6-5, p324.
    Input Parameters:
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
    return None


def val_one_tan_burn(r_init: float, r_final: float, nu_trans_b: float, mu_trans: float):
    """
    Vallado One-Tangent Burn Transfer, algorithm 38, pp.333.

    TODO: account for quadrants; R_ratio & (peri vs. apo) for ecc and sma, Vallado p.332
        2024-08-23, done, but not completely tested; parabolic & hyperbolic.
    TODO: manage transfer start E0, now E0=0; Vallado p.335.
        2024-08-23, note Curtis section 6.6, pp.338
    TODO: manage transfer time vs angle (acos(E)); Vallado p.335.
        2024-08-24, done, but not completely tested; parabolic & hyperbolic.

    Input Parameters:
    ----------
        r_init     : float, initial orbit radius magnitude
        r_final    : float, final orbit radius magnitude
        nu_trans_b : float, [deg] true anomaly (angle to point)
        mu_trans   : float, transfer central body gravitational constant

    Returns:
    -------
        ecc_trans  : float, transfer eccentricity
        a_trans    : float, transfer semi-major axis (sma)
        tof_trans  : float, transfer time-of-flight
        dv_a       : float, delta velocity at departure (point a)
        dv_b       : float, delta velocity at arrival (point b)

    Notes:
    -------
        Lots of temporary prints commented out, not yet figured out
            python testing tools.
        Assume one central body, transfer to an orbit, not a planet.
            See Vallado fig 6-7, p.331.
        Assume initial and final orbits are circular!
        Notation: b= point b, the transfer intercept point.

        References: see list at file beginning.
    """
    import math  # some math functions are more effficient than numpy

    # print(f"\n*** Vallado One-Tangent Burn Transfer ***")  # temporary print

    r1 = r_init
    r2 = r_final
    mu = mu_trans
    # convert input degrees to radians
    nu_trans_b1 = nu_trans_b * np.pi / 180

    R_ratio = r1 / r2
    print(f"orbital radius ratio = {R_ratio:0.8g}")

    # periapsis or apoapsis launch sets ecc (eccentricity), Vallado p.332
    cos_R_ratio = math.cos(nu_trans_b1)
    # print(f"cos_R_ratio= {cos_R_ratio:.4g}")
    if (R_ratio > 1.0 and cos_R_ratio > R_ratio) or (
        R_ratio < 1.0 and cos_R_ratio < R_ratio
    ):
        print(f"periapsis launch")

        ecc_trans = (R_ratio - 1) / (np.cos(nu_trans_b1) - R_ratio)
        a_trans = r1 / (1 - ecc_trans)  # transfer semi-major axis
    else:
        print(f"apoapsis launch")

        ecc_trans = (R_ratio - 1) / (np.cos(nu_trans_b1) + R_ratio)
        a_trans = r1 / (1 + ecc_trans)  # transfer semi-major axis

    # print(f"transfer eccentricity = {ecc_trans:0.8g}")
    # print(f"semimajor axis, transfer: {a_trans:0.8g} [km]")

    v_1 = calc_v1(mu, r1)  # circular orbit velocity, for launch
    v_2 = calc_v1(mu, r2)  # circular orbit velocity, for destination
    # print(f"v1 initial velocity: {v_1:0.8g} [km/s]")
    # print(f"v2 final velocity: {v_2:0.8g} [km/s]")

    # launch transfer ellipse orbit velocity; point a
    v_trans_a = calc_v2(mu, r1, a_trans)  # v2=elliptical calc
    # arrive ellipse orbit velocity; point b
    v_trans_b = calc_v2(mu, r2, a_trans)  # ellipse orbit velocity
    # print(f"velocity, transfer a: {v_trans_a:0.8g} [km/s]")
    # print(f"velocity, transfer b: {v_trans_b:0.8g} [km/s]")

    # delta-velocity relation at launch; point a
    dv_a = v_trans_a - v_1  # [km/s]
    # print(f"delta-velocity at launch (point a): {dv_a:0.8g} [km/s]")

    # fpa (flight path angle)
    fpa_tan = ecc_trans * np.sin(nu_trans_b1) / (1 + ecc_trans * np.cos(nu_trans_b1))
    fpa_trans_b = np.arctan(fpa_tan)  # [rad]
    # print(f"flight path angle, transfer = {fpa_trans_b*180/np.pi:0.8g} [deg]")

    # delta-velocity at arrive; point b
    dv_b = np.sqrt(v_2**2 + v_trans_b**2 - 2 * v_2 * v_trans_b * np.cos(fpa_trans_b))
    dv_orb = abs(dv_a) + abs(dv_b)
    # print(f"delta velocity b = {dv_b:0.8g} [km/s]")
    # print(f"orbit delta velocity = {dv_orb:0.8g} [km/s]")

    # calculate time-of-flight (tof), see Vallado p.335
    tof_cosE = (ecc_trans + np.cos(nu_trans_b1)) / (1 + ecc_trans * np.cos(nu_trans_b1))
    # print(f"tof_cosE = {tof_cosE:0.8g}")

    # tof_cosE always has two allowable solutions
    tof_E = np.arccos(tof_cosE)
    if nu_trans_b > 180 and nu_trans_b <= 360:
        tof_E = math.pi + tof_E
    # print(f"tof_E = {(tof_E * 180/np.pi):0.8g} [deg]")

    # Notes, Vallado p.335:
    #   1) transfer starts at periapsis, thus E0=0,
    #   2) transfer does not pass periapsis, thus k=0 in 2*pi*k relation
    tof_trans = np.sqrt(a_trans**3 / mu) * (
        2 * np.pi * 0
        + (tof_E - ecc_trans * np.sin(tof_E))
        - (0 - ecc_trans * np.sin(0))
    )
    # print(
    #     f"tof_trans= {tof_trans:0.8g} [sec], {tof_trans/60:0.8g} [min], {(tof_trans/(60*60)):0.8g} [hr]"
    # )
    # my return variable names: ecc_tx, sma_tx, tof_tx, dv_0, dv_1
    return ecc_trans, a_trans, tof_trans, dv_a, dv_b


def plot_sp_vs_sma(r0_mag, r1_mag, delta_nu, sp=1.0, clip1=True) -> None:
    """
    Plot sp (semi-parameter) vs. sma (semi-major axis); note BMWS, p.205, fig.5-3.
    Plot range recognizes difference between ellipse and hyperbolic trajectories.
    User may choose to clip calculated sma, since sma may calculate to infinite.
    Plotting may take some fooling with to get the outcome u want.
    Feel free to improve plot lables, etc.

    Input Parameters
    ----------
        r0_mag   : float, focus to initial radius
        r1_mag   : float, focus to 2nd radius
        delta_nu : float, angular distance between r0, r1; focus is central point
        sp       : float, optional, default 1.0 turns off sp marker in plot
        clip1    : boolean, optional, default true; NOTE sma maybe near infinite

    Returns
    -------
        Plot sp vs. sma.
    """

    import math

    import matplotlib.pyplot as plt
    import numpy as np

    # constants for given r0, r1, angle
    k = r0_mag * r1_mag * (1 - np.cos(delta_nu))  # BMWS [1], p.204, eqn 5-42
    l = r0_mag + r1_mag
    m = abs(r0_mag) * abs(r1_mag) * (1 + np.cos(delta_nu))
    sma = k_l_m_sp(k, l, m, sp)  # semi-major axis

    # bracket p values for ellipse, p_i & p_ii; BMWS [1], p.205
    # values > p_ii, hyperbolic trajectories
    # value at p_ii, parabolic trajectory
    # minimum sp for ellipse; calculated value maybe degenerate...
    sp_i = k / (l + np.sqrt(2 * m))  # BMWS [1], p.208, eqn 5-52
    sp_i_min = sp_i * 1.001  # practical minimum for ellipse
    # maximum sp for ellipse; calculated value is actually a parabola
    sp_ii = k / (l - np.sqrt(2 * m))  # BMWS [1], p.208, eqn 5-53
    sp_ii_max = sp_ii * 1.01  # will show part of hyperbola
    sp_i_mid = (sp_ii - sp_i) / 2

    if sma > 0 and sp > 1.0:  # ellipse
        x = np.linspace(
            sp_i_min, sp_ii_max, 100
        )  # between sp_min & sp_max plot 100 points
    elif sma <= 0 and sp > 1.0:  # parabolic or hyperbolic
        x = np.linspace(sp_ii_max, sp * 10)  # between sp_min & sp_max plot 100 points
    y = k_l_m_sp(k, l, m, x)

    # if clipping enabled, limit y value excursions
    if clip1 == True and sma > 0:
        y = np.clip(y, -sp_i_mid * 200, sp_i_mid * 500)
    else:  # parabolic or hyperbolic plot range near extremes...
        y = np.clip(y, -sp * 5, sp * 5)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    # if optional sp in calling routine, add marker
    if sp != 1.0:  # optional input parameter sp defined
        ecc = math.sqrt(1 - sp / sma)  # BMWS [2] p.21, eqn.1-44
        ax.plot(sp, sma, marker="h", ms=10, mfc="red")
        plt.text(
            sp,
            sma,
            f"sp={sp:.6g}\nsma={sma:.6g}\n ecc={ecc:.6g}",  # move ecc text away from marker
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    text1 = "Ellipse= between positive peaks.\nHyperbola= -sma"
    text2 = (
        f"r0={r0_mag:8g}\nr1={r1_mag:.8g}\ndelta angle={delta_nu*180/np.pi:.6g} [deg]"
    )
    plt.text(
        0.5,
        0.9,
        text2,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.text(
        0.4,
        0.75,
        text1,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.xlabel("sp (semi-parameter, aka p)")
    plt.ylabel("sma (semi-major axis, aka a)")
    plt.grid(True)
    plt.title("SMA vs. SP")
    # not so sure about a graphic fill, below, to graphically denote ellipse
    # ax.fill_between(x, np.max(y), where=y > 0, facecolor="green", alpha=0.5)
    plt.show()
    return None  # plot_sp_vs_sma()


def k_l_m_sp(k, l, m, sp):
    """
    Calculate sma from input parameters
    """
    # BMWS [2], p.204, eqn 5-42
    sma = (m * k * sp) / ((2 * m - l**2) * sp**2 + (2 * k * l * sp - k**2))
    return sma
