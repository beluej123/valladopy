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
    
    2024-09-14:
    For functions ccheck out https://github.com/cosinekitty/astronomy/tree/master/source

References:
----------
    [1] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020, 2nd ed.).
        Fundamentals of Astrodynamics. Dover Publications Inc.
    [2] Vallado, David A., (2013, 4th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
    [3] Curtis, H.W. (2009 2nd ed.).
        Orbital Mechanics for Engineering Students. Elsevier Ltd.
    [4] Vallado, David A., (2022, 5th ed.).
        Fundamentals of Astrodynamics and Applications. Microcosm Press.
"""

import math

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


def calc_ecc(r_peri: float, r_apo: float) -> float:
    """
    Calculate eccentricity, given r_periapsis, r_apoapsis.
    Assume ellipse.

    Input Parameters:
    ----------
        r_peri : float, radius of periapsis
        r_apo : float, radius of apoapsis

    Returns:
    -------
        eccentricity : float
    """
    return (r_apo - r_peri) / (r_apo + r_peri)


def val_hohmann(r_init: float, r_final: float, mu_trans: float):
    """
    Vallado Hohmann Transfer, p.325, algorithm 36.
    Assume one central body; inner and outer circular orbits.
    See Vallado fig 6-4, p323.

    Input Parameters:
    ----------
        r_init   : float, initial orbit radius
        r_final  : float, final orbit radius
        mu_trans : float, transfer central body gravitational constant

    Returns:
    -------
        tof_hoh   :
        ecc_trans :
        dv_total  :

    """
    r1 = r_init
    r2 = r_final
    mu = mu_trans

    v_init = calc_v1(mu, r1)  # circular orbit velocity
    # print(f"v1 initial velocity: {v_init:0.8g} [km/s]")
    v_final = calc_v1(mu, r2)
    # print(f"v2 final velocity: {v_final:0.8g} [km/s]")
    a_trans = cal_a(r1, r2)  # transfer semi-major axis
    # print(f"transfer semimajor axis (a): {a_trans:0.8g} [km]")

    # transfer ellipse relations: point a and point b
    v_trans_a = calc_v2(mu, r1, a_trans)  # elliptical orbit velocity at a
    # print(f"velocity, transfer periapsis: {v_trans_a:0.8g} [km/s]")
    v_trans_b = calc_v2(mu, r2, a_trans)  # elliptical orbit velocity at b
    # print(f"velocity, transfer apoapsis: {v_trans_b:0.8g} [km/s]")

    # delta-velocity relations
    dv_total = abs(v_trans_a - v_init) + abs(v_final - v_trans_b)
    # print(f"total delta-velocity: {dv_total:0.8g} [km/s]")

    # time of flight (tof) for hohmann transfer
    tof_hoh = calc_tof(mu, a_trans)
    # print(f"\nHohmann transfer time: {(tof_hoh/2):0.8g} [s], {tof_hoh/(2*60):0.8g} [m]")

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


def val_bielliptic(r_init: float, r_b: float, r_final: float, mu_trans: float):
    """
    Vallado Bi-Elliptic Transfer, algorithm 37.
        Assume one central body.  See Vallado fig 6-5, p324.
        Bi-Elliptic uses one central body for the transfer ellipse.
        Two transfer ellipses; one at periapsis, the other at apoapsis.

    Input Parameters:
    ----------
        r_init   : float, initial orbit radius
        r_b      : float, point b, beyond r_final orbit
        r_final  : float, final orbit radius
        mu_trans : float, transfer central body gravitational constant
    Returns:
    -------
        a_trans1
        a_trans2
        t_trans1
        t_trans2
        dv_total
    """
    r1 = r_init
    r2 = r_final
    mu = mu_trans

    # calculate the two transfer semi-major axis (aka sma)
    a_trans1 = cal_a(r1, r_b)  # transfer 1 semi-major axis
    # print(f"semimajor axis, transfer 1: {a_trans1:0.8g} [km]")
    a_trans2 = cal_a(r_b, r2)  # transfer 1 semi-major axis
    # print(f"semimajor axis, transfer 2: {a_trans2:0.8g} [km]")

    # note, calc_v1()=circular, calc_v1()=elliptical
    # velocity(a1), initial orbit velocity; assume circular orbit
    v_init = calc_v1(mu, r1)
    # print(f"v1 initial velocity: {v_init:0.8g} [km/s]")

    # velocity(a1), start transfer ellipse velocity: point a -> b
    v_trans_1a = calc_v2(mu, r1, a_trans1)  # v at elipse peri- or apo-apsis
    # print(f"velocity, transfer 1a: {v_trans_1a:0.8g} [km/s]")

    # velocity(b1), finish transfer ellipse velocity: point b
    v_trans_1b = calc_v2(mu, r_b, a_trans1)  # v at elipse peri- or apo-apsis
    # print(f"velocity, transfer 1b: {v_trans_1b:0.8g} [km/s]")

    # velocity(b2), start transfer ellipse velocity: point b -> c
    v_trans_2b = calc_v2(mu, r_b, a_trans2)  # v at elipse peri- or apo-apsis
    # print(f"velocity, transfer 2b: {v_trans_2b:0.8g} [km/s]")

    # velocity(c1), finish transfer ellipse: point c
    v_trans_2c = calc_v2(mu, r2, a_trans2)  # v at elipse peri- or apo-apsis
    # print(f"velocity, transfer 2c: {v_trans_2c:0.8g} [km/s]")

    # velocity(c1), final transfer circlar orbit: point c
    v_final = calc_v1(mu, r2)
    # print(f"v2 final velocity: {v_final:0.8g} [km/s]")

    # delta-velocity relations
    dv_a = v_trans_1a - v_init
    dv_b = v_trans_2b - v_trans_1b
    dv_c = v_final - v_trans_2c
    dv_total = abs(dv_a) + abs(dv_b) + abs(dv_c)
    # print(f"total delta-velocity: {dv_total:0.8g} [km/s]")

    # time of flight (tof) for bi-elliptic transfer
    t_trans1 = calc_tof(mu, a_trans1) / 2  # tof time-of-flight
    t_trans2 = calc_tof(mu, a_trans2) / 2  # tof time-of-flight
    return a_trans1, a_trans2, t_trans1, t_trans2, dv_total


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


def planet_ele_0(planet_id: int, eph_data=0, au_units=True, rad_units=False):
    """
    Planet orbital elements from user chosen data set.

    Input Parameters:
    ----------
        planet_id  : int,
            JPL Horizons eph_data=0, 1->8 Mercury->Neptune
            Standish eph_data=1, 1->8 Mercury->Neptune
        eph_data   : int, 0 or 1:
            0 = JPL horizons data set, Table 1
            1 = Standish 1992 data
        au_units   : boolean; output true=distance units in au
        rad_units  : boolean; output true=angular units in radians

    Returns (for planet_id input):
    -------
        J2000_coe   : numpy.array, J2000 clasic orbital elements (Kepler).
        J2000_rates : numpy.array, coe rate change (x/century) from 2000-01-01.
    Notes:
    ----------
        Note, I want to prevent array auto formatting (with vscode black), so you
            will see various schemes; mostly they do not work - frustraiting.
        Orbital element naming amongst many authors is still challenging...
        Element list output:
            sma   = [km] semi-major axis (aka a)
            ecc   = [--] eccentricity
            incl  = [deg] inclination angle; to the ecliptic
            RAAN  = [deg] right ascension of ascending node (aka capital W)
                    longitude node
            w_bar = [deg] longitude of periapsis
            L     = [deg] mean longitude

        References: see list at file beginning.
    """
    # Keplerian Elements and Rates, JPL Horizons, Table 1; EXCLUDING Pluto.
    #   From JPL Horizons:
    #   https://ssd.jpl.nasa.gov/planets/approx_pos.html
    #   https://ssd.jpl.nasa.gov/tools/orbit_viewer.html
    #   Mean ecliptic and equinox of J2000; time-interval 1800 AD - 2050 AD.

    #   NOTICE !!
    #   *** UNITS are different between the two data sets. ***
    
    if eph_data == 0:  # JPL horizons data set
    # Horizons Table 1 data set COPIED DIRECTLY from the web-site noted above.
    
    #   JPL Table 1 order of the elements is different then the other list below.
    #   Also note, Table 1 list earth-moon barycenter, not just earth.
    #           sma   |    ecc      |     incl    | long.node   | long.peri   | mean.long
    #       au, au/cy | ecc, ecc/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy
        J2000=np.array(
            [
            [0.38709927, 0.20563593,  7.00497902,  252.25032350, 77.45779628, 48.33076593],#xxx
            [0.00000037,  0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
            [0.72333566,  0.00677672, 3.39467605,   181.97909950, 131.60246718, 76.67984255],
            [0.00000390,  -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
            [1.00000261,  0.01671123,  -0.00001531, 100.46457166,  102.93768193, 0.0],
            [0.00000562,  -0.00004392, -0.01294668, 35999.37244981,  0.32327364, 0.0],
            [1.52371034,  0.09339410,  1.84969142,  -4.55343205, -23.94362959, 49.55953891],
            [0.00001847,  0.00007882,  -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
            [5.20288700,  0.04838624,  1.30439695,  34.39644051,  14.72847983, 100.47390909],
            [-0.00011607, -0.00013253, -0.00183714, 3034.74612775,  0.21252668, 0.20469106],
            [9.53667594,  0.05386179,  2.48599187,  49.95424423,  92.59887831, 113.66242448],
            [-0.00125060, -0.00050991, 0.00193609,  1222.49362201, -0.41897216, -0.28867794],
            [19.18916464, 0.04725744, 0.77263783, 313.23810451,  170.95427630, 74.01692503],
            [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
            [30.06992276, 0.00859048, 1.77004347, -55.12002969,  44.96476227, 131.78422574],
            [0.00026291,  0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]
            ]
            )
        J2000_elements=J2000[::2] # every other row, starting row 0
        cent_rates=J2000[1::2] # every other row, starting row 1
        
    # Data below, copied Curtis tbl 8.1, Standish et.al. 1992
    # Elements, numpy.array
    # (semi-major axis)|             |             |(RAAN, Omega)| (omega_bar) |
    #            sma   |    ecc      |     incl    | long.node   | long.peri   |  mean.long (L)
    #        au, au/cy | ecc, ecc/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy
    elif eph_data == 1:
        J2000_elements = np.array([
            [0.38709893, 0.20563069, 7.00487, 48.33167, 77.4545, 252.25084],#xxx
            [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
            [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
            [1.52366231, 0.09341233, 1.845061, 49.57854, 336.04084, 355.45332],
            [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
            [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
            [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
            [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
            [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881]
        ])
        # century [cy] rates, numpy.array
        # Data below, copied Curtis tbl 8.1, Standish et.al. 1992
        # Units of rates table:
        # "au/cy", "1/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy"
        cent_rates = np.array([
            [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],#xxx
            [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
            [-0.0000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
            [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
            [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
            [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
            [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
            [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
            [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90]
        ])
        # convert arc-sec/cy to deg/cy
        # There must be a better way for this conversion; this gets the job done
        cent_rates[:,2] /= 3600.0
        cent_rates[:,3] /= 3600.0
        cent_rates[:,4] /= 3600.0
        cent_rates[:,5] /= 3600.0
        
        # elements & rates conversions
        au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
        deg2rad = math.pi/180
        if au_units == False: # then convert units to km
            J2000_elements[:,0] *= au  # [km] sma (semi-major axis, aka a) convert
            cent_rates[:,0] *= au
        if rad_units == True:
            J2000_elements[:,2] *= deg2rad  # [rad]
            cent_rates[:,2] *= deg2rad
            J2000_elements[:,3] *= deg2rad  # [rad]
            cent_rates[:,3] *= deg2rad
            J2000_elements[:,4] *= deg2rad  # [rad]
            cent_rates[:,4] *= deg2rad
            J2000_elements[:,5] *= deg2rad  # [rad]
            cent_rates[:,5] *= deg2rad
    
        # np.set_printoptions(precision=4)
        # print(f"J2000_elements=\n{J2000_elements}")
        # print(f"cent_rates=\n{cent_rates}")
        
    # extract user requested planet coe data & rates;
    #   reminder, coe=classic orbital elements (Kepler)
    J2000_coe = J2000_elements[planet_id - 1]
    J2000_rates = cent_rates[planet_id - 1]
    return J2000_coe, J2000_rates
    
def planet_ele_1(planet_id=4, au_units=True, rad_units=False):
    """ 
    Planet elements coefficients table, heliocentric; polynomial in t_TDB.
        t_TDB = julian centuries of tdb (barycentric dynamic time).
        Format: x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    
    Notes:
    ----------
    Data below, Vallado [4], appendix D.4, Planetary Ephemerides, pp.1062
        Ecliptic coordinates, mean equator, mean equinox of IAU-76/FK5.
    """
    
    if planet_id ==0: # mercury
        print(f"mercury not yet copied, 2024-09-15")
    if planet_id ==1: # venus
        print(f"venus not yet copied, 2024-09-15")
    if planet_id ==2: # earth
        print(f"earth not yet copied, 2024-09-15")
    if planet_id ==3: # mars
        print(f"mars not yet copied, 2024-09-15")
        
    if planet_id == 4: # jupiter
        J2000_coefs = np.array([
                [5.202603191, 0.0000001913, 0, 0, 0],# [au] sma
                [0.048494850, 0.0001632440, -0.0000004719, -0.00000000197, 0],# [--] ecc
                [1.303270000, -0.001987200, 0.0000331800, 0.00000009200, 0],# [deg] incl
                [100.4644410, 0.1766828000, 0.0009038700, -0.0000070320, 0],# [deg] raan
                [14.33130900, 0.2155525000, 0.0007225200, -0.0000045900, 0],# [deg] w_bar
                [34.35148400, 3034.9056746, -0.000085010, 0.00000000400, 0]# [deg] Lm
            ])
    elif planet_id == 5: # saturn
        print(f"saturn not yet copied, 2024-09-15")
    elif planet_id == 6: # uranus
        print(f"uranus not yet copied, 2024-09-15")
    elif planet_id == 7: # neptune
        print(f"neptune not yet copied, 2024-09-15")
    elif planet_id == 8: # pluto
        print(f"pluto not yet copied, 2024-09-15")
    else:
        print(f"Not a valid planet id.")
    
    return J2000_coefs