"""
Vallado function collection.
Edits 2024-08-21 +, Jeff Belue.
Be careful of Vallado book example answers. Some Vallado examples have calculation errors,
    after all the book is 1000+ pages.

Notes:
----------
    Note, some code below, have AUTO-formatting instructions for "black" formatter!
        Generally, donot format numeric arrays (tables); use # fmt: off/on
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
    See references.py for references list.
"""

import math

import numpy as np

import astro_time
from kepler import coe2rv, eccentric_to_true, kep_eqtnE


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
    Hohmann Transfer, calculate tof, eccentricity, delta-v; Vallado [4] example 6-1, pp.330.
        Vallado [2] p.325, algorithm 36.
        Vallado [4] p.329, algorithm 36.
    Assume one central body; inner and outer circular orbits.
    See Vallado [4] fig 6-4, p327.

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
    Planet orbital elements from ephemeris data set:
        1) JPL Horizons Table 1.
        2) Curtis [3] Table 8.1, p.472

    Input Parameters:
    ----------
        planet_id  : int,
            0=Mercury, 1=Venus, 2=Earth, 3=Mars, 4=Jupiter,
            5=Saturn, 6 = Urnaus, 7=Neptune
        eph_data   : int, ephemeris data set
            0 = JPL horizons Table 1 data set
            1 = Curtis [3] Table 8.1, p.472
        au_units   : boolean; output true=distance units in au
        rad_units  : boolean; output true=angular units in radians

    Returns: (for planet_id input):
    -------
        J2000_coe   : numpy.array, J2000 clasic orbital elements (Kepler).
        J2000_rates : numpy.array, coe rate change (x/century) from 2000-01-01.
    Notes:
    ----------
        Note, I want to prevent **array** auto formatting (with vscode black),
            so I use the "# fmt: off" and "# fmt: on" commands.
        Orbital element naming amongst many authors is still challenging...
        Element list output, Curtis [3] order; not same as Horizons order !
            sma   = [km] semi-major axis (aka a)
            ecc   = [--] eccentricity
            incl  = [deg] inclination angle; to the ecliptic
            RAAN  = [deg] right ascension of ascending node (aka capital W)
                    (aka longitude node)
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
    #   *** ANGLE UNITS are different between the two data sets. ***

    if eph_data == 0:  # JPL horizons data set
        # Horizons Table 1 data set COPIED DIRECTLY from the web-site noted above.

        #   JPL Table 1 order of the elements is different then the other list below.
        #   Also note, Table 1 list earth-moon barycenter, not just earth; ecliptic data!
        #           sma   |    ecc      |     incl    | mean.long   | long.peri   | long.node
        #                 |             |             |     L       |  w_bar      | raan
        #       au, au/cy | ecc, ecc/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy
        # fmt: off
        J2000=np.array(
            [
            [0.38709927,   0.20563593,  7.00497902,  252.25032350, 77.45779628, 48.33076593],
            [0.00000037,   0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
            [0.72333566,   0.00677672, 3.39467605,   181.97909950, 131.60246718, 76.67984255],
            [0.00000390,  -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
            [1.00000261,   0.01671123,  -0.00001531, 100.46457166,  102.93768193, 0.0],
            [0.00000562,  -0.00004392, -0.01294668, 35999.37244981,  0.32327364, 0.0],
            [1.52371034,   0.09339410,  1.84969142,  -4.55343205, -23.94362959, 49.55953891],
            [0.00001847,   0.00007882,  -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
            [5.20288700,   0.04838624,  1.30439695,  34.39644051,  14.72847983, 100.47390909],
            [-0.00011607, -0.00013253, -0.00183714, 3034.74612775,  0.21252668, 0.20469106],
            [9.53667594,   0.05386179,  2.48599187,  49.95424423,  92.59887831, 113.66242448],
            [-0.00125060, -0.00050991, 0.00193609,  1222.49362201, -0.41897216, -0.28867794],
            [19.18916464,  0.04725744, 0.77263783, 313.23810451,  170.95427630, 74.01692503],
            [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
            [30.06992276,  0.00859048, 1.77004347, -55.12002969,  44.96476227, 131.78422574],
            [0.00026291,   0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
            ]
            )
        # fmt: on
        J2000_elements = J2000[::2]  # every other row, starting row 0
        cent_rates = J2000[1::2]  # every other row, starting row 1

    # Data below, Curtis [3] p.472, tbl 8.1; not sure Standish et.al. 1992?
    # Earth-moon barycenter ?, ecliptic data!
    # (semi-major axis)|             |             |(RAAN, Omega)| (omega_bar) |
    #            sma   |    ecc      |     incl    | long.node   | long.peri   |  mean.long (L)
    #        au, au/cy | ecc, ecc/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy | deg, deg/cy
    elif eph_data == 1:
        # fmt: off
        J2000_elements = np.array(
            [
                [0.38709893, 0.20563069, 7.00487, 48.33167, 77.4545, 252.25084],
                [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
                [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
                [1.52366231, 0.09341233, 1.845061, 49.57854, 336.04084, 355.45332],
                [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
                [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
                [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
                [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
                [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881],
            ]
        )
        # fmt: on
        # julian century [cy] rates
        # Data below, Curtis [3] p.472, tbl 8.1; not sure Standish et.al. 1992?
        # Units of rates table:
        # "au/cy", "1/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy", "arc-sec/cy"
        # fmt: off
        cent_rates = np.array(
            [
                [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
                [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
                [-0.0000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
                [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
                [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
                [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
                [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
                [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
                [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90],
            ]
        )
        # fmt: on
        # convert arc-sec/cy to deg/cy
        # There must be a better way for this conversion; this gets the job done
        cent_rates[:, 2] /= 3600.0
        cent_rates[:, 3] /= 3600.0
        cent_rates[:, 4] /= 3600.0
        cent_rates[:, 5] /= 3600.0

        # elements & rates conversions
        au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
        deg2rad = math.pi / 180
        if au_units == False:  # then convert units to km
            J2000_elements[:, 0] *= au  # [km] sma (semi-major axis, aka a) convert
            cent_rates[:, 0] *= au
        if rad_units == True:
            J2000_elements[:, 2] *= deg2rad  # [rad]
            cent_rates[:, 2] *= deg2rad
            J2000_elements[:, 3] *= deg2rad  # [rad]
            cent_rates[:, 3] *= deg2rad
            J2000_elements[:, 4] *= deg2rad  # [rad]
            cent_rates[:, 4] *= deg2rad
            J2000_elements[:, 5] *= deg2rad  # [rad]
            cent_rates[:, 5] *= deg2rad

        # np.set_printoptions(precision=4)
        # print(f"J2000_elements=\n{J2000_elements}")
        # print(f"cent_rates=\n{cent_rates}")

    # extract user requested planet coe data & rates;
    #   reminder, coe=classic orbital elements (Kepler)
    # J2000_coe = J2000_elements[planet_id - 1]
    J2000_coe = J2000_elements[planet_id]
    J2000_rates = cent_rates[planet_id]
    return J2000_coe, J2000_rates


def planet_ele_1(planet_id):
    """
    Vallado [4], planet elements coefficients table, heliocentric/equatorial.
        Table of polynomial coefficients in t_TDB.
        t_TDB = julian centuries of tdb (barycentric dynamic time).
        Format: x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    Input Parameters:
    ----------
        planet_id : int
            0 = Mercury, not yet entered
            1 = Venus
            2 = Earth
            3 = Mars
            4 = Jupiter
            5 = Saturn
            6 = Urnaus, not yet entered
            7 = Neptune, not yet entered
            8 = Pluto, not yet entered
    Returns: (for planet_id input):
    -------
        J2000_coefs   : numpy.array, J2000 clasic orbital elements (Kepler).
    Notes:
    ----------
    Data below, Vallado [4], appendix D.4, Planetary Ephemerides, pp.1062.
        Heliocentric, equatorial (not ecliptic), mean equator, mean equinox of IAU-76/FK5.
    """

    if planet_id == 0:  # mercury
        print(f"mercury not yet copied, 2024-09-15")
    elif planet_id == 1:  # venus
        # fmt: off
        J2000_coefs = np.array(
            [
                [ 0.723329820, 0.0, 0.0, 0.0, 0],  # [au] sma
                [ 0.006771880, -0.000047766,  0.0000000975, 0.00000000044, 0],  # [--] ecc
                [ 3.394662000, -0.000856800, -0.000032440,  0.00000001000, 0],  # [deg] incl
                [ 76.67992000, -0.278008000, -0.000142560, -0.0000001980,  0],  # [deg] raan
                [131.5637070,  0.0048646000, -0.001382320, -0.0000053320,  0],  # [deg] w_bar
                [181.9798010,  58517.815676, 0.0000016500,  -0.0000000020, 0],  # [deg] Lm
            ]
        )
        # fmt: on
    elif planet_id == 2:  # earth
        # fmt: off
        J2000_coefs = np.array(
            [
                [1.000001018,          0.0,           0.0,           0.0, 0],  # [au] sma
                [0.016708620, -0.000042037, -0.0000001236, 0.00000000004, 0],  # [--] ecc
                [0.000000000, 0.0130546000, -0.000009310, -0.0000000340,  0],  # [deg] incl
                [174.8731740, -0.241090800,  0.0000406700, -0.0000013270, 0],  # [deg] raan
                [102.9373480, 0.3225550000,  0.0001502600, 0.00000047800, 0],  # [deg] w_bar
                [100.4664490, 35999.372851, -0.000005680, 0.00000000000, 0]    # [deg] Lm
            ]
        )
        # fmt: on
    elif planet_id == 3:  # mars
        # fmt: off
        J2000_coefs = np.array(
            [
                [1.523679342,          0.0,           0.0,            0.0, 0],  # [au] sma
                [0.093400620, 0.0000904830, -0.0000000806, -0.00000000035, 0],  # [--] ecc
                [1.849726000, -0.008147900, -0.0000225500, -0.00000002700, 0],  # [deg] incl
                [49.55809300, -0.294984600, -0.0006399300, -0.00000214300, 0],  # [deg] raan
                [336.0602340, 0.4438898000, -0.0001732100,  0.00000030000, 0],  # [deg] w_bar
                [355.4332750, 19140.2993313, 0.0000026100, -0.00000000300, 0]  # [deg] Lm
            ]
        )
        # fmt: on
    elif planet_id == 4:  # jupiter
        # fmt: off
        J2000_coefs = np.array(
            [
                [5.202603191, 0.0000001913,             0,              0, 0],  # [au] sma
                [0.048494850, 0.0001632440, -0.0000004719, -0.00000000197, 0],  # [--] ecc
                [1.303270000, -0.001987200, 0.00003318000, 0.0000000920, 0],  # [deg] incl
                [100.4644410, 0.1766828000, 0.0009038700, -0.0000070320, 0],  # [deg] raan
                [14.33130900, 0.2155525000, 0.0007225200, -0.0000045900, 0],  # [deg] w_bar
                [34.35148400, 3034.9056746, -0.000085010, 0.00000000400, 0]  # [deg] Lm
            ]
        )
        # fmt: on
    elif planet_id == 5:  # saturn
        # fmt: off
        J2000_coefs = np.array(
            [
                [9.554909596, -0.000002138,             0,              0],  # [au] sma
                [0.055508620, -0.000346818, -0.0000006456, 0.00000000338, 0],  # [--] ecc
                [2.488878000, 0.0025510000, -0.000049030, 0.00000001800, 0],  # [deg] incl
                [113.6655240, -0.256664900, -0.000183450, 0.00000035700, 0],  # [deg] raan
                [93.05678700, 0.5665496000, 0.0005280900, 0.00000488200, 0],  # [deg] w_bar
                [50.07747100, 1222.1137943, 0.0002100400, -0.0000000190, 0]  # [deg] Lm
            ]
        )
        # fmt: on
    elif planet_id == 6:  # uranus
        # fmt: on
        print(f"uranus not yet copied, 2024-09-15")
    elif planet_id == 7:  # neptune
        print(f"neptune not yet copied, 2024-09-15")
    elif planet_id == 8:  # pluto
        print(f"pluto not yet copied, 2024-09-15")
    else:
        print(f"Not a valid planet id, {planet_id}")
        raise NameError("Not a valid planet id.")  # figure out print format

    return J2000_coefs


def planet_rv(planet_id, date_):
    """
    Find planet position, r_vec and v_vec; given planet_id, and date.
        Outputs in both equatorial and ecliptic frames.
    From Vallado [4], example 5-5, pp.304; called algorithm 33, pp.303.

    Input Parameters:
    ----------
        planet_id : int, 0=mercury, 1=venus. 2=earth, 3=mars, 4=jupiter
                            5=saturn, 6=uranus, 7=neptune, 8=pluto
        date_     : python date object
    Returns:
    ----------
        r_vec     : np.array row [au] position; equatorial frame
        v_vec     : np.array row [au/day] velocity; equatorial frame
        r1_vec    : np.array row [au] position; ecliptic frame
        v1_vec    : np.array row [au/day] velocity; ecliptic frame
    Notes:
    ----------
    Planet elements coefficients table, heliocentric/equatorial,
        comes from planet_ele_1() function. Consists of polynomial coefficients
        table, in powers of t_TDB.  t_TDB = julian centuries of tdb
        (barycentric dynamic time).
        Format: x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    """
    # fmt: on
    np.set_printoptions(precision=6)  # numpy, set vector printing size
    deg2rad = math.pi / 180  # used multiple times
    rad2deg = 180 / math.pi  # used multiple times

    au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    mu_sun_km = 1.32712428e11  # [km^3/s^2], Vallado [2] p.1043, tbl.D-5
    mu_sun_au = mu_sun_km / (au**3)  # [au^3/s^2], unit conversion
    # print(f"mu_sun_au= {mu_sun_au}")

    year, month, day, hour, minute, second = (
        date_.year,
        date_.month,
        date_.day,
        date_.hour,
        date_.minute,
        date_.second,
    )
    # print(f"Parameters Date: {year}-{month}-{day} {hour}:{minute}:{second}")

    # jd_convTime(), calculates julian date, acd calculates other
    #   time conversions; i.e. c_type=0, julian centuries from J2000.0 TT.
    jd, jd_cJ2000 = astro_time.jd_convTime(
        year, month, day, hour, minute, second, c_type=0
    )
    # print(f"jd= {jd:.10g}, jd_cent={jd_cJ2000:.10g}")

    # 2024-09-21, not all planets are in coefficients table so far
    J2000_coefs = planet_ele_1(planet_id)
    # coeffs format; x0*t_TDB^0 + x1*t_TDB^1 + x2*t_TDB^2 + ...
    #   time, t_tdb = julian centuries of barycentric dynamical time
    x1 = np.arange(5)  # number of exponent values; power series
    x2 = np.full(5, jd_cJ2000)  # base time value
    x3 = x2**x1  # time multiplier series
    sma = np.sum(J2000_coefs[0, :] * x3)  # [au]

    # make sure angles, modulo +- 2*pi; phi%(2*math.pi) # %=modulo
    ecc = np.sum(J2000_coefs[1, :] * x3)  # [--]
    incl_deg = np.sum(J2000_coefs[2, :] * x3)  # [deg]
    raan_deg = np.sum(J2000_coefs[3, :] * x3)  # [deg]
    w_bar_deg = np.sum(J2000_coefs[4, :] * x3)  # [deg]
    L_bar_deg = np.sum(J2000_coefs[5, :] * x3)  # [deg]

    incl_rad = incl_deg * deg2rad
    raan_rad = raan_deg * deg2rad
    w_bar_rad = w_bar_deg * deg2rad
    L_bar_rad = L_bar_deg * deg2rad

    # print(f"sma= {sma:.8g} [au]")
    # print(f"ecc= {ecc:.8g}")
    # print(f"incl= {incl_deg:.8g} [deg]")
    # print(f"raan= {raan_deg:.8g} [deg]")
    # print(f"w_bar= {w_bar_deg:.8g} [deg]")  # longitude of periapsis
    # print(f"L_bar= {L_bar_deg:.8g} [deg]")

    M_deg = L_bar_deg - w_bar_deg  # [deg] mean angle/anomaly
    M_rad = M_deg * deg2rad
    w_deg = w_bar_deg - raan_deg  # [deg] argument of periapsis (aka aop, or arg_p)
    w_rad = w_deg * deg2rad
    # print(f"\nM_deg= {M_deg:.8g} [deg], {M_rad:.8g} [rad]")
    # print(f"w_deg= {w_deg:.8g} [deg]")

    E_rad = kep_eqtnE(M=M_rad, e=ecc)
    E_deg = E_rad * rad2deg
    # TA_rad below, no quadrent ambiguity; addresses near pi values
    TA_rad = eccentric_to_true(E=E_rad, e=ecc)
    # below, commented out, is Curtis [3], p.160, eqn 3.13b.
    # TA_rad = 2 * math.atan(math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E_rad / 2))
    TA_deg = TA_rad * rad2deg

    # print(f"E_deg= {E_deg:.8g} [deg]")
    # print(f"TA_deg= {TA_deg:.8g} [deg]")

    # s=semi-parameter (aka p), sma=semi-major axis (aka a)
    sp = sma * (1 - ecc**2)
    # function inputs, coe2rv(p, ecc, inc, raan, aop, anom, mu=Earth.mu)
    r_vec, v_vec = coe2rv(
        p=sp,
        ecc=ecc,
        inc=incl_rad,
        raan=raan_rad,
        aop=w_rad,
        anom=TA_rad,
        mu=mu_sun_au,
    )
    r_vec = np.ravel(r_vec)  # [au] convert column array to row vector
    v_vec = np.ravel(v_vec)  # [au/s]
    v_vec *= 86400  # [au/day] convert seconds to days
    # print(f"\nEquatorial/Heliocentric, XYZ")
    # print(f"r_vec= {r_vec} [au]")
    # print(f"v_vec= {v_vec} [au/day]")  # convert, seconds to days

    # rotate r_vec and v_vec from equatorial to ecliptic/heliocentric
    #   ecliptic angle, Vallado [4] p.217, eqn.3-69.
    e_angle = ecliptic_angle(jd_cJ2000)
    # print(f"ecliptic angle, e_angle= {e_angle:.8g} [deg]")

    r1_vec = r_vec @ rot_matrix(angle=-e_angle * deg2rad, axis=0)  # [au]
    v1_vec = v_vec @ rot_matrix(angle=-e_angle * deg2rad, axis=0)  # [au/s]
    v1_vec *= 86400  # [au/day] convert seconds to days
    # print(f"\nEcliptic/Heliocentric, XYZ")
    # print(f"r1_vec= {r1_vec} [au]")
    # print(f"v1_vec= {v1_vec} [au/day]")

    # equatorial = r_vec, v_vec; ecliptic = r_vec, v_vec
    return r_vec, v_vec, r1_vec, v1_vec


def rot_matrix(angle, axis: int):
    """
    Returns rotation matrix based on user axis choice.
        Function from github, lamberthub, utilities->elements.py

    Input Parameters:
    ----------
        angle      : [rad]
        axis       : axis=0 rotate x-axis;
                    axis=1 rotate y-axis
                    axis=2 rotate z-axis
    Returns:
    -------
        np.array   : rotation matrix, 3x3

    Raises:
    ------
        ValueError : if invalid axis
    Notes:
    -----------

    """
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid axis: axis=0, x; axis=1, y; axis=2, z")


def ecliptic_angle(jd_cJ2000):
    """
    Calculate mean angle/obliquity of the ecliptic from polynomial.
    Vallado [4], section 3.7, "classical equinox based transformation with
        IAU-2006/2000A", p.217, equation 3-69. 2nd eqn.

    Input Parameters:
    ----------
        time : float [sec] julian centuries from J2000.00 (TT)
    Returns:
    -------
        e_angle : float [deg]
    Notes:
        Vallado [4] notes the data comes from Kaplan, 2005:44.
        coefs format; coef0*jd_cJ2000^0 + coef1*jd_cJ2000^1 + coef2*jd_cJ2000^2 + ...
    """
    ecliptic_coefs = np.array(
        [23.439279, -0.0130102, -5.086e-8, 5.565e-7, -1.6e-10, -1.21e-11]
    )
    # coeffs format; coef0*jd_cJ2000^0 + coef1*jd_cJ2000^1 + coef2*jd_cJ2000^2 + ...
    #   jd_cJ2000 = julian centuries since J2000.0
    x1 = np.arange(6)  # exponent values; power series
    x2 = np.full(6, jd_cJ2000)  # base time value
    x3 = x2**x1  # time multiplier series
    # print(f"{ecliptic_coefs[:] * x3}") # troubleshooting
    e_angle = np.sum(ecliptic_coefs[:] * x3)  # [deg]
    return e_angle


def sunPosition(yr, mo, day, hr=0, min=0, sec=0.0):
    """
    Find geocentric position of sun vector.
        Vallado [4] algorithm 29, pp.285. Associated example 5-1, pp.286.
    Input Parameters:
    ----------
        yr   : int, year
        mo   : int, month
        day  : int, day
    Returns:
    ----------
        sun_vec : np.array [au]
    Notes:
    ----------
        This is a low precision solution; see Vallado [4] section 5.1.1, p283.
        L_bar_ecl : ecliptic mean longitude in TOD frame, Vallado [4] p.283, eqn5-1
        L_ecl     : ecliptic longitude
        M_sun     : mean anomaly of sun
    """
    deg2rad = math.pi / 180
    rad2deg = 180 / math.pi
    # au = 149597870.7  # [km/au] Vallado [2] p.1043, tbl.D-5
    # two choices for julian date; g_date2jd() and jd_convTime().
    #   g_date2jd() calculates only the julian date, while
    #   jd_convTime() calculates both julian date, julian centuries since J2000.0.
    #   if c_type=0, find julian centuries from J2000.0 TT.

    jd, jd_cJ2000 = astro_time.jd_convTime(yr=yr, mo=mo, d=day, c_type=0)
    print(f"jd= {jd:.10g}, jd_cent={jd_cJ2000:.10g}")  # troubleshooting
    # reminder, angle values are modulo 360.0 for degrees
    L_bar_ecl_deg = math.fmod(280.460 + 36000.771285 * jd_cJ2000, 360.0)  # ecliptic
    M_sun_deg = math.fmod(357.528 + 35999.050957 * jd_cJ2000, 360.0)
    M_sun_rad = M_sun_deg * deg2rad
    L_ecl_deg = math.fmod(
        L_bar_ecl_deg + 1.915 * math.sin(M_sun_rad) + 0.020 * math.sin(2 * M_sun_rad),
        360.0,
    )
    L_ecl_rad = L_ecl_deg * deg2rad
    phi_ecl = 0.0
    # obliquity of the ecliptic; meaning angle of ecliptic to equatorial
    ob_ecl_deg = 23.439291 - 0.01461 * jd_cJ2000
    ob_ecl_rad = ob_ecl_deg * deg2rad
    print(f"L_bar_ecl= {L_bar_ecl_deg} [deg]")
    print(f"M_sun= {M_sun_deg} [deg]")
    print(f"L_ecl= {L_ecl_deg} [deg]")
    print(f"ob_ecl= {ob_ecl_deg} [deg]")

    sun_mag = (
        1.00014 - 0.01671 * math.cos(M_sun_rad) - 0.00014 * math.cos(2 * M_sun_rad)
    )
    print(f"sun_mag= {sun_mag} [au]")
    # sun_vec, below,  is column array
    sun_vec = np.array(
        [
            [sun_mag * math.cos(L_ecl_rad)],
            [sun_mag * math.cos(ob_ecl_rad) * math.sin(L_ecl_rad)],
            [sun_mag * math.sin(ob_ecl_rad) * math.sin(L_ecl_rad)],
        ]
    )
    # print(f"sun_vec= {sun_vec}")

    # find declination and right ascension; remember, manage quadrants with atan2()
    # assumes phi_ecl = 0 [deg]
    decl_rad = math.asin(math.sin(ob_ecl_rad) * math.sin(L_ecl_rad))
    sin_ra = math.cos(ob_ecl_rad) * math.sin(L_ecl_rad) / math.cos(decl_rad)
    cos_ra = math.cos(L_ecl_rad) / math.cos(decl_rad)
    ra_rad = math.atan2(sin_ra, cos_ra)
    print(f"decl_deg, {decl_rad*rad2deg} [deg]")
    print(f"ra_deg, {ra_rad*rad2deg} [deg]")

    # if u want to flatten the array (make a row vector)
    sun_vec = np.ravel(sun_vec)

    return sun_vec  # sunPosition()


def sun_r_s(jd_, phi_gc_rad):
    """
    Interal calculations supporting sunRiseSet()
    Frees-up what would be redundant calculations related to the sun position.

    Input Parameters:
    ----------
        jd_        : float [jd] julian date
        phi_gc_rad : float [rad] geocentric latitude, positive north

    Returns:
    ----------
        decl_rad : float [rad] declination
        ra_rad   : float [rad] right ascension
    """
    deg2rad = math.pi / 180
    rad2deg = 180 / math.pi

    jd_cent = (jd_ - 2451545) / 36525  # centuries since J2000
    print(f"jd_= {jd_}, jd_r_cent= {jd_cent}")

    # reminder, angle values are modulo 360.0 for degrees; be careful of minus signs
    # L_bar_ecl_deg = math.fmod((280.4606184 + 36000.77005361*jd_cJ2000), 360.0) # ecliptic
    L_bar_ecl_deg = (280.4606184 + 36000.77005361 * jd_cent) % 360.0
    M_sun_deg = (357.5291092 + 35999.05034 * jd_cent) % 360.0
    M_sun_rad = M_sun_deg * deg2rad
    L_ecl_deg = (
        L_bar_ecl_deg
        + 1.914666471 * math.sin(M_sun_rad)
        + 0.019994643 * math.sin(2 * M_sun_rad)
    ) % 360.0
    L_ecl_rad = L_ecl_deg * deg2rad
    phi_ecl = 0.0
    # obliquity of the ecliptic; meaning angle of ecliptic to equatorial
    ob_ecl_deg = 23.439291 - 0.0130042 * jd_cent
    ob_ecl_rad = ob_ecl_deg * deg2rad
    print(f"L_bar_ecl= {L_bar_ecl_deg} [deg]")
    print(f"M_sun= {M_sun_deg} [deg]")
    print(f"L_ecl= {L_ecl_deg} [deg]")
    print(f"ob_ecl= {ob_ecl_deg} [deg]")

    # find declination and right ascension; remember, manage quadrants with atan2()
    # assumes phi_ecl = 0 [deg]
    decl_rad = math.asin(math.sin(ob_ecl_rad) * math.sin(L_ecl_rad))
    sin_ra = math.cos(ob_ecl_rad) * math.sin(L_ecl_rad) / math.cos(decl_rad)
    cos_ra = math.cos(L_ecl_rad) / math.cos(decl_rad)
    ra_rad = math.atan2(sin_ra, cos_ra)
    print(f"decl_deg, {decl_rad*rad2deg} [deg]")
    print(f"ra_deg, {ra_rad*rad2deg} [deg]")

    # zeta=angle between sun and site; sunrise & sunset, plus any refractions...
    zeta_rad = (90 + 50 / 60) * deg2rad  # given in Vallado [4] ex-2, p.290
    LHA_rad = math.acos(
        (math.cos(zeta_rad) - math.sin(decl_rad) * math.sin(phi_gc_rad))
        / (math.cos(decl_rad) * math.cos(phi_gc_rad))
    )
    print(f"LHA_rad, {LHA_rad:.8g} [rad], {LHA_rad*rad2deg:.8g} [deg]")

    # gmst_rise_deg = astro_time.find_gmst(jd_ut1=2448855.009722) # test case ok
    # gmst_rise_deg from Vallado [4] p.189, eqn3-46; modulo 360
    gmst_deg = (
        100.4606184
        + (36000.77005361) * jd_cent
        + 0.00038793 * jd_cent * jd_cent
        - 6.2e-8 * jd_cent * jd_cent * jd_cent
    ) % 360
    print(f"gmst_deg= {gmst_deg} [deg]")

    return decl_rad, ra_rad, gmst_deg, LHA_rad


def sunRiseSet(yr, mo, day, lat, lon):
    """
    Find sun-rise and sun-set.
    Vallado [4] algorithm 30, pp.289. Associated example 5-2, pp.290.

    Input Parameters:
    ----------
        yr  : int, year
        mo  : int, month
        day : int, day
        lat : site lattitude [deg] called phi_gc in function
        lon : site longitude [deg], I believe; Vallado [4] is not clear
    Returns:
    ----------
        sun_vec : np.array [au]
    Notes:
    ----------
        This is a low precision solution; see Vallado [4] section 5.1.1, p283.
        L_bar_ecl : ecliptic mean longitude in TOD frame, Vallado [4] p.283, eqn5-1
        L_ecl     : ecliptic longitude
        M_sun     : mean anomaly of sun
    """
    deg2rad = math.pi / 180
    rad2deg = 180 / math.pi
    # !! normally extract lattitude from site input that has lat & lon
    phi_gc_deg = lat
    phi_gc_rad = phi_gc_deg * deg2rad

    # two choices for julian date; g_date2jd() and jd_convTime().
    #   g_date2jd() converts gregorian date to julian date only, while
    #   jd_convTime() converts gregorian date to julian date, and
    #       julian centuries since J2000.0.
    #       if c_type=0, find julian centuries from J2000.0 TT.
    jd = astro_time.g_date2jd(yr=yr, mo=mo, d=day)
    # print(f"jd= {jd:.10g}") # troubleshooting print
    jd_rise = jd + 6 / 24 - lon / 360.0

    print(f"*** start sunrise ***")
    decl_rise_rad, ra_rise_rad, gmst_rise_deg, LHA_rad = sun_r_s(
        jd_=jd_rise, phi_gc_rad=phi_gc_rad
    )
    # return values: from sun_r_s(), decl_rad, ra_rad, gmst_deg, LHA_rad
    LHA_rise_rad = 2 * math.pi - LHA_rad
    UT_sunRise = (LHA_rise_rad * rad2deg + ra_rise_rad * rad2deg - gmst_rise_deg) % 360

    print(f"\n*** start sunset ****")
    jd_set = jd + 18 / 24 - lon / 360.0  # julian date for sunset
    print(f"jd_set= {jd_set}")

    # return values: from sun_r_s(), decl_rad, ra_rad, gmst_deg, LHA_rad
    decl_set_rad, ra_set_rad, gmst_set_deg, LHA_set_rad = sun_r_s(
        jd_=jd_set, phi_gc_rad=phi_gc_rad
    )
    UT_sunSet = (LHA_set_rad * rad2deg + ra_set_rad * rad2deg - gmst_set_deg) % 360

    return UT_sunRise, UT_sunSet  # sunRiseSet()
