# Vallado functions
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
    return (r1+r2)/2
def calc_v1(mu, r1): # circular orbit velocity
    return np.sqrt(mu/r1)
def calc_v2(mu, r1, a): # elliptical orbit velocity
    return np.sqrt(mu*((2/r1)-(1/a)))

def calc_tof(mu, a):  # time of flight for complete orbit
    return (2*np.pi*np.sqrt(a**3/mu))

def val_hohmann(r_init, r_final,mu_init, mu_final):
    """Vallado Hohmann Transfer, algorithm 36

    Parameters
    ----------
    r_init : _type_
        _description_
    r_final : _type_
        _description_
    mu_init : _type_
        _description_
    mu_final : _type_
        _description_
    Returns
    -------
    
    """
    r1 = r_init
    mu_1 = mu_init
    r2 = r_final
    mu_2 = mu_final
    
    v_init = calc_v1(mu_1, r1)  # circular orbit velocity
    print(f"v1 initial velocity: {v_init} [km/s]")
    v_final = calc_v1(mu_2, r2)
    print(f"v2 final velocity: {v_final} [km/s]")
    a_trans = cal_a(r1,r2) # transfer semi-major axis
    print(f"transfer semimajor axis (a): {a_trans} [km]")

    # transfer ellipse relations
    v_trans_a = calc_v2(mu_1, r1, a_trans) # elliptical orbit velocity
    print(f"velocity, transfer periapsis: {v_trans_a} [km/s]")
    v_trans_b = calc_v2(mu_2, r2, a_trans) # elliptical orbit velocity
    print(f"velocity, transfer apoapsis: {v_trans_b} [km/s]")
    
    # delta-velocity relations
    dv_total = abs(v_trans_a - v_init) + abs(v_final - v_trans_b)
    print(f"total delta-velocity: {dv_total} [km/s]")

    # time of flight (tof) for hohmann transfer
    tof_hoh = calc_tof(mu_2, a_trans)
    print(f"\nHohmann transfer time: {(tof_hoh/2):0.8g} [s]")
    print(f"Hohmann transfer time: {tof_hoh/(2*60):0.8g} [m]")
    