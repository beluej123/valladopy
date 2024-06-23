# Vallado_examples
import numpy as np

import vallado_func as vfunc  # Vallado functions


def hohmann_ex6_1():
    """Vallado, Hohmann Transfer, example 6-1
    This example assumes the same central body for all 3 orbits:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival
    
    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth=6378.137  #[km]
    mu_earth=3.986012e5  #[km^3 / s^2] gravatational constant, earth
    mu_sun=1.327e11  #[km^3 / s^2] gravatational constant, sun

    # define inputs
    r1 = r_earth+191.34411 #[km]
    mu_1 = mu_earth # [km^3 / s^2] gravatational constant
    r2 = r_earth+35781.34857 #[km]
    mu_2 = mu_earth # [km^3 / s^2] gravatational constant

    v_init = vfunc.calc_v1(mu_1, r1)  # circular orbit velocity
    print(f"v1 initial velocity: {v_init} [km/s]")
    v_final = vfunc.calc_v1(mu_2, r2)
    print(f"v2 final velocity: {v_final} [km/s]")
    a_trans = vfunc.cal_a(r1,r2) # transfer semi-major axis
    print(f"transfer semimajor axis (a): {a_trans} [km]")

    # transfer ellipse relations
    v_trans_a = vfunc.calc_v2(mu_1, r1, a_trans) # elliptical orbit velocity
    print(f"velocity, transfer periapsis: {v_trans_a} [km/s]")
    v_trans_b = vfunc.calc_v2(mu_1, r2, a_trans) # elliptical orbit velocity
    print(f"velocity, transfer apoapsis: {v_trans_b} [km/s]")
    
    # delta-velocity relations
    dv_total = abs(v_trans_a - v_init) + abs(v_final - v_trans_b)
    print(f"total delta-velocity: {dv_total} [km/s]")

    # time of flight (tof) for hohmann transfer
    tof_hoh = vfunc.calc_tof(mu_earth, a_trans)
    print(f"\nHohmann transfer time: {(tof_hoh/2):0.8g} [s]")
    print(f"Hohmann transfer time: {tof_hoh/(2*60):0.8g} [m]")

def main():
    print("main function, placeholder")

# Main code. Functions and class methods are called from main.
if __name__ == "__main__":
    hohmann_ex6_1() # hohmann transfer, vallado example 6-1
    main() # placeholder function