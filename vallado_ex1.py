# Vallado_examples
# Cannot get black formatter to work within VSCode, in terminal type; black vallado_func.py
import numpy as np

import vallado_func as vfunc  # Vallado functions


def hohmann_ex6_1():
    """Vallado, Hohmann Transfer, example 6-1.
    Hohmann uses one central body for the transfer ellipse.
    Interplanetary missions use the patched conic; 3 orbit types:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth = 6378.137  # [km]
    mu_earth = 3.986012e5  # [km^3 / s^2] gravatational constant, earth
    mu_sun = 1.327e11  # [km^3 / s^2] gravatational constant, sun

    # define inputs for hohmann transfer
    r1 = r_earth + 191.34411  # [km]
    r2 = r_earth + 35781.34857  # [km]
    mu_trans = mu_earth  # [km^3 / s^2] gravatational constant

    vfunc.val_hohmann(r1, r2, mu_trans)  # vallado hohmann transfer


def bielliptic_ex6_2():
    """Vallado, Bi-Elliptic Transfer, example 6-2.
    Hohmann uses one central body for the transfer ellipse.
    Interplanetary missions use the patched conic; 3 orbit types:
    1) initial body - departure
    2) transfer body - transfer
    3) final body - arrival

    Interplanetary missions with patched conic in chapter 12.
    """
    # define constants
    r_earth = 6378.137  # [km]
    mu_earth = 3.986012e5  # [km^3 / s^2] gravatational constant, earth
    mu_sun = 1.327e11  # [km^3 / s^2] gravatational constant, sun

    # define inputs; bi-elliptic transfer
    r1 = r_earth + 191.34411  # [km]
    rb = r_earth + 503873  # [km] point b, beyond r2
    r2 = r_earth + 376310  # [km]
    mu_trans = mu_earth  # [km^3 / s^2] gravatational constant

    vfunc.val_bielliptic(r1, rb, r2, mu_trans)  # vallado bi-elliptic transfer


def main() -> None:
    pass  # placeholder


# Main code. Functions and class methods are called from main.
if __name__ == "__main__":
    # hohmann_ex6_1()  # hohmann transfer, vallado example 6-1
    bielliptic_ex6_2()  # bi-elliptic transfer, vallado example 6-2
    main()  # placeholder function
