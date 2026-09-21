"""Solar geometry and a clear-sky irradiance model.

Deliberately dependency-light: solar position and the Haurwitz clear-sky model are
short enough to write out, and doing so keeps the demonstrator runnable with nothing
but numpy. For production work against a real site I would swap this for pvlib,
which handles the edge cases this does not.
"""

from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0  # W/m2


def day_angle(day_of_year: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * (day_of_year - 1) / 365.0


def declination(day_of_year: np.ndarray) -> np.ndarray:
    """Solar declination in radians (Spencer's series)."""
    b = day_angle(day_of_year)
    return (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.001480 * np.sin(3 * b)
    )


def equation_of_time(day_of_year: np.ndarray) -> np.ndarray:
    """Equation of time in minutes."""
    b = day_angle(day_of_year)
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.040849 * np.sin(2 * b)
    )


def hour_angle(
    clock_hour: np.ndarray,
    day_of_year: np.ndarray,
    longitude: float,
    timezone_offset_h: float,
) -> np.ndarray:
    """Solar hour angle in radians; zero at solar noon, negative in the morning."""
    standard_meridian = 15.0 * timezone_offset_h
    solar_time = (
        clock_hour
        + (4.0 * (longitude - standard_meridian) + equation_of_time(day_of_year)) / 60.0
    )
    return np.deg2rad(15.0 * (solar_time - 12.0))


def cos_zenith(
    latitude: float,
    day_of_year: np.ndarray,
    clock_hour: np.ndarray,
    longitude: float,
    timezone_offset_h: float,
) -> np.ndarray:
    lat = np.deg2rad(latitude)
    dec = declination(day_of_year)
    omega = hour_angle(clock_hour, day_of_year, longitude, timezone_offset_h)
    cz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(omega)
    return np.clip(cz, 0.0, 1.0)


def haurwitz_ghi(cos_z: np.ndarray, altitude_m: float = 0.0) -> np.ndarray:
    """Clear-sky global horizontal irradiance, W/m2.

    Haurwitz (1945) with a mild altitude correction. Crude, but it gets the shape
    and magnitude of a desert clear-sky day right, which is all the detector needs
    in order to be exercised.
    """
    cz = np.clip(cos_z, 1e-6, 1.0)
    ghi = 1098.0 * cz * np.exp(-0.059 / cz)
    ghi *= 1.0 + 4.0e-5 * altitude_m
    return np.where(cos_z <= 0.0, 0.0, ghi)


def extraterrestrial(day_of_year: np.ndarray) -> np.ndarray:
    b = day_angle(day_of_year)
    return SOLAR_CONSTANT * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b))


def plane_of_array(
    ghi: np.ndarray,
    cos_z: np.ndarray,
    day_of_year: np.ndarray,
    latitude: float,
    clock_hour: np.ndarray,
    longitude: float,
    timezone_offset_h: float,
    tilt_deg: float,
    azimuth_deg: float,
    albedo: float = 0.2,
) -> np.ndarray:
    """Irradiance on a fixed tilted plane, W/m2 (isotropic sky, Erbs split)."""
    zenith = np.arccos(np.clip(cos_z, 0.0, 1.0))
    e0 = extraterrestrial(day_of_year)
    kt = np.divide(ghi, np.maximum(e0 * cos_z, 1e-6))
    kt = np.clip(kt, 0.0, 1.0)

    # Erbs correlation for the diffuse fraction.
    df = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511
            - 0.1604 * kt
            + 4.388 * kt**2
            - 16.638 * kt**3
            + 12.336 * kt**4,
            0.165,
        ),
    )
    dhi = ghi * df
    dni = np.divide(ghi - dhi, np.maximum(cos_z, 1e-6))
    dni = np.clip(dni, 0.0, 1100.0)

    beta = np.deg2rad(tilt_deg)
    gamma = np.deg2rad(azimuth_deg)
    lat = np.deg2rad(latitude)
    dec = declination(day_of_year)
    omega = hour_angle(clock_hour, day_of_year, longitude, timezone_offset_h)

    # Angle of incidence on the tilted plane.
    cos_theta = (
        np.sin(dec) * np.sin(lat) * np.cos(beta)
        - np.sin(dec) * np.cos(lat) * np.sin(beta) * np.cos(gamma)
        + np.cos(dec) * np.cos(lat) * np.cos(beta) * np.cos(omega)
        + np.cos(dec) * np.sin(lat) * np.sin(beta) * np.cos(gamma) * np.cos(omega)
        + np.cos(dec) * np.sin(beta) * np.sin(gamma) * np.sin(omega)
    )
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    beam = dni * cos_theta
    sky = dhi * (1.0 + np.cos(beta)) / 2.0
    ground = ghi * albedo * (1.0 - np.cos(beta)) / 2.0
    poa = beam + sky + ground
    return np.where(cos_z <= 0.0, 0.0, poa)


def cell_temperature(
    poa: np.ndarray, ambient_c: np.ndarray, noct_c: float = 45.0
) -> np.ndarray:
    """NOCT cell temperature model."""
    return ambient_c + (noct_c - 20.0) / 800.0 * poa
