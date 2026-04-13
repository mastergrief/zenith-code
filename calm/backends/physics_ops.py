"""
CALM Physics backend — kinematics, electricity, energy, waves.

Models approximate formulas, confuse units, botch multi-step physics.
Pure deterministic computation.
"""

from __future__ import annotations

import math


# --- Kinematics ---

def velocity(distance: float, time: float) -> float:
    """Average velocity: v = d/t."""
    t = float(time)
    if t == 0:
        return 0.0
    return round(float(distance) / t, 4)


def acceleration(v_final: float, v_initial: float, time: float) -> float:
    """Acceleration: a = (vf - vi) / t."""
    t = float(time)
    if t == 0:
        return 0.0
    return round((float(v_final) - float(v_initial)) / t, 4)


def displacement(v_initial: float, time: float, acceleration: float) -> float:
    """Displacement: s = vi*t + 0.5*a*t^2."""
    vi, t, a = float(v_initial), float(time), float(acceleration)
    return round(vi * t + 0.5 * a * t ** 2, 4)


def final_velocity(v_initial: float, acceleration: float, time: float) -> float:
    """Final velocity: vf = vi + a*t."""
    return round(float(v_initial) + float(acceleration) * float(time), 4)


def free_fall_time(height: float, g: float = 9.81) -> float:
    """Time for free fall from height: t = sqrt(2h/g)."""
    return round(math.sqrt(2 * float(height) / float(g)), 4)


def projectile_range(v: float, angle_deg: float, g: float = 9.81) -> float:
    """Projectile range: R = v^2 * sin(2θ) / g."""
    theta = math.radians(float(angle_deg))
    return round(float(v) ** 2 * math.sin(2 * theta) / float(g), 4)


def projectile_max_height(v: float, angle_deg: float, g: float = 9.81) -> float:
    """Projectile max height: H = v^2 * sin^2(θ) / (2g)."""
    theta = math.radians(float(angle_deg))
    return round(float(v) ** 2 * math.sin(theta) ** 2 / (2 * float(g)), 4)


# --- Forces ---

def force(mass: float, acceleration: float) -> float:
    """Newton's second law: F = m*a."""
    return round(float(mass) * float(acceleration), 4)


def weight(mass: float, g: float = 9.81) -> float:
    """Weight: W = m*g."""
    return round(float(mass) * float(g), 4)


def gravitational_force(m1: float, m2: float, r: float) -> float:
    """Newton's gravitational force: F = G*m1*m2/r^2."""
    G = 6.674e-11
    return G * float(m1) * float(m2) / float(r) ** 2


def momentum(mass: float, velocity: float) -> float:
    """Linear momentum: p = m*v."""
    return round(float(mass) * float(velocity), 4)


# --- Energy ---

def kinetic_energy(mass: float, velocity: float) -> float:
    """Kinetic energy: KE = 0.5*m*v^2."""
    return round(0.5 * float(mass) * float(velocity) ** 2, 4)


def potential_energy(mass: float, height: float, g: float = 9.81) -> float:
    """Gravitational potential energy: PE = m*g*h."""
    return round(float(mass) * float(g) * float(height), 4)


def work(force: float, distance: float, angle_deg: float = 0) -> float:
    """Work done: W = F*d*cos(θ)."""
    return round(float(force) * float(distance) * math.cos(math.radians(float(angle_deg))), 4)


def power(work: float, time: float) -> float:
    """Power: P = W/t."""
    t = float(time)
    if t == 0:
        return 0.0
    return round(float(work) / t, 4)


# --- Electricity ---

def ohms_law(voltage: float = None, current: float = None, resistance: float = None) -> dict:
    """Ohm's law: V = I*R. Provide any 2, get the third."""
    v = float(voltage) if voltage is not None else None
    i = float(current) if current is not None else None
    r = float(resistance) if resistance is not None else None

    if v is not None and i is not None:
        return {"voltage": v, "current": i, "resistance": round(v / i, 4) if i != 0 else None}
    elif v is not None and r is not None:
        return {"voltage": v, "current": round(v / r, 4) if r != 0 else None, "resistance": r}
    elif i is not None and r is not None:
        return {"voltage": round(i * r, 4), "current": i, "resistance": r}
    return {"error": "Provide exactly 2 of: voltage, current, resistance"}


def electrical_power(voltage: float, current: float) -> float:
    """Electrical power: P = V*I."""
    return round(float(voltage) * float(current), 4)


def resistance_series(*resistors) -> float:
    """Total resistance in series: R = R1 + R2 + ..."""
    return round(sum(float(r) for r in resistors), 4)


def resistance_parallel(*resistors) -> float:
    """Total resistance in parallel: 1/R = 1/R1 + 1/R2 + ..."""
    total = sum(1.0 / float(r) for r in resistors if float(r) != 0)
    return round(1.0 / total, 4) if total != 0 else 0.0


# --- Waves ---

def wave_speed(frequency: float, wavelength: float) -> float:
    """Wave speed: v = f*λ."""
    return round(float(frequency) * float(wavelength), 4)


def frequency_from_period(period: float) -> float:
    """Frequency from period: f = 1/T."""
    p = float(period)
    if p == 0:
        return 0.0
    return round(1.0 / p, 6)


def doppler_frequency(f_source: float, v_sound: float, v_observer: float, v_source: float = 0) -> float:
    """Doppler effect: f_obs = f_source * (v_sound + v_observer) / (v_sound + v_source).
    +v_observer = moving toward source, +v_source = moving away from observer."""
    vs = float(v_sound)
    denom = vs + float(v_source)
    if denom == 0:
        return 0.0
    return round(float(f_source) * (vs + float(v_observer)) / denom, 4)


PHYSICS_FUNCTIONS = {
    "velocity": velocity,
    "acceleration": acceleration,
    "displacement": displacement,
    "final_velocity": final_velocity,
    "free_fall_time": free_fall_time,
    "projectile_range": projectile_range,
    "projectile_max_height": projectile_max_height,
    "force": force,
    "weight": weight,
    "gravitational_force": gravitational_force,
    "momentum": momentum,
    "kinetic_energy": kinetic_energy,
    "potential_energy": potential_energy,
    "work": work,
    "power": power,
    "ohms_law": ohms_law,
    "electrical_power": electrical_power,
    "resistance_series": resistance_series,
    "resistance_parallel": resistance_parallel,
    "wave_speed": wave_speed,
    "frequency_from_period": frequency_from_period,
    "doppler_frequency": doppler_frequency,
}

PHYSICS_NL_PATTERNS = [
    (r'kinetic energy.*?mass\s+(?:of\s+)?([\d.]+)\s*(?:kg)?.*?velocity\s+(?:of\s+)?([\d.]+)', 'kinetic_energy({0}, {1})'),
    (r'potential energy.*?mass\s+(?:of\s+)?([\d.]+)\s*(?:kg)?.*?height\s+(?:of\s+)?([\d.]+)', 'potential_energy({0}, {1})'),
    (r'(?:force|F)\s*=.*?mass\s+(?:of\s+)?([\d.]+).*?acceleration\s+(?:of\s+)?([\d.]+)', 'force({0}, {1})'),
    (r'free fall.*?height\s+(?:of\s+)?([\d.]+)', 'free_fall_time({0})'),
    (r'projectile range.*?velocity\s+(?:of\s+)?([\d.]+).*?angle\s+(?:of\s+)?([\d.]+)', 'projectile_range({0}, {1})'),
    (r'wave speed.*?frequency\s+(?:of\s+)?([\d.]+).*?wavelength\s+(?:of\s+)?([\d.]+)', 'wave_speed({0}, {1})'),
    (r'(?:ohm|voltage|current|resistance).*?(\d[\d.]*)\s*V.*?(\d[\d.]*)\s*(?:A|amp)', 'ohms_law(voltage={0}, current={1})'),
]
