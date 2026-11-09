"""
Modified Crazyflow Control Environment
When using remote PC for control over attitude rates and f the problem is that the thrust is actually limited
way lower than the actual values due to battery preservation and other checks
So this version implements it from that view
"""

from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.constants import GRAVITY, MASS, MIX_MATRIX

class Control(str, Enum):
    """Control type of the simulated onboard controller."""
    state = "state"
    attitude = "attitude"
    thrust = "thrust"
    default = thrust

# #CF21 THRUST VS PWM CONSTANTS FOR EACH MOTOR'S THRUST
# # T = A*p^2 + B*p + C (where p is 0.0 to 1.0)
# POLY_A = -0.091502/4.0
# POLY_B = 0.579921/4.0
# POLY_C = -0.045246/4.0
# #Note that in real system MIN and MAX are not zeros
# MAX_PWM_VAL = 65535 #We need to ensure the MAX PWM val is actually less than intended here
# MIN_PWM_VAL = 6000 #Directly from the thrust curve
# # Normalize PWM limits to 0.0 - 1.0 range for the polynomial
# MIN_PWM_NORM = MIN_PWM_VAL / MAX_PWM_VAL
# MAX_PWM_NORM = 1.0

# _min_thrust_raw = POLY_A * (MIN_PWM_NORM**2) + POLY_B * MIN_PWM_NORM + POLY_C
# MIN_THRUST = max(0.0, _min_thrust_raw)
# MAX_THRUST = POLY_A * (1.0**2) + POLY_B * (1.0) + POLY_C  # Approx 0.6884 N

# #True values
# KF: float = 3.16e-10
# KM: float = 7.94e-12

# PWM2RPM_SCALE: float = 0.2685
# PWM2RPM_CONST: float = 4070.3
# MIN_PWM: float = float(MIN_PWM_VAL)
# MAX_PWM: float = float(MAX_PWM_VAL)
# MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
# MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
# _ESTIMATED_MAX_RPM = MAX_RPM
# #PID gains, unused here
# P_F: Array = np.array([0.4, 0.4, 1.25])
# I_F: Array = np.array([0.05, 0.05, 0.05])
# D_F: Array = np.array([0.2, 0.2, 0.5])
# I_F_RANGE: Array = np.array([2.0, 2.0, 0.4])
# P_T: Array = np.array([70000.0, 70000.0, 60000.0])
# I_T: Array = np.array([0.0, 0.0, 500.0])
# D_T: Array = np.array([20000.0, 20000.0, 12000.0])
# MIN_OMEGA: float = -2.0
# MAX_OMEGA: float = 2.0 #Rads
# #CF21 values from firmware for Rate Control
# P_RATE: Array = np.array([250.0, 250.0, 120.0])
# I_RATE: Array = np.array([500.0, 500.0, 16.7])
# D_RATE: Array = np.array([2.5, 2.5, 0.0])
# I_RATE_RANGE: Array = np.array([33.3, 33.3, 166.7])

#CF21BL THRUST VS PWM CONSTANTS FOR EACH MOTOR'S THRUST
#T = A*p^2 + B*p + C (where p is 0.0 to 1.0)
# POLY_A = 0.2593/4.0
# POLY_B = 0.4694/4.0
# POLY_C = -0.0403/4.0
#Is this the new mappings we found? yeah but I did it in wrong sequence (A and B are switched but even that it should show z higher
# POLY_A = 0.35/4.0
# POLY_B = 0.26/4.0
# POLY_C = 0.0/4.0
POLY_A = -2.199045963119e-02 /4.0
POLY_B = 7.659665118497e-01 /4.0
POLY_C = -2.384504350144e-02/4.0
DEFAULT_TOTAL_THRUST_CURVE = np.array([
    4.0 * POLY_A,
    4.0 * POLY_B,
    4.0 * POLY_C,
])
# POLY_A = 0.35/4.0
# POLY_B = 0.26/4.0
# POLY_C = -0.024/4.0
#Where is lee log?
#Note that in real system MIN and MAX are not zeros
MAX_PWM_VAL = 65535 #We need to ensure the MAX PWM val is actually less than intended here
MIN_PWM_VAL = 0 #Directly from the thrust curve
# Normalize PWM limits to 0.0 - 1.0 range for the polynomial
MIN_PWM_NORM = MIN_PWM_VAL / MAX_PWM_VAL
MAX_PWM_NORM = 1.0

_min_thrust_raw = POLY_A * (MIN_PWM_NORM**2) + POLY_B * MIN_PWM_NORM + POLY_C
MIN_THRUST = max(0.0, _min_thrust_raw)
#MAX_THRUST = POLY_A * (1.0**2) + POLY_B * (1.0) + POLY_C  # Approx 0.6884 N
MAX_THRUST = 0.759/4

#Estimate other constants
_ESTIMATED_MAX_RPM = 30000.0 #Assumed max 30k RPM
KF: float = 3.5e-10
# KF: float = MAX_THRUST / (_ESTIMATED_MAX_RPM**2)  # Back-calculated KF
_ORIG_RATIO = 8.5e-12 / 3.41e-10 #Previously computed ratio
KM: float = KF * _ORIG_RATIO

#Legacy/Unused constants kept for compatibility with other imports
PWM2RPM_SCALE: float = 0.436
PWM2RPM_CONST: float = 1884.0
MIN_PWM: float = float(MIN_PWM_VAL)
MAX_PWM: float = float(MAX_PWM_VAL)
MIN_RPM: float = 0.0
MAX_RPM: float = _ESTIMATED_MAX_RPM

#PID gains
P_F: Array = np.array([0.4, 0.4, 1.25])
I_F: Array = np.array([0.05, 0.05, 0.05])
D_F: Array = np.array([0.2, 0.2, 0.5])
I_F_RANGE: Array = np.array([2.0, 2.0, 0.4])
P_T: Array = np.array([70000.0, 70000.0, 60000.0])
I_T: Array = np.array([0.0, 0.0, 500.0])
D_T: Array = np.array([20000.0, 20000.0, 12000.0])
MIN_OMEGA: float = -4.0
MAX_OMEGA: float = 4.0 #Rads
#CF21BL
P_RATE: Array = np.array([200.0, 200.0, 120.0])
I_RATE: Array = np.array([450.0, 450.0, 16.7])
D_RATE: Array = np.array([2.5, 2.5, 0.0])
I_RATE_RANGE: Array = np.array([33.3, 33.3, 166.7]) # Anti-windup clip

@partial(jnp.vectorize, signature="(3),(3),(4),(3),(3),(1),(3)->(4),(3)", excluded=[7])
def state2attitude(
    pos: Array,
    vel: Array,
    quat: Array,
    des_pos: Array,
    des_vel: Array,
    des_yaw: Array,
    i_error: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
    pos_error, vel_error = des_pos - pos, des_vel - vel
    i_error = jnp.clip(i_error + pos_error * dt, -I_F_RANGE, I_F_RANGE)
    thrust = P_F * pos_error + I_F * i_error + D_F * vel_error
    thrust = thrust.at[2].add(MASS * GRAVITY)
    z_axis = R.from_quat(quat).as_matrix()[:, 2]
    # Project thrust onto z-axis
    thrust_desired = jnp.clip(thrust @ z_axis, MIN_THRUST, MAX_THRUST) # Updated limits
    z_axis = thrust / jnp.linalg.norm(thrust)
    yaw_axis = jnp.concatenate([jnp.cos(des_yaw), jnp.sin(des_yaw), jnp.array([0.0])])
    y_axis = jnp.cross(z_axis, yaw_axis)
    y_axis = y_axis / jnp.linalg.norm(y_axis)
    x_axis = jnp.cross(y_axis, z_axis)
    euler_desired = R.from_matrix(jnp.vstack([x_axis, y_axis, z_axis]).T).as_euler("xyz")
    return jnp.concatenate([jnp.atleast_1d(thrust_desired), euler_desired]), i_error

@partial(jnp.vectorize, signature="(4),(4),(3),(3)->(4),(3)", excluded=[4])
def attitude2rpm(
    controls: Array, quat: Array, last_rpy: Array, rpy_err_i: Array, dt: float
) -> tuple[Array, Array]:
    """Convert the desired collective thrust and attitude into motor RPMs/PWMs."""
    rot = R.from_quat(quat)
    target_rot = R.from_euler("xyz", controls[1:])
    drot = (target_rot.inv() * rot).as_matrix()
    rot_e = jnp.array([drot[2, 1] - drot[1, 2], drot[0, 2] - drot[2, 0], drot[1, 0] - drot[0, 1]])
    rpy_rates_e = -(rot.as_euler("xyz") - last_rpy) / dt
    rpy_err_i = rpy_err_i - rot_e * dt
    rpy_err_i = jnp.clip(rpy_err_i, -1500.0, 1500.0)
    rpy_err_i = rpy_err_i.at[:2].set(jnp.clip(rpy_err_i[:2], -1.0, 1.0))
    target_torques = -P_T * rot_e + D_T * rpy_rates_e + I_T * rpy_err_i
    target_torques = jnp.clip(target_torques, -3200, 3200)
    thrust_per_motor = jnp.atleast_1d(controls[0]) / 4
    base_pwm = thrust2pwm(thrust_per_motor)
    pwm = jnp.clip(base_pwm + MIX_MATRIX @ target_torques, MIN_PWM, MAX_PWM)
    return pwm2rpm(pwm), rpy_err_i

@partial(jnp.vectorize, signature="(4)->(4)")
def thrust2pwm(thrust: Array) -> Array:
    """Convert desired thrust (N) to PWM (0-65535) using inverse quadratic.
    Solves: A*p^2 + B*p + (C - Thrust) = 0 for p
    """
    thrust = jnp.clip(thrust, MIN_THRUST, MAX_THRUST)
    c_term = POLY_C - thrust
    delta = POLY_B**2 - 4 * POLY_A * c_term
    delta = jnp.maximum(delta, 0.0)
    p_norm = (-POLY_B + jnp.sqrt(delta)) / (2 * POLY_A)
    pwm_val = p_norm * MAX_PWM

    # jax.debug.print(
    # "thrust={} MIN={} MAX={} p_norm={} MAX_PWM={} pwm={}",
    # thrust, MIN_THRUST, MAX_THRUST, p_norm, MAX_PWM, pwm_val
    # )

    return jnp.clip(pwm_val, MIN_PWM, MAX_PWM)

@jax.jit
def thrust2pwm_kernel(thrust: Array) -> Array:
    """JIT compiled version of thrust2pwm."""
    thrust = jnp.clip(thrust, MIN_THRUST, MAX_THRUST)
    c_term = POLY_C - thrust
    delta = POLY_B**2 - 4 * POLY_A * c_term
    delta = jnp.maximum(delta, 0.0)
    p_norm = (-POLY_B + jnp.sqrt(delta)) / (2 * POLY_A)
    pwm_val = p_norm * MAX_PWM
    return jnp.clip(pwm_val, MIN_PWM, MAX_PWM)

@partial(jnp.vectorize, signature="(4)->(4)")
def pwm2thrust(pwm: Array) -> Array:
    """Convert PWM (0-65535) to Thrust (N) using polynomial.
    T = A*p^2 + B*p + C
    """
    p_norm = pwm / MAX_PWM
    thrust = POLY_A * (p_norm**2) + POLY_B * p_norm + POLY_C
    return jnp.clip(thrust, 0.0, MAX_THRUST)

@jax.jit
def pwm2thrust_kernel(pwm: Array) -> Array:
    """JIT compiled version of pwm2thrust."""
    p_norm = pwm / MAX_PWM
    thrust = POLY_A * (p_norm**2) + POLY_B * p_norm + POLY_C
    return jnp.clip(thrust, 0.0, MAX_THRUST)


@jax.jit
def thrust2pwm_curve_kernel(
    thrust: Array,
    total_curve: Array,
    pwm_scale: float,
) -> Array:
    """Invert a configurable *total-thrust* curve for each environment.

    ``total_curve`` has shape ``(n_envs, 3)`` and stores ``[A, B, C]`` for
    ``T_total = A*d**2 + B*d + C``.  The controller works per motor, so the
    coefficients are divided by four here.  Keeping the command curve separate
    from the plant curve lets domain randomization change the response without
    changing the action definition seen by the policy.
    """
    thrust = jnp.clip(thrust, MIN_THRUST, MAX_THRUST)
    a = total_curve[:, 0:1] / 4.0
    b = total_curve[:, 1:2] / 4.0
    c = total_curve[:, 2:3] / 4.0
    discriminant = jnp.maximum(b**2 - 4.0 * a * (c - thrust), 0.0)
    quadratic_root = (-b + jnp.sqrt(discriminant)) / jnp.where(
        jnp.abs(2.0 * a) > 1e-8, 2.0 * a, 1.0
    )
    linear_root = (thrust - c) / jnp.where(jnp.abs(b) > 1e-8, b, 1.0)
    duty = jnp.where(jnp.abs(a) > 1e-8, quadratic_root, linear_root)
    return jnp.clip(duty * pwm_scale, MIN_PWM, MAX_PWM)


@jax.jit
def pwm2thrust_curve_kernel(
    pwm: Array,
    total_curve: Array,
    pwm_scale: float,
) -> Array:
    """Apply a configurable plant-side total-thrust curve per environment."""
    duty = pwm / pwm_scale
    a = total_curve[:, 0:1] / 4.0
    b = total_curve[:, 1:2] / 4.0
    c = total_curve[:, 2:3] / 4.0
    thrust = a * duty**2 + b * duty + c
    # Do not clip to the nominal MAX_THRUST: a randomized plant is allowed to
    # produce more or less thrust at the same PWM.  Negative thrust is invalid.
    return jnp.maximum(thrust, 0.0)

#Legacy Wrappers
@partial(jnp.vectorize, signature="(4)->(4)")
def pwm2rpm(pwm: Array) -> Array:
    """Legacy RPM conversion (mostly for compatibility)."""
    return (pwm / MAX_PWM) * _ESTIMATED_MAX_RPM

@jax.jit
def thrust_curve(thrust: Array) -> Array:
    """Legacy interface."""
    return thrust2pwm_kernel(thrust)
