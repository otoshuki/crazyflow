"""
New 100Hz Gym environment (CustomDroneEnv) that wraps the 500Hz
simulator (which acts as the "inner_env").

This file is self-contained and includes all the necessary PID
controller logic.
"""
from typing import Callable, Literal, NamedTuple
from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np
import warnings
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array
from gymnasium.vector import VectorEnv
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim import Sim
from crazyflow.sim.physics import Physics
from crazyflow.sim.structs import SimData
from crazyflow.utils import leaf_replace
from crazyflow.sim.integration import Integrator

from crazyflow.control.control import (
    Control,
    MIN_THRUST,MAX_THRUST,MIX_MATRIX,
    MAX_PWM,MIN_PWM,
    P_RATE,I_RATE,D_RATE,I_RATE_RANGE,
    DEFAULT_TOTAL_THRUST_CURVE,
    thrust2pwm_curve_kernel,pwm2thrust_curve_kernel,
)

RAD_TO_DEG = 180.0 / jnp.pi

@jax.jit
def rates_to_motor_thrusts_pid_kernel(
    controls: Array,      # Input: (B, 1, 4) [T_des, p_des, q_des, r_des]
    current_rates: Array, # Input: (B, 1, 3) [p_actual, q_actual, r_actual]
    rate_err_i: Array,    # State: (B, 1, 3) integral of rate error
    last_err_p: Array,    # State: (B, 1, 3) last proportional error
    command_curve: Array, # (B, 3), total-thrust [A, B, C]
    plant_curve: Array,   # (B, 3), total-thrust [A, B, C]
    pwm_scale: float,
    rate_p: Array,        # (B, 3)
    rate_i: Array,        # (B, 3)
    rate_d: Array,        # (B, 3)
    dt: float,
) -> tuple[Array, Array, Array]:
    """
    JIT-compiled, fully-batched PID controller kernel similar to firmware
    """
    desired_thrust = controls[..., 0:1]
    desired_rates = controls[..., 1:4]
    rate_error_p = (desired_rates - current_rates)*RAD_TO_DEG
    rate_err_i = rate_err_i + rate_error_p * dt
    rate_err_i = jnp.clip(rate_err_i, -I_RATE_RANGE, I_RATE_RANGE)
    rate_error_d = (rate_error_p - last_err_p) / dt
    target_torques = (
        rate_p * rate_error_p +
        rate_i * rate_err_i +
        rate_d * rate_error_d
    )
    #Thrust component
    base_thrust_per_motor = jnp.atleast_1d(desired_thrust) / 4.0
    base_pwm = thrust2pwm_curve_kernel(
        base_thrust_per_motor, command_curve, pwm_scale
    )
    #Torque component
    target_torques = jnp.clip(target_torques, -3200, 3200)
    target_torques_T = jnp.swapaxes(target_torques, -1, -2)
    pwm_torques = MIX_MATRIX @ target_torques_T
    pwm_torques = jnp.swapaxes(pwm_torques, -1, -2)
    #Total pwm
    pwm = jnp.clip(base_pwm + pwm_torques, MIN_PWM, MAX_PWM)
    motor_thrusts = pwm2thrust_curve_kernel(pwm, plant_curve, pwm_scale)

    return motor_thrusts, rate_err_i, rate_error_p

class FOmegaControlEnv(DroneEnv):
    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["sys_id", "analytical"] | Physics = Physics.analytical,
        device: str = "cpu",
        reset_randomization: Callable[[SimData, Array], SimData] | None = None,
        integrator: Literal["euler", "rk4", "symplectic_euler"] | Integrator = Integrator.default
    ):
        self.num_envs = num_envs
        self.device = jax.devices(device)[0]
        self.freq = 100  #f_omega control frequency
        self.sim_freq = 500 #thrust control frequency
        self.control_dt = 1/self.freq
        self.sim_dt = 1/self.sim_freq
        self.n_substeps = self.sim_freq // self.freq #5
        self.max_episode_time = max_episode_time
        assert Physics(physics) in Physics, f"Invalid physics type {physics}"
        assert Integrator(integrator) in Integrator, f"Invalid integrator type {integrator}"
        print(f"Using Physics: {physics} with Integrator: {integrator}")
        self.sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            freq=self.sim_freq,
            attitude_freq=self.sim_freq,
            thrust_freq=self.sim_freq,
            device=device,
            physics=physics,
            control=Control.thrust,
            integrator=integrator
        )
        assert self.sim.freq >= self.sim.control_freq, "Sim freq must be higher than control freq"
        if not self.sim.freq % self.freq == 0:
            warnings.warn("Simulation frequency should be a multiple of env frequency.")
        if reset_randomization is None:
            reset_randomization = self._reset_randomization
        self.sim.reset_pipeline += (reset_randomization,)
        self.sim.build_reset_fn()
        self._marked_for_reset = jnp.zeros((self.sim.n_worlds), dtype=jnp.bool_, device=self.device)
        ang_vel_limit = 10
        self.single_action_space = spaces.Box(
            low=np.array([4 * MIN_THRUST, -ang_vel_limit, -ang_vel_limit, -ang_vel_limit], dtype=np.float32),
            high=np.array([4 * MAX_THRUST, ang_vel_limit, ang_vel_limit, ang_vel_limit], dtype=np.float32),
        )
        self.action_space = batch_space(self.single_action_space, self.sim.n_worlds)
        self.single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)
        self.integral_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
        self.last_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
        self._motor_response_state = jnp.zeros((self.num_envs, 4), device=self.sim.device)
        self._motor_response_initialized = jnp.zeros((self.num_envs, 1), dtype=jnp.bool_, device=self.sim.device)
        self._max_action_delay_steps = 0
        self._action_delay_buffer = jnp.zeros((1, self.num_envs, 4), device=self.sim.device)
        self._action_delay_initialized = jnp.zeros((self.num_envs,), dtype=jnp.bool_, device=self.sim.device)

    def configure_action_delay(self, max_delay_steps: int) -> None:
        """Allocate a fixed-size control-rate delay line for JAX-friendly use."""
        self._max_action_delay_steps = max(0, int(max_delay_steps))
        self._action_delay_buffer = jnp.zeros(
            (self._max_action_delay_steps + 1, self.num_envs, 4),
            device=self.sim.device,
        )
        self._action_delay_initialized = jnp.zeros(
            (self.num_envs,), dtype=jnp.bool_, device=self.sim.device
        )

    def _apply_action_delay(self, action: Array, delay_steps: Array) -> Array:
        if self._max_action_delay_steps == 0:
            return action
        delay_steps = jnp.clip(
            jnp.asarray(delay_steps, dtype=jnp.int32).reshape(self.num_envs),
            0,
            self._max_action_delay_steps,
        )
        shifted = jnp.concatenate(
            [action[None, ...], self._action_delay_buffer[:-1]], axis=0
        )
        filled = jnp.broadcast_to(action[None, ...], shifted.shape)
        self._action_delay_buffer = jnp.where(
            self._action_delay_initialized[None, :, None], shifted, filled
        )
        self._action_delay_initialized = jnp.ones_like(
            self._action_delay_initialized
        )
        return self._action_delay_buffer[
            delay_steps, jnp.arange(self.num_envs)
        ]


    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        self._apply_action(action)
        self._reset(mask=self._marked_for_reset)
        terminated, truncated = self.terminated(), self.truncated()
        self._marked_for_reset = self._done(terminated, truncated)
        return self.obs(), self.reward(), terminated, truncated, {}

    def _apply_action(
        self,
        action: Array,
        thrust_noise: Array | None = None,
        thrust_gain: Array | None = None,
        thrust_tau: Array | None = None,
        action_delay_steps: Array | None = None,
        command_curve: Array | None = None,
        plant_curve: Array | None = None,
        pwm_scale: float = MAX_PWM,
        rate_p: Array | None = None,
        rate_i: Array | None = None,
        rate_d: Array | None = None,
    ):
        if thrust_noise is None:
            thrust_noise = jnp.zeros((self.num_envs, 4), device=self.sim.device)
        if thrust_gain is None:
            thrust_gain = jnp.ones((self.num_envs, 1), device=self.sim.device)
        if thrust_tau is None:
            thrust_tau = jnp.ones((self.num_envs, 1), device=self.sim.device) * 0.045
        if action_delay_steps is None:
            action_delay_steps = jnp.zeros((self.num_envs,), dtype=jnp.int32, device=self.sim.device)
        if command_curve is None:
            command_curve = jnp.broadcast_to(
                jnp.asarray(DEFAULT_TOTAL_THRUST_CURVE, device=self.sim.device),
                (self.num_envs, 3),
            )
        if plant_curve is None:
            plant_curve = command_curve
        if rate_p is None:
            rate_p = jnp.broadcast_to(jnp.asarray(P_RATE), (self.num_envs, 3))
        if rate_i is None:
            rate_i = jnp.broadcast_to(jnp.asarray(I_RATE), (self.num_envs, 3))
        if rate_d is None:
            rate_d = jnp.broadcast_to(jnp.asarray(D_RATE), (self.num_envs, 3))
        low, high = self.action_space.low, self.action_space.high
        ref_action = self._sanitize_action(action, low, high, self.sim.n_worlds, self.sim.device)
        ref_action = ref_action.squeeze(axis=1)
        ref_action = self._apply_action_delay(ref_action, action_delay_steps)
        rate_err_i = self.integral_error.squeeze(axis=1)
        last_err = self.last_error.squeeze(axis=1)
        for _ in range(self.n_substeps):
            current_ang_vel = self.sim.data.states.ang_vel.squeeze(axis=1)
            motor_thrusts, new_rate_err_i, new_last_err = rates_to_motor_thrusts_pid_kernel(
                ref_action,
                current_ang_vel,
                rate_err_i,
                last_err,
                command_curve,
                plant_curve,
                pwm_scale,
                rate_p,
                rate_i,
                rate_d,
                self.sim_dt
            )
            rate_err_i, last_err = new_rate_err_i, new_last_err
            noisy_thrusts = motor_thrusts * thrust_gain + thrust_noise

            # Physical first-order motor/thrust response with time constant tau.
            alpha = self.sim_dt / (thrust_tau + self.sim_dt)  # per-env stable 0<alpha<1

            if not hasattr(self, "_motor_response_state"):
                self._motor_response_state = noisy_thrusts
            if not hasattr(self, "_motor_response_initialized"):
                self._motor_response_initialized = jnp.ones((self.num_envs, 1), dtype=jnp.bool_, device=self.sim.device)

            response_prev = jnp.where(
                self._motor_response_initialized,
                self._motor_response_state,
                noisy_thrusts,
            )
            self._motor_response_state = (
                response_prev + alpha * (noisy_thrusts - response_prev)
            )
            self._motor_response_initialized = jnp.ones_like(
                self._motor_response_initialized
            )
            delayed_thrusts = self._motor_response_state

            #print(f"total motor thrust: {noisy_thrusts.sum()}")
            self.sim.thrust_control(delayed_thrusts.reshape(self.num_envs, 1, 4))
            self.sim.step(1)
        self.integral_error = rate_err_i.reshape(self.num_envs, 1, 3)
        self.last_error = last_err.reshape(self.num_envs, 1, 3)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Overrides DroneEnv.reset to also reset the PID controller state."""
        """This one does not return anything to ensure obs is not called"""
        VectorEnv.reset(self, seed=seed)
        self._reset(options=options)
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        if seed is not None:
            self.sim.seed(seed)
        self.integral_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
        self.last_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
        if hasattr(self, "_motor_response_state"):
            self._motor_response_state = jnp.zeros((self.num_envs, 4), device=self.sim.device)
        if hasattr(self, "_motor_response_initialized"):
            self._motor_response_initialized = jnp.zeros((self.num_envs, 1), dtype=jnp.bool_, device=self.sim.device)
        if hasattr(self, "_action_delay_buffer"):
            self._action_delay_buffer = jnp.zeros_like(self._action_delay_buffer)
            self._action_delay_initialized = jnp.zeros_like(self._action_delay_initialized)


    def _reset(self, mask: Array | None = None, options: dict | None = None) -> None:
        """Overrides DroneEnv._reset to reset PID states for masked envs."""
        self.sim.reset(mask=mask)
        if mask is None:
             self.integral_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
             self.last_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
             if hasattr(self, "_motor_response_state"):
                 self._motor_response_state = jnp.zeros((self.num_envs, 4), device=self.sim.device)
             if hasattr(self, "_motor_response_initialized"):
                 self._motor_response_initialized = jnp.zeros((self.num_envs, 1), dtype=jnp.bool_, device=self.sim.device)
             if hasattr(self, "_action_delay_buffer"):
                 self._action_delay_buffer = jnp.zeros_like(self._action_delay_buffer)
                 self._action_delay_initialized = jnp.zeros_like(self._action_delay_initialized)
        else:
            mask_broadcast = mask.reshape(self.num_envs, 1, 1)
            self.integral_error = jnp.where(
                mask_broadcast,
                0.0,
                self.integral_error
            )
            self.last_error = jnp.where(
                mask_broadcast,
                0.0,
                self.last_error
            )
            if hasattr(self, "_motor_response_state"):
                mask_2d = mask.reshape(self.num_envs, 1)
                self._motor_response_state = jnp.where(mask_2d, 0.0, self._motor_response_state)
            if hasattr(self, "_motor_response_initialized"):
                mask_2d = mask.reshape(self.num_envs, 1)
                self._motor_response_initialized = jnp.where(mask_2d, False, self._motor_response_initialized)
            if hasattr(self, "_action_delay_buffer"):
                mask_3d = mask.reshape(1, self.num_envs, 1)
                self._action_delay_buffer = jnp.where(
                    mask_3d, 0.0, self._action_delay_buffer
                )
                self._action_delay_initialized = jnp.where(
                    mask, False, self._action_delay_initialized
                )


    def reward(self) -> Array:
        return jnp.ones(self.num_envs)

    @staticmethod
    def _reset_randomization(data: SimData, mask: Array) -> SimData:
        """Unused"""
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        pos_min = jnp.array([-1.0, -1.0, 1.0])
        pos_max = jnp.array([1.0, 1.0, 2.0])
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pos_min, maxval=pos_max)
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=-1.0, maxval=1.0)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data
