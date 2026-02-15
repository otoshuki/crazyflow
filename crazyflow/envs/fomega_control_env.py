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
    thrust2pwm_kernel,pwm2thrust_kernel
)

RAD_TO_DEG = 180.0 / jnp.pi

@jax.jit
def rates_to_motor_thrusts_pid_kernel(
    controls: Array,      # Input: (B, 1, 4) [T_des, p_des, q_des, r_des]
    current_rates: Array, # Input: (B, 1, 3) [p_actual, q_actual, r_actual]
    rate_err_i: Array,    # State: (B, 1, 3) integral of rate error
    last_err_p: Array,    # State: (B, 1, 3) last proportional error
    dt: float,
) -> tuple[Array, Array, Array]:
    """
    JIT-compiled, fully-batched PID controller kernel similar to firmware
    """
    desired_thrust = controls[..., 0:1]
    desired_rates = controls[..., 1:4]
    rate_error_p = (desired_rates - current_rates)* RAD_TO_DEG
    rate_err_i = rate_err_i + rate_error_p * dt
    rate_err_i = jnp.clip(rate_err_i, -I_RATE_RANGE, I_RATE_RANGE)
    rate_error_d = (rate_error_p - last_err_p) / dt
    target_torques = (
        P_RATE * rate_error_p +
        I_RATE * rate_err_i +
        D_RATE * rate_error_d
    )
    #Thrust component
    base_thrust_per_motor = jnp.atleast_1d(desired_thrust) / 4.0
    base_pwm = thrust2pwm_kernel(base_thrust_per_motor)
    #Torque component
    target_torques = jnp.clip(target_torques, -3200, 3200)
    target_torques_T = jnp.swapaxes(target_torques, -1, -2)
    pwm_torques = MIX_MATRIX @ target_torques_T
    pwm_torques = jnp.swapaxes(pwm_torques, -1, -2)
    #Total pwm
    pwm = jnp.clip(base_pwm + pwm_torques, MIN_PWM, MAX_PWM)
    motor_thrusts = pwm2thrust_kernel(pwm)
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

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        self._apply_action(action)
        self._reset(mask=self._marked_for_reset)
        terminated, truncated = self.terminated(), self.truncated()
        self._marked_for_reset = self._done(terminated, truncated)
        return self.obs(), self.reward(), terminated, truncated, {}

    def _apply_action(self, action: Array, thrust_noise: Array | None = None):
        if thrust_noise is None:
            thrust_noise = jnp.zeros((self.num_envs, 4), device=self.sim.device)
        low, high = self.action_space.low, self.action_space.high
        ref_action = self._sanitize_action(action, low, high, self.sim.n_worlds, self.sim.device)
        ref_action = ref_action.squeeze(axis=1)
        rate_err_i = self.integral_error.squeeze(axis=1)
        last_err = self.last_error.squeeze(axis=1)
        for _ in range(self.n_substeps):
            current_ang_vel = self.sim.data.states.ang_vel.squeeze(axis=1)
            motor_thrusts, new_rate_err_i, new_last_err = rates_to_motor_thrusts_pid_kernel(
                ref_action,
                current_ang_vel,
                rate_err_i,
                last_err,
                self.sim_dt
            )
            noisy_thrusts = motor_thrusts + thrust_noise
            self.sim.thrust_control(noisy_thrusts.reshape(self.num_envs, 1, 4))
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

    def _reset(self, mask: Array | None = None, options: dict | None = None) -> None:
        """Overrides DroneEnv._reset to reset PID states for masked envs."""
        self.sim.reset(mask=mask)
        if mask is None:
             self.integral_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
             self.last_error = jnp.zeros((self.num_envs, 1, 3), device=self.device)
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
