"""Bloque 2 del entorno: reset() y la observación."""

from __future__ import annotations

import math

import numpy as np

from furuta.sim.pendulum_env import (
    MOTOR_SAFE_LIMIT_RAD,
    RotaryInvertedPendulumEnv,
    _wrap_pi,
)


def test_estado_inicial_tras_reset() -> None:
    env = RotaryInvertedPendulumEnv()
    for seed in range(20):
        env.reset(seed=seed)
        motor = env.data.qpos[env._motor_qpos_addr]
        phi = env.data.qpos[env._pen_qpos_addr]
        assert abs(motor) <= 0.7 * MOTOR_SAFE_LIMIT_RAD
        assert abs(phi) <= 0.05
        assert env.data.qvel[env._motor_qvel_addr] == 0.0
        assert env.data.qvel[env._pen_qvel_addr] == 0.0
        assert env.data.time == 0.0


def test_observacion_colgando() -> None:
    env = RotaryInvertedPendulumEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (24,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    frame = obs[-6:]                         # frame más nuevo
    assert frame[2] < -0.99                  # cos θ ≈ -1 (colgando)
    assert abs(frame[1]) < 0.06              # sin θ ≈ 0
    assert frame[3] == 0.0 and frame[4] == 0.0 and frame[5] == 0.0
    assert info == {}


def test_misma_semilla_misma_observacion() -> None:
    env = RotaryInvertedPendulumEnv()
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    c, _ = env.reset(seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_varios_reset_seguidos() -> None:
    env = RotaryInvertedPendulumEnv()
    for _ in range(5):
        env.reset()


def test_wrap_pi() -> None:
    assert math.isclose(_wrap_pi(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(_wrap_pi(2 * math.pi + 0.1), 0.1, abs_tol=1e-9)
    assert math.isclose(_wrap_pi(-math.pi), math.pi, abs_tol=1e-9)   # rango (-π, π]
    assert -math.pi < _wrap_pi(-0.03 - math.pi) <= math.pi


if __name__ == "__main__":
    test_estado_inicial_tras_reset()
    test_observacion_colgando()
    test_misma_semilla_misma_observacion()
    test_varios_reset_seguidos()
    test_wrap_pi()
    print("OK")
