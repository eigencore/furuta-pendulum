"""Aleatorización de dominio, paso 3: sesgo del cero del péndulo (solo en la observación)."""

from __future__ import annotations

import math

import numpy as np

from furuta.sim.pendulum_env import (
    DR_THETA_BIAS_MAX_RAD,
    RotaryInvertedPendulumEnv,
    _wrap_pi,
)
from furuta.sim.reward import compute_reward


def _theta_observado(obs: np.ndarray) -> float:
    return math.atan2(obs[-5], obs[-4])     # frame más nuevo: sin θ, cos θ


def _theta_verdadero(env: RotaryInvertedPendulumEnv) -> float:
    return _wrap_pi(float(env.data.qpos[env._pen_qpos_addr]) - math.pi)


def test_por_defecto_el_sesgo_esta_activo_aunque_no_haya_dr() -> None:
    env = RotaryInvertedPendulumEnv()               # domain_randomization=False
    sesgos = []
    for seed in range(100):
        env.reset(seed=seed)
        sesgos.append(env._theta_bias_rad)
    assert max(abs(b) for b in sesgos) <= DR_THETA_BIAS_MAX_RAD
    assert len(set(sesgos)) > 90, "el sesgo debe cambiar en cada episodio"
    # Cubre ambos signos y llega cerca de los extremos.
    assert min(sesgos) < -0.8 * DR_THETA_BIAS_MAX_RAD
    assert max(sesgos) > 0.8 * DR_THETA_BIAS_MAX_RAD


def test_la_observacion_es_theta_verdadero_mas_el_sesgo() -> None:
    env = RotaryInvertedPendulumEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(30):
        diferencia = _wrap_pi(_theta_observado(obs) - _theta_verdadero(env))
        assert abs(diferencia - env._theta_bias_rad) < 1e-5
        obs, *_ = env.step(np.array([0.5]))


def test_el_sesgo_es_constante_dentro_del_episodio() -> None:
    env = RotaryInvertedPendulumEnv()
    obs, _ = env.reset(seed=4)
    difs = []
    for _ in range(50):
        obs, *_ = env.step(np.array([-0.4]))
        difs.append(_wrap_pi(_theta_observado(obs) - _theta_verdadero(env)))
    assert max(difs) - min(difs) < 1e-5


def test_con_maximo_cero_no_hay_sesgo() -> None:
    env = RotaryInvertedPendulumEnv(dr_theta_bias_max_rad=0.0)
    obs, _ = env.reset(seed=0)
    assert env._theta_bias_rad == 0.0
    assert abs(_theta_observado(obs) - _theta_verdadero(env)) < 1e-6


def test_respeta_un_maximo_personalizado() -> None:
    env = RotaryInvertedPendulumEnv(dr_theta_bias_max_rad=0.02)
    for seed in range(50):
        env.reset(seed=seed)
        assert abs(env._theta_bias_rad) <= 0.02


def test_un_maximo_negativo_se_rechaza() -> None:
    try:
        RotaryInvertedPendulumEnv(dr_theta_bias_max_rad=-0.01)
    except ValueError:
        return
    raise AssertionError("debió lanzar ValueError")


def test_la_recompensa_y_la_fisica_no_ven_el_sesgo() -> None:
    """La recompensa coincide con la fórmula evaluada en el estado verdadero."""
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=6)
    assert env._theta_bias_rad != 0.0
    for _ in range(20):
        _, r, *_ = env.step(np.array([0.0]))
        esperado = compute_reward(
            theta=_theta_verdadero(env),
            pen_vel=float(env.data.qvel[env._pen_qvel_addr]),
            motor_pos=float(env.data.qpos[env._motor_qpos_addr]),
            motor_vel=float(env.data.qvel[env._motor_qvel_addr]),
            action=0.0,
            weights=env.reward_weights,
        )
        assert math.isclose(r, esperado, rel_tol=1e-12), (r, esperado)


def test_es_reproducible_con_semilla() -> None:
    a, b = RotaryInvertedPendulumEnv(), RotaryInvertedPendulumEnv()
    a.reset(seed=9)
    b.reset(seed=9)
    assert a._theta_bias_rad == b._theta_bias_rad


if __name__ == "__main__":
    test_por_defecto_el_sesgo_esta_activo_aunque_no_haya_dr()
    test_la_observacion_es_theta_verdadero_mas_el_sesgo()
    test_el_sesgo_es_constante_dentro_del_episodio()
    test_con_maximo_cero_no_hay_sesgo()
    test_respeta_un_maximo_personalizado()
    test_un_maximo_negativo_se_rechaza()
    test_la_recompensa_y_la_fisica_no_ven_el_sesgo()
    test_es_reproducible_con_semilla()
    print("OK")
