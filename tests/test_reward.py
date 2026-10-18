"""Verifica la recompensa: función pura (compute_reward) y su uso en step().

Se puede correr con pytest o directamente: python tests/test_reward.py
"""

from __future__ import annotations

import math

import numpy as np

from furuta.sim import pendulum_env
from furuta.sim.pendulum_env import (
    HARD_STOP_PENALTY,
    MOTOR_LIMIT_RAD,
    RotaryInvertedPendulumEnv,
)
from furuta.sim.reward import RewardWeights, compute_reward

W = RewardWeights()
META = dict(theta=0.0, pen_vel=0.0, motor_pos=0.0, motor_vel=0.0, action=0.0)


def test_la_meta_da_cero() -> None:
    assert compute_reward(**META, weights=W) == 0.0


def test_colgado_cuesta_pi_al_cuadrado() -> None:
    r = compute_reward(**{**META, "theta": math.pi}, weights=W)
    assert math.isclose(r, -math.pi**2)


def test_cada_termino_con_su_peso() -> None:
    cases = {
        "pen_vel": (2.0, W.k_pen_vel * 4.0),
        "motor_pos": (1.0, W.k_motor_pos * 1.0),
        "motor_vel": (3.0, W.k_motor_vel * 9.0),
        "action": (-1.0, W.k_action * 1.0),
        "theta": (0.5, 0.25),
    }
    for name, (value, expected_cost) in cases.items():
        r = compute_reward(**{**META, name: value}, weights=W)
        assert math.isclose(r, -expected_cost), name


def test_sin_offset_nunca_es_positiva() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = compute_reward(
            theta=rng.uniform(-math.pi, math.pi), pen_vel=rng.uniform(-30, 30),
            motor_pos=rng.uniform(-2.4, 2.4), motor_vel=rng.uniform(-12, 12),
            action=rng.uniform(-1, 1), weights=W,
        )
        assert r <= 0.0


def test_el_offset_se_suma() -> None:
    w = RewardWeights(k_alive_offset=15.0)
    assert compute_reward(**META, weights=w) == 15.0
    assert math.isclose(compute_reward(**{**META, "theta": math.pi}, weights=w),
                        15.0 - math.pi**2)


def test_step_usa_la_funcion_con_el_estado_posterior() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    _, reward, term, _, info = env.step(np.array([0.0]))
    assert not term
    # Colgado y casi quieto: θ ≈ ±π, así que el costo ronda π² y α² pesa poco.
    assert -(math.pi**2) - 3.0 < reward < -(math.pi**2) + 0.5


def test_choque_con_el_tope_resta_la_penalizacion() -> None:
    """Se baja el límite del módulo a 1.0 rad para provocar el choque.

    No se puede forzar con el brazo en 135°: el servo lo devuelve a ≤125°
    (el objetivo está recortado) antes de acabar los 20 ms del paso.
    """
    def un_paso(limite: float) -> tuple[float, bool]:
        env = RotaryInvertedPendulumEnv()      # modelo con el límite real
        env.reset(seed=0)
        env.data.qpos[env._motor_qpos_addr] = 1.2
        env._motor_target = env._prev_cmd_pos = 1.2
        original = pendulum_env.MOTOR_LIMIT_RAD
        pendulum_env.MOTOR_LIMIT_RAD = limite  # solo afecta a la comprobación de step()
        try:
            _, reward, term, _, _ = env.step(np.array([0.0]))
        finally:
            pendulum_env.MOTOR_LIMIT_RAD = original
        return reward, term

    r_choque, term_choque = un_paso(limite=1.0)     # brazo en 1.2 ≥ 1.0
    r_libre, term_libre = un_paso(limite=MOTOR_LIMIT_RAD)
    assert term_choque and not term_libre
    assert math.isclose(r_libre - r_choque, HARD_STOP_PENALTY, abs_tol=1e-6)


def test_offset_alto_hace_que_terminar_no_convenga() -> None:
    w = RewardWeights(k_alive_offset=15.0)
    env = RotaryInvertedPendulumEnv(reward_weights=w)
    env.reset(seed=0)
    _, reward, *_ = env.step(np.array([0.0]))
    assert reward > 0.0          # colgado ≈ 15 − 9.9 > 0


if __name__ == "__main__":
    test_la_meta_da_cero()
    test_colgado_cuesta_pi_al_cuadrado()
    test_cada_termino_con_su_peso()
    test_sin_offset_nunca_es_positiva()
    test_el_offset_se_suma()
    test_step_usa_la_funcion_con_el_estado_posterior()
    test_choque_con_el_tope_resta_la_penalizacion()
    test_offset_alto_hace_que_terminar_no_convenga()
    print("OK")
