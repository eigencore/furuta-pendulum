"""Bloque 7 del entorno: suavizado de acción (promedio de las últimas N).

Se puede correr con pytest o directamente: python tests/test_action_smoothing.py
"""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import ACTION_SMOOTH_WINDOW, RotaryInvertedPendulumEnv

N = ACTION_SMOOTH_WINDOW


def test_escalon_sube_en_n_pasos() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    vistas = [env.step(np.array([1.0]))[4]["smoothed_action"] for _ in range(N + 1)]
    esperado = [min(k, N) / N for k in range(1, N + 2)]      # 0.25, 0.5, 0.75, 1, 1
    assert np.allclose(vistas, esperado)


def test_zigzag_se_cancela() -> None:
    """+1, -1, +1, -1... (25 Hz) debe quedar en 0 para el motor."""
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    ultimas = []
    for k in range(12):
        info = env.step(np.array([1.0 if k % 2 == 0 else -1.0]))[4]
        ultimas.append(info["smoothed_action"])
    assert np.allclose(ultimas[N:], 0.0)                     # tras llenar la ventana


def test_la_recompensa_y_prev_action_usan_la_accion_cruda() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    obs, _, _, _, info = env.step(np.array([0.8]))
    assert abs(obs[-1] - 0.8) < 1e-6                         # prev_action = cruda
    assert abs(info["smoothed_action"] - 0.8 / N) < 1e-9     # motor: promedio


def test_ventana_1_no_suaviza() -> None:
    env = RotaryInvertedPendulumEnv(action_smooth_window=1)
    env.reset(seed=0)
    info = env.step(np.array([0.6]))[4]
    assert abs(info["smoothed_action"] - 0.6) < 1e-9


def test_reset_limpia_el_buffer() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.array([1.0]))
    env.reset(seed=1)
    info = env.step(np.array([0.0]))[4]
    assert info["smoothed_action"] == 0.0                    # sin restos del episodio previo


def test_ventana_invalida_da_error() -> None:
    try:
        RotaryInvertedPendulumEnv(action_smooth_window=0)
    except ValueError:
        return
    raise AssertionError("action_smooth_window=0 debería lanzar ValueError")


if __name__ == "__main__":
    test_escalon_sube_en_n_pasos()
    test_zigzag_se_cancela()
    test_la_recompensa_y_prev_action_usan_la_accion_cruda()
    test_ventana_1_no_suaviza()
    test_reset_limpia_el_buffer()
    test_ventana_invalida_da_error()
    print("OK")
