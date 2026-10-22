"""Bloque 6 del entorno: apilado de frames en la observación.

Se puede correr con pytest o directamente: python tests/test_obs_stacking.py
"""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import OBS_FRAME_DIM, OBS_HISTORY_LEN, RotaryInvertedPendulumEnv

K = OBS_HISTORY_LEN
D = OBS_FRAME_DIM


def _frames(obs: np.ndarray) -> np.ndarray:
    """(K, D): fila 0 = frame más viejo, fila K-1 = más nuevo."""
    return obs.reshape(K, D)


def test_tras_reset_los_k_frames_son_identicos() -> None:
    env = RotaryInvertedPendulumEnv()
    obs, _ = env.reset(seed=0)
    frames = _frames(obs)
    assert all(np.array_equal(frames[0], f) for f in frames)


def test_el_frame_nuevo_entra_al_final_y_los_demas_se_desplazan() -> None:
    env = RotaryInvertedPendulumEnv()
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(np.array([0.5]))
    obs2, *_ = env.step(np.array([-0.5]))
    f0, f1, f2 = _frames(obs0)[-1], _frames(obs1)[-1], _frames(obs2)[-1]
    assert not np.array_equal(f0, f1)                   # el estado cambió
    # Tras 2 steps: [f0, f0, f1, f2]
    assert np.array_equal(_frames(obs2)[0], f0)
    assert np.array_equal(_frames(obs2)[1], f0)
    assert np.array_equal(_frames(obs2)[2], f1)
    assert np.array_equal(_frames(obs2)[3], f2)


def test_reset_borra_el_historial_anterior() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.array([1.0]))
    obs, _ = env.reset(seed=1)
    frames = _frames(obs)
    assert all(np.array_equal(frames[0], f) for f in frames)   # sin restos del episodio previo


def test_con_un_frame_vuelve_a_la_observacion_simple() -> None:
    env = RotaryInvertedPendulumEnv(obs_history_len=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (D,) and env.observation_space.shape == (D,)


def test_historial_invalido_da_error() -> None:
    try:
        RotaryInvertedPendulumEnv(obs_history_len=0)
    except ValueError:
        return
    raise AssertionError("obs_history_len=0 debería lanzar ValueError")


if __name__ == "__main__":
    test_tras_reset_los_k_frames_son_identicos()
    test_el_frame_nuevo_entra_al_final_y_los_demas_se_desplazan()
    test_reset_borra_el_historial_anterior()
    test_con_un_frame_vuelve_a_la_observacion_simple()
    test_historial_invalido_da_error()
    print("OK")
