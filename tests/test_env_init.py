"""Bloque 1 del entorno: construcción, constantes de tiempo y espacios."""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import (
    OBS_FRAME_DIM,
    OBS_HISTORY_LEN,
    RotaryInvertedPendulumEnv,
)


def test_pasos_de_fisica_y_de_episodio() -> None:
    env = RotaryInvertedPendulumEnv()
    assert env._n_substeps == 20            # 20 ms / 1 ms
    assert env._max_steps == 400            # 8 s × 50 Hz


def test_espacios() -> None:
    env = RotaryInvertedPendulumEnv()
    assert env.observation_space.shape == (OBS_FRAME_DIM * OBS_HISTORY_LEN,) == (24,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0


def test_direcciones_de_articulaciones_distintas() -> None:
    env = RotaryInvertedPendulumEnv()
    assert env._motor_qpos_addr != env._pen_qpos_addr
    assert np.isfinite(env.model.opt.timestep)


if __name__ == "__main__":
    test_pasos_de_fisica_y_de_episodio()
    test_espacios()
    test_direcciones_de_articulaciones_distintas()
    print("OK")
