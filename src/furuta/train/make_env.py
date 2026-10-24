"""Fábricas de entornos para entrenar y evaluar.

Devuelven funciones sin argumentos (lo que espera `DummyVecEnv`), no entornos
ya construidos. Las comparten el SAC de SB3 y, más adelante, el propio.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from furuta.sim.pendulum_env import RotaryInvertedPendulumEnv

EnvFactory = Callable[[], gym.Env]


def make_train_env_fn(*, monitor_path: Path | None = None, **env_kwargs: Any) -> EnvFactory:
    """Entorno de entrenamiento: los `env_kwargs` van directo al constructor.

    El `Monitor` registra retorno y duración de cada episodio (en un CSV si se
    da `monitor_path`); SB3 lo necesita para sus estadísticas.
    """
    def _init() -> gym.Env:
        env = RotaryInvertedPendulumEnv(**env_kwargs)
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)
    return _init


def make_eval_env_fn(*, firmware_obs_model: bool = True, **env_kwargs: Any) -> EnvFactory:
    """Entorno de evaluación: determinista, sin DR y sin sesgo θ.

    Es la referencia fija con la que se elige el mejor modelo: si evaluara con
    DR, el ruido de un episodio a otro taparía las diferencias entre modelos.
    Por defecto usa el modelo de medición del firmware, que es parte de la
    planta del robot real (no una aleatorización): sin ruido, pero con la misma
    cuantización y antigüedad nominal con la que se entrenó.
    """
    env_kwargs = {
        **env_kwargs,
        "firmware_obs_model": firmware_obs_model,
        "domain_randomization": False,
        "dr_theta_bias_max_rad": 0.0,
    }

    def _init() -> gym.Env:
        return Monitor(RotaryInvertedPendulumEnv(**env_kwargs))
    return _init
