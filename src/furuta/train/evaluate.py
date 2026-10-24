"""Evalúa un modelo entrenado: retorno, % de episodios equilibrados y topes duros.

Uso:
    uv run python -m furuta.train.evaluate runs/<run>/best_model.zip --episodes 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from furuta.sim.pendulum_env import CONTROL_FREQ_HZ, _wrap_pi
from furuta.train.make_env import EnvFactory, make_eval_env_fn, make_train_env_fn

# Un episodio cuenta como "equilibrado" si durante los últimos 2 s (100 pasos
# a 50 Hz) el péndulo nunca se aleja más de 0.2 rad (≈ 11°) de la vertical.
BALANCE_WINDOW_STEPS = 100
BALANCE_THETA_RAD = 0.2

# "Recaptura": tras llegar arriba por primera vez, el péndulo se aleja más de
# 1 rad (≈ 57°) antes de asentarse. Es el péndulo que pasa de largo y da una
# vuelta rápida antes de quedar atrapado: inofensivo en simulación, brusco en
# el robot real.
RECAPTURE_THETA_RAD = 1.0

# Condiciones de despliegue para evaluar la robustez (etapa 3 del currículo):
# DR completa, retardo de 1 tick y lag de hasta 15 ms.
DEPLOY_DELAY_STEPS_RANGE = (1, 1)
DEPLOY_LAG_TAU_RANGE_S = (0.0, 0.015)
DEPLOY_OBS_STALENESS_RANGE_S = (0.002, 0.020)    # cubre los 15.6 ms medidos en el robot


def make_deploy_env_fn(*, firmware_obs_model: bool = True) -> EnvFactory:
    """Entorno con las condiciones de despliegue (DR completa)."""
    return make_train_env_fn(
        domain_randomization=True,
        dr_action_delay_steps_range=DEPLOY_DELAY_STEPS_RANGE,
        dr_action_lag_tau_range_s=DEPLOY_LAG_TAU_RANGE_S,
        firmware_obs_model=firmware_obs_model,
        dr_obs_staleness_range_s=DEPLOY_OBS_STALENESS_RANGE_S,
    )


def evaluate_model(
    model: Any, *, n_episodes: int = 20, seed: int = 0, env_fn: EnvFactory | None = None,
) -> dict[str, float]:
    """Corre `n_episodes` episodios deterministas.

    `model` es cualquier objeto con `predict(obs, deterministic=True)`. Sin
    `env_fn` usa el entorno de evaluación (sin DR); con `make_deploy_env_fn()`
    mide la robustez bajo DR.
    """
    env = (env_fn or make_eval_env_fn())()
    returns, steps, balanced, hard_stops, recaptures = [], [], 0, 0, 0
    tiempos_arriba = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, thetas, done = 0.0, [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            thetas.append(_wrap_pi(info["phi"] - math.pi))
            done = terminated or truncated
        returns.append(total)
        steps.append(len(thetas))
        hard_stops += int(terminated)
        absth = np.abs(thetas)
        tail = absth[-BALANCE_WINDOW_STEPS:]
        balanced += int(len(thetas) >= BALANCE_WINDOW_STEPS and tail.max() < BALANCE_THETA_RAD)
        arriba = np.where(absth < BALANCE_THETA_RAD)[0]
        if len(arriba):
            tiempos_arriba.append(arriba[0] / CONTROL_FREQ_HZ)
            recaptures += int(absth[arriba[0]:].max() > RECAPTURE_THETA_RAD)
    return {
        "episodios": float(n_episodes),
        "retorno_medio": float(np.mean(returns)),
        "retorno_std": float(np.std(returns)),
        "pasos_medios": float(np.mean(steps)),
        "equilibrado_%": 100.0 * balanced / n_episodes,
        "tope_duro_%": 100.0 * hard_stops / n_episodes,
        "recaptura_%": 100.0 * recaptures / n_episodes,
        "tiempo_arriba_s": float(np.mean(tiempos_arriba)) if tiempos_arriba else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", type=Path, help="archivo .zip de SB3")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dr", action="store_true",
                    help="evalúa con DR completa y retardo de despliegue (robustez)")
    ap.add_argument("--no-firmware-obs-model", action="store_true",
                    help="observación directa del estado, sin el modelo del firmware "
                         "(para modelos entrenados sin él)")
    args = ap.parse_args()

    model = SAC.load(args.modelo, device=args.device)
    fw = not args.no_firmware_obs_model
    env_fn = make_deploy_env_fn(firmware_obs_model=fw) if args.dr else make_eval_env_fn(
        firmware_obs_model=fw)
    res = evaluate_model(model, n_episodes=args.episodes, seed=args.seed, env_fn=env_fn)
    for clave, valor in res.items():
        print(f"{clave:15s} {valor:10.1f}")


if __name__ == "__main__":
    main()
