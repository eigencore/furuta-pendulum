"""Prueba de humo del entorno: ¿se comporta como debe antes de entrenar?

1. `check_env` de Gymnasium: formas, tipos, semillas y contrato reset/step.
2. Tres políticas SIN aprendizaje (acción 0, constante +1, aleatoria) durante
   varios episodios: retorno, duración y cómo terminan.

Referencia: colgado y quieto cuesta ≈ -π² ≈ -9.87 por paso, o sea ≈ -3950 por
episodio de 400 pasos. Si los números se alejan mucho, hay un error.

Uso:
    uv run python scripts/probar_entorno.py
    uv run python scripts/probar_entorno.py --episodios 50
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from gymnasium.utils.env_checker import check_env

from furuta.sim.pendulum_env import RotaryInvertedPendulumEnv

Policy = Callable[[np.random.Generator], np.ndarray]

POLITICAS: dict[str, Policy] = {
    "acción 0": lambda rng: np.array([0.0]),
    "acción +1": lambda rng: np.array([1.0]),
    "aleatoria": lambda rng: rng.uniform(-1.0, 1.0, size=1),
}


def correr(policy: Policy, episodios: int) -> dict[str, float]:
    env = RotaryInvertedPendulumEnv()
    rng = np.random.default_rng(0)
    retornos, largos, terminados = [], [], 0
    r_min, r_max = np.inf, -np.inf
    for ep in range(episodios):
        env.reset(seed=ep)
        total, pasos, done = 0.0, 0, False
        while not done:
            _, r, term, trunc, _ = env.step(policy(rng))
            total += r
            pasos += 1
            r_min, r_max = min(r_min, r), max(r_max, r)
            done = term or trunc
        terminados += int(term)
        retornos.append(total)
        largos.append(pasos)
    return {
        "retorno medio": float(np.mean(retornos)),
        "retorno min": float(np.min(retornos)),
        "retorno max": float(np.max(retornos)),
        "pasos medios": float(np.mean(largos)),
        "% tope duro": 100.0 * terminados / episodios,
        "r/paso min": float(r_min),
        "r/paso max": float(r_max),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodios", type=int, default=20)
    args = ap.parse_args()

    print("== check_env de Gymnasium ==")
    check_env(RotaryInvertedPendulumEnv(), skip_render_check=True)
    print("OK: el entorno cumple el contrato.\n")

    print(f"== Políticas sin aprendizaje ({args.episodios} episodios cada una) ==")
    for nombre, policy in POLITICAS.items():
        res = correr(policy, args.episodios)
        print(f"\n[{nombre}]")
        for clave, valor in res.items():
            print(f"  {clave:14s} {valor:10.1f}")


if __name__ == "__main__":
    main()
