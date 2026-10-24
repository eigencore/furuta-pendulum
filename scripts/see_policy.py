"""Muestra una política entrenada (SB3) controlando el péndulo en el visor.

Corre varios episodios seguidos con la política determinista y el entorno de
evaluación (sin DR ni sesgo θ). Con --dr usa el entorno de entrenamiento con
toda la aleatorización, para ver qué tan robusta es.

Uso (en macOS el visor exige `mjpython`):
    uv run mjpython scripts/see_policy.py runs/etapa1/best_model.zip
    uv run mjpython scripts/see_policy.py runs/etapa1/best_model.zip --dr --episodios 5
    uv run python  scripts/see_policy.py runs/etapa1/best_model.zip --sin-visor

El visor dibuja las piezas reales (STL) con las dimensiones NOMINALES; con
--dr la masa y la fricción de cada episodio cambian en la física, no en el dibujo.

Cada segundo simulado imprime el ángulo del péndulo (0 = arriba), la posición
del motor y la acción de la política.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from furuta.sim.pendulum_env import CONTROL_FREQ_HZ, RotaryInvertedPendulumEnv, _wrap_pi
from furuta.sim.visual import build_visual_model


def _linea(env: RotaryInvertedPendulumEnv, paso: int, accion: float) -> str:
    theta = math.degrees(_wrap_pi(float(env.data.qpos[env._pen_qpos_addr]) - math.pi))
    motor = math.degrees(float(env.data.qpos[env._motor_qpos_addr]))
    return (f"t={paso / CONTROL_FREQ_HZ:5.2f} s | theta={theta:+7.1f}° | "
            f"motor={motor:+6.1f}° | acción={accion:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", type=Path, help="archivo .zip de SB3")
    ap.add_argument("--episodios", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true",
                    help="entorno de entrenamiento con DR completa")
    ap.add_argument("--params", type=Path, default=None,
                    help="sysid_params_*.json (por defecto, el de moso)")
    ap.add_argument("--sin-mallas", action="store_true",
                    help="cápsulas simples en lugar de los STL")
    ap.add_argument("--sin-visor", action="store_true",
                    help="no abre ventana: solo imprime números, sin esperar")
    args = ap.parse_args()

    model = SAC.load(args.modelo, device="cpu")
    if args.dr:
        env = RotaryInvertedPendulumEnv(params_path=args.params, domain_randomization=True)
    else:
        env = RotaryInvertedPendulumEnv(params_path=args.params, dr_theta_bias_max_rad=0.0)
    cada = int(CONTROL_FREQ_HZ)            # una línea por segundo simulado

    def episodio(ep: int, al_paso=None) -> None:
        """Corre un episodio; `al_paso` se llama tras cada paso (dibujar y esperar)."""
        obs, _ = env.reset(seed=args.seed + ep)
        if al_paso:
            al_paso(0, 0.0)
        total, paso, terminado, truncado = 0.0, 0, False, False
        print(f"--- episodio {ep + 1}/{args.episodios} (semilla {args.seed + ep}) ---")
        while not (terminado or truncado):
            accion, _ = model.predict(obs, deterministic=True)
            obs, r, terminado, truncado, _ = env.step(accion)
            total += r
            paso += 1
            if al_paso and not al_paso(paso, float(accion[0])):
                return
            if paso % cada == 0 or terminado or truncado:
                print(_linea(env, paso, float(accion[0])))
        print(f"retorno del episodio: {total:.1f}")

    if args.sin_visor:
        for ep in range(args.episodios):
            episodio(ep)
        return

    import mujoco.viewer

    # Modelo con mallas, solo para dibujar: mismas articulaciones que el del
    # entorno, así que basta copiar qpos/qvel en cada paso.
    vmodel = build_visual_model(env.params, with_meshes=not args.sin_mallas)
    vdata = mujoco.MjData(vmodel)
    assert vmodel.nq == env.model.nq and vmodel.nv == env.model.nv

    with mujoco.viewer.launch_passive(vmodel, vdata) as viewer:
        viewer.cam.lookat[:] = [0.03, 0.0, -0.05]
        viewer.cam.distance = 0.55

        def al_paso(paso: int, _accion: float) -> bool:
            vdata.qpos[:] = env.data.qpos
            vdata.qvel[:] = env.data.qvel
            vdata.ctrl[:] = env.data.ctrl
            mujoco.mj_forward(vmodel, vdata)
            viewer.sync()
            # Un paso de control = 20 ms: se espera al reloj de pared.
            t_objetivo = t_inicio + paso / CONTROL_FREQ_HZ
            while viewer.is_running() and time.monotonic() < t_objetivo:
                time.sleep(0.002)
            return viewer.is_running()

        for ep in range(args.episodios):
            t_inicio = time.monotonic()
            episodio(ep, al_paso)
            if not viewer.is_running():
                break
            time.sleep(1.0)                # pausa breve entre episodios


if __name__ == "__main__":
    main()
