"""Abre el modelo del péndulo en el visor interactivo de MuJoCo.

Sirve para ver con tus propios ojos la física de `build_mjcf`: el brazo
sostenido por el servo y el péndulo oscilando libremente.

Uso (en macOS el visor exige `mjpython`, no `python`):
    uv run mjpython scripts/ver_modelo.py
    uv run mjpython scripts/ver_modelo.py --amplitud 1.5     # suelta desde ~86°
    uv run mjpython scripts/ver_modelo.py --sin-mallas       # cápsulas simples
    uv run mjpython scripts/ver_modelo.py --params mediciones/sysid_params_<robot>.json

Por defecto muestra las piezas reales (STL de model/meshes/), solo como dibujo:
no afectan a la física ni al entrenamiento.

En el visor:
    - Arrastra con el botón izquierdo para rotar la cámara; rueda para zoom.
    - Doble clic sobre el péndulo y Ctrl + botón derecho arrastrando: le aplica
      una fuerza (para empujarlo y ver cómo reacciona).
    - Espacio pausa; Backspace reinicia la simulación.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from furuta.sim.pendulum_env import PendulumParams
from furuta.sim.visual import build_visual_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplitud", type=float, default=1.0,
                    help="ángulo inicial del péndulo en rad (0 = colgando)")
    ap.add_argument("--params", type=Path, default=None,
                    help="sysid_params_*.json (por defecto, el de referencia)")
    ap.add_argument("--sin-mallas", action="store_true",
                    help="usar las cápsulas simples en lugar de los STL reales")
    args = ap.parse_args()

    params = PendulumParams.load(args.params)
    # Con mallas reales (STL, solo visuales) o con las cápsulas simples de
    # build_mjcf. La física es idéntica en ambos casos.
    model = build_visual_model(params, with_meshes=not args.sin_mallas)
    data = mujoco.MjData(model)

    # Estado inicial: péndulo desplazado `amplitud` rad; el brazo, en 0.
    data.qpos[model.joint("pendulum_joint").qposadr[0]] = args.amplitud
    data.ctrl[0] = 0.0          # objetivo del servo del motor: 0 rad
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # El robot mide unos 20 cm de alto (base bajo el brazo, péndulo
        # colgando): se centra la cámara en la zona brazo + péndulo.
        viewer.cam.lookat[:] = [0.03, 0.0, -0.05]
        viewer.cam.distance = 0.55

        # Se avanza la física en tiempo real: cada vuelta ejecuta los pasos
        # de 1 ms que corresponden al tiempo de reloj transcurrido.
        t_wall = time.monotonic()
        while viewer.is_running():
            elapsed = time.monotonic() - t_wall
            while data.time < elapsed:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()
