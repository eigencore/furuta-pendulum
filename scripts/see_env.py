"""Ejecuta `RotaryInvertedPendulumEnv` con un patrón de acciones y lo muestra.

Sirve para ver el entorno completo en movimiento (ley P, integración, servo)
y comprobar cuánto se aleja el motor de su objetivo `ctrl[0]`.

Uso (en macOS el visor exige `mjpython`):
    uv run mjpython scripts/ver_entorno.py
    uv run mjpython scripts/ver_entorno.py --patron constante --duracion 4
    uv run python  scripts/ver_entorno.py --sin-visor        # solo números

Patrones: bangbang (±1 cada 10 pasos), constante (+1), seno (0.5 Hz), cero.

El visor dibuja las piezas reales (STL), pero la física la calcula el entorno
con su modelo ligero: el visor solo copia el estado (qpos/qvel).

Cada segundo simulado imprime:
    objetivo = ctrl[0]        posición a la que empuja el servo
    pos      = posición real del motor
    error    = objetivo − pos (mrad)
    v_cmd    = velocidad ordenada;  v_real = velocidad medida en MuJoCo
    theta    = ángulo del péndulo (0 = arriba, ±180° = colgando)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from furuta.sim.pendulum_env import CONTROL_FREQ_HZ, RotaryInvertedPendulumEnv
from furuta.sim.visual import build_visual_model


def _accion(patron: str, paso: int) -> float:
    if patron == "bangbang":
        return 1.0 if (paso // 10) % 2 == 0 else -1.0
    if patron == "constante":
        return 1.0
    if patron == "seno":
        return math.sin(2.0 * math.pi * 0.5 * paso / CONTROL_FREQ_HZ)
    return 0.0


def _linea(env: RotaryInvertedPendulumEnv, paso: int) -> str:
    pos = float(env.data.qpos[env._motor_qpos_addr])
    objetivo = float(env.data.ctrl[0])
    v_real = float(env.data.qvel[env._motor_qvel_addr])
    phi = float(env.data.qpos[env._pen_qpos_addr])
    theta = math.degrees(math.remainder(phi - math.pi, 2.0 * math.pi))
    return (f"t={paso / CONTROL_FREQ_HZ:5.2f} s | objetivo={objetivo:+.4f} rad | "
            f"pos={pos:+.4f} rad | error={(objetivo - pos) * 1e3:+7.2f} mrad | "
            f"v_cmd={env._motor_vel:+.2f} v_real={v_real:+.2f} rad/s | "
            f"theta={theta:+7.1f}°")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patron", choices=["bangbang", "constante", "seno", "cero"],
                    default="bangbang")
    ap.add_argument("--duracion", type=float, default=8.0, help="segundos simulados")
    ap.add_argument("--params", type=Path, default=None,
                    help="sysid_params_*.json (por defecto, el de referencia)")
    ap.add_argument("--sin-mallas", action="store_true",
                    help="cápsulas simples en lugar de los STL")
    ap.add_argument("--sin-visor", action="store_true",
                    help="no abre ventana: solo imprime números, sin esperar")
    args = ap.parse_args()

    env = RotaryInvertedPendulumEnv(params_path=args.params)
    env.reset(seed=0)
    n_pasos = int(args.duracion * CONTROL_FREQ_HZ)
    cada = int(CONTROL_FREQ_HZ)            # una línea por segundo simulado

    if args.sin_visor:
        for paso in range(n_pasos):
            _, _, terminado, truncado, _ = env.step(np.array([_accion(args.patron, paso)]))
            if paso % cada == 0 or terminado or truncado:
                print(_linea(env, paso + 1))
            if terminado or truncado:
                break
        return

    import mujoco.viewer

    # Modelo con mallas, solo para dibujar. Tiene las mismas articulaciones
    # que el del entorno, así que basta copiar qpos/qvel en cada paso.
    vmodel = build_visual_model(env.params, with_meshes=not args.sin_mallas)
    vdata = mujoco.MjData(vmodel)
    assert vmodel.nq == env.model.nq and vmodel.nv == env.model.nv

    def sincronizar() -> None:
        vdata.qpos[:] = env.data.qpos
        vdata.qvel[:] = env.data.qvel
        vdata.ctrl[:] = env.data.ctrl
        mujoco.mj_forward(vmodel, vdata)

    sincronizar()
    with mujoco.viewer.launch_passive(vmodel, vdata) as viewer:
        viewer.cam.lookat[:] = [0.03, 0.0, -0.05]
        viewer.cam.distance = 0.55

        paso = 0
        t_inicio = time.monotonic()
        while viewer.is_running() and paso < n_pasos:
            # Un paso de control = 20 ms: se espera al reloj de pared.
            if time.monotonic() - t_inicio < paso / CONTROL_FREQ_HZ:
                time.sleep(0.002)
                continue
            _, _, terminado, truncado, _ = env.step(np.array([_accion(args.patron, paso)]))
            sincronizar()
            viewer.sync()
            paso += 1
            if paso % cada == 0 or terminado or truncado:
                print(_linea(env, paso))
            if terminado or truncado:
                break


if __name__ == "__main__":
    main()
