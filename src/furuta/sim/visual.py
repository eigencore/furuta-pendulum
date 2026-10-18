"""Modelo MuJoCo con las mallas reales (STL) de las piezas, SOLO para visualizar.

`build_mjcf` dibuja el robot con cápsulas porque la física solo necesita masa,
COM, inercia y ejes de giro. Este módulo toma ese mismo modelo y le añade, como
geometrías puramente visuales, los STL que imprimiste (base, tapa, brazo y
péndulo), con las transformaciones que define `model/model.urdf`.

La física NO cambia (comprobado en tests/test_visual_model.py):
  - las mallas tienen contype=conaffinity=0, así que no colisionan;
  - los cuerpos móviles tienen un <inertial> explícito, por lo que MuJoCo no
    calcula su masa/inercia a partir de las geometrías;
  - el entrenamiento sigue usando `build_mjcf` (ligero): los STL pesan ~21 MB.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from furuta.sim.pendulum_env import PendulumParams, REPO_ROOT, build_mjcf

MESH_DIR = REPO_ROOT / "model" / "meshes"

# Los STL están diseñados en milímetros; MuJoCo trabaja en metros.
_MM_TO_M = 0.001

# Colores del URDF (material enclosure / arm_blue / pendulum_red).
_ENCLOSURE_RGBA = (0.25, 0.25, 0.28, 1.0)
_ARM_RGBA = (0.4, 0.6, 0.8, 1.0)
_PENDULUM_RGBA = (0.8, 0.4, 0.4, 1.0)

# --- Posiciones relativas, tomadas de model/model.urdf ----------------------
# El URDF coloca la articulación del péndulo en (0.062, 0, 0.014) del marco del
# brazo: 14 mm "por encima" del origen de la malla del brazo (a media altura de
# su alojamiento del rodamiento) y a 62 mm del eje del motor.
# En el modelo de física, el eje del péndulo es la recta y = z = 0 (eje x) y su
# pivote está a ARM_LENGTH_M = 65 mm. Para que la malla del brazo y la del
# péndulo queden alineadas con ESE eje se desplaza lo visual:
_PENDULUM_BORE_Z_M = 0.014    # altura del eje del péndulo sobre el origen de la malla del brazo
_URDF_PIVOT_X_M = 0.062       # alcance del brazo en el URDF
_BASE_TO_ARM_Z_M = 0.075      # altura de la articulación del brazo sobre la base
_BASE_TO_LID_Z_M = 0.070      # altura de la tapa sobre la base


def _quat_wxyz(rotation: Rotation) -> list[float]:
    """Cuaternión en el orden de MuJoCo (w, x, y, z); scipy usa (x, y, z, w)."""
    x, y, z, w = rotation.as_quat()
    return [w, x, y, z]


def build_visual_model(params: PendulumParams, with_meshes: bool = True) -> mujoco.MjModel:
    """Compila el modelo de `build_mjcf`; con `with_meshes` le añade los STL visuales."""
    spec = mujoco.MjSpec.from_string(build_mjcf(params))
    if not with_meshes:
        return spec.compile()

    # Las cápsulas de relleno se ocultan (grupo 4: el visor no lo muestra por
    # defecto) para que no se vean dentro de las piezas reales.
    for name in ("arm_visual", "pendulum_visual", "pendulum_tip"):
        spec.geom(name).group = 4

    for name in ("base", "lid", "arm", "pendulum"):
        spec.add_mesh(name=name, file=str(MESH_DIR / f"{name}.stl"),
                      scale=[_MM_TO_M] * 3)

    def add_mesh_geom(body, mesh, rgba, pos, rotation: Rotation | None = None) -> None:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, rgba=list(rgba),
            pos=list(pos),
            quat=_quat_wxyz(rotation) if rotation is not None else [1, 0, 0, 0],
            contype=0, conaffinity=0,        # solo visual: sin colisiones
        )

    # Brazo: marco del URDF = marco del cuerpo "arm" (eje del motor en z, brazo
    # a lo largo de +x). Se baja 14 mm para que su alojamiento del rodamiento
    # quede sobre el eje del péndulo (y = z = 0).
    arm_pos = (0.0, 0.0, -_PENDULUM_BORE_Z_M)
    add_mesh_geom(spec.body("arm"), "arm", _ARM_RGBA, arm_pos)

    # Péndulo: el URDF aplica rpy = (π, 0, −π/2) (ejes fijos X-Y-Z) para llevar
    # el eje del cubo de la malla (su y) al eje de giro (x) y su varilla a −z,
    # de modo que cuelgue con el ángulo de articulación en 0.
    # OJO: en scipy, minúsculas ("xyz") = ejes FIJOS (extrínsecos), que es la
    # convención rpy del URDF; las mayúsculas ("XYZ") son intrínsecas y darían
    # el cubo del péndulo del lado equivocado del eje.
    # Además se desplaza −3 mm en x (visual): del pivote del modelo (65 mm) al
    # del URDF (62 mm). Es un desplazamiento A LO LARGO del eje de giro, así que
    # no cambia cómo se ve el balanceo.
    pen_rot = Rotation.from_euler("xyz", [math.pi, 0.0, -math.pi / 2])
    pen_pos = (_URDF_PIVOT_X_M - 0.065, 0.0, 0.0)
    add_mesh_geom(spec.body("pendulum"), "pendulum", _PENDULUM_RGBA, pen_pos, pen_rot)

    # Base y tapa: en el URDF el marco del brazo es el de la base girado −90° en
    # z (base_to_arm). El modelo de física pone el brazo a lo largo de +x con el
    # motor en 0, así que la base se gira +90° para que cuadren. En z, la base
    # queda 75 mm por debajo de la articulación del brazo (más los 14 mm del
    # desplazamiento visual), y la tapa 70 mm sobre la base.
    base_rot = Rotation.from_euler("z", math.pi / 2)
    base_z = -_PENDULUM_BORE_Z_M - _BASE_TO_ARM_Z_M
    world = spec.worldbody
    add_mesh_geom(world, "base", _ENCLOSURE_RGBA, (0.0, 0.0, base_z), base_rot)
    add_mesh_geom(world, "lid", _ENCLOSURE_RGBA,
                  (0.0, 0.0, base_z + _BASE_TO_LID_Z_M), base_rot)

    return spec.compile()
