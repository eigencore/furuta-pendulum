"""Verifica que las mallas del visor son SOLO visuales: no cambian la física.

Se puede correr con pytest o directamente: python tests/test_visual_model.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from furuta.sim.pendulum_env import PendulumParams
from furuta.sim.visual import build_visual_model


def _free_swing(model: mujoco.MjModel, amplitude_rad: float = 0.8, steps: int = 2000) -> np.ndarray:
    data = mujoco.MjData(model)
    addr = model.joint("pendulum_joint").qposadr[0]
    data.qpos[addr] = amplitude_rad
    data.ctrl[0] = 0.0
    out = np.empty(steps)
    for i in range(steps):
        mujoco.mj_step(model, data)
        out[i] = data.qpos[addr]
    return out


def test_las_mallas_no_cambian_los_parametros_fisicos() -> None:
    params = PendulumParams.load()
    plain = build_visual_model(params, with_meshes=False)
    meshes = build_visual_model(params, with_meshes=True)

    for attr in ("body_mass", "body_inertia", "body_ipos", "body_pos",
                 "dof_damping", "dof_frictionloss", "jnt_range",
                 "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange"):
        assert np.array_equal(getattr(plain, attr), getattr(meshes, attr)), attr


def test_las_mallas_no_colisionan() -> None:
    model = build_visual_model(PendulumParams.load(), with_meshes=True)
    is_mesh = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    assert is_mesh.sum() == 4                       # base, tapa, brazo, péndulo
    assert not model.geom_contype[is_mesh].any()
    assert not model.geom_conaffinity[is_mesh].any()


def test_la_trayectoria_es_identica_con_y_sin_mallas() -> None:
    params = PendulumParams.load()
    a = _free_swing(build_visual_model(params, with_meshes=False))
    b = _free_swing(build_visual_model(params, with_meshes=True))
    assert np.array_equal(a, b)


if __name__ == "__main__":
    test_las_mallas_no_cambian_los_parametros_fisicos()
    test_las_mallas_no_colisionan()
    test_la_trayectoria_es_identica_con_y_sin_mallas()
    print("OK")
