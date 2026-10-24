"""Verifica que el modelo MJCF reproduce la física esperada del péndulo.

La prueba más importante: el periodo de oscilación libre simulado debe
coincidir con el teórico de amplitud pequeña,
    T₀ = 2π · sqrt(I / (m·g·d))
Se puede correr con pytest o directamente: python tests/test_build_mjcf.py
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from furuta.sim.pendulum_env import (
    DEFAULT_PARAMS_PATH,
    GRAVITY,
    MOTOR_LIMIT_RAD,
    PendulumParams,
    build_mjcf,
)

# Estas pruebas verifican el constructor del MJCF, no un robot concreto: se fija
# el JSON de referencia (fricción baja) para que no dependan del robot por defecto.
REFERENCE_PARAMS_PATH = DEFAULT_PARAMS_PATH.parent / "sysid_params_reference.json"


def _simulate_free_swing(
    params: PendulumParams, amplitude_rad: float = 0.15, duration_s: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """Suelta el péndulo desde `amplitude_rad` con el motor sosteniendo 0.

    Devuelve (tiempos, ángulos del péndulo). Ángulo 0 = colgando.

    Amplitud por defecto 0.15 rad (~8.6°): lo bastante pequeña para que el
    periodo sea el de amplitud pequeña (la corrección elíptica ≈ 0.14 %), y lo
    bastante grande para que la fricción de Coulomb no lo detenga antes de
    tener ciclos suficientes. La fricción seca resta ~2·F_c/(m·g·d) ≈ 5 mrad
    por medio ciclo (decae en línea recta), y detiene el péndulo dentro de una
    banda de ±F_c/(m·g·d) ≈ ±2.6 mrad.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf(params))
    data = mujoco.MjData(model)
    pen_addr = model.joint("pendulum_joint").qposadr[0]

    data.qpos[pen_addr] = amplitude_rad
    data.ctrl[0] = 0.0                      # el servo mantiene el brazo en 0
    mujoco.mj_forward(model, data)

    n = int(duration_s / model.opt.timestep)
    t = np.empty(n)
    y = np.empty(n)
    for i in range(n):
        t[i] = data.time
        y[i] = data.qpos[pen_addr]
        mujoco.mj_step(model, data)
    return t, y


def _mean_period(t: np.ndarray, y: np.ndarray, max_cycles: int = 5) -> float:
    """Periodo medio de los primeros `max_cycles` ciclos, por cruces por cero
    de pendiente negativa.

    Solo se usan los primeros ciclos a propósito: la fricción seca resta
    amplitud en línea recta y, cuando la amplitud baja a unos pocos mrad (la
    banda de ±F_c/(m·g·d)), el periodo medido se alarga (en la simulación,
    del ciclo 9 en adelante) y ya no representa el periodo de amplitud pequeña.
    """
    idx = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    # Interpolación lineal del instante exacto del cruce.
    crossings = t[idx] + (y[idx] / (y[idx] - y[idx + 1])) * (t[idx + 1] - t[idx])
    assert len(crossings) > max_cycles, "muy pocos ciclos para medir el periodo"
    return float(np.mean(np.diff(crossings[: max_cycles + 1])))


def test_periodo_oscilacion_libre_coincide_con_teoria() -> None:
    params = PendulumParams.load(REFERENCE_PARAMS_PATH)
    t, y = _simulate_free_swing(params)

    measured = _mean_period(t, y)
    theory = 2.0 * math.pi * math.sqrt(
        params.pendulum_inertia_kg_m2
        / (params.pendulum_mass_kg * GRAVITY * params.pendulum_com_m)
    )
    rel_err = abs(measured - theory) / theory
    print(f"T simulado = {measured:.4f} s | T teórico = {theory:.4f} s | "
          f"error = {rel_err * 100:.2f} %")
    assert rel_err < 0.01, f"el periodo difiere {rel_err * 100:.2f} % de la teoría"


def test_el_pendulo_se_amortigua() -> None:
    """Con fricción > 0 la amplitud debe decaer; sin energía añadida."""
    params = PendulumParams.load(REFERENCE_PARAMS_PATH)
    t, y = _simulate_free_swing(params, amplitude_rad=0.3, duration_s=10.0)
    first = np.max(np.abs(y[: len(y) // 5]))
    last = np.max(np.abs(y[-len(y) // 5:]))
    assert last < first


def test_limites_de_articulacion_en_radianes() -> None:
    """El rango del motor debe ser ±135° (2.356 rad), no ±2.36°.

    MuJoCo interpreta los ángulos del XML en grados salvo que se declare
    <compiler angle="radian"/>; sin eso el rango quedaba en ±0.0411 rad.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf(PendulumParams.load()))
    low, high = model.joint("motor_joint").range
    assert math.isclose(high, MOTOR_LIMIT_RAD, rel_tol=1e-9)
    assert math.isclose(low, -MOTOR_LIMIT_RAD, rel_tol=1e-9)


if __name__ == "__main__":
    test_periodo_oscilacion_libre_coincide_con_teoria()
    test_el_pendulo_se_amortigua()
    test_limites_de_articulacion_en_radianes()
    print("OK")
