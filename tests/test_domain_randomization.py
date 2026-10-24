"""Aleatorización de dominio, paso 1: parámetros físicos por episodio."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from furuta.sim.pendulum_env import (
    DR_BASE_TILT_MAX_RAD,
    DR_MOTOR_FRICTIONLOSS_RANGE_N_M,
    DR_PENDULUM_COM_FRAC,
    DR_PENDULUM_FRICTION_MULT_RANGE,
    DR_PENDULUM_MASS_FRAC,
    GRAVITY,
    PENDULUM_I_COM_SWING_KG_M2,
    RotaryInvertedPendulumEnv,
)


def _fisica(env: RotaryInvertedPendulumEnv) -> dict[str, float]:
    """Lee del modelo los parámetros que la DR modifica."""
    m, pen, mot = env.model, env._pen_body_id, env._motor_qvel_addr
    return {
        "masa": float(m.body_mass[pen]),
        "com": float(-m.body_ipos[pen, 2]),
        "viscosa": float(m.dof_damping[env._pen_qvel_addr]),
        "coulomb": float(m.dof_frictionloss[env._pen_qvel_addr]),
        "stiction": float(m.dof_frictionloss[mot]),
    }


def test_sin_dr_la_fisica_es_nominal_en_todos_los_episodios() -> None:
    env = RotaryInvertedPendulumEnv()
    nominal = env.params
    for seed in range(5):
        env.reset(seed=seed)
        f = _fisica(env)
        assert f["masa"] == nominal.pendulum_mass_kg
        assert math.isclose(f["com"], nominal.pendulum_com_m)
        assert f["viscosa"] == nominal.pendulum_friction
        assert f["coulomb"] == nominal.pendulum_coulomb
        assert f["stiction"] == 0.0
        assert np.allclose(env.model.opt.gravity, [0.0, 0.0, -GRAVITY])


def test_con_dr_los_parametros_caen_en_sus_rangos() -> None:
    env = RotaryInvertedPendulumEnv(domain_randomization=True)
    nom = env.params
    mult_lo, mult_hi = DR_PENDULUM_FRICTION_MULT_RANGE
    st_lo, st_hi = DR_MOTOR_FRICTIONLOSS_RANGE_N_M
    masas = set()
    for seed in range(100):
        env.reset(seed=seed)
        f = _fisica(env)
        masas.add(f["masa"])
        assert abs(f["masa"] / nom.pendulum_mass_kg - 1.0) <= DR_PENDULUM_MASS_FRAC + 1e-12
        assert abs(f["com"] / nom.pendulum_com_m - 1.0) <= DR_PENDULUM_COM_FRAC + 1e-12
        # Mismo multiplicador para la fricción viscosa y la de Coulomb.
        mult_v = f["viscosa"] / nom.pendulum_friction
        mult_c = f["coulomb"] / nom.pendulum_coulomb
        assert mult_lo - 1e-12 <= mult_v <= mult_hi + 1e-12
        assert math.isclose(mult_v, mult_c, rel_tol=1e-9)
        assert st_lo <= f["stiction"] <= st_hi
        # Inclinación: la gravedad conserva su módulo y se desvía ≤ máx.
        g = env.model.opt.gravity
        assert math.isclose(float(np.linalg.norm(g)), GRAVITY, rel_tol=1e-9)
        assert math.acos(-g[2] / GRAVITY) <= DR_BASE_TILT_MAX_RAD + 1e-12
    assert len(masas) > 90, "la masa debería cambiar casi en cada episodio"


def test_dr_reproducible_con_semilla_y_distinta_entre_episodios() -> None:
    a = RotaryInvertedPendulumEnv(domain_randomization=True)
    b = RotaryInvertedPendulumEnv(domain_randomization=True)
    a.reset(seed=7)
    b.reset(seed=7)
    assert _fisica(a) == _fisica(b)
    primero = _fisica(a)
    a.reset()                       # sin semilla: sigue la secuencia, otro robot
    assert _fisica(a) != primero


def _periodo_libre(env: RotaryInvertedPendulumEnv, ciclos: int = 4) -> float:
    """Periodo del péndulo suelto (sin fricción) con la física ya sorteada."""
    m = env.model
    m.dof_damping[env._pen_qvel_addr] = 0.0
    m.dof_frictionloss[env._pen_qvel_addr] = 0.0
    env.data.qpos[env._pen_qpos_addr] = 0.05        # amplitud pequeña
    env.data.qvel[env._pen_qvel_addr] = 0.0
    mujoco.mj_forward(m, env.data)

    n = int(6.0 / m.opt.timestep)
    t = np.empty(n)
    y = np.empty(n)
    for i in range(n):
        t[i] = env.data.time
        y[i] = env.data.qpos[env._pen_qpos_addr]
        mujoco.mj_step(m, env.data)
    idx = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    cruces = t[idx] + (y[idx] / (y[idx] - y[idx + 1])) * (t[idx + 1] - t[idx])
    assert len(cruces) > ciclos
    return float(np.mean(np.diff(cruces[: ciclos + 1])))


def test_el_periodo_sigue_a_la_masa_y_al_com_sorteados() -> None:
    """Si masa/COM no llegaran a la dinámica, el periodo seguiría siendo el nominal."""
    env = RotaryInvertedPendulumEnv(domain_randomization=True)
    periodos = []
    for seed in range(5):
        env.reset(seed=seed)
        m = float(env.model.body_mass[env._pen_body_id])
        d = float(-env.model.body_ipos[env._pen_body_id, 2])
        i_pivote = m * d * d + PENDULUM_I_COM_SWING_KG_M2
        teoria = 2.0 * math.pi * math.sqrt(i_pivote / (m * GRAVITY * d))
        medido = _periodo_libre(env)
        periodos.append(medido)
        err = abs(medido - teoria) / teoria
        assert err < 0.01, f"seed {seed}: T={medido:.4f} vs teoría {teoria:.4f} ({err*100:.2f} %)"
    assert max(periodos) - min(periodos) > 0.005, "el periodo debería variar entre episodios"


if __name__ == "__main__":
    test_sin_dr_la_fisica_es_nominal_en_todos_los_episodios()
    test_con_dr_los_parametros_caen_en_sus_rangos()
    test_dr_reproducible_con_semilla_y_distinta_entre_episodios()
    test_el_periodo_sigue_a_la_masa_y_al_com_sorteados()
    print("OK")
