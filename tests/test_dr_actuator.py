"""Aleatorización de dominio, paso 4: envolvente de aceleración, lag y retardo del comando."""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import (
    DR_ACTION_LAG_TAU_RANGE_S,
    DR_MOTOR_ACCEL_RANGE_RAD_S2,
    MAX_ACCEL_RAD_S2,
    RotaryInvertedPendulumEnv,
)

UNO = np.array([1.0])


def _env(**kw) -> RotaryInvertedPendulumEnv:
    """DR activa, sin suavizado de acción (para que la cuenta salga a mano)."""
    kw.setdefault("domain_randomization", True)
    kw.setdefault("action_smooth_window", 1)
    kw.setdefault("dr_motor_accel_range_rad_s2", (MAX_ACCEL_RAD_S2, MAX_ACCEL_RAD_S2))
    kw.setdefault("dr_action_delay_steps_range", (0, 0))
    kw.setdefault("dr_action_lag_tau_range_s", (0.0, 0.0))
    return RotaryInvertedPendulumEnv(**kw)


def _accels(env: RotaryInvertedPendulumEnv, n: int, seed: int = 0) -> list[float]:
    env.reset(seed=seed)
    return [env.step(UNO)[4]["accel_cmd"] for _ in range(n)]


def test_sin_dr_el_actuador_es_neutro() -> None:
    env = RotaryInvertedPendulumEnv(action_smooth_window=1)
    env.reset(seed=0)
    assert env._motor_max_accel_rad_s2 == MAX_ACCEL_RAD_S2
    assert env._action_delay_steps == 0 and env._action_lag_tau_s == 0.0
    assert env.step(UNO)[4]["accel_cmd"] == MAX_ACCEL_RAD_S2


def test_los_rangos_no_hacen_nada_sin_dr() -> None:
    env = _env(
        domain_randomization=False,
        dr_motor_accel_range_rad_s2=(110.0, 110.0),
        dr_action_delay_steps_range=(2, 2),
        dr_action_lag_tau_range_s=(0.05, 0.05),
    )
    assert _accels(env, 1) == [MAX_ACCEL_RAD_S2]


def test_la_envolvente_recorta_la_aceleracion() -> None:
    env = _env(dr_motor_accel_range_rad_s2=(110.0, 110.0))
    # El primer tick la ley P pide 150 (su tope) y el techo del motor en este
    # episodio es 110. Después la velocidad ya subió y la ley pide menos.
    accels = _accels(env, 10)
    assert accels[0] == 110.0
    assert all(abs(a) <= 110.0 for a in accels)


def test_envolvente_lag_y_retardo_por_defecto_caen_en_sus_rangos() -> None:
    env = RotaryInvertedPendulumEnv(domain_randomization=True)
    envolventes, taus = set(), set()
    for seed in range(100):
        env.reset(seed=seed)
        lo, hi = DR_MOTOR_ACCEL_RANGE_RAD_S2
        assert lo <= env._motor_max_accel_rad_s2 <= hi
        lo, hi = DR_ACTION_LAG_TAU_RANGE_S
        assert lo <= env._action_lag_tau_s <= hi
        assert env._action_delay_steps == 0          # la cola viene apagada
        envolventes.add(env._motor_max_accel_rad_s2)
        taus.add(env._action_lag_tau_s)
    assert len(envolventes) > 90 and len(taus) > 90


def test_el_lag_filtra_el_comando_de_aceleracion() -> None:
    # τ = 20 ms = periodo de control → alpha = 0.5: mitad comando actual, mitad el anterior.
    env = _env(dr_action_lag_tau_range_s=(0.02, 0.02))
    env.reset(seed=0)
    a1 = env.step(UNO)[4]
    assert a1["accel_cmd"] == 0.5 * MAX_ACCEL_RAD_S2             # 0.5·150 = 75
    v1 = a1["motor_vel_cmd"]
    crudo2 = min(MAX_ACCEL_RAD_S2, (3.5 - v1) * 50.0)             # lo que pide la ley P
    a2 = env.step(UNO)[4]
    assert abs(a2["accel_cmd"] - (0.5 * a1["accel_cmd"] + 0.5 * crudo2)) < 1e-9


def test_la_cola_retrasa_el_comando_n_ticks() -> None:
    env = _env(dr_action_delay_steps_range=(2, 2))
    env.reset(seed=0)
    primeros = [env.step(UNO)[4] for _ in range(2)]
    assert [i["accel_cmd"] for i in primeros] == [0.0, 0.0]
    assert primeros[-1]["motor_vel_cmd"] == 0.0                   # el motor aún no se enteró
    assert env.step(UNO)[4]["accel_cmd"] == MAX_ACCEL_RAD_S2      # llega el comando del tick 1


def test_lag_y_cola_se_componen_primero_el_filtro_y_luego_la_cola() -> None:
    env = _env(dr_action_lag_tau_range_s=(0.02, 0.02), dr_action_delay_steps_range=(1, 1))
    accels = _accels(env, 3)
    assert accels[0] == 0.0
    assert accels[1] == 75.0            # sale lo filtrado en el tick 1
    assert accels[2] == 112.5           # sale lo filtrado en el tick 2: 0.5·75 + 0.5·150


def test_reset_vacia_el_transporte() -> None:
    env = _env(dr_action_delay_steps_range=(2, 2), dr_action_lag_tau_range_s=(0.02, 0.02))
    _accels(env, 20)                                  # deja filtro y cola con contenido
    assert env._lagged_accel_cmd != 0.0
    env.reset(seed=1)
    assert env._lagged_accel_cmd == 0.0
    assert list(env._accel_cmd_queue) == [0.0, 0.0]
    assert [env.step(UNO)[4]["accel_cmd"] for _ in range(2)] == [0.0, 0.0]


def test_el_sorteo_es_reproducible_con_semilla() -> None:
    def sorteo(seed: int) -> tuple[float, int, float]:
        env = RotaryInvertedPendulumEnv(
            domain_randomization=True, dr_action_delay_steps_range=(0, 2)
        )
        env.reset(seed=seed)
        return env._motor_max_accel_rad_s2, env._action_delay_steps, env._action_lag_tau_s
    assert sorteo(4) == sorteo(4)
    assert sorteo(4) != sorteo(5)


def test_el_retardo_sorteado_cubre_todo_el_rango() -> None:
    env = RotaryInvertedPendulumEnv(
        domain_randomization=True, dr_action_delay_steps_range=(0, 2)
    )
    vistos = set()
    for seed in range(60):
        env.reset(seed=seed)
        vistos.add(env._action_delay_steps)
    assert vistos == {0, 1, 2}


def test_rangos_invalidos_se_rechazan() -> None:
    for kw in (
        {"dr_motor_accel_range_rad_s2": (190.0, 110.0)},
        {"dr_motor_accel_range_rad_s2": (-1.0, 110.0)},
        {"dr_action_delay_steps_range": (-1, 0)},
        {"dr_action_delay_steps_range": (2, 1)},
        {"dr_action_lag_tau_range_s": (-0.01, 0.02)},
    ):
        try:
            RotaryInvertedPendulumEnv(**kw)
        except ValueError:
            continue
        raise AssertionError(f"debió lanzar ValueError con {kw}")


if __name__ == "__main__":
    test_sin_dr_el_actuador_es_neutro()
    test_los_rangos_no_hacen_nada_sin_dr()
    test_la_envolvente_recorta_la_aceleracion()
    test_envolvente_lag_y_retardo_por_defecto_caen_en_sus_rangos()
    test_el_lag_filtra_el_comando_de_aceleracion()
    test_la_cola_retrasa_el_comando_n_ticks()
    test_lag_y_cola_se_componen_primero_el_filtro_y_luego_la_cola()
    test_reset_vacia_el_transporte()
    test_el_sorteo_es_reproducible_con_semilla()
    test_el_retardo_sorteado_cubre_todo_el_rango()
    test_rangos_invalidos_se_rechazan()
    print("OK")
