"""Modelo de medición del firmware: cuantización, antigüedad y velocidades por diferencias."""

from __future__ import annotations

import math

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from furuta.sim.pendulum_env import (
    DR_OBS_NOISE_STD_POS_RAD,
    DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S,
    FIRMWARE_VEL_WINDOW_S,
    MOTOR_STEP_LSB_RAD,
    OBS_STALENESS_NOMINAL_S,
    PENDULUM_LSB_RAD,
    RotaryInvertedPendulumEnv,
)


def _env(**kw) -> RotaryInvertedPendulumEnv:
    kw.setdefault("firmware_obs_model", True)
    kw.setdefault("dr_theta_bias_max_rad", 0.0)      # aquí no interesa el sesgo θ
    return RotaryInvertedPendulumEnv(**kw)


def _es_multiplo(valor: float, paso: float) -> bool:
    celdas = valor / paso
    return abs(celdas - round(celdas)) < 1e-6


def test_las_posiciones_medidas_estan_cuantizadas() -> None:
    env = _env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, *_ = env.step(rng.uniform(-1, 1, size=1))
        assert _es_multiplo(env._meas_motor_pos, MOTOR_STEP_LSB_RAD)
        assert _es_multiplo(env._meas_pen_phi, PENDULUM_LSB_RAD)
        assert obs[-6] == np.float32(env._meas_motor_pos)     # sin DR no hay ruido encima


def test_las_velocidades_son_diferencias_finitas_de_la_ventana() -> None:
    """vel = k·LSB/ventana, con k entero: de ahí salen los picos de ±1 LSB."""
    env = _env()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    pico_pen = PENDULUM_LSB_RAD / FIRMWARE_VEL_WINDOW_S      # ≈ 0.19 rad/s
    pico_mot = MOTOR_STEP_LSB_RAD / FIRMWARE_VEL_WINDOW_S    # ≈ 0.12 rad/s (1/32)
    assert abs(pico_pen - 0.19) < 0.01 and abs(pico_mot - 0.12) < 0.01
    for _ in range(80):
        env.step(rng.uniform(-1, 1, size=1))
        assert _es_multiplo(env._meas_pen_vel, pico_pen)
        assert _es_multiplo(env._meas_motor_vel, pico_mot)


def test_un_pendulo_quieto_mide_velocidad_cero() -> None:
    env = _env()
    env.reset(seed=0)
    env.data.qpos[env._pen_qpos_addr] = 0.0
    env.data.qvel[env._pen_qvel_addr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    for _ in range(10):
        env.step(np.array([0.0]))
        assert env._meas_pen_vel == 0.0 and env._meas_motor_vel == 0.0


def test_la_antiguedad_retrasa_la_medicion_del_pendulo() -> None:
    fresco = _env(obs_staleness_s=0.0)
    viejo = _env(obs_staleness_s=0.010)
    for e in (fresco, viejo):
        e.reset(seed=3)
        e.data.qpos[e._pen_qpos_addr] = 1.0           # suelto a ~57°
        e.data.qvel[e._pen_qvel_addr] = 0.0
        mujoco.mj_forward(e.model, e.data)
        e.seed_measurement_ring()
    difs, esperadas = [], []
    for _ in range(40):
        fresco.step(np.array([0.0]))
        viejo.step(np.array([0.0]))
        # Con acción 0 el brazo no se mueve: la física es idéntica y los anillos coinciden.
        assert fresco._meas_ring[-1] == viejo._meas_ring[-1]
        # Estructura: la lectura vieja es la muestra de hace 10 substeps, cuantizada.
        muestra = viejo._meas_ring[-1 - 10][1]
        assert viejo._meas_pen_phi == round(muestra / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        difs.append(fresco._meas_pen_phi - viejo._meas_pen_phi)
        esperadas.append(float(fresco.data.qvel[fresco._pen_qvel_addr]) * 0.010)
    # Comportamiento: lo que se pierde es ≈ velocidad × antigüedad.
    assert max(abs(d) for d in difs) > 0.02
    assert np.median(np.abs(np.array(difs) - np.array(esperadas))) < 0.01


def test_la_antiguedad_nominal_sin_dr_y_la_sorteada_con_dr() -> None:
    sin_dr = _env()
    sin_dr.reset(seed=0)
    assert sin_dr._obs_staleness_s == OBS_STALENESS_NOMINAL_S
    assert sin_dr.step(np.array([0.0]))[4]["obs_staleness_s"] == OBS_STALENESS_NOMINAL_S

    fija = _env(obs_staleness_s=0.007)
    fija.reset(seed=0)
    assert fija._obs_staleness_s == 0.007

    # Con DR se sortea en cada episodio dentro del rango (aquí el de producción).
    con_dr = _env(domain_randomization=True, dr_obs_staleness_range_s=(0.002, 0.020))
    vistos = set()
    for seed in range(60):
        con_dr.reset(seed=seed)
        assert 0.002 <= con_dr._obs_staleness_s <= 0.020
        vistos.add(con_dr._obs_staleness_s)
    assert len(vistos) > 50

    pinned = _env(domain_randomization=True, dr_obs_staleness_range_s=(0.015, 0.015))
    pinned.reset(seed=0)
    assert pinned._obs_staleness_s == 0.015


def test_sin_el_modelo_no_hay_antiguedad_ni_medicion() -> None:
    env = RotaryInvertedPendulumEnv(domain_randomization=True)
    env.reset(seed=0)
    info = env.step(np.array([0.0]))[4]
    assert info["obs_staleness_s"] == 0.0 and info["motor_vel_meas"] is None


def test_con_el_modelo_el_ruido_de_velocidad_baja_al_residuo() -> None:
    env = _env(domain_randomization=True)
    e_pos, e_mvel, e_pvel = [], [], []
    for seed in range(4):
        env.reset(seed=seed)
        for _ in range(399):
            obs, *_ = env.step(np.array([0.0]))
            e_pos.append(obs[-6] - env._meas_motor_pos)
            e_mvel.append(obs[-3] - env._meas_motor_vel)
            e_pvel.append(obs[-2] - env._meas_pen_vel)
    for nombre, errores, esperado in [
        ("pos motor", e_pos, DR_OBS_NOISE_STD_POS_RAD),
        ("vel motor", e_mvel, DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S),
        ("vel péndulo", e_pvel, DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S),
    ]:
        sigma = float(np.std(errores))
        assert abs(sigma / esperado - 1.0) < 0.10, f"{nombre}: σ={sigma:.4f} vs {esperado:.4f}"


def test_el_integrador_de_velocidad_sigue_a_la_orden_con_la_correccion() -> None:
    env = _env()
    env.reset(seed=0)
    for paso in range(20):
        info = env.step(np.array([1.0]))[4]
        if paso >= 5:
            assert abs(info["motor_vel_cmd"] - 3.5) < 0.2, info["motor_vel_cmd"]
            assert abs(info["motor_vel_meas"] - 3.5) < 0.4, info["motor_vel_meas"]
    # Quieto, la medición y el integrador se quedan en 0.
    quieto = _env()
    quieto.reset(seed=0)
    for _ in range(50):
        info = quieto.step(np.array([0.0]))[4]
        assert info["motor_vel_cmd"] == 0.0 and info["motor_vel_meas"] == 0.0


def test_el_anillo_sembrado_lee_la_velocidad_inicial_verdadera() -> None:
    env = _env()
    env.reset(seed=0)
    env.data.qvel[env._pen_qvel_addr] = 3.0
    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    assert abs(env._meas_pen_vel - 3.0) < 2 * PENDULUM_LSB_RAD / FIRMWARE_VEL_WINDOW_S


def test_es_reproducible_con_semilla_y_cumple_check_env() -> None:
    def correr() -> np.ndarray:
        env = _env(domain_randomization=True, dr_obs_staleness_range_s=(0.002, 0.020))
        env.reset(seed=5)
        obs = None
        for _ in range(80):
            obs, *_ = env.step(np.array([0.4]))
        return obs
    assert np.array_equal(correr(), correr())
    check_env(_env(domain_randomization=True), warn=True)


def test_argumentos_invalidos_se_rechazan() -> None:
    for kw in ({"obs_staleness_s": -0.001}, {"dr_obs_staleness_range_s": (0.02, 0.002)},
               {"dr_obs_staleness_range_s": (-0.001, 0.01)}):
        try:
            RotaryInvertedPendulumEnv(**kw)
        except ValueError:
            continue
        raise AssertionError(f"debió lanzar ValueError con {kw}")


if __name__ == "__main__":
    test_las_posiciones_medidas_estan_cuantizadas()
    test_las_velocidades_son_diferencias_finitas_de_la_ventana()
    test_un_pendulo_quieto_mide_velocidad_cero()
    test_la_antiguedad_retrasa_la_medicion_del_pendulo()
    test_la_antiguedad_nominal_sin_dr_y_la_sorteada_con_dr()
    test_sin_el_modelo_no_hay_antiguedad_ni_medicion()
    test_con_el_modelo_el_ruido_de_velocidad_baja_al_residuo()
    test_el_integrador_de_velocidad_sigue_a_la_orden_con_la_correccion()
    test_el_anillo_sembrado_lee_la_velocidad_inicial_verdadera()
    test_es_reproducible_con_semilla_y_cumple_check_env()
    test_argumentos_invalidos_se_rechazan()
    print("OK")
