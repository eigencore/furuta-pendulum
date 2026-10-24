"""Aleatorización de dominio, paso 2: jitter del paso de control y ruido de observación."""

from __future__ import annotations

import math

import numpy as np

from furuta.sim import pendulum_env
from furuta.sim.reward import compute_reward
from furuta.sim.pendulum_env import (
    DR_OBS_NOISE_STD_POS_RAD,
    DR_OBS_NOISE_STD_VEL_RAD_S,
    PENDULUM_LSB_RAD,
    RotaryInvertedPendulumEnv,
    _wrap_pi,
)

CERO = np.array([0.0])


def _tiempos_de_paso(env: RotaryInvertedPendulumEnv, n: int) -> list[float]:
    env.reset(seed=0)
    out = []
    for _ in range(n):
        antes = env.data.time
        env.step(CERO)
        out.append(env.data.time - antes)
    return out


def test_sin_dr_el_paso_dura_siempre_20_ms() -> None:
    dts = _tiempos_de_paso(RotaryInvertedPendulumEnv(), 100)
    assert all(abs(dt - 0.020) < 1e-9 for dt in dts)


def test_con_dr_el_paso_dura_19_20_o_21_ms() -> None:
    dts = _tiempos_de_paso(RotaryInvertedPendulumEnv(domain_randomization=True), 400)
    cuentas = {k: sum(abs(dt - k / 1000) < 1e-9 for dt in dts) for k in (19, 20, 21)}
    assert sum(cuentas.values()) == 400, f"duraciones inesperadas: {set(np.round(dts, 6))}"
    # Con ±5 % sobre 20 substeps: ~25 % / 50 % / 25 %. Margen amplio.
    assert all(c > 40 for c in cuentas.values()), cuentas


def test_sin_dr_la_observacion_es_el_estado_verdadero() -> None:
    env = RotaryInvertedPendulumEnv()
    obs, _ = env.reset(seed=0)
    assert obs[-6] == np.float32(env.data.qpos[env._motor_qpos_addr])
    obs, *_ = env.step(CERO)
    assert obs[-6] == np.float32(env.data.qpos[env._motor_qpos_addr])
    assert obs[-3] == np.float32(env.data.qvel[env._motor_qvel_addr])
    assert obs[-2] == np.float32(env.data.qvel[env._pen_qvel_addr])


def test_el_ruido_de_observacion_tiene_la_desviacion_esperada() -> None:
    """Compara lo que ve la política con el estado verdadero de MuJoCo."""
    # Sin sesgo θ: aquí se mide solo el ruido (el sesgo tiene su propio test).
    env = RotaryInvertedPendulumEnv(domain_randomization=True, dr_theta_bias_max_rad=0.0)
    e_mot, e_pen, e_mvel, e_pvel = [], [], [], []
    for seed in range(4):
        env.reset(seed=seed)
        for _ in range(399):
            obs, *_ = env.step(CERO)
            f = obs[-6:]
            e_mot.append(f[0] - env.data.qpos[env._motor_qpos_addr])
            theta_obs = math.atan2(f[1], f[2])
            theta_real = _wrap_pi(env.data.qpos[env._pen_qpos_addr] - math.pi)
            e_pen.append(_wrap_pi(theta_obs - theta_real))
            e_mvel.append(f[3] - env.data.qvel[env._motor_qvel_addr])
            e_pvel.append(f[4] - env.data.qvel[env._pen_qvel_addr])
    # Posición del péndulo: ruido gaussiano + error de cuantización (LSB/√12).
    esperado_pen = math.sqrt(DR_OBS_NOISE_STD_POS_RAD**2 + PENDULUM_LSB_RAD**2 / 12.0)
    for nombre, errores, esperado in [
        ("pos motor", e_mot, DR_OBS_NOISE_STD_POS_RAD),
        ("pos péndulo", e_pen, esperado_pen),
        ("vel motor", e_mvel, DR_OBS_NOISE_STD_VEL_RAD_S),
        ("vel péndulo", e_pvel, DR_OBS_NOISE_STD_VEL_RAD_S),
    ]:
        sigma = float(np.std(errores))
        media = float(np.mean(errores))
        assert abs(sigma / esperado - 1.0) < 0.10, f"{nombre}: σ={sigma:.4f} vs {esperado:.4f}"
        assert abs(media) < 4 * esperado / math.sqrt(len(errores)), f"{nombre}: sesgo {media:.5f}"


def test_el_angulo_del_pendulo_llega_cuantizado_al_lsb() -> None:
    """Sin ruido de posición, el ángulo observado cae exactamente en la rejilla del AS5600."""
    original = pendulum_env.DR_OBS_NOISE_STD_POS_RAD
    pendulum_env.DR_OBS_NOISE_STD_POS_RAD = 0.0
    try:
        env = RotaryInvertedPendulumEnv(domain_randomization=True, dr_theta_bias_max_rad=0.0)
        env.reset(seed=1)
        for _ in range(50):
            obs, *_ = env.step(np.array([0.5]))
            phi_obs = math.atan2(obs[-5], obs[-4]) + math.pi       # θ = phi − π
            celdas = phi_obs / PENDULUM_LSB_RAD
            assert abs(celdas - round(celdas)) < 1e-2, celdas
    finally:
        pendulum_env.DR_OBS_NOISE_STD_POS_RAD = original


def test_con_dr_es_reproducible_con_semilla() -> None:
    def correr() -> np.ndarray:
        env = RotaryInvertedPendulumEnv(domain_randomization=True)
        env.reset(seed=5)
        obs = None
        for _ in range(60):
            obs, *_ = env.step(np.array([0.3]))
        return obs
    assert np.array_equal(correr(), correr())


def test_la_recompensa_usa_el_estado_verdadero_no_el_ruidoso() -> None:
    """La recompensa debe coincidir EXACTO con la fórmula evaluada en el estado de MuJoCo."""
    env = RotaryInvertedPendulumEnv(domain_randomization=True)
    env.reset(seed=2)
    for _ in range(20):
        _, r, *_ = env.step(np.array([0.0]))
        esperado = compute_reward(
            theta=_wrap_pi(env.data.qpos[env._pen_qpos_addr] - math.pi),
            pen_vel=float(env.data.qvel[env._pen_qvel_addr]),
            motor_pos=float(env.data.qpos[env._motor_qpos_addr]),
            motor_vel=float(env.data.qvel[env._motor_qvel_addr]),
            action=0.0,
            weights=env.reward_weights,
        )
        assert math.isclose(r, esperado, rel_tol=1e-12), (r, esperado)


if __name__ == "__main__":
    test_sin_dr_el_paso_dura_siempre_20_ms()
    test_con_dr_el_paso_dura_19_20_o_21_ms()
    test_sin_dr_la_observacion_es_el_estado_verdadero()
    test_el_ruido_de_observacion_tiene_la_desviacion_esperada()
    test_el_angulo_del_pendulo_llega_cuantizado_al_lsb()
    test_con_dr_es_reproducible_con_semilla()
    test_la_recompensa_usa_el_estado_verdadero_no_el_ruidoso()
    print("OK")
