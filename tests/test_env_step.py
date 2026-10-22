"""Bloque 4 del entorno: step()."""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import (
    MAX_VELOCITY_RAD_S,
    MOTOR_SAFE_LIMIT_RAD,
    RotaryInvertedPendulumEnv,
)


def test_accion_cero_no_mueve_el_brazo() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    start = env.data.qpos[env._motor_qpos_addr]
    for _ in range(25):
        _, _, term, trunc, info = env.step(np.array([0.0]))
        assert info["motor_vel_cmd"] == 0.0
    assert abs(env.data.qpos[env._motor_qpos_addr] - start) < 2e-3
    assert not term and not trunc


def test_velocidad_ordenada_llega_a_3_5_en_dos_pasos_y_no_pasa() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    _, _, _, _, i1 = env.step(np.array([1.0]))
    _, _, _, _, i2 = env.step(np.array([1.0]))
    assert abs(i1["motor_vel_cmd"] - 3.0) < 1e-9      # 150 rad/s² · 20 ms
    assert abs(i2["motor_vel_cmd"] - MAX_VELOCITY_RAD_S) < 1e-9
    _, _, _, _, i3 = env.step(np.array([5.0]))        # se recorta a 1
    assert i3["motor_vel_cmd"] <= MAX_VELOCITY_RAD_S + 1e-9


def test_objetivo_nunca_supera_el_limite_seguro() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    for _ in range(200):                               # 4 s empujando
        _, _, _, _, info = env.step(np.array([1.0]))
        assert abs(info["motor_target"]) <= MOTOR_SAFE_LIMIT_RAD + 1e-9
    assert abs(info["motor_target"] - MOTOR_SAFE_LIMIT_RAD) < 1e-9
    assert info["motor_vel_cmd"] == 0.0


def test_el_servo_sigue_el_objetivo() -> None:
    """Con el rango de la articulación bien escrito, el error es sub-mrad."""
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    worst = 0.0
    for _ in range(150):
        env.step(np.array([1.0]))
        worst = max(worst, abs(env.data.ctrl[0] - env.data.qpos[env._motor_qpos_addr]))
    assert worst < 2e-3, f"error de seguimiento {worst * 1e3:.2f} mrad"


def test_truncado_a_los_400_pasos_y_tiempo_avanza() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    trunc = False
    for i in range(400):
        _, _, term, trunc, _ = env.step(np.array([0.0]))
        assert not term
        assert trunc == (i == 399)
    assert abs(env.data.time - 8.0) < 1e-6             # 400 · 20 ms


def test_determinista() -> None:
    def run() -> np.ndarray:
        env = RotaryInvertedPendulumEnv()
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        obs = None
        for _ in range(100):
            obs, *_ = env.step(rng.uniform(-1, 1, size=1))
        return obs
    assert np.array_equal(run(), run())


def test_observacion_valida_tras_step() -> None:
    env = RotaryInvertedPendulumEnv()
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(np.array([0.3]))
    assert obs.shape == (24,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert abs(obs[-1] - 0.3) < 1e-6                   # acción previa del frame más nuevo
    assert reward < 0.0                                # colgado: lejos de la meta


if __name__ == "__main__":
    test_accion_cero_no_mueve_el_brazo()
    test_velocidad_ordenada_llega_a_3_5_en_dos_pasos_y_no_pasa()
    test_objetivo_nunca_supera_el_limite_seguro()
    test_el_servo_sigue_el_objetivo()
    test_truncado_a_los_400_pasos_y_tiempo_avanza()
    test_determinista()
    test_observacion_valida_tras_step()
    print("OK")
