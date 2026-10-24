"""Simetría de espejo: las funciones puras y, sobre todo, que el ENTORNO sea simétrico."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from furuta.sim.pendulum_env import OBS_FRAME_DIM, OBS_HISTORY_LEN, RotaryInvertedPendulumEnv
from furuta.sim.symmetry import (
    FRAME_DIM,
    asymmetry,
    engage_obs,
    mirror_action,
    mirror_obs,
    obs_signs,
    obs_signs_for_dim,
)


def test_el_frame_coincide_con_el_del_entorno() -> None:
    assert FRAME_DIM == OBS_FRAME_DIM


def test_los_signos_voltean_todo_menos_cos_theta() -> None:
    s = obs_signs(OBS_HISTORY_LEN)
    assert s.shape == (OBS_HISTORY_LEN * FRAME_DIM,)
    por_frame = s[:FRAME_DIM]
    assert list(por_frame) == [-1, -1, +1, -1, -1, -1]         # solo cos θ (índice 2) se conserva
    assert np.array_equal(s, np.tile(por_frame, OBS_HISTORY_LEN))


def test_espejar_dos_veces_devuelve_la_observacion() -> None:
    s = obs_signs(4)
    obs = np.random.default_rng(0).normal(size=(5, 24)).astype(np.float32)
    assert np.array_equal(mirror_obs(mirror_obs(obs, s), s), obs)
    a = np.array([[0.3], [-0.7]], dtype=np.float32)
    assert np.array_equal(mirror_action(mirror_action(a)), a)


def test_obs_signs_for_dim_rechaza_tamanos_incoherentes() -> None:
    assert obs_signs_for_dim(24, obs_history_len=4).shape == (24,)
    for obs_dim, k in ((20, 4), (24, 3), (24, 0)):
        try:
            obs_signs_for_dim(obs_dim, obs_history_len=k)
        except ValueError:
            continue
        raise AssertionError(f"debió rechazar obs_dim={obs_dim}, K={k}")


def test_engage_obs_es_su_propio_espejo() -> None:
    e = engage_obs(obs_history_len=4)
    # Punto fijo: motor 0, θ = π (sin ≈ 0), velocidades y acción 0.
    assert np.allclose(mirror_obs(e, obs_signs(4)), e, atol=1e-6)


def test_asymmetry_detecta_politicas_equivariantes_y_no() -> None:
    s = obs_signs(4)
    obs = np.random.default_rng(1).normal(size=(200, 24)).astype(np.float32)
    # Equivariante: π(Ms) = -π(s). Una función impar del estado lo cumple.
    impar = lambda x: np.tanh(x[:, [0, 1, 3, 4, 5]].sum(axis=1, keepdims=True))   # solo canales que cambian
    rep = asymmetry(impar, obs, s)
    assert rep.rms_error < 1e-6 and rep.relative_error < 1e-5
    # Sesgada: siempre empuja a la derecha.
    rep2 = asymmetry(lambda x: np.full((len(x), 1), 0.5), obs, s)
    assert math.isclose(rep2.mean_action, 0.5) and rep2.relative_error > 1.9


# --- El entorno es simétrico: la premisa del aumento por espejo -----------------------

def _par_espejo(**kw) -> tuple[RotaryInvertedPendulumEnv, RotaryInvertedPendulumEnv]:
    """Dos entornos iguales; el segundo parte del estado reflejado del primero."""
    kw.setdefault("dr_theta_bias_max_rad", 0.0)
    e1, e2 = RotaryInvertedPendulumEnv(**kw), RotaryInvertedPendulumEnv(**kw)
    e1.reset(seed=0)
    e2.reset(seed=0)
    # Reflejo del estado inicial: el motor y el péndulo cambian de signo.
    e2.data.qpos[e2._motor_qpos_addr] = -e1.data.qpos[e1._motor_qpos_addr]
    e2.data.qpos[e2._pen_qpos_addr] = -e1.data.qpos[e1._pen_qpos_addr]
    e2._motor_target = float(e2.data.qpos[e2._motor_qpos_addr])
    e2._prev_cmd_pos = e2._motor_target
    e2.data.ctrl[0] = e2._motor_target
    mujoco.mj_forward(e2.model, e2.data)
    if e2.firmware_obs_model:
        e2.seed_measurement_ring()
    e2._obs_history.clear()
    e2._obs()                               # reconstruye el historial con el estado reflejado
    return e1, e2


def _comparar_trayectorias(tol_obs: float, **kw) -> None:
    e1, e2 = _par_espejo(**kw)
    signs = obs_signs(OBS_HISTORY_LEN)
    rng = np.random.default_rng(7)
    peor = 0.0
    for _ in range(100):
        a = rng.uniform(-1, 1, size=1)
        o1, r1, te1, tr1, _ = e1.step(a)
        o2, r2, te2, tr2, _ = e2.step(-a)
        peor = max(peor, float(np.max(np.abs(o2 - mirror_obs(o1, signs)))))
        assert math.isclose(r1, r2, rel_tol=1e-6, abs_tol=1e-6), (r1, r2)
        assert (te1, tr1) == (te2, tr2)
    assert peor < tol_obs, f"la observación espejo se aleja hasta {peor:.2e}"



def test_el_entorno_es_simetrico_sin_modelo_del_firmware() -> None:
    _comparar_trayectorias(1e-6)


def test_el_entorno_es_simetrico_con_modelo_del_firmware() -> None:
    # La cuantización (redondeo al par) es simétrica: el resultado coincide al bit.
    _comparar_trayectorias(1e-6, firmware_obs_model=True)


def test_la_simetria_se_conserva_con_lag_y_retardo_fijos() -> None:
    _comparar_trayectorias(1e-6, action_delay_steps=1, action_lag_tau_s=0.01)


if __name__ == "__main__":
    test_el_frame_coincide_con_el_del_entorno()
    test_los_signos_voltean_todo_menos_cos_theta()
    test_espejar_dos_veces_devuelve_la_observacion()
    test_obs_signs_for_dim_rechaza_tamanos_incoherentes()
    test_engage_obs_es_su_propio_espejo()
    test_asymmetry_detecta_politicas_equivariantes_y_no()
    test_el_entorno_es_simetrico_sin_modelo_del_firmware()
    test_el_entorno_es_simetrico_con_modelo_del_firmware()
    test_la_simetria_se_conserva_con_lag_y_retardo_fijos()
    print("OK")
