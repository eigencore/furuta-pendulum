"""MirrorReplayBuffer: cada transición se guarda junto con su imagen espejo."""

from __future__ import annotations

import numpy as np

from furuta.sim.pendulum_env import OBS_HISTORY_LEN, RotaryInvertedPendulumEnv
from furuta.sim.symmetry import obs_signs
from furuta.train.mirror_buffer import MirrorReplayBuffer


def _buffer(n: int = 20, **kw) -> tuple[MirrorReplayBuffer, RotaryInvertedPendulumEnv]:
    env = RotaryInvertedPendulumEnv()
    kw.setdefault("obs_sign", obs_signs(OBS_HISTORY_LEN))
    buf = MirrorReplayBuffer(n, env.observation_space, env.action_space, device="cpu", **kw)
    return buf, env


def test_cada_add_ocupa_dos_casillas_y_la_segunda_es_el_espejo() -> None:
    buf, env = _buffer()
    signs = obs_signs(OBS_HISTORY_LEN)
    obs, _ = env.reset(seed=0)
    a = np.array([[0.4]], dtype=np.float32)
    nxt, r, term, trunc, info = env.step(a[0])
    buf.add(obs[None], nxt[None], a, np.array([r]), np.array([term]), [info])
    assert buf.pos == 2
    assert np.array_equal(buf.observations[0, 0], obs)
    assert np.array_equal(buf.observations[1, 0], obs * signs)
    assert np.array_equal(buf.next_observations[1, 0], nxt * signs)
    assert np.array_equal(buf.actions[0], a) and np.array_equal(buf.actions[1], -a)
    assert buf.rewards[0] == buf.rewards[1] and buf.dones[0] == buf.dones[1]


def test_se_puede_muestrear_y_el_espejo_conserva_el_lote() -> None:
    buf, env = _buffer(200)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1, 1, size=(1, 1)).astype(np.float32)
        nxt, r, term, trunc, info = env.step(a[0])
        buf.add(obs[None], nxt[None], a, np.array([r]), np.array([term]), [info])
        obs = nxt
    assert buf.pos == 100
    lote = buf.sample(32)
    assert lote.observations.shape == (32, 24) and lote.actions.shape == (32, 1)


def test_rechaza_configuraciones_incoherentes() -> None:
    env = RotaryInvertedPendulumEnv()
    casos = [
        dict(obs_sign=None),                                       # falta el vector de signos
        dict(obs_sign=obs_signs(3)),                               # tamaño equivocado
        dict(obs_sign=obs_signs(OBS_HISTORY_LEN), optimize_memory_usage=True),
    ]
    for kw in casos:
        try:
            MirrorReplayBuffer(10, env.observation_space, env.action_space, device="cpu", **kw)
        except ValueError:
            continue
        raise AssertionError(f"debió lanzar ValueError con {kw}")


if __name__ == "__main__":
    test_cada_add_ocupa_dos_casillas_y_la_segunda_es_el_espejo()
    test_se_puede_muestrear_y_el_espejo_conserva_el_lote()
    test_rechaza_configuraciones_incoherentes()
    print("OK")
