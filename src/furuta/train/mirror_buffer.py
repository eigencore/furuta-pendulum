"""Buffer de repetición de SB3 que guarda cada transición Y su imagen espejo.

Método DUP (Abdolhosseini et al., "On Learning Symmetric Locomotion", 2019).
Para SAC es exacto, no aproximado: (Ms, -a, r, Ms') es una transición genuina del
mismo MDP, con la misma recompensa (ver `furuta.sim.symmetry` y los tests que
comprueban que el entorno es simétrico). El buffer cubre las dos mitades del
espacio de estados en lugar de solo la que la política prefiere.

Cada `add()` ocupa DOS casillas, así que un `buffer_size` guarda la mitad de
experiencia real; quien lo use debe duplicarlo. `optimize_memory_usage` se
rechaza porque guarda next_obs en la casilla siguiente, y las escrituras
intercaladas del espejo la corromperían.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

from furuta.sim.symmetry import mirror_action, mirror_obs


class MirrorReplayBuffer(ReplayBuffer):
    def __init__(self, *args, obs_sign: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError(
                "MirrorReplayBuffer no admite optimize_memory_usage (next_obs comparte la "
                "casilla siguiente, que las escrituras intercaladas del espejo corromperían)"
            )
        if obs_sign is None:
            raise ValueError(
                "MirrorReplayBuffer necesita obs_sign: constrúyelo con "
                "symmetry.obs_signs_for_dim(obs_dim, obs_history_len=K)"
            )
        sign = np.asarray(obs_sign, dtype=np.float32)
        esperado = int(np.prod(self.obs_shape))
        if sign.shape != (esperado,):
            raise ValueError(
                f"obs_sign tiene forma {sign.shape}, se esperaba ({esperado},) para que "
                "coincida con el espacio de observación del buffer"
            )
        self.obs_sign = sign

    def add(self, obs, next_obs, action, reward, done, infos):  # noqa: D102
        super().add(obs, next_obs, action, reward, done, infos)
        super().add(
            mirror_obs(obs, self.obs_sign),
            mirror_obs(next_obs, self.obs_sign),
            mirror_action(action),
            reward,
            done,
            infos,
        )
