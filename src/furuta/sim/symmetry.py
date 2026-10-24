"""Simetría de espejo del robot (funciones puras, sin dependencia de SB3).

El péndulo de Furuta tiene una sola simetría no trivial: reflejar toda la
máquina respecto al plano vertical que contiene al brazo cuando motor_pos = 0.
El grupo es Z2 = {identidad, espejo}. Bajo el espejo:

    motor_pos -> -motor_pos      theta   -> -theta
    motor_vel -> -motor_vel      pen_vel -> -pen_vel
    acción    -> -acción

Como el frame guarda el ángulo del péndulo como (sin θ, cos θ), `sin` cambia de
signo y `cos` NO. En un frame, el espejo es una diagonal fija de ±1, repetida
para los K frames apilados:

    [motor_pos, sin θ, cos θ, motor_vel, pen_vel, prev_action]
    [   -1        -1     +1       -1        -1        -1     ]

Es una simetría de la planta (geometría reflejada, misma física), de la
recompensa (todos sus términos son pares en esas magnitudes) y de la
distribución de reset (motor_pos y sesgo θ se sortean simétricos; el acimut de
la inclinación es uniforme). Entonces, para cualquier transición (s, a, r, s')
la transición espejo (Ms, -a, r, Ms') es una transición igualmente válida del
MISMO MDP: no es una aproximación.

Una política ideal es *equivariante*, π(Ms) = -π(s), con crítico *invariante*,
Q(Ms, -a) = Q(s, a). SAC no lo impone, así que una política determinista suele
romper la simetría y elegir una dirección de swing-up al azar, distinta en cada
entrenamiento (se ve como un brazo ladeado: lo medimos con `asymmetry()`).

CUIDADO — el punto fijo del espejo. Los estados con motor_pos = 0, θ = ±π (o
0), velocidades cero y prev_action cero son SU PROPIA imagen: una política
exactamente equivariante daría 0 ahí. Ese es justo el estado en que arranca el
robot tras la tara. La simetrización suave (aumento de datos) deja una pequeña
asimetría residual que rompe el empate y no tiene ese problema.
`engage_obs()` construye ese estado para probar cualquier política.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Signos por frame: [motor_pos, sin θ, cos θ, motor_vel, pen_vel, prev_action].
# El índice 2 es cos θ y es el único que NO cambia de signo.
FRAME_SIGNS = (-1.0, -1.0, +1.0, -1.0, -1.0, -1.0)
FRAME_DIM = len(FRAME_SIGNS)


def obs_signs(obs_history_len: int) -> np.ndarray:
    """Vector de ±1 para una observación apilada de `obs_history_len` frames."""
    if int(obs_history_len) < 1:
        raise ValueError(f"obs_history_len debe ser >= 1, recibí {obs_history_len}")
    return np.tile(np.array(FRAME_SIGNS, dtype=np.float32), int(obs_history_len))


def obs_signs_for_dim(obs_dim: int, *, obs_history_len: int) -> np.ndarray:
    """Vector de signos para `obs_dim`, verificando que cuadre con K frames.

    Más seguro que fiarse de una bandera: si el tamaño no es K·6, espejar con la
    disposición equivocada voltearía cos θ y dejaría sin θ, es decir, un estado
    "espejo" que ni siquiera está en la variedad válida. Mejor fallar.
    """
    obs_dim, k = int(obs_dim), int(obs_history_len)
    if k < 1 or obs_dim != k * FRAME_DIM:
        raise ValueError(
            f"obs_dim={obs_dim} no es K·{FRAME_DIM} con K={k}: no es una observación "
            f"apilada de este entorno"
        )
    return obs_signs(k)


def mirror_obs(obs, signs: np.ndarray) -> np.ndarray:
    """Ms: espeja una observación o un lote de ellas."""
    return np.asarray(obs, dtype=np.float32) * signs


def mirror_action(action) -> np.ndarray:
    """-a: la acción es una orden con signo, así que solo cambia de signo."""
    return -np.asarray(action, dtype=np.float32)


def engage_obs(*, obs_history_len: int, theta: float = np.pi) -> np.ndarray:
    """Estado de arranque tras la tara: brazo centrado, péndulo colgando, quieto.

    Es un punto fijo del espejo: una política exactamente equivariante da 0 aquí.
    Cualquier acción distinta de 0 es su dirección de swing-up "cableada".
    """
    frame = [0.0, float(np.sin(theta)), float(np.cos(theta)), 0.0, 0.0, 0.0]
    return np.tile(np.array(frame, dtype=np.float32), int(obs_history_len))


@dataclass(frozen=True)
class AsymmetryReport:
    """Qué tan lejos está una política de π(Ms) = -π(s) en un conjunto de estados."""

    n: int
    rms_error: float      # RMS de π(s) + π(Ms): 0 si es exactamente equivariante
    rms_action: float     # RMS de π(s), para tener escala
    max_error: float
    mean_action: float    # 0 en una política equivariante sobre estados simétricos
    engage_action: float | None = None   # π en el punto fijo del espejo

    @property
    def relative_error(self) -> float:
        """`rms_error` como fracción del RMS de la acción.

        0 = exactamente equivariante. ≈ 1.41 = π(Ms) sin relación con -π(s) (dos
        sorteos independientes de la misma magnitud).
        """
        return float(self.rms_error / self.rms_action) if self.rms_action > 0 else float("nan")


def _acciones_planas(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim > 1 else arr.reshape(-1)


def asymmetry(predict, obs, signs: np.ndarray, *,
              engage: np.ndarray | None = None) -> AsymmetryReport:
    """Mide qué tan equivariante es una política sobre las observaciones `obs`.

    `predict` recibe un lote de observaciones y devuelve acciones; para un modelo
    de SB3: `lambda x: model.predict(x, deterministic=True)[0]`. Lo útil es medir
    sobre estados que la política de verdad visita (un rollout).
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"obs debe tener forma (n, obs_dim), recibí {obs.shape}")
    a = _acciones_planas(predict(obs))
    a_espejo = _acciones_planas(predict(mirror_obs(obs, signs)))
    err = a + a_espejo
    engage_action = None
    if engage is not None:
        engage_action = float(_acciones_planas(predict(engage.reshape(1, -1)))[0])
    return AsymmetryReport(
        n=int(obs.shape[0]),
        rms_error=float(np.sqrt((err ** 2).mean())),
        rms_action=float(np.sqrt((a ** 2).mean())),
        max_error=float(np.abs(err).max()),
        mean_action=float(a.mean()),
        engage_action=engage_action,
    )
