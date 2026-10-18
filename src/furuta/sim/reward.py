"""Recompensa del entorno: costo cuadrático alrededor de la meta.

Meta: péndulo arriba (θ = 0), quieto, brazo centrado y acciones suaves.
Cada término vale 0 en la meta y crece con el cuadrado de la desviación:

    costo  = θ² + k_pen_vel·θ̇² + k_motor_pos·α² + k_motor_vel·α̇² + k_action·a²
    reward = k_alive_offset − costo

(θ: ángulo del péndulo, α: posición del brazo, a: acción en [-1, 1].)

El peso de θ² es 1 por definición; los demás pesos son relativos a él.
Es una función pura: no toca el entorno, así que se prueba con números.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Pesos del costo cuadrático (valores canónicos del original) + offset opcional."""

    k_pen_vel: float = 0.001     # velocidad del péndulo (pequeño: no estorba el swing-up)
    k_motor_pos: float = 0.5     # posición del brazo, en rad (lo mantiene centrado)
    k_motor_vel: float = 0.005   # velocidad del brazo
    k_action: float = 0.20       # tamaño de la acción

    # Constante que se SUMA en cada paso. Con 0 la recompensa es siempre ≤ 0,
    # y como chocar con el tope termina el episodio, "ir al tope" puede salir
    # más barato que seguir intentando (colgado un episodio ≈ -4000; chocar
    # ≈ -200..-500). Un offset ≥ el peor costo por paso (~14) hace la
    # recompensa ≥ 0 y terminar siempre pierde valor. El original recomienda
    # 15.0; el valor por defecto 0 es la recompensa canónica.
    k_alive_offset: float = 0.0


def compute_reward(
    *,
    theta: float,
    pen_vel: float,
    motor_pos: float,
    motor_vel: float,
    action: float,
    weights: RewardWeights,
) -> float:
    """Recompensa de un paso. θ = 0 es péndulo arriba; velocidades en rad/s."""
    cost = (
        theta * theta
        + weights.k_pen_vel * pen_vel * pen_vel
        + weights.k_motor_pos * motor_pos * motor_pos
        + weights.k_motor_vel * motor_vel * motor_vel
        + weights.k_action * action * action
    )
    return float(weights.k_alive_offset - cost)
