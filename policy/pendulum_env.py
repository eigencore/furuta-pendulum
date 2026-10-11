"""Entorno Gymnasium para el péndulo rotatorio invertido, parametrizado a
partir de los ajustes de identificación de sistema (system identification)
guardados en un archivo sysid_params_*.json por cada robot.

Geometria (Furuta pendulum):
    - El brazo gira al rededor de un eje vertical, accionado por un "motor"
    controlado en posicion (que modela AccelStepperr + el dirver como un PD rigido).
    - El péndulo está articulado en el extremo del brazo y oscila libremente en el
    plano vertical perpendicular a la dirección del brazo. En reposo, cuelga
    recto hacia abajo.

Observacion (5-dim):
    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel]
    donde theta = 0 es la posicion vertical hacia arriba (por lo que cos(theta) = 1 en la meta).

Accion (1-dim, en [-1, 1]) - interpretada segun 'action_mode':
    - "accel" (default): aceleracion angular comandada en
    [-max_accel, +max_accel] rad/s**2; se integra hasta una 
    velocidad limitada
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class PendulumParams:
    pendulum_mass_kg: float
    pendulum_com_m: float           # distancia perpendicular del eje de la articulación al COM
    pendulum_inertia_kg_m2: float   # respecto al eje de la articulación (m·d² + I_com_swing; teorema de Steiner)
    pendulum_friction: float        # fricción viscosa; N·m·s
    pendulum_coulomb: float         # torque de fricción de Coulomb (seca); N·m

