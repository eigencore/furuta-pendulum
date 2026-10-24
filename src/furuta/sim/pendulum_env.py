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

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from furuta.sim.reward import RewardWeights, compute_reward
from furuta.sim.pendulum_geometry import (
    PENDULUM_COM_M,
    PENDULUM_I_COM_SWING_KG_M2,
    PENDULUM_MASS_KG,
)


# Raíz del repositorio. Disposición:
#   <repo>/src/furuta/sim/pendulum_env.py   <- este archivo
#   <repo>/mediciones/sysid_params_*.json
# parents[0] = sim, [1] = furuta, [2] = src, [3] = <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]

# Valor por defecto cuando no se da `params_path`. Lleva el nombre del robot
# que describe, en lugar de un genérico `sysid_params.json`, para que en una
# configuración con varios robots no se entrene en silencio con el robot que
# se haya medido último. Para cualquier otro robot, pasa `--params-path`.
DEFAULT_PARAMS_PATH = REPO_ROOT / "mediciones" / "sysid_params_moso.json"

# --- Límites del motor -----------------------------------------------------

# Tope duro de la articulación del motor. Coincide con el límite mecánico
# del saliente de la tapa (±135°), pero se limita la política a ±125° para
# que nunca *ordene* golpear el tope.
MOTOR_LIMIT_RAD = math.radians(135.0)

# Penalización que se resta al chocar con el tope duro (termina el episodio).
HARD_STOP_PENALTY = 5.0
MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)

# --- Geometría del brazo ---------------------------------------------------

# Medida el 2026-05-02 contra el CAD de Onshape y una báscula de cocina. El
# brazo mide 65 mm desde el eje del stepper hasta la articulación del péndulo
# (donde un solo rodamiento 608 sostiene el eje del eslabón del péndulo; la
# reconstrucción quitó el segundo rodamiento del lado del eje del motor).
# Masa total 30 g; el COM está a 35 mm del eje del motor, algo más allá de la
# mitad del brazo porque el rodamiento que queda está en el extremo del péndulo.
#
# Nota: el URDF dice 0.026 kg y ~62 mm de alcance, pero nada lee el URDF para
# el brazo. Cambiarlo cambia la planta con la que se entrenó la política: es
# una decisión de modelo, no un refactor. Pendiente: medir el brazo propio.
ARM_LENGTH_M = 0.065     # m, eje del motor -> articulación del péndulo
ARM_MASS_KG = 0.030      # kg
ARM_COM_M = 0.035        # m, centro de masa medido desde el eje del motor

GRAVITY = 9.81           # m/s²

# Resolución del encoder AS5600 (12 bits): un LSB = 2π/4096 rad ≈ 0.088°.
PENDULUM_LSB_RAD = 2.0 * math.pi / 4096.0

# Modelo de medición del firmware (LowLevelServer.ino). El firmware guarda un
# anillo de muestras del encoder a 500 Hz; GET_STATE devuelve las posiciones de
# la muestra más nueva y las velocidades como (nueva − vieja)/Δt sobre una
# ventana de 5 muestras (8 ms). La posición del motor es su contador de pasos;
# la del péndulo, el AS5600 de 12 bits. Esas diferencias finitas de posiciones
# CUANTIZADAS reproducen, por el mismo mecanismo, los picos de velocidad del
# robot: ±1 LSB del péndulo en la ventana ≈ 0.19 rad/s y ±1 paso del motor a
# 1/32 ≈ 0.12 rad/s (los ~±0.4 rad/s medidos con el péndulo quieto).
#
# MOTOR_MICROSTEPS debe coincidir con MICROSTEPS de los sketches: fija qué tan
# gruesa es la cuantización de la posición del motor.
MOTOR_MICROSTEPS = 32
MOTOR_STEP_LSB_RAD = 2.0 * math.pi / (200.0 * MOTOR_MICROSTEPS)
FIRMWARE_VEL_WINDOW_S = 0.008

# Antigüedad de la lectura cuando el host actúa: edad de la muestra del encoder
# (0–2 ms a 500 Hz) + ida y vuelta serie del GET_STATE. Sin DR queda fija en el
# valor nominal; con DR se sortea por episodio. El rango de producción llega a
# 20 ms porque el robot autónomo mide 15.6 ms; el rango por defecto de abajo
# NO los contiene y hay que ampliarlo al entrenar.
OBS_STALENESS_NOMINAL_S = 0.004
DR_OBS_STALENESS_RANGE_S = (0.002, 0.010)

# --- Control ---------------------------------------------------------------

# Frecuencia de control canónica del robot y valor por defecto del entorno.
# Todo lo que quiera *declarar* la frecuencia de despliegue debe leer esta
# constante en lugar de repetirla (el wizard llegó a tener 35 Hz escritos a
# mano y siguió imprimiéndolos mucho después de que el sistema pasara a 50).
CONTROL_FREQ_HZ = 50.0   # Hz  -> un paso de control cada 20 ms

# Modo de acción "velocidad" (el de producción): la acción en [-1, 1] se
# escala a una velocidad objetivo en [-3.5, +3.5] rad/s.
MAX_VELOCITY_RAD_S = 3.5

# Tope de la aceleración angular comandada al motor.
# Subió de 100 tras ver, en el primer despliegue en modo aceleración, que la
# política saturaba su comando en ±99 una y otra vez: necesitaba más autoridad.
# 150 es deliberadamente conservador para dejar margen contra salto de pasos.
# TODO: el original justifica el margen con "100 kpasos/s² ≈ 393 rad/s²", pero
# ese equivalente corresponde a 1/8 de micropaso; con 1/32 (el valor actual del
# firmware) 100 kpasos/s² serían ≈ 98 rad/s². Verificar contra el firmware.
MAX_ACCEL_RAD_S2 = 150.0

# --- Aleatorización de dominio (DR) ------------------------------------------
# Se activa con `domain_randomization=True`. Cada episodio se sortea un robot
# distinto alrededor de los valores nominales (sysid + URDF), para que la
# política no se sobreajuste a un solo conjunto de números. Los rangos cubren
# lo medido con algo de margen; no deben incluir regímenes imposibles.

DR_PENDULUM_MASS_FRAC = 0.10      # masa del péndulo: nominal × (1 ± 0.10)
DR_PENDULUM_COM_FRAC = 0.10       # distancia al COM: nominal × (1 ± 0.10)

# Multiplicador de la fricción del péndulo. Es UNO solo y se aplica a la
# viscosa y a la de Coulomb: ambas salen del mismo rodamiento (grasa,
# temperatura, asiento de las bolas) y varían juntas.
DR_PENDULUM_FRICTION_MULT_RANGE = (0.5, 2.0)

# Fricción seca del joint del motor (N·m). El stepper real tiene un par de
# detención que el actuador de posición no captura. El 0 está dentro del rango
# para que aparezcan episodios sin stiction.
DR_MOTOR_FRICTIONLOSS_RANGE_N_M = (0.0, 0.005)

# Inclinación de la base: la mesa no está perfectamente nivelada. Se modela
# girando el vector de gravedad (ángulo en [0, máx], dirección en [0, 2π)).
# 0.017 rad ≈ 1°. Mueve la vertical VERDADERA según la posición del brazo.
DR_BASE_TILT_MAX_RAD = 0.017

# Jitter del paso de control: cada tick, el número de substeps de física se
# multiplica por (1 ± 0.05) (a 50 Hz: 19, 20 o 21 substeps en vez de 20 fijos).
# Según el autor es la DR MÁS valiosa: con tiempos estrictos SAC cae en un
# "atractor de corrección activa" (el motor serrucha ±0.5 rad aun balanceado);
# con jitter encuentra el atractor "calmo de acción mínima", que es el que
# funciona en el robot real.
DR_CONTROL_DT_JITTER_FRAC = 0.05

# Ruido de la observación (solo con DR). Imita el jitter de las diferencias
# finitas y el ruido del encoder medidos en el robot. La velocidad pesa más:
# el firmware la calcula como (nuevo − viejo)/Δt sobre ~8 ms de muestras de
# 12 bits, y ±1 LSB ya produce picos de ~0.2–0.5 rad/s con el péndulo quieto.
DR_OBS_NOISE_STD_POS_RAD = 0.005     # σ en la posición del motor y del péndulo
DR_OBS_NOISE_STD_VEL_RAD_S = 0.15    # σ en las velocidades

# Con el modelo del firmware encendido los picos de velocidad salen del
# mecanismo (diferencias finitas cuantizadas), y el ruido gaussiano baja a un
# residuo que cubre lo que el mecanismo no explica (jitter de I²C y de la
# hora de muestreo). Dejar 0.15 encima contaría dos veces el mismo ruido.
DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S = 0.05

# Sesgo del cero del péndulo, sorteado por episodio. Al arrancar, el firmware
# toma como cero el ángulo en que quedó el péndulo, que por la fricción seca
# solo está a ±F_c/(m·g·d) ≈ ±2° de la vertical colgando. Sin esta DR la
# política aprende a llevar el θ OBSERVADO a 0; en el robot eso es llevar el θ
# real a −sesgo, la gravedad deja un par restaurador constante y nunca logra
# quedarse quieta. Solo afecta a la OBSERVACIÓN (la física y la recompensa no).
# 0.05 rad ≈ 2.9° cubre la banda medida con margen. A diferencia del resto de
# la DR, está activo por defecto aunque `domain_randomization` sea False; la
# evaluación lo apaga con `dr_theta_bias_max_rad=0.0`.
DR_THETA_BIAS_MAX_RAD = 0.05

# Envolvente de aceleración del motor, sorteada por episodio: el techo REAL
# del stepper depende de la carga y de su margen de par. Rodea al tope de la
# ley P (MAX_ACCEL_RAD_S2 = 150) para entrenar con episodios por encima y por
# debajo; si solo hubiera valores menores, la política nunca tendría la
# autoridad completa.
DR_MOTOR_ACCEL_RANGE_RAD_S2 = (110.0, 190.0)

# Transporte del comando: lo que pasa entre que el host calcula la aceleración
# y el motor la ejecuta (laptop → USB → Arduino → ISR del stepper). Se modela
# en el comando de ACELERACIÓN, que es lo que de verdad viaja por el cable
# (la ley P corre en el host, sin retardo). Dos piezas que se componen:
#  - cola de ticks enteros: apagada por defecto (0, 0). El currículo la
#    enciende en las etapas 2 y 3 (con el retardo total medido, ≈ 1 tick).
#  - filtro de primer orden con constante τ: retardo fraccionario. Con
#    τ = periodo de control sale "mitad comando actual, mitad el anterior",
#    que es lo medido en el robot.
DR_ACTION_DELAY_STEPS_RANGE = (0, 0)
DR_ACTION_LAG_TAU_RANGE_S = (0.0, 0.030)


@dataclass(frozen=True)
class PendulumParams:
    pendulum_mass_kg: float
    pendulum_com_m: float           # distancia perpendicular del eje de la articulación al COM
    pendulum_inertia_kg_m2: float   # respecto al eje de la articulación (m·d² + I_com_swing; teorema de Steiner)
    pendulum_friction: float        # fricción viscosa; N·m·s
    pendulum_coulomb: float         # torque de fricción de Coulomb (seca); N·m

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PendulumParams":
        """Construye los parámetros a partir de un sysid_params_*.json propio de cada
        robot. La masa, el COM y la inercia respecto al pivote salen del URDF (a través
        de `pendulum_geometry`): son constantes geométricas del cuerpo del péndulo, no
        mediciones de cada robot. Solo los términos de fricción (que dependen de los
        rodamientos, la grasa y la temperatura) vienen de la identificación de sistema.
        """
        path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
        with open(path) as f:
            doc = json.load(f)

        # El bloque "pendulum.derived" lo escribe sysid_wizard en su fase `fit`.
        # Si falta una clave se falla con un mensaje claro: asumir 0 en silencio
        # produciría una simulación que "funciona" pero con la física mal.
        try:
            pen = doc["pendulum"]["derived"]
            viscous = float(pen["viscous_friction_N_m_s"])
            coulomb = float(pen["coulomb_friction_N_m"])
        except KeyError as e:
            raise ValueError(
                f"{path}: falta la clave {e} en pendulum.derived. "
                "¿Es un sysid_params_*.json generado por sysid_wizard?"
            ) from e

        return cls(
            pendulum_mass_kg=PENDULUM_MASS_KG,
            pendulum_com_m=PENDULUM_COM_M,
            # Teorema de Steiner (ejes paralelos):
            #   I_eje = m·d² + I_com
            # m·d² traslada la inercia desde el COM hasta el eje de giro.
            pendulum_inertia_kg_m2=(
                PENDULUM_MASS_KG * PENDULUM_COM_M ** 2
                + PENDULUM_I_COM_SWING_KG_M2
            ),
            pendulum_friction=viscous,
            pendulum_coulomb=coulomb,
        )


def build_mjcf(p: PendulumParams) -> str:
    """Construye un modelo MJCF (texto XML) parametrizado con los parámetros de sysid.

    Descomposición de la inercia del péndulo: MJCF espera el tensor de
    inercia en el marco *inercial* del cuerpo, expresado respecto al COM. El
    eje x del marco del cuerpo del péndulo está alineado con el eje de giro de
    la articulación (axis="1 0 0"). Para la rotación alrededor del eje de la
    articulación, el teorema de los ejes paralelos da
        I_eje = I_com_xx + m * (distancia perpendicular del eje x al COM)²
              = I_com_xx + m * (y_com² + z_com²)
    Con el COM en (0, 0, -d), la distancia perpendicular es d.
    Tomamos I_com_xx del CAD (PENDULUM_I_COM_SWING_KG_M2) en lugar de despejar
    I_eje − m·d² a partir de sysid: la I_eje de sysid coincide con el valor
    del CAD dentro de un 5 %, y usar la constante del CAD mantiene `(m, d,
    I_com)` independientes, de modo que la randomización de dominio (DR) pueda
    muestrear m y d sin arrastrar I_com con la aproximación de masa puntual.
    Para las componentes fuera del eje: un péndulo en forma de varilla que se
    extiende a lo largo de z tiene una extensión apreciable perpendicular a y,
    así que iyy ≈ ixx; izz (eje longitudinal) es casi cero. Una ligera
    asimetría no afecta la dinámica de oscilación de 1 grado de libertad, pero
    mantiene bien condicionado el solver de MuJoCo.

    Modelo del motor: el stepper real, cuando está activado, mantiene su
    posición comandada de forma prácticamente rígida frente al diminuto par de
    reacción del péndulo al oscilar. Lo modelamos con un actuador de posición
    rígido (kp=1000, críticamente amortiguado frente a la inercia que ve el
    motor; ver más abajo por qué ese valor).
    """
    m = p.pendulum_mass_kg
    d = p.pendulum_com_m
    I_com_swing = PENDULUM_I_COM_SWING_KG_M2
    # diaginertia[0] = ixx = inercia del eje de oscilación respecto al COM
    #                  (LA que importa)
    # diaginertia[1] = iyy ≈ ixx para un cuerpo en forma de varilla con
    #                  extensión perpendicular a su eje longitudinal
    # diaginertia[2] = izz ≈ diminuta (inercia del eje longitudinal)
    diag_inertia = (I_com_swing, I_com_swing, 1e-7)

    # Inercia del brazo respecto a su COM (a lo largo del eje de giro del
    # stepper). Tras la reconstrucción con 1 rodamiento, la distribución de
    # masa se parece más a una varilla delgada que al modelo anterior de
    # "masas puntuales en ambos extremos". La aproximación de varilla,
    #     I = m·L² / 12
    # da ~1.06e-5 kg·m² para el brazo actual de 30 g / 65 mm. Se usa en el
    # diaginertia del cuerpo MJCF.
    arm_I = ARM_MASS_KG * ARM_LENGTH_M ** 2 / 12.0

    # Ganancias del actuador de posición PD. La articulación del motor ve la
    # inercia efectiva COMPLETA (brazo por ejes paralelos + masa del péndulo en
    # la punta del brazo + inercia propia del péndulo respecto a su
    # articulación), no solo arm_I; kv es el amortiguamiento crítico frente a
    # esa inercia.
    #
    # kp modela la rigidez del stepper, y un stepper es una fuente de posición
    # casi RÍGIDA: su par de retención (~0.4 N·m) supera por mucho al par de
    # reacción del péndulo (~0.02 N·m), así que el brazo físico sigue el perfil
    # de pasos prácticamente de forma exacta. El antiguo kp=100 era lo bastante
    # blando como para que la reacción del péndulo hiciera oscilar el brazo
    # simulado a ±rad/s: una textura de la planta que el robot real no tiene y
    # a la que las políticas se sobreajustaban. El viejo "techo" de que
    # "kp ≥ 200 desestabiliza el integrador" NO era un límite del paso de
    # tiempo: era el objetivo en ESCALERA a la frecuencia de control golpeando
    # al servo con un gran impulso por tick. Con el objetivo interpolado a
    # resolución de subpaso (ver step()), kp=1000 es estable bajo implicitfast
    # con paso de 1 ms y sigue la trayectoria comandada con ~2 mrad / ~0.08 rad/s.
    I_arm_about_motor = arm_I + ARM_MASS_KG * ARM_COM_M ** 2     # Steiner
    I_pen_at_arm_tip  = p.pendulum_mass_kg * ARM_LENGTH_M ** 2   # m·L²
    I_pen_self        = p.pendulum_mass_kg * p.pendulum_com_m ** 2 + PENDULUM_I_COM_SWING_KG_M2
    I_motor_joint     = I_arm_about_motor + I_pen_at_arm_tip + I_pen_self
    kp = 1000.0
    # Amortiguamiento crítico de un sistema masa-resorte:  c = 2·√(k·I)
    # (con ζ = 1 no oscila ni se arrastra).
    kv = 2.0 * math.sqrt(kp * I_motor_joint)

    # Nota: en el modelo, ángulo del péndulo 0 = colgando (su COM está en −z).
    # La convención del entorno (0 = vertical hacia arriba) se aplica después,
    # en la observación.
    return f"""<?xml version="1.0"?>
<mujoco model="rotary_inverted_pendulum">
  <!-- Unidades angulares en RADIANES. MuJoCo lee por defecto los ángulos del
       XML en grados: sin esta línea, range="-2.356 2.356" se interpretaba
       como ±2.36° (el original del autor tiene ese error; ver
       tests/test_build_mjcf.py::test_limites_de_articulacion_en_radianes). -->
  <compiler angle="radian"/>

  <!-- Paso de física de 1 ms. Gravedad hacia -z. Integrador implícito "fast",
       estable con el servo rígido de abajo. -->
  <option timestep="0.001" gravity="0 0 -{GRAVITY}" integrator="implicitfast"/>

  <default>
    <joint armature="0" damping="0"/>
    <!-- contype/conaffinity = 0: sin colisiones; las formas son solo visuales -->
    <geom contype="0" conaffinity="0" rgba="0.6 0.6 0.6 1"/>
  </default>

  <worldbody>
    <light diffuse="0.7 0.7 0.7" pos="0 0 1" dir="0 0 -1"/>

    <!-- Brazo: gira alrededor del eje vertical z, accionado por el motor -->
    <body name="arm" pos="0 0 0">
      <joint name="motor_joint" type="hinge" axis="0 0 1"
             range="-{MOTOR_LIMIT_RAD} {MOTOR_LIMIT_RAD}" limited="true"
             damping="0.0001" frictionloss="0"/>
      <!-- COM del brazo a lo largo de x; inercia diagonal (x, y, z) -->
      <inertial pos="{ARM_COM_M} 0 0" mass="{ARM_MASS_KG}"
                diaginertia="1e-6 {arm_I} {arm_I}"/>
      <geom name="arm_visual" type="capsule"
            fromto="0 0 0 {ARM_LENGTH_M} 0 0" size="0.004"
            rgba="0.4 0.6 0.8 1"/>

      <!-- Péndulo: articulado en la punta del brazo, gira alrededor de x
           (a lo largo del brazo) -->
      <body name="pendulum" pos="{ARM_LENGTH_M} 0 0">
        <!-- damping = fricción viscosa (N·m·s); frictionloss = Coulomb (N·m),
             ambos medidos con sysid -->
        <joint name="pendulum_joint" type="hinge" axis="1 0 0"
               damping="{p.pendulum_friction}"
               frictionloss="{p.pendulum_coulomb}"/>
        <!-- COM a -d en z (colgando); inercia respecto al COM -->
        <inertial pos="0 0 -{d}" mass="{m}"
                  diaginertia="{diag_inertia[0]} {diag_inertia[1]} {diag_inertia[2]}"/>
        <geom name="pendulum_visual" type="capsule"
              fromto="0 0 0 0 0 -{d * 2}" size="0.003"
              rgba="0.8 0.4 0.4 1"/>
        <geom name="pendulum_tip" type="sphere" pos="0 0 -{d * 2}"
              size="0.006" rgba="0.9 0.7 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <!-- Motor: servo de posición. Fuerza = kp·(objetivo − pos) − kv·vel.
       ctrlrange limita el objetivo al límite seguro de ±125°. -->
  <actuator>
    <position name="motor" joint="motor_joint"
              kp="{kp}" kv="{kv}"
              ctrlrange="-{MOTOR_SAFE_LIMIT_RAD} {MOTOR_SAFE_LIMIT_RAD}"/>
  </actuator>
</mujoco>
"""

def _wrap_pi(angle: float) -> float:
    """Lleva un ángulo al rango (-π, π]."""
    return math.pi - ((math.pi - angle) % (2.0 * math.pi))


# --- Entorno Gymnasium -------------------------------------------------------

# Dimensión de un frame de observación:
#   [motor_pos, sin θ, cos θ, vel_motor, vel_péndulo, acción_previa]
OBS_FRAME_DIM = 6

# Cuántos frames consecutivos se apilan en la observación (del más viejo al
# más nuevo). Con 4 frames a 50 Hz la política ve 80 ms de historia: puede
# inferir aceleraciones y el efecto de sus propias acciones previas. Debe
# coincidir con OBS_FRAMES del firmware (RLControl.ino); si no, la política
# desplegada recibiría una observación distinta de la que entrenó.
OBS_HISTORY_LEN = 4

# Suavizado de acción: al motor llega el PROMEDIO de las últimas N acciones
# de la política (N=1 lo desactiva). Con N=4 a 50 Hz el promedio anula las
# oscilaciones de 25 y 12.5 Hz (el "PWM" que aprenden las redes), deja pasar
# casi intacto el control lento (ganancia 0.96 a 2 Hz) y añade un retardo de
# (N-1)/2 = 1.5 pasos (30 ms). Debe coincidir con ACTION_SMOOTH_WINDOW del
# firmware (RLControl.ino).
ACTION_SMOOTH_WINDOW = 4


class RotaryInvertedPendulumEnv(gym.Env):
    """Simulación MuJoCo del péndulo invertido rotatorio (versión mínima).

    Convención del ángulo: θ = 0 es péndulo ARRIBA (vertical). En el modelo
    MuJoCo el ángulo de la articulación `phi` es 0 con el péndulo colgando,
    así que θ = phi − π (se convierte al construir la observación).

    Bloques 1-7: construcción, reset(), observación apilada, step(),
    recompensa y suavizado de acción. Con `domain_randomization=True` cada
    episodio sortea parámetros físicos (masa, COM, fricción, stiction del
    motor, inclinación de la base), la envolvente de aceleración del motor, el
    lag/retardo del comando, el jitter del paso de control y el ruido de
    observación. Con `firmware_obs_model=True` la observación sale de un
    modelo mecanístico de la lectura del firmware (cuantización, antigüedad y
    diferencias finitas). Aún falta el render.
    """

    metadata = {"render_modes": ["human"], "render_fps": CONTROL_FREQ_HZ}

    def __init__(
        self,
        *,
        params_path: str | Path | None = None,
        control_freq_hz: float = CONTROL_FREQ_HZ,
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,
        episode_length_s: float = 8.0,
        obs_history_len: int = OBS_HISTORY_LEN,
        action_smooth_window: int = ACTION_SMOOTH_WINDOW,
        reward_weights: RewardWeights | None = None,
        hard_stop_penalty: float = HARD_STOP_PENALTY,
        domain_randomization: bool = False,
        dr_theta_bias_max_rad: float = DR_THETA_BIAS_MAX_RAD,
        dr_motor_accel_range_rad_s2: tuple[float, float] = DR_MOTOR_ACCEL_RANGE_RAD_S2,
        dr_action_delay_steps_range: tuple[int, int] = DR_ACTION_DELAY_STEPS_RANGE,
        dr_action_lag_tau_range_s: tuple[float, float] = DR_ACTION_LAG_TAU_RANGE_S,
        action_delay_steps: int = 0,
        action_lag_tau_s: float = 0.0,
        firmware_obs_model: bool = False,
        obs_staleness_s: float | None = None,
        dr_obs_staleness_range_s: tuple[float, float] = DR_OBS_STALENESS_RANGE_S,
    ) -> None:
        super().__init__()
        self.params = PendulumParams.load(params_path)
        self.control_freq_hz = control_freq_hz
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.episode_length_s = episode_length_s
        if obs_history_len < 1:
            raise ValueError(f"obs_history_len debe ser >= 1, recibí {obs_history_len}")
        self.obs_history_len = obs_history_len
        if action_smooth_window < 1:
            raise ValueError(
                f"action_smooth_window debe ser >= 1, recibí {action_smooth_window}"
            )
        self.action_smooth_window = action_smooth_window
        self.reward_weights = reward_weights or RewardWeights()
        self.hard_stop_penalty = hard_stop_penalty
        # False = parámetros nominales (evaluación y depuración); True = un
        # robot distinto por episodio (entrenamiento).
        self.domain_randomization = domain_randomization
        # El sesgo θ es independiente de la DR (ver DR_THETA_BIAS_MAX_RAD).
        if dr_theta_bias_max_rad < 0:
            raise ValueError(
                f"dr_theta_bias_max_rad debe ser >= 0, recibí {dr_theta_bias_max_rad}"
            )
        self._dr_theta_bias_max_rad = float(dr_theta_bias_max_rad)
        self._theta_bias_rad = 0.0     # se sortea en cada reset()

        # Rangos de la DR del actuador (se pueden cambiar por etapa del
        # currículo) y los valores del episodio en curso. Sin DR quedan en el
        # valor neutro: envolvente = tope de la ley P, sin lag ni retardo.
        for nombre, (lo, hi), minimo in [
            ("dr_motor_accel_range_rad_s2", dr_motor_accel_range_rad_s2, 0.0),
            ("dr_action_delay_steps_range", dr_action_delay_steps_range, 0),
            ("dr_action_lag_tau_range_s", dr_action_lag_tau_range_s, 0.0),
        ]:
            if not (minimo <= lo <= hi):
                raise ValueError(f"{nombre} debe cumplir {minimo} <= min <= max, recibí ({lo}, {hi})")
        self._dr_motor_accel_range_rad_s2 = dr_motor_accel_range_rad_s2
        self._dr_action_delay_steps_range = dr_action_delay_steps_range
        self._dr_action_lag_tau_range_s = dr_action_lag_tau_range_s
        self._motor_max_accel_rad_s2 = max_accel_rad_s2
        # Retardo y lag FIJOS, para un entorno determinista con el transporte del
        # robot real (el de evaluación). Con DR se vuelven a sortear en cada
        # reset() y estos valores solo son los del primer episodio.
        if action_delay_steps < 0 or action_lag_tau_s < 0:
            raise ValueError(
                "action_delay_steps y action_lag_tau_s deben ser >= 0, "
                f"recibí ({action_delay_steps}, {action_lag_tau_s})"
            )
        self._action_delay_steps = int(action_delay_steps)
        self._action_lag_tau_s = float(action_lag_tau_s)

        # Modelo de medición del firmware (ver arriba). Apagado, la observación
        # sale directo del estado de MuJoCo. Encendido, sale de un anillo de
        # posiciones a ritmo de física, cuantizadas y con antigüedad.
        lo, hi = dr_obs_staleness_range_s
        if not (0.0 <= lo <= hi):
            raise ValueError(
                f"dr_obs_staleness_range_s debe cumplir 0 <= min <= max, recibí ({lo}, {hi})"
            )
        if obs_staleness_s is not None and obs_staleness_s < 0:
            raise ValueError(f"obs_staleness_s debe ser >= 0, recibí {obs_staleness_s}")
        self.firmware_obs_model = bool(firmware_obs_model)
        self._fixed_obs_staleness_s = (
            float(obs_staleness_s) if obs_staleness_s is not None
            else OBS_STALENESS_NOMINAL_S
        )
        self._dr_obs_staleness_range_s = dr_obs_staleness_range_s
        self._obs_staleness_s = self._fixed_obs_staleness_s
        self._meas_motor_pos = 0.0
        self._meas_pen_phi = 0.0
        self._meas_motor_vel = 0.0
        self._meas_pen_vel = 0.0
        self._lagged_accel_cmd = 0.0
        self._accel_cmd_queue: deque[float] = deque()

        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params))
        self.data = mujoco.MjData(self.model)

        # Pasos de física (1 ms) por cada paso de control (20 ms a 50 Hz).
        physics_dt = self.model.opt.timestep
        self._n_substeps = max(1, int(round(1.0 / control_freq_hz / physics_dt)))

        # Anillo de (pos. motor ordenada, ángulo del péndulo), una entrada por
        # substep de física; cubre la ventana de velocidad + la peor antigüedad.
        self._meas_win_substeps = max(1, int(round(FIRMWARE_VEL_WINDOW_S / physics_dt)))
        stale_max_s = max(self._fixed_obs_staleness_s, self._dr_obs_staleness_range_s[1])
        self._meas_ring: deque[tuple[float, float]] = deque(
            maxlen=self._meas_win_substeps + int(math.ceil(stale_max_s / physics_dt)) + 2
        )

        # Pasos de control por episodio (8 s × 50 Hz = 400).
        self._max_steps = int(episode_length_s * control_freq_hz)
        self._step_count = 0

        # Direcciones en qpos/qvel de cada articulación. Se buscan una vez
        # por nombre; después se indexa el array directamente (más rápido).
        motor = self.model.joint("motor_joint")
        pendulum = self.model.joint("pendulum_joint")
        self._motor_qpos_addr = int(motor.qposadr[0])
        self._motor_qvel_addr = int(motor.dofadr[0])
        self._pen_qpos_addr = int(pendulum.qposadr[0])
        self._pen_qvel_addr = int(pendulum.dofadr[0])
        # Id del cuerpo del péndulo: la DR modifica su masa, inercia y COM.
        self._pen_body_id = int(self.model.body("pendulum").id)

        # Acción: un número en [-1, 1] que luego se escala a ±max_velocity.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observación: obs_history_len frames de OBS_FRAME_DIM valores, uno
        # tras otro. Las cotas son holgadas; solo documentan la escala de
        # cada canal (np.tile repite las de un frame para cada copia).
        frame_high = np.array(
            [MOTOR_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0], dtype=np.float32
        )
        obs_high = np.tile(frame_high, obs_history_len)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Historial de frames, del más viejo al más nuevo. Solo lo modifican
        # reset() y step() (a través de _obs), una vez por paso de control.
        self._obs_history: deque[np.ndarray] = deque(maxlen=obs_history_len)

        self._prev_action = 0.0

        # Últimas acciones CRUDAS de la política (arranca en ceros). Su
        # promedio es la acción que ve el motor.
        self._action_buf: deque[float] = deque(
            [0.0] * action_smooth_window, maxlen=action_smooth_window
        )

        # Estado de la cadena aceleración → velocidad → posición (ver step()).
        self._motor_vel = 0.0       # velocidad ORDENADA al motor (rad/s), no la medida
        self._motor_target = 0.0    # posición objetivo integrada (rad)
        self._prev_cmd_pos = 0.0    # objetivo al final del paso anterior (rad)

    # --- API de Gymnasium ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Empieza un episodio nuevo: péndulo colgando, motor en posición aleatoria."""
        super().reset(seed=seed)       # inicializa self.np_random con la semilla
        mujoco.mj_resetData(self.model, self.data)

        # `mj_resetData` solo limpia `data`; los parámetros físicos viven en
        # `model`, así que la DR los reescribe aquí, una vez por episodio.
        if self.domain_randomization:
            self._sample_dr_params()

        # Sesgo del cero del encoder: fuera del `if` de arriba a propósito (ver
        # DR_THETA_BIAS_MAX_RAD). Con 0.0 no se gasta ningún número aleatorio.
        if self._dr_theta_bias_max_rad > 0.0:
            self._theta_bias_rad = float(self.np_random.uniform(
                -self._dr_theta_bias_max_rad, self._dr_theta_bias_max_rad
            ))
        else:
            self._theta_bias_rad = 0.0

        # Motor: posición aleatoria dentro del 70 % del límite seguro (≈ ±87°),
        # para que la política aprenda a volver desde cualquier punto del
        # rango y no solo desde el centro.
        self.data.qpos[self._motor_qpos_addr] = self.np_random.uniform(
            -0.7 * MOTOR_SAFE_LIMIT_RAD, 0.7 * MOTOR_SAFE_LIMIT_RAD
        )
        # Péndulo: phi = 0 es colgando; se añade un ruido pequeño (±0.05 rad).
        self.data.qpos[self._pen_qpos_addr] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[self._motor_qvel_addr] = 0.0
        self.data.qvel[self._pen_qvel_addr] = 0.0

        # El servo debe sostener la posición inicial, no tirar hacia 0:
        # el objetivo arranca en la posición real y la velocidad ordenada en 0.
        self._motor_target = float(self.data.qpos[self._motor_qpos_addr])
        self._prev_cmd_pos = self._motor_target
        self._motor_vel = 0.0
        self.data.ctrl[0] = self._motor_target

        self._step_count = 0
        self._prev_action = 0.0
        self._action_buf.extend([0.0] * self.action_smooth_window)  # maxlen descarta lo viejo
        self._obs_history.clear()      # _obs() lo rellena con el frame inicial

        # Transporte vacío al empezar: el filtro en 0 y la cola llena de ceros
        # (los primeros `retardo` ticks no llega ninguna aceleración al motor).
        # Va DESPUÉS de sortear los parámetros, porque el retardo se sortea ahí.
        self._lagged_accel_cmd = 0.0
        self._accel_cmd_queue = deque(
            [0.0] * self._action_delay_steps, maxlen=self._action_delay_steps + 1
        )

        # Recalcula las cantidades derivadas con el nuevo estado (sin avanzar
        # el tiempo).
        mujoco.mj_forward(self.model, self.data)
        if self.firmware_obs_model:
            self.seed_measurement_ring()
        return self._obs(), {}

    def seed_measurement_ring(self) -> None:
        """(Re)llena el anillo de medición con el estado ACTUAL de las articulaciones.

        Extrapola hacia atrás con las velocidades actuales, para que la primera
        diferencia finita lea la velocidad inicial verdadera. También sirve tras
        sobrescribir qpos/qvel a mano (herramientas de reproducción y tests).
        """
        motor0 = float(self.data.qpos[self._motor_qpos_addr])
        pen0 = float(self.data.qpos[self._pen_qpos_addr])
        motor_vel0 = float(self.data.qvel[self._motor_qvel_addr])
        pen_vel0 = float(self.data.qvel[self._pen_qvel_addr])
        self._prev_cmd_pos = motor0         # la trayectoria ordenada parte de la posición real
        dt_p = self.model.opt.timestep
        n = self._meas_ring.maxlen
        self._meas_ring.clear()
        self._meas_ring.extend(
            (motor0 - motor_vel0 * (n - 1 - k) * dt_p,
             pen0 - pen_vel0 * (n - 1 - k) * dt_p)
            for k in range(n)
        )
        self._capture_measured_state()

    def _capture_measured_state(self) -> None:
        """Modela UNA lectura de GET_STATE a partir del anillo de posiciones.

        Las posiciones salen cuantizadas a la resolución del contador de pasos y
        del AS5600, y con `_obs_staleness_s` de antigüedad; las velocidades son
        diferencias finitas de esas posiciones cuantizadas en la ventana de 8 ms,
        así que los picos de un LSB salen del mismo mecanismo que en el robot.
        """
        dt_p = self.model.opt.timestep
        ring = self._meas_ring
        k_stale = int(round(self._obs_staleness_s / dt_p))
        idx_new = max(0, len(ring) - 1 - k_stale)
        idx_old = max(0, idx_new - self._meas_win_substeps)
        m_new, p_new = ring[idx_new]
        m_old, p_old = ring[idx_old]
        mq_new = round(m_new / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        mq_old = round(m_old / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        pq_new = round(p_new / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        pq_old = round(p_old / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        span_s = max(1, idx_new - idx_old) * dt_p
        self._meas_motor_pos = mq_new
        self._meas_pen_phi = pq_new
        self._meas_motor_vel = (mq_new - mq_old) / span_s
        self._meas_pen_vel = (pq_new - pq_old) / span_s

    def _sample_dr_params(self) -> None:
        """Sortea los parámetros físicos de este episodio y los escribe en `model`.

        Usa `self.np_random`, así que `reset(seed=...)` lo hace reproducible.
        """
        rng = self.np_random
        nominal = self.params
        pen = self._pen_body_id

        # Masa y distancia al COM. La inercia propia respecto al COM es una
        # propiedad geométrica de la pieza (viene del CAD) y NO se sortea:
        # MuJoCo aplica Steiner solo, así que la inercia respecto al pivote
        # resulta m·d² + I_com con los valores sorteados.
        m = rng.uniform(
            nominal.pendulum_mass_kg * (1.0 - DR_PENDULUM_MASS_FRAC),
            nominal.pendulum_mass_kg * (1.0 + DR_PENDULUM_MASS_FRAC),
        )
        d = rng.uniform(
            nominal.pendulum_com_m * (1.0 - DR_PENDULUM_COM_FRAC),
            nominal.pendulum_com_m * (1.0 + DR_PENDULUM_COM_FRAC),
        )
        self.model.body_mass[pen] = m
        self.model.body_inertia[pen] = (
            PENDULUM_I_COM_SWING_KG_M2, PENDULUM_I_COM_SWING_KG_M2, 1e-7
        )
        self.model.body_ipos[pen, 2] = -d       # COM a -d en z (colgando)

        # Fricción del péndulo: mismo multiplicador para viscosa y Coulomb.
        fric_mult = rng.uniform(*DR_PENDULUM_FRICTION_MULT_RANGE)
        self.model.dof_damping[self._pen_qvel_addr] = nominal.pendulum_friction * fric_mult
        self.model.dof_frictionloss[self._pen_qvel_addr] = nominal.pendulum_coulomb * fric_mult

        # Stiction del motor.
        self.model.dof_frictionloss[self._motor_qvel_addr] = rng.uniform(
            *DR_MOTOR_FRICTIONLOSS_RANGE_N_M
        )

        # Inclinación de la base: se gira la gravedad un ángulo pequeño alrededor
        # de un eje horizontal al azar. Su módulo se conserva (g sigue valiendo 9.81).
        tilt = rng.uniform(0.0, DR_BASE_TILT_MAX_RAD)
        azim = rng.uniform(0.0, 2.0 * math.pi)
        self.model.opt.gravity[:] = (
            GRAVITY * math.sin(tilt) * math.cos(azim),
            GRAVITY * math.sin(tilt) * math.sin(azim),
            -GRAVITY * math.cos(tilt),
        )

        # Actuador: envolvente de aceleración y transporte del comando. Estos
        # sorteos van al final para no alterar la secuencia de los anteriores.
        self._motor_max_accel_rad_s2 = float(rng.uniform(*self._dr_motor_accel_range_rad_s2))
        lo, hi = self._dr_action_delay_steps_range
        self._action_delay_steps = int(rng.integers(lo, hi + 1))
        self._action_lag_tau_s = float(rng.uniform(*self._dr_action_lag_tau_range_s))
        if self.firmware_obs_model:
            self._obs_staleness_s = float(rng.uniform(*self._dr_obs_staleness_range_s))

    def _obs(self) -> np.ndarray:
        """Observación apilada: los últimos `obs_history_len` frames, del más viejo al nuevo.

        Llamar EXACTAMENTE una vez por reset() y una por step(): cada llamada
        añade un frame al historial. Tras reset() el historial está vacío y se
        rellena con el frame inicial repetido, así no hay ceros falsos.
        """
        frame = self._frame()
        if not self._obs_history:
            self._obs_history.extend([frame] * self.obs_history_len)
        else:
            self._obs_history.append(frame)
        return np.concatenate(self._obs_history)

    def _frame(self) -> np.ndarray:
        """Un frame: [motor_pos, sin θ, cos θ, vel_motor, vel_péndulo, acción_previa].

        θ = 0 es péndulo arriba; el ángulo de la articulación `phi` es 0
        colgando, por eso θ = phi − π. Se usan (sin, cos) en vez de θ para que
        la posición sea continua al cruzar ±π.

        Con `domain_randomization` el frame sale cuantizado y con ruido, y por
        eso consume números aleatorios: sin DR es una función pura del estado.
        """
        if self.firmware_obs_model:
            # Lo que el host leería de verdad: la última lectura de GET_STATE,
            # ya cuantizada, con antigüedad y con velocidades por diferencias.
            motor_pos = self._meas_motor_pos
            phi = self._meas_pen_phi
            motor_vel = self._meas_motor_vel
            pen_vel = self._meas_pen_vel
        else:
            motor_pos = float(self.data.qpos[self._motor_qpos_addr])
            phi = float(self.data.qpos[self._pen_qpos_addr])
            motor_vel = float(self.data.qvel[self._motor_qvel_addr])
            pen_vel = float(self.data.qvel[self._pen_qvel_addr])

        if self.domain_randomization:
            # Lo que el robot real mide: el ángulo del péndulo llega cuantizado
            # al LSB del AS5600, y todas las lecturas traen algo de ruido. Solo
            # la OBSERVACIÓN se ensucia; la física y la recompensa usan el
            # estado verdadero.
            if not self.firmware_obs_model:     # con el modelo ya viene cuantizado
                phi = round(phi / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
            rng = self.np_random
            vel_std = (DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S if self.firmware_obs_model
                       else DR_OBS_NOISE_STD_VEL_RAD_S)
            motor_pos += rng.normal(0.0, DR_OBS_NOISE_STD_POS_RAD)
            phi += rng.normal(0.0, DR_OBS_NOISE_STD_POS_RAD)
            motor_vel += rng.normal(0.0, vel_std)
            pen_vel += rng.normal(0.0, vel_std)

        # Sesgo del cero del encoder: solo la observación, nunca la física.
        phi += self._theta_bias_rad

        theta = _wrap_pi(phi - math.pi)
        return np.array(
            [motor_pos, math.sin(theta), math.cos(theta),
             motor_vel, pen_vel, self._prev_action],
            dtype=np.float32,
        )

    def step(
        self, action: np.ndarray | float
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Avanza un paso de control (20 ms a 50 Hz).

        Cadena (modo velocidad):
            acción ∈ [-1, 1] → promedio de las últimas N → v_des = promedio·3.5 rad/s
            → ley P saturada: accel = (v_des − v_ordenada)·f_control, tope ±150 rad/s²
            → transporte (con DR): filtro de 1.er orden y cola de ticks, sobre accel
            → techo de aceleración del motor en este episodio (110–190 con DR)
            → v_ordenada += accel·dt (tope ±3.5 rad/s)
            → objetivo += v_ordenada·dt (tope ±125°)
            → 20 substeps de física con ctrl[0] interpolado hacia el objetivo.

        Recompensa: ver reward.py; si choca con el tope duro se resta
        `hard_stop_penalty` y el episodio termina.
        """
        a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # Suavizado: la acción CRUDA `a` va a la recompensa y a prev_action
        # (la política ve lo que ella decidió); solo el motor recibe el
        # promedio de las últimas N acciones.
        self._action_buf.append(a)
        smoothed = sum(self._action_buf) / self.action_smooth_window

        # Duración de ESTE paso de control. Con DR, jitter de ±5 % en el número
        # de substeps (el reloj real nunca es perfecto); sin DR, siempre fijo.
        if self.domain_randomization:
            jitter = self.np_random.uniform(
                -DR_CONTROL_DT_JITTER_FRAC, DR_CONTROL_DT_JITTER_FRAC
            )
            n_sub = max(1, int(round(self._n_substeps * (1.0 + jitter))))
        else:
            n_sub = self._n_substeps
        dt = n_sub * self.model.opt.timestep

        # --- Ley P: velocidad deseada → aceleración ordenada ---
        # Realimenta la velocidad ORDENADA, no la medida (ver explicación):
        # el stepper ejecuta casi perfecto, y la medida añadiría temblor.
        v_des = smoothed * self.max_velocity_rad_s
        accel_cmd = (v_des - self._motor_vel) * self.control_freq_hz
        accel_cmd = float(np.clip(accel_cmd, -self.max_accel_rad_s2, self.max_accel_rad_s2))

        # --- Transporte host → motor, sobre el comando de ACELERACIÓN ---
        # La ley P corre en el host sin retardo; lo que viaja (y se retrasa) es
        # la aceleración ya calculada. Primero el filtro de 1.er orden (retardo
        # fraccionario) y luego la cola de ticks enteros; se componen.
        # alpha = dt/(τ+dt): τ=0 → alpha=1 (sin lag); τ=dt → mitad actual y
        # mitad anterior. Se usa el periodo NOMINAL de control, sin jitter.
        if self._action_lag_tau_s > 0.0:
            dt_ctrl = 1.0 / self.control_freq_hz
            alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
            self._lagged_accel_cmd = (
                (1.0 - alpha) * self._lagged_accel_cmd + alpha * accel_cmd
            )
            accel_cmd = self._lagged_accel_cmd
        if self._action_delay_steps > 0:
            self._accel_cmd_queue.append(accel_cmd)
            accel_cmd = float(self._accel_cmd_queue.popleft())
        # Techo del stepper en ESTE episodio (sin DR coincide con el de la ley P).
        accel_cmd = float(np.clip(
            accel_cmd, -self._motor_max_accel_rad_s2, self._motor_max_accel_rad_s2
        ))

        # --- Aceleración → velocidad → posición objetivo ---
        self._motor_vel = float(np.clip(
            self._motor_vel + accel_cmd * dt,
            -self.max_velocity_rad_s, self.max_velocity_rad_s,
        ))
        # Corrección complementaria, como el host real: acerca suavemente (10 %
        # por tick) el integrador de velocidad ordenada a la velocidad MEDIDA,
        # para que topes o pasos perdidos no acumulen error a lazo abierto. Deja
        # pasar ~10 % del ruido de cuantización de la medición, que es el
        # temblor residual con el que entrena la política. La ley P realimenta
        # el integrador, NO la medición: hacerlo con ganancia f inyecta un
        # temblor de ±17 rad/s² que destruye el equilibrio (medido en el robot).
        if self.firmware_obs_model:
            self._motor_vel += 0.1 * (self._meas_motor_vel - self._motor_vel)
        # En el tope seguro, si la velocidad empuja hacia fuera se anula
        # (imita el recorte del firmware).
        if self._motor_target >= MOTOR_SAFE_LIMIT_RAD and self._motor_vel > 0.0:
            self._motor_vel = 0.0
        elif self._motor_target <= -MOTOR_SAFE_LIMIT_RAD and self._motor_vel < 0.0:
            self._motor_vel = 0.0
        self._motor_target = float(np.clip(
            self._motor_target + self._motor_vel * dt,
            -MOTOR_SAFE_LIMIT_RAD, MOTOR_SAFE_LIMIT_RAD,
        ))

        # --- Física: rampa del objetivo a lo largo de los substeps ---
        # Un escalón por paso de control golpearía al servo rígido (kp=1000)
        # con un impulso cada 20 ms; el stepper real avanza de forma continua.
        cmd_start, cmd_end = self._prev_cmd_pos, self._motor_target
        n = n_sub
        for k in range(n):
            cmd_k = cmd_start + (cmd_end - cmd_start) * (k + 1) / n
            self.data.ctrl[0] = cmd_k
            mujoco.mj_step(self.model, self.data)
            if self.firmware_obs_model:
                # MOTOR: el firmware lee su propio contador de pasos, o sea la
                # trayectoria ORDENADA (no la articulación de MuJoCo: el PD es
                # solo nuestra aproximación del stepper, y su error de
                # seguimiento metido en la medición crea una oscilación solo de
                # simulación). PÉNDULO: el AS5600 mide la articulación física.
                self._meas_ring.append((cmd_k, float(self.data.qpos[self._pen_qpos_addr])))
        if self.firmware_obs_model:
            # Una lectura por tick de control, como el GET_STATE del host: esta
            # instantánea alimenta esta observación y la corrección del siguiente.
            self._capture_measured_state()

        self._prev_cmd_pos = cmd_end
        self._step_count += 1
        self._prev_action = a

        motor_pos = float(self.data.qpos[self._motor_qpos_addr])
        terminated = abs(motor_pos) >= MOTOR_LIMIT_RAD     # tope físico (±135°)
        truncated = self._step_count >= self._max_steps    # se acabó el tiempo

        # Recompensa con el estado DESPUÉS de la física y la acción aplicada.
        phi = float(self.data.qpos[self._pen_qpos_addr])
        reward = compute_reward(
            theta=_wrap_pi(phi - math.pi),
            pen_vel=float(self.data.qvel[self._pen_qvel_addr]),
            motor_pos=motor_pos,
            motor_vel=float(self.data.qvel[self._motor_qvel_addr]),
            action=a,
            weights=self.reward_weights,
        )
        if terminated:
            reward -= self.hard_stop_penalty

        info = {
            "motor_pos": motor_pos,
            "phi": phi,
            "motor_target": self._motor_target,
            "motor_vel_cmd": self._motor_vel,
            "accel_cmd": accel_cmd,          # aceleración que llegó al motor
            "smoothed_action": smoothed,
            "obs_staleness_s": self._obs_staleness_s if self.firmware_obs_model else 0.0,
            "motor_vel_meas": self._meas_motor_vel if self.firmware_obs_model else None,
        }
        return self._obs(), reward, terminated, truncated, info
