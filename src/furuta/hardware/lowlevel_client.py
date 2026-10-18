"""Cliente Python para el sketch de Arduino LowLevelServer.

Habla el protocolo binario definido al inicio de
firmware/LowLevelServer/LowLevelServer.ino:
    0x01 READY               -> la placa responde con 0x01
    0x02 GET_STATE           -> la placa devuelve 20 bytes:
                                  uint32 time_us
                                  float32 motor_pos_rad
                                  float32 pendulum_pos_rad
                                  float32 motor_vel_rad_s
                                  float32 pendulum_vel_rad_s
    0x03 SET_ACCEL + float   aceleración angular comandada en rad/s²
    0x04 ENGAGE_MOTOR
    0x05 DISENGAGE_MOTOR
    0x07 SET_TARGET + float  posición comandada del motor en radianes (modo posición)

Convención de signos: LowLevelServer invierte el signo de las posiciones Y
las velocidades del motor y del péndulo al enviarlas. Aquí pasamos los bytes
tal cual; quien llama decide si aplica transformaciones adicionales. El código
de despliegue (`run_policy.py`) y el entorno real (`real_env.py`) los consumen
sin cambios: la observación coincide con la convención de la simulación.

Este cliente funciona por petición (una lectura de 20 bytes por cada llamada a
get_state). El ancho de banda sostenido es mínimo (~2 kB/s a 100 Hz), así que
2 Mbaud queda de sobra.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

import serial


CMD_READY = 0x01
CMD_GET_STATE = 0x02
CMD_SET_ACCEL = 0x03  # antes era CMD_SET_TARGET (modo posición); ahora aceleración angular en rad/s²
CMD_ENGAGE_MOTOR = 0x04
CMD_DISENGAGE_MOTOR = 0x05
CMD_TARE_PENDULUM = 0x06
CMD_SET_TARGET = 0x07  # modo posición: posición comandada del motor en radianes (moveTo)
CMD_ZERO_MOTOR = 0x08  # pone a cero el contador de pasos del motor en la posición actual del brazo

# time_us (uint32 LE), motor_pos, pen_pos, motor_vel, pen_vel (4× float32 LE).
_STATE_STRUCT = struct.Struct("<Iffff")
_STATE_SIZE = _STATE_STRUCT.size  # 20 bytes


@dataclass(frozen=True)
class State:
    time_us: int           # micros() del Arduino cuando se tomó la lectura
    motor_pos_rad: float   # ángulo del motor con signo, marco del firmware (contador de pasos)
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float


class LowLevelClient:
    """Maneja el sketch LowLevelServer a través de un puerto serie.

    Úsalo como gestor de contexto:

        with LowLevelClient(port) as c:
            c.wait_until_ready()
            c.engage_motor()
            c.set_acceleration(50.0)   # rad/s² (ver la entrada del modo aceleración en RL_PLAN)
            s = c.get_state()
            ...
            c.disengage_motor()  # también se llama automáticamente en __exit__
    """

    def __init__(self, port: str, baud: int = 2_000_000, timeout: float = 0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        # Las transacciones seriales se protegen con un lock para que el
        # cliente sea seguro entre varios hilos (p. ej. un lazo de control
        # asíncrono donde un hilo hace get_state/set_acceleration de forma
        # periódica mientras el hilo principal puede emitir un
        # disengage_motor de emergencia al recibir una señal).
        self._lock = threading.Lock()

    def __enter__(self) -> "LowLevelClient":
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Abrir el puerto serie reinicia el bootloader, y setup() (que se
        # bloquea esperando el imán del AS5600) suele tardar ~2 s.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            try:
                # Estado seguro al salir, en la medida de lo posible: detener el motor.
                self.disengage_motor()
            except Exception:
                pass
            self._ser.close()
            self._ser = None

    @property
    def ser(self) -> serial.Serial:
        if self._ser is None:
            raise RuntimeError("LowLevelClient debe usarse como gestor de contexto")
        return self._ser

    # --- Saludo inicial --------------------------------------------------

    def wait_until_ready(self, *, retries: int = 5, retry_delay_s: float = 0.5) -> bool:
        """Envía READY hasta que la placa lo devuelva. Devuelve True si lo logra."""
        for _ in range(retries):
            with self._lock:
                self.ser.reset_input_buffer()
                self.ser.write(bytes([CMD_READY]))
                self.ser.flush()
                resp = self.ser.read(1)
            if resp == bytes([CMD_READY]):
                return True
            time.sleep(retry_delay_s)
        return False

    # --- Comandos --------------------------------------------------------

    def get_state(self) -> State:
        """Pide una lectura de estado. Se bloquea hasta recibir 20 bytes o agotar el timeout."""
        with self._lock:
            self.ser.write(bytes([CMD_GET_STATE]))
            self.ser.flush()
            buf = self.ser.read(_STATE_SIZE)
        if len(buf) != _STATE_SIZE:
            raise RuntimeError(
                f"get_state: se esperaban {_STATE_SIZE} bytes, llegaron {len(buf)}. "
                "¿Está cargado el sketch LowLevelServer? ¿Se detecta el imán?"
            )
        time_us, motor_rad, pen_rad, motor_vel, pen_vel = _STATE_STRUCT.unpack(buf)
        return State(time_us=int(time_us), motor_pos_rad=float(motor_rad),
                     pendulum_pos_rad=float(pen_rad),
                     motor_vel_rad_s=float(motor_vel),
                     pendulum_vel_rad_s=float(pen_vel))

    def set_acceleration(self, accel_rad_s2: float) -> None:
        """Comanda una nueva aceleración angular del motor, en rad/s².

        Corresponde a moveByAcceleration() de FastAccelStepper, con el valor
        equivalente en pasos/s² (int32) en el lado del firmware. El motor
        acelera suavemente hacia el límite de velocidad correspondiente
        (fijado una vez al arrancar con setSpeedInUs) y pasa por cero al
        invertir el sentido, sin ningún tratamiento extra. Reemplazó al
        comando anterior de posición objetivo a partir del cambio a modo
        aceleración (ver RL_PLAN.md).

        No hay `flush()` después de `write()`: los 5 bytes quedan en el búfer
        USB del sistema operativo hasta transmitirse, pero a nuestra
        frecuencia (50-100 Hz × 5 B = 250-500 B/s frente a un enlace de
        2 Mbaud) el búfer nunca se acumula, y los bytes quedan en el cable
        antes de que empiece el intercambio de get_state del siguiente tick.
        Así se evita ~1 ms de sobrecarga de tcdrain en macOS por tick. Los
        demás comandos (engage/disengage/saludo READY) conservan el flush
        porque se ejecutan una vez por sesión, no por tick, y les conviene
        la semántica determinista de "completado al retornar".
        """
        payload = bytes([CMD_SET_ACCEL]) + struct.pack("<f", float(accel_rad_s2))
        with self._lock:
            self.ser.write(payload)

    def set_target(self, target_rad: float) -> None:
        """Comanda una nueva posición objetivo del motor, en radianes (modo posición).

        Corresponde al moveTo() del firmware. Convención de signos: el marco
        del firmware, igual que get_state y set_acceleration: un único marco
        en todas partes.
        Sin flush: un payload de 5 bytes por tick nunca satura el enlace de 2 Mbaud.
        """
        payload = bytes([CMD_SET_TARGET]) + struct.pack("<f", float(target_rad))
        with self._lock:
            self.ser.write(payload)

    def engage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_ENGAGE_MOTOR]))
            self.ser.flush()

    def disengage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_DISENGAGE_MOTOR]))
            self.ser.flush()

    def tare_pendulum(self, *, timeout_s: float = 0.5) -> bool:
        """Pone a cero pen_position_rad en la lectura actual del AS5600.

        Lo usa `real_env.reset()` para que cada episodio de fine-tune tome un
        sesgo nuevo de la distribución real de reposo del robot, limitada por
        la fricción estática (stiction). El firmware desplaza de forma
        atómica cada entrada de su búfer circular del péndulo por el
        pen_position_rad actual, así que el cálculo de velocidad continúa sin
        interrupción a través del tare.

        Se bloquea hasta que el Arduino confirma (eco de un byte del comando)
        o pasa `timeout_s`. Devuelve True si tiene éxito.
        """
        with self._lock:
            self.ser.write(bytes([CMD_TARE_PENDULUM]))
            self.ser.flush()
            resp = self.ser.read(1)
        return resp == bytes([CMD_TARE_PENDULUM])

    def zero_motor(self, *, timeout_s: float = 0.5) -> bool:
        """Pone a cero el contador de pasos del motor en la posición actual del brazo.

        Lo llama `real_env.reset()` después de que el operador recentra el
        brazo a mano: el contador es de lazo abierto, así que de lo contrario
        seguiría reportando la posición anterior al movimiento. Devuelve True
        si recibe la confirmación.
        """
        with self._lock:
            self.ser.write(bytes([CMD_ZERO_MOTOR]))
            self.ser.flush()
            resp = self.ser.read(1)
        return resp == bytes([CMD_ZERO_MOTOR])
