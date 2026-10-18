"""Asistente interactivo (wizard) de identificación de sistema.

Ejecuta todo el flujo de sysid contra `LowLevelServer.ino`, el mismo firmware
que usa el despliegue de RL, de modo que los parámetros de sysid describen
con seguridad al robot en las condiciones que la política realmente va a ver.

El asistente se divide limpiamente en dos fases:

    collect    — adquisición de datos guiada por el operador. Escribe las
                 grabaciones crudas (NumPy .npz + un metadata.json) en un
                 directorio con marca de tiempo. Los únicos efectos
                 secundarios son en disco.
    fit        — post-procesamiento puro. Lee un directorio de grabaciones,
                 deriva los parámetros, genera gráficas de validación
                 sim-vs-real y escribe el sysid_params_*.json de cada robot.
                 No requiere el dispositivo.

Separar las dos permite volver a ajustar tras un cambio de código (por
ejemplo, una fórmula de derivación mejorada) sin volver a grabar en el robot:
es útil porque iterar sobre las matemáticas es rápido, pero las sesiones con
el robot son lentas y dependen de que el operador esté presente.

La fricción es la única magnitud que se mide por robot, así que `--out-json`
es obligatorio en lugar de tener un valor por defecto: escribir en un nombre
de archivo genérico es la forma en que las mediciones de un robot terminan
describiendo en silencio a otro.

Uso:
    # flujo completo (collect+fit+validate)
    python sysid_wizard.py --out-json sysid_params_<rig>.json
    python sysid_wizard.py collect               # solo graba datos
    python sysid_wizard.py fit --in-dir <ruta> --out-json sysid_params_<rig>.json
    python sysid_wizard.py validate-motor        # solo el barrido de verificación del motor a ±90°
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from furuta.hardware.lowlevel_client import LowLevelClient
from furuta.sysid.sysid_core import (
    DEFAULT_FREE_SWING_DURATION_S,
    DEFAULT_FREE_SWING_RUNS,
    aggregate_free_swing_fits,
    derive_pendulum_friction,
    fit_free_swing,
    validate_free_swing,
    write_params_json,
)


# ---------------------------------------------------------------------------
# Utilidades de terminal (ANSI; sin color cuando stdout no es una terminal)
# ---------------------------------------------------------------------------

# Si la salida se redirige a un archivo o a un pipe, los códigos de color
# aparecerían como basura; por eso solo se colorea cuando es una terminal real.
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    # Secuencia de escape ANSI: ESC [ <código> m  texto  ESC [ 0 m
    # (el "0" al final restablece el estilo para no teñir lo que sigue).
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


# Códigos ANSI: 1 = negrita, 2 = tenue, 31 = rojo, 32 = verde, 33 = amarillo, 36 = cian.
def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)
def _yellow(s: str) -> str: return _c("33", s)
def _red(s: str) -> str: return _c("31", s)
def _green(s: str) -> str: return _c("32", s)
def _cyan(s: str) -> str: return _c("36", s)


def _hr(title: str = "") -> None:
    """Imprime un separador horizontal de 72 columnas, con título opcional."""
    width = 72
    if title:
        # ljust rellena con "─" hasta completar el ancho.
        print("\n" + _bold(f"── {title} ".ljust(width, "─")))
    else:
        print("\n" + ("─" * width))


def _print_warnings(warnings) -> None:
    """Imprime las advertencias de validación, coloreadas según su severidad."""
    if not warnings:
        return
    for w in warnings:
        if w.severity == "error":
            print(_red(f"  [error] {w.message}"))
        elif w.severity == "warn":
            print(_yellow(f"  [warn]  {w.message}"))
        else:
            print(_dim(f"  [info]  {w.message}"))


def _prompt_yes_no(label: str, *, default: bool = True) -> bool:
    """Pregunta sí/no. Enter vacío devuelve `default` (la letra en mayúscula)."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        s = input(f"{label} {suffix}: ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def _prompt_enter(label: str) -> None:
    input(f"{label} (press Enter)")


def _countdown(seconds: int = 3) -> None:
    """Cuenta regresiva para que el operador suelte el péndulo justo en "GO"."""
    for k in range(seconds, 0, -1):
        print(f"    {k}...")
        time.sleep(1)
    print(_bold("    GO."))


# ---------------------------------------------------------------------------
# Detección de puerto (tomada del antiguo sysid_client; misma heurística)
# ---------------------------------------------------------------------------

def auto_detect_port() -> list[str]:
    """Devuelve los puertos serie-USB candidatos que parecen de un Arduino."""
    import serial.tools.list_ports
    # Fragmentos de nombre típicos: macOS (usbserial, usbmodem, wchusbserial)
    # y Linux (ttyUSB, ttyACM). Es una heurística por nombre: no verifica que
    # el dispositivo sea realmente un Arduino.
    needles = ("usbserial", "usbmodem", "wchusbserial", "ttyUSB", "ttyACM")
    return [
        p.device for p in serial.tools.list_ports.comports()
        if any(n in p.device for n in needles)
    ]


def pick_port(arg_port: str | None) -> str:
    """Elige el puerto: el pasado por `--port`, el único detectado, o pregunta."""
    if arg_port:
        return arg_port
    candidates = auto_detect_port()
    if not candidates:
        print(_red("No USB-serial ports detected. Pass --port explicitly."))
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]
    print(_bold("Multiple candidate ports found:"))
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    while True:
        s = input("Pick one [0]: ").strip() or "0"
        try:
            idx = int(s)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print(_red("  invalid choice, try again"))


def make_output_dir(arg_out_dir: str | None) -> Path:
    """Crea (y devuelve) el directorio donde se guardan las grabaciones.

    Sin `--out-dir`, usa `sysid_runs/<AAAA-MM-DD_HHMMSS>/` junto a este
    archivo, de modo que cada sesión tiene su propia carpeta y nunca
    sobrescribe a otra.
    """
    if arg_out_dir:
        out = Path(arg_out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path(__file__).resolve().parent / "sysid_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Funciones auxiliares de grabación: usan LowLevelClient con el motor
# desactivado (oscilación libre) o activado con un controlador PD simple
# (barrido del motor).
# ---------------------------------------------------------------------------

def _record_state_stream(
    client: LowLevelClient, duration_s: float, sample_hz: float
) -> dict:
    """Muestrea el estado a `sample_hz` durante `duration_s`. Devuelve un dict de arrays.

    Los valores llegan en el marco del firmware: el único marco que se usa en todas partes.
    """
    # Número de muestras y periodo de muestreo:
    #   n  = duración · frecuencia      (p. ej. 10 s · 100 Hz = 1000 muestras)
    #   dt = 1 / frecuencia             (p. ej. 1/100 Hz = 10 ms)
    n = int(duration_s * sample_hz)
    dt = 1.0 / sample_hz
    # Arrays preasignados: más rápido que ir añadiendo a una lista dentro del bucle.
    t = np.zeros(n, dtype=np.float64)
    pen = np.zeros(n, dtype=np.float64)      # ángulo del péndulo (rad)
    motor = np.zeros(n, dtype=np.float64)    # ángulo del brazo (rad)
    pvel = np.zeros(n, dtype=np.float64)     # velocidad del péndulo (rad/s)
    mvel = np.zeros(n, dtype=np.float64)     # velocidad del brazo (rad/s)

    t_start = time.monotonic()
    next_tick = t_start
    for i in range(n):
        s = client.get_state()
        t[i] = time.monotonic() - t_start
        pen[i] = s.pendulum_pos_rad
        motor[i] = s.motor_pos_rad
        pvel[i] = s.pendulum_vel_rad_s
        mvel[i] = s.motor_vel_rad_s
        # Se espera hasta un instante ABSOLUTO (next_tick += dt) en lugar de
        # hacer sleep(dt): así, si una lectura tarda más, la siguiente espera
        # menos y el ritmo no acumula deriva.
        next_tick += dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    return dict(t=t, pen=pen, motor=motor, pvel=pvel, mvel=mvel)


def record_tare(client: LowLevelClient, out_dir: Path,
                duration_s: float = 3.0, sample_hz: float = 100.0) -> float:
    """Muestrea unos segundos con el péndulo en reposo para fijar el cero "colgado".

    El acumulador de ángulo del péndulo en el firmware tiene un desfase
    arbitrario (la lectura cruda del AS5600 al arrancar el Arduino es su
    cero). Grabar con todo en reposo nos permite fijar "colgado físicamente"
    a un valor conocido en el marco de coordenadas guardado, y luego expresar
    cada grabación posterior relativa a él. Elimina la clase de error del
    operador en la que se inicia el asistente con el péndulo en un ángulo
    desconocido.
    """
    _hr("Tare hanging position")
    print(_dim(
        "  Let the pendulum hang at rest (motor disengaged). Don't touch\n"
        "  it during the recording. We sample for a few seconds and use\n"
        "  the median as 'physical hanging = 0' for all downstream fits."
    ))
    client.disengage_motor()
    time.sleep(0.5)
    _prompt_enter("\n  Pendulum still and undisturbed?")

    print(_dim(f"  Recording {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)
    # Mediana en vez de media: es robusta a algún valor atípico (una lectura
    # I²C corrupta o un golpe) que sí desplazaría a la media.
    hanging_zero = float(np.median(rec["pen"]))
    # Desviación estándar = cuánto se movió durante el tare. En reposo
    # debería ser solo ruido del sensor (unos pocos mrad).
    motion = float(np.std(rec["pen"]))
    print(_dim(f"  Hanging zero: {hanging_zero:+.4f} rad   "
               f"(std during tare: {motion*1000:.2f} mrad)"))
    if motion > 0.01:   # 0.01 rad = 10 mrad ≈ 0.57°
        print(_yellow(
            "  WARNING: pendulum moved during tare (>10 mrad std). "
            "The hanging-zero estimate may be off; consider re-doing."))
    np.savez(out_dir / "tare.npz", **rec, hanging_zero=hanging_zero)
    return hanging_zero


def record_free_swing(
    client: LowLevelClient, out_dir: Path, run_i: int, n_runs: int,
    duration_s: float, sample_hz: float = 100.0,
) -> Path:
    """Graba un ensayo de oscilación libre. El motor está desactivado; el operador suelta."""
    print(_bold(f"\n  Run {run_i + 1}/{n_runs}"))
    print(_dim(
        "    Hold the motor arm firmly. Slide a box under the pendulum so\n"
        "    its tip rests on the box (the lift angle doesn't have to be\n"
        "    exact — the sensor will record whatever angle you start from).\n"
        "    When ready, press Enter. After the countdown, slide the box\n"
        "    out perpendicular to the swing plane to release."
    ))
    _prompt_enter("    Ready to release")
    print("    Releasing in...")
    _countdown(3)
    print(_dim(f"    Recording for {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)

    # Verificación rápida: ¿se movió realmente el péndulo?
    # Amplitud = máxima desviación respecto al valor medio:
    #   amp = max |θ(t) − media(θ)|
    amp = float(np.max(np.abs(rec["pen"] - np.mean(rec["pen"]))))
    if amp < 0.05:   # menos de ~2.9°: casi seguro que la suelta falló
        print(_yellow(f"    WARNING: only {amp:.3f} rad of pendulum motion "
                      "— did the release work?"))
    else:
        print(_dim(f"    -> max amplitude during recording: {amp:.3f} rad "
                   f"({math.degrees(amp):.0f}°)"))

    path = out_dir / f"free_run_{run_i + 1}.npz"
    np.savez(path, **rec)
    return path


def move_motor_to(
    client: LowLevelClient, target_pos: float, *,
    max_accel: float = 40.0, kp: float = 20.0, kd: float = 10.0,
    tolerance_rad: float = 0.03, control_hz: float = 50.0,
    timeout_s: float = 4.0, log: list | None = None,
    log_t_offset: float = 0.0,
) -> float:
    """Lleva el motor a `target_pos` con un controlador PD de lazo cerrado
    sobre la aceleración.

    Tanto bang-bang (demostrablemente óptimo en tiempo, en tiempo continuo)
    como el trapezoidal en lazo abierto con prealimentación (demostrablemente
    correcto con un actuador perfecto) fallaron en este robot, porque la
    cadena motor + FastAccelStepper tiene un retardo de respuesta de ~75 ms
    entre llamar a `moveByAcceleration` y que el motor cambie realmente de
    velocidad. Ese retardo es comparable al tiempo de movimiento por tramo y
    arruina ambos enfoques:
      - El "cambia a frenar justo aquí" de bang-bang nunca coincide con la
        realidad, porque el freno no hace efecto hasta 75 ms después.
      - El lazo abierto programa el freno demasiado pronto; el motor nunca
        alcanza el tope de velocidad y se queda muy corto del objetivo.

    Un PD con ganancias moderadas esquiva el problema. Las aceleraciones
    comandadas se mantienen bien por debajo de `max_accel` para errores
    típicos, así que nunca empujamos al motor al régimen donde importa el
    retardo de respuesta. El motor simplemente se rezaga un poco y el PD
    sigue empujando hasta que todo converge.

    Ganancias elegidas para mantener la aceleración comandada
    aproximadamente en [-max_accel/2, max_accel/2] en un movimiento de
    ±π/2. Ganancias más duras vuelven al régimen de chatter; más blandas
    tardan más en asentarse.
    """
    dt = 1.0 / control_hz
    next_tick = time.monotonic()
    t_start = next_tick
    # "Asentado" = cumplir la tolerancia durante 0.3 s seguidos
    # (a 50 Hz son 15 ticks consecutivos).
    settle_count = 0
    settle_needed = int(0.3 * control_hz)
    while time.monotonic() - t_start < timeout_s:
        s = client.get_state()
        motor_pos = s.motor_pos_rad
        motor_vel = s.motor_vel_rad_s

        # Ley de control PD sobre la aceleración:
        #     error = objetivo − posición
        #     accel = kp·error − kd·velocidad
        # El término kp empuja hacia el objetivo; el término kd frena en
        # proporción a la velocidad, para no pasarse (amortiguamiento).
        #
        # Sin retardo, esto equivale a  ë + kd·ė + kp·e = 0, es decir un
        # oscilador amortiguado con
        #     ω_n = √kp = √20 ≈ 4.47 rad/s
        #     ζ   = kd / (2·√kp) = 10 / (2·4.47) ≈ 1.12   (≥ 1: sin oscilar)
        # Con kp=20, el comando inicial para un movimiento de π/2 es
        # 20·(π/2) ≈ 31 rad/s², por debajo de max_accel = 40.
        error = target_pos - motor_pos
        accel = kp * error - kd * motor_vel
        accel = float(np.clip(accel, -max_accel, max_accel))

        client.set_acceleration(accel)

        if log is not None:
            log.append((time.monotonic() - t_start + log_t_offset,
                        motor_pos, motor_vel, accel, target_pos))

        # Criterio de llegada: cerca del objetivo (0.03 rad ≈ 1.7°) Y casi
        # parado (< 0.3 rad/s). Si se rompe en algún tick, el contador reinicia.
        if abs(error) < tolerance_rad and abs(motor_vel) < 0.3:
            settle_count += 1
            if settle_count >= settle_needed:
                break
        else:
            settle_count = 0

        next_tick += dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Al terminar, aceleración cero: el motor deja de ser empujado.
    client.set_acceleration(0.0)
    return time.monotonic() - t_start


def record_motor_sweep(client: LowLevelClient, out_dir: Path) -> Path:
    """Mueve el motor por 0 → +90° → 0 → −90° → 0 y graba la trayectoria.

    El operador observa el robot: si el motor se queda visiblemente corto de
    las marcas de ±90°, se están saltando pasos y el contador de posición del
    firmware nos está mintiendo. También se detecta en la traza grabada.
    """
    _hr("Motor ±90° sanity sweep")
    print(_dim(
        "  We drive the motor through 0 → +90° → 0 → −90° → 0 in accel\n"
        "  mode (same as the policy uses). Watch the rig: the motor arm\n"
        "  should reach roughly perpendicular to its starting position\n"
        "  at the peaks. If it visibly under-shoots, the stepper has\n"
        "  skipped steps and the firmware position no longer matches\n"
        "  the actual mechanical position — that's a deployment-blocker.\n"
    ))
    print(_yellow("  Clear space around the rig; the arm will sweep ~180°."))
    _prompt_enter("  Ready")

    client.set_acceleration(0.0)
    client.engage_motor()
    time.sleep(0.5)

    # Reinicio del acumulador: se asume que el operador colocó el brazo donde
    # quiere que esté el "0", y que el contador de pasos del firmware vale 0 ahí.
    log: list = []
    cumulative_t = 0.0
    targets = [
        (0.0, "centre"),
        (+math.pi / 2, "+90°"),    # π/2 rad = 90°
        (0.0, "centre"),
        (-math.pi / 2, "−90°"),
        (0.0, "centre"),
    ]
    for tgt, label in targets:
        print(_dim(f"  → {label} ({math.degrees(tgt):+.0f}°)..."))
        elapsed = move_motor_to(client, tgt, log=log, log_t_offset=cumulative_t)
        cumulative_t += elapsed
        # Se mantiene un instante en el pico para que el operador lo verifique
        # a ojo. Esa pausa también se suma al tiempo acumulado del registro.
        time.sleep(0.4)
        cumulative_t += 0.4

    client.set_acceleration(0.0)
    time.sleep(0.2)
    client.disengage_motor()

    # El registro es una lista de tuplas; se pasa a una matriz (n × 5) y se
    # guarda cada columna con su nombre.
    log_arr = np.asarray(log, dtype=np.float64)
    path = out_dir / "motor_sweep.npz"
    np.savez(path,
             t=log_arr[:, 0],
             motor=log_arr[:, 1],
             mvel=log_arr[:, 2],
             accel_cmd=log_arr[:, 3],
             target=log_arr[:, 4])
    print(_dim(f"  Saved {len(log_arr)} samples to {path.name}"))
    return path


# ---------------------------------------------------------------------------
# Guardado y carga de metadatos
# ---------------------------------------------------------------------------

def save_metadata(out_dir: Path, **kwargs) -> None:
    """Escribe (o actualiza) `metadata.json` en el directorio de la sesión."""
    meta_path = out_dir / "metadata.json"
    existing: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.update(kwargs)
    # setdefault: la marca de tiempo se fija solo la primera vez que se guarda.
    existing.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_metadata(in_dir: Path) -> dict:
    with open(in_dir / "metadata.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fase de recolección (collect) — depende del dispositivo
# ---------------------------------------------------------------------------

def collect_data(args: argparse.Namespace) -> Path:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    print(_bold(f"\nRecording session: {out_dir}"))
    print(_dim(f"Port: {port}"))
    print(_dim(
        "  Pendulum geometry (mass, COM, I_com) is loaded from model/model.urdf"
        " — no operator measurement needed."
    ))

    # El `with` abre el puerto y, al salir (incluso por error), desactiva el motor.
    with LowLevelClient(port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print(_red("Arduino did not respond. Is LowLevelServer.ino flashed?"))
            sys.exit(2)
        print(_green("  Arduino ready.\n"))

        # Paso 1: tare (fija el cero "colgado")
        hanging_zero = record_tare(client, out_dir)

        # Se guarda ya el cero y los parámetros de la sesión: así `fit` puede
        # correr después sin haber visto el robot.
        save_metadata(out_dir,
                      port=port,
                      baud=args.baud,
                      hanging_zero_rad=hanging_zero,
                      n_free_runs=args.n_free_runs,
                      free_swing_duration_s=args.duration_s,
                      sample_hz=args.sample_hz,
                      firmware="LowLevelServer.ino")

        # Paso 2: oscilación libre × N (varias corridas para promediar el ruido)
        _hr(f"Free-swing × {args.n_free_runs}")
        for i in range(args.n_free_runs):
            while True:
                try:
                    record_free_swing(client, out_dir, i, args.n_free_runs,
                                       args.duration_s, args.sample_hz)
                    break
                except Exception as e:
                    print(_red(f"    recording failed: {e}"))
                    if not _prompt_yes_no("    Retry?", default=True):
                        sys.exit(3)

        # Paso 3: barrido de verificación del motor (opcional)
        if not args.skip_motor:
            record_motor_sweep(client, out_dir)

    print(_green(f"\nCollection complete: {out_dir}\n"))
    return out_dir


# ---------------------------------------------------------------------------
# Fase de ajuste (fit) — post-procesamiento puro
# ---------------------------------------------------------------------------

def fit_data(in_dir: Path, out_json: Path) -> dict:
    _hr("Fit + analysis")
    print(_dim(f"  In:  {in_dir}"))
    print(_dim(f"  Out: {out_json}\n"))

    meta = load_metadata(in_dir)
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        print(_red("  No free_run_*.npz files in directory."))
        sys.exit(4)

    print(_dim(f"  {len(free_files)} free-swing recording(s)"))
    fits: list[dict] = []
    for path in free_files:
        rec = dict(np.load(path))
        # Se resta el cero "colgado" para que θ = 0 sea el péndulo en reposo
        # abajo, que es el equilibrio alrededor del cual oscila.
        pen_centered = rec["pen"] - hanging_zero
        try:
            fit = fit_free_swing(rec["t"], pen_centered)
        except Exception as e:
            print(_red(f"    {path.name}: fit failed ({e})"))
            continue
        fits.append(fit)
        # T = periodo medido; T₀ = periodo extrapolado a amplitud pequeña;
        # α = constante de decaimiento de la envolvente.
        print(_dim(
            f"    {path.name}: T={fit['period_s']*1000:.0f} ms  "
            f"T₀={fit['small_amp_period_s']*1000:.0f} ms  "
            f"α={fit['decay_constant_s_inv']:.4f} 1/s  "
            f"peaks={fit['n_extrema_used']}"
        ))
    if not fits:
        print(_red("  All fits failed."))
        sys.exit(5)

    # Se combinan las corridas (mediana) y de ahí se derivan las fricciones.
    fit = aggregate_free_swing_fits(fits)
    derived = derive_pendulum_friction(fit)

    # Comprobación cruzada de la inercia: la medida a partir del periodo
    # pequeño frente a la que predice el CAD.
    #   error relativo = |I_medida − I_CAD| / I_CAD
    # Un error de pocos % confirma que el URDF describe bien la pieza impresa.
    rel_err = abs(
        derived["inertia_measured_kg_m2"] - derived["inertia_predicted_kg_m2"]
    ) / derived["inertia_predicted_kg_m2"]
    print(_bold("\n  Derived parameters:"))
    print(f"    viscous friction:        {derived['viscous_friction_N_m_s']:.4e} N·m·s")
    print(f"    Coulomb friction:        {derived['coulomb_friction_N_m']:.4e} N·m")
    print(_dim(
        f"    I (CAD-predicted):       {derived['inertia_predicted_kg_m2']:.4e} kg·m²"
    ))
    print(_dim(
        f"    I (measured from T₀):    {derived['inertia_measured_kg_m2']:.4e} kg·m²  "
        f"({rel_err*100:+.1f}% vs CAD)"
    ))

    # Validación: avisa si algo se ve fuera de lo razonable.
    warnings = validate_free_swing(fit, derived)
    if warnings:
        print()
        _print_warnings(warnings)

    # Ventana sugerida de frecuencia de control. (Sin una medición del ancho
    # de banda del motor, solo imprimimos la mitad del péndulo; el operador
    # decide.)
    print(_bold("\n  Control rate suggestion (based on pendulum dynamics alone):"))
    # Import local: la ruta `fit` es deliberadamente libre de dispositivo y de
    # MuJoCo, e importar pendulum_env a nivel de módulo arrastraría MuJoCo
    # (misma razón por la que el import del entorno en la función de gráficas
    # también es local).
    from furuta.sim.pendulum_env import CONTROL_FREQ_HZ
    # Frecuencia natural del péndulo:  f_n = 1 / T₀
    # Regla práctica: muestrear/controlar a al menos 5·f_n para no perder la dinámica.
    # (Con el robot de referencia, T₀ ≈ 0.50 s → f_n ≈ 2 Hz → mínimo ≈ 10 Hz.)
    pendulum_period = fit["small_amp_period_s"]
    f_n = 1.0 / pendulum_period
    print(f"    pendulum natural freq f_n ≈ {f_n:.2f} Hz")
    print(f"    minimum useful control rate (5·f_n) ≈ {5*f_n:.1f} Hz")
    print(_dim(f"    (the rig deploys at {CONTROL_FREQ_HZ:.0f} Hz; this run won't change that)"))

    # Se escribe el JSON. Esquema: solo la fricción es por robot y vive aquí;
    # masa/COM/I_com salen de model/model.urdf vía pendulum_geometry. El par
    # de inercia medida/predicha se incluye para trazabilidad.
    pendulum_payload = {
        "derived": {
            "viscous_friction_N_m_s": derived["viscous_friction_N_m_s"],
            "coulomb_friction_N_m": derived["coulomb_friction_N_m"],
            "inertia_predicted_kg_m2": derived["inertia_predicted_kg_m2"],
            "inertia_measured_kg_m2": derived["inertia_measured_kg_m2"],
        },
        "fit": fit,
    }
    run_meta = {
        "wizard_version": "2026-05-20",
        "collected_at": meta.get("timestamp"),
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir.resolve()),
        "n_free_swing_runs": len(fits),
        "hanging_zero_rad": hanging_zero,
        "firmware": meta.get("firmware", "unknown"),
    }
    write_params_json(str(out_json),
                      pendulum_payload=pendulum_payload, run_meta=run_meta)
    print(_green(f"\n  Wrote {out_json}"))

    # Genera las gráficas de validación
    plot_free_swing_compare(in_dir, derived, out_dir=in_dir)
    if (in_dir / "motor_sweep.npz").exists():
        plot_motor_sweep(in_dir / "motor_sweep.npz", out_dir=in_dir)

    return derived


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def plot_free_swing_compare(in_dir: Path, derived: dict, *, out_dir: Path) -> None:
    """Reproduce en simulación la oscilación libre más larga y compara envolventes.

    Se alinea en el tiempo con el momento real de la suelta, no con el inicio
    de la grabación: el operador suele sostener el péndulo sobre la caja
    durante una fracción de segundo tras pulsar GO. Detectamos la suelta como
    la primera muestra cuya velocidad del péndulo supera un umbral pequeño,
    inicializamos la simulación con el estado (pos, vel) de esa muestra y
    graficamos ambas trazas con ese instante como t=0.
    """
    # Imports locales: matplotlib y MuJoCo solo hacen falta aquí.
    import matplotlib.pyplot as plt
    import mujoco
    from furuta.sim.pendulum_env import RotaryInvertedPendulumEnv

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        return
    rec = dict(np.load(free_files[0]))
    meta = load_metadata(in_dir)
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    real_t = rec["t"]
    real_pen = rec["pen"] - hanging_zero
    real_pvel = rec["pvel"]
    sample_dt = float(np.median(np.diff(real_t)))

    # Busca el índice de la suelta: primera muestra con |vel_péndulo| > 0.5
    # rad/s (bien por encima del ruido del sensor, bien por debajo de las
    # velocidades típicas de suelta).
    moving = np.where(np.abs(real_pvel) > 0.5)[0]
    rel_idx = int(moving[0]) if len(moving) else 0
    # Se retrocede una muestra para que la traza muestre el inicio del movimiento.
    rel_idx = max(0, rel_idx - 1)

    # Se recortan ambas trazas para que empiecen en la suelta.
    real_t_trim = real_t[rel_idx:] - real_t[rel_idx]
    real_pen_trim = real_pen[rel_idx:]
    release_pos = float(real_pen_trim[0])
    release_vel = float(real_pvel[rel_idx])

    # Simulación con los parámetros derivados que carga el entorno. Nota: el
    # entorno lee los parámetros de sysid al construirse, así que dependemos
    # de que `fit_data` acabe de escribir ese archivo.
    env = RotaryInvertedPendulumEnv(
        control_freq_hz=200.0, domain_randomization=False,
        episode_length_s=real_t_trim[-1] + 0.5,
    )
    env.reset(seed=0)
    # Se fuerza el estado inicial de MuJoCo: brazo quieto en 0 y péndulo con
    # la posición y velocidad reales del instante de la suelta.
    # (qpos/qvel = posiciones/velocidades generalizadas de las articulaciones.)
    env.data.qpos[env._motor_qpos_addr] = 0.0
    env.data.qpos[env._pen_qpos_addr]   = release_pos
    env.data.qvel[env._motor_qvel_addr] = 0.0
    env.data.qvel[env._pen_qvel_addr]   = release_vel
    env._motor_target = 0.0; env._motor_vel = 0.0
    mujoco.mj_forward(env.model, env.data)

    n = len(real_t_trim)
    sim_pen = np.zeros(n)
    sim_dt = 1.0 / 200.0     # un paso del entorno = 5 ms a 200 Hz
    t = 0.0; idx = 0; next_t = 0.0
    while idx < n:
        # Acción 0: el motor no se mueve, igual que en la grabación real
        # (péndulo libre).
        env.step(np.array([0.0], dtype=np.float32))
        t += sim_dt
        # Remuestreo: la simulación avanza a 200 Hz y la grabación real está
        # a ~100 Hz; se toma una muestra simulada cada vez que se alcanza el
        # siguiente instante de muestreo real.
        while idx < n and t >= next_t:
            sim_pen[idx] = float(env.data.qpos[env._pen_qpos_addr])
            idx += 1; next_t += sample_dt

    # Se usan los arrays recortados en el resto de la función.
    real_t = real_t_trim
    real_pen = real_pen_trim

    # Ajuste de envolventes
    def _fit(t, y):
        # 1) Cruces por cero: índices donde cambia el signo de y.
        crossings = np.where(np.diff(np.sign(y)) != 0)[0]
        pt, pa = [], []
        # 2) Entre dos cruces consecutivos hay una semioscilación; su pico
        #    es el punto de mayor |y| en ese tramo.
        for k in range(len(crossings) - 1):
            a, b = crossings[k] + 1, crossings[k + 1] + 1
            j = a + int(np.argmax(np.abs(y[a:b])))
            pt.append(t[j]); pa.append(abs(y[j]))
        pt, pa = np.array(pt), np.array(pa)
        if len(pt) < 3:
            return None
        # 3) Dos picos consecutivos (de signo alterno) están separados medio
        #    periodo:  T = 2 · mediana(Δt entre picos).
        period = 2 * np.median(np.diff(pt))
        # 4) La envolvente decae como  A(t) = A₀·e^(−α·t),  luego
        #       ln A = ln A₀ − α·t
        #    es una recta de pendiente −α. Un ajuste lineal (polyfit grado 1)
        #    sobre (t, ln A) da la pendiente; α = −pendiente.
        #    El 1e-9 evita ln(0) si algún pico fuera exactamente cero.
        decay = -float(np.polyfit(pt, np.log(pa + 1e-9), 1)[0])
        return period, decay, pt, pa

    # Se descartan los primeros 0.3 s (el transitorio de la suelta).
    ix_start = int(0.3 / sample_dt)
    fr = _fit(real_t[ix_start:], real_pen[ix_start:])    # real
    fs = _fit(real_t[ix_start:], sim_pen[ix_start:])     # simulación

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(real_t, real_pen, label="real", lw=1.0)
    axes[0].plot(real_t, sim_pen, label="sim (from derived params)",
                  lw=1.0, alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("angle from hanging (rad)")
    axes[0].grid(alpha=0.3)
    if fr and fs:
        # Δ% = (sim/real − 1)·100: cuánto se desvía la simulación del robot real.
        axes[0].set_title(
            f"real T={fr[0]*1000:.0f} ms  τ={fr[1]:.3f}/s  vs  "
            f"sim T={fs[0]*1000:.0f} ms  τ={fs[1]:.3f}/s  "
            f"(Δ {(fs[0]/fr[0]-1)*100:+.1f}%, {(fs[1]/fr[1]-1)*100:+.1f}%)"
        )
        # Eje logarítmico: una exponencial decreciente se ve como una recta,
        # y la pendiente es −α. Si real y sim son paralelas, la fricción coincide.
        axes[1].semilogy(fr[2], fr[3], "o-", label=f"real peaks (τ={fr[1]:.3f}/s)")
        axes[1].semilogy(fs[2], fs[3], "s--", label=f"sim peaks (τ={fs[1]:.3f}/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("|peak amplitude| (rad, log)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = out_dir / "freeswing_compare.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))


def plot_motor_sweep(npz_path: Path, *, out_dir: Path) -> None:
    """Grafica el objetivo del motor frente al real durante el barrido de ±90°.

    Las grabaciones antiguas, hechas antes del 2026-05-20, escribían el
    tiempo por tramo (cada `move_motor_to` reiniciaba t a 0), lo que hace
    que la traza parezca tramos superpuestos. Se detecta ese caso y se
    reconstruye un eje de tiempo acumulado a partir de los reinicios.
    """
    import matplotlib.pyplot as plt
    d = dict(np.load(npz_path))
    t = d["t"].copy()

    # Reconstrucción del tiempo acumulado para grabaciones antiguas por tramo.
    # Si t retrocede entre dos muestras, cada retroceso marca el inicio de una
    # nueva llamada a move_motor_to. Se usa el valor acumulado de la muestra
    # anterior como nuevo desplazamiento.
    dt = np.diff(t)
    if (dt < 0).any():
        offset = 0.0
        prev = 0.0
        for i in range(len(t)):
            if i > 0 and d["t"][i] < d["t"][i - 1]:
                offset = prev
            t[i] = d["t"][i] + offset
            prev = t[i]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, np.degrees(d["motor"]),  label="motor (firmware-reported)")
    axes[0].plot(t, np.degrees(d["target"]), "--", label="target", alpha=0.7)
    axes[0].axhline(+90, color="k", lw=0.3); axes[0].axhline(-90, color="k", lw=0.3)
    axes[0].axhline(0, color="k", lw=0.2)
    axes[0].set_ylabel("angle (°)"); axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor ±90° sanity sweep — firmware tracks commanded angle")
    axes[1].plot(t, d["accel_cmd"], label="commanded accel (rad/s²)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("accel (rad/s²)")
    axes[1].grid(alpha=0.3); axes[1].legend(loc="upper right")
    plt.tight_layout()
    fig_path = out_dir / "motor_sweep.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))


# ---------------------------------------------------------------------------
# Punto de entrada independiente de validación del motor (solo graba su
# propio barrido del motor, sin oscilación libre)
# ---------------------------------------------------------------------------

def validate_motor(args: argparse.Namespace) -> None:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    with LowLevelClient(port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print(_red("Arduino did not respond.")); sys.exit(2)
        record_motor_sweep(client, out_dir)
    plot_motor_sweep(out_dir / "motor_sweep.npz", out_dir=out_dir)


# ---------------------------------------------------------------------------
# Principal (main)
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common_device(sp):
        # Opciones comunes a todo lo que habla con el robot.
        sp.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
        sp.add_argument("--baud", type=int, default=2_000_000)
        sp.add_argument("--out-dir", default=None,
                        help="output directory (auto-timestamped if omitted)")

    pc = sub.add_parser("collect", help="acquire data only")
    common_device(pc)
    pc.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    pc.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    pc.add_argument("--sample-hz", type=float, default=100.0)
    pc.add_argument("--skip-motor", action="store_true",
                    help="skip the motor ±90° sanity sweep")

    pf = sub.add_parser("fit", help="fit parameters from a collected directory")
    pf.add_argument("--in-dir", required=True, type=Path)
    pf.add_argument("--out-json", required=True, type=Path,
                    help="where to write the fitted parameters, e.g. "
                         "sysid_params_rig1.json. Required rather than "
                         "defaulted: friction is per-rig, so writing to a "
                         "generic filename is how one rig's measurements end "
                         "up silently describing another.")

    pv = sub.add_parser("validate-motor", help="motor ±90° sanity sweep only")
    common_device(pv)

    # Por defecto (sin subcomando): el flujo completo.
    common_device(p)
    p.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    p.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    p.add_argument("--sample-hz", type=float, default=100.0)
    p.add_argument("--skip-motor", action="store_true")
    p.add_argument("--out-json", required=True, type=Path,
                   help="where to write the fitted parameters, e.g. "
                        "sysid_params_rig1.json. Required rather than "
                        "defaulted: friction is per-rig, so writing to a "
                        "generic filename is how one rig's measurements end "
                        "up silently describing another.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd == "collect":
        collect_data(args)
    elif args.cmd == "fit":
        fit_data(args.in_dir, args.out_json)
    elif args.cmd == "validate-motor":
        validate_motor(args)
    else:
        # Flujo completo: grabar y luego ajustar sobre lo recién grabado.
        out_dir = collect_data(args)
        fit_data(out_dir, args.out_json)
        print(_bold("\nDone. Use the policy training scripts next:"))
        print(_dim(f"  ./curriculum_train.sh <run-prefix>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
