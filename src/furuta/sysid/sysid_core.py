"""Funciones matemáticas de sysid compartidas, usadas por `sysid_wizard.py`.

Este módulo es la fuente única de verdad para:
  - el ajuste de oscilación libre del péndulo amortiguado (`fit_free_swing`)
  - la derivación de fricción con corrección elíptica (`derive_pendulum_friction`)
  - la agregación de varias corridas de oscilación libre
  - los mensajes de verificación / validación

La geometría del péndulo (masa, COM, I_com) *no* se ajusta aquí: viene del
URDF a través de `pendulum_geometry`. El pipeline de sysid solo mide las
magnitudes que realmente varían por robot: la fricción viscosa y la de Coulomb.

Nada aquí imprime ni pregunta. El asistente (wizard) se encarga de toda la E/S.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks
from scipy.special import ellipk


GRAVITY = 9.81  # m/s^2


def _elliptic_period_factor(amplitude_rad: float) -> float:
    """Devuelve T(amplitud) / T₀ para un péndulo gravitatorio simple.

    El periodo exacto de un péndulo con amplitud pico θ_max es
        T(θ_max) = T₀ · (2/π) · K(sin²(θ_max/2))
    donde K es la integral elíptica completa de primera especie y
    T₀ = 2π·sqrt(I/(m·g·d)) es el periodo de amplitud pequeña. Para
    θ_max → 0 el factor → 1; a 90° (π/2) vale ≈ 1.18.

    Sirve para corregir el periodo medido (a amplitud finita) hasta el T₀ de
    amplitud pequeña, que es el que entra en la fórmula de la inercia.
    """
    # scipy.special.ellipk recibe el parámetro m = sin²(θ_max/2).
    m = math.sin(amplitude_rad / 2.0) ** 2
    return float((2.0 / math.pi) * ellipk(m))


DEFAULT_FREE_SWING_DURATION_S: float = 10.0
DEFAULT_FREE_SWING_RUNS: int = 3


# ---------------------------------------------------------------------------
# Ajuste de oscilación libre
# ---------------------------------------------------------------------------

def fit_free_swing(time_s: np.ndarray, pendulum_rad: np.ndarray) -> dict:
    """Ajusta los parámetros de un oscilador amortiguado a datos de oscilación libre.

    Periodo: separación media entre extremos consecutivos del mismo lado.

    Amortiguamiento: ajusta un modelo combinado de fricción viscosa +
    Coulomb. La envolvente de picos de un péndulo con fricción viscosa
    (b·θ̇) y de Coulomb (F_c·sign(θ̇)) cumple
        |P_{n+1}| = r * |P_n| - delta_c
    por cada medio ciclo, donde r = exp(-π ζ / sqrt(1-ζ²)) (viscosa) y
    delta_c = 2 F_c / (m g l) (escalón de Coulomb). Es lineal en (r, delta_c)
    y se ajusta trivialmente por mínimos cuadrados. También se calcula el
    decremento logarítmico de viscosidad pura, solo para comparar.
    """
    n = len(pendulum_rad)
    # Posición de reposo = media del último cuarto de la grabación (cuando el
    # péndulo ya casi se detuvo); se resta para oscilar alrededor de 0.
    rest = float(np.mean(pendulum_rad[3 * n // 4:]))
    p = pendulum_rad - rest

    abs_max = float(np.max(np.abs(p)))
    if abs_max < 1e-3:
        raise RuntimeError(
            "Pendulum did not move appreciably during the recording. "
            "Did you displace it before release?"
        )
    # Prominencia mínima para aceptar un pico: 2 % de la amplitud máxima (y al
    # menos 5 mrad), para no contar ruido como oscilaciones.
    prominence = max(0.02 * abs_max, 5e-3)

    # Picos positivos (máximos de p) y negativos (máximos de −p).
    pos_peaks, _ = find_peaks(p, prominence=prominence)
    neg_peaks, _ = find_peaks(-p, prominence=prominence)
    extrema = np.sort(np.concatenate([pos_peaks, neg_peaks]))
    if len(extrema) < 4:
        raise RuntimeError(
            f"Only {len(extrema)} oscillation extrema detected; need at least 4. "
            "Record longer or release with a larger initial angle."
        )

    extrema_t = time_s[extrema]
    extrema_v = p[extrema]

    # Periodo por parejas, a partir de picos del mismo lado (de pico a pico a
    # lo largo de un ciclo completo). Los extremos alternan de signo, así que
    # tomar uno de cada dos deja los del mismo lado.
    same_side_times = extrema_t[::2]
    same_side_amps = np.abs(extrema_v[::2])
    same_side = np.diff(same_side_times)
    period_s = float(np.mean(same_side))
    if period_s <= 0:
        raise RuntimeError("Failed to estimate a positive oscillation period.")
    # Frecuencia angular amortiguada:  ω_d = 2π / T
    omega_d = 2.0 * math.pi / period_s

    # T₀ de amplitud pequeña mediante la corrección de la integral elíptica.
    # El periodo medido T(θ) a amplitud finita es más largo que T₀; sin esta
    # corrección, la inercia derivada queda sesgada al alza con amplitudes
    # grandes (la fórmula de amplitud pequeña I = m·g·d·T²/(4π²) necesita T₀,
    # no T(θ)). Se calcula T₀ por cada pareja consecutiva de picos del mismo
    # lado usando la media de sus amplitudes, y luego se promedia. Si alguna
    # corrección falla (p. ej. amplitud > π) se usa el periodo sin corregir.
    pair_amps = (same_side_amps[:-1] + same_side_amps[1:]) / 2.0
    try:
        t0_estimates = np.array([
            period / _elliptic_period_factor(a)
            for period, a in zip(same_side, pair_amps)
            if a < math.pi
        ])
        small_amp_period_s = float(np.mean(t0_estimates)) if len(t0_estimates) else period_s
    except Exception:
        small_amp_period_s = period_s
    omega_n_small_amp = 2.0 * math.pi / small_amp_period_s

    # Modelo A — solo viscosa: la envolvente decae como A(t) = A₀·e^(−α·t),
    # luego  ln|pico| = ln A₀ − α·t  es una recta. Mínimos cuadrados sobre
    # (t, ln|pico|): la pendiente es −α y la ordenada es ln A₀.
    log_amp = np.log(np.abs(extrema_v))
    A = np.vstack([extrema_t, np.ones_like(extrema_t)]).T
    slope, intercept = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    alpha_visc_only = -float(slope)
    if alpha_visc_only <= 0:
        raise RuntimeError(
            "Decay rate is non-positive; the recording does not show a "
            "decaying envelope. Check that the pendulum is freely swinging."
        )

    # Modelo B — viscosa + Coulomb: regresión lineal de cada pico contra el
    # anterior,
    #     |P_{n+1}| = r·|P_n| − δ_c
    # La matriz M tiene columnas [P_n, 1]; la solución es [r, −δ_c].
    abs_peaks = np.abs(extrema_v)
    Pn = abs_peaks[:-1]
    Pn1 = abs_peaks[1:]
    M = np.vstack([Pn, np.ones_like(Pn)]).T
    sol, *_ = np.linalg.lstsq(M, Pn1, rcond=None)
    # r debe estar en (0, 1): se acota para evitar log(0) o log de algo ≥ 1.
    r_per_half = float(np.clip(sol[0], 1e-6, 1.0 - 1e-9))
    delta_c_per_half = float(max(-sol[1], 0.0))

    # De r al factor de amortiguamiento ζ. De  r = exp(−π ζ / √(1−ζ²))  sale
    #     ζ = (−ln r / π) · √(1 − ζ²)
    # que tiene ζ a ambos lados: se resuelve iterando (punto fijo, 8 pasadas,
    # partiendo de la aproximación ζ ≈ −ln r / π).
    log_r = -math.log(r_per_half)
    zeta = log_r / math.pi
    for _ in range(8):
        zeta = log_r / math.pi * math.sqrt(max(1 - zeta * zeta, 1e-12))
    # Frecuencia natural sin amortiguar:  ω_n = ω_d / √(1 − ζ²)
    omega_n_combined = omega_d / math.sqrt(max(1 - zeta * zeta, 1e-12))

    # Qué tan bien reproduce cada modelo los picos medidos (error RMS):
    # el combinado se simula con la recurrencia; el viscoso, con la exponencial.
    pred_combined = np.empty_like(abs_peaks)
    pred_combined[0] = abs_peaks[0]
    for i in range(1, len(pred_combined)):
        pred_combined[i] = max(r_per_half * pred_combined[i - 1] - delta_c_per_half, 0.0)
    pred_visc_only = abs_peaks[0] * np.exp(-alpha_visc_only * (extrema_t - extrema_t[0]))
    env_rmse_combined = float(np.sqrt(np.mean((pred_combined - abs_peaks) ** 2)))
    env_rmse_visc_only = float(np.sqrt(np.mean((pred_visc_only - abs_peaks) ** 2)))

    return {
        "n_extrema_used": int(len(extrema)),
        "period_s": period_s,
        "small_amp_period_s": small_amp_period_s,
        "mean_pair_amplitude_rad": float(np.mean(pair_amps)) if len(pair_amps) else 0.0,
        "omega_d_rad_s": omega_d,
        "omega_n_rad_s": omega_n_combined,
        "omega_n_small_amp_rad_s": omega_n_small_amp,
        "zeta": zeta,
        "rest_angle_rad": rest,
        # α = ζ·ω_n : constante de decaimiento de la envolvente (1/s)
        "decay_constant_s_inv": float(zeta * omega_n_combined),
        "initial_amplitude_rad": float(math.exp(intercept)),
        "r_per_half_cycle": r_per_half,
        "delta_c_per_half_cycle_rad": delta_c_per_half,
        "decay_constant_visc_only_s_inv": alpha_visc_only,
        "envelope_rmse_combined_rad": env_rmse_combined,
        "envelope_rmse_visc_only_rad": env_rmse_visc_only,
    }


def derive_pendulum_friction(fit: dict, *, gravity: float = GRAVITY) -> dict:
    """Deriva la fricción viscosa + Coulomb a partir de un ajuste de oscilación
    libre, usando como constantes la geometría del péndulo definida en el URDF
    (masa, COM, I_com).

    La masa y el COM ya no provienen de mediciones tecleadas por el operador:
    son propiedades geométricas del cuerpo del péndulo, fijadas por el CAD de
    Onshape y materializadas en `model/model.urdf`. El pipeline de sysid solo
    mide lo que realmente varía por robot (rodamientos, grasa, temperatura).

    Fórmulas de inercia:
    - I_predicted = m·d² + I_com_swing — el modelo CAD (teorema de Steiner).
    - I_measured  = m·g·d / ω_n_small² — a partir del periodo de amplitud
      pequeña (corregido con la integral elíptica desde el ajuste a amplitud
      finita). Se reporta para contrastar con I_predicted; una discrepancia
      apunta a un URDF desactualizado o a un péndulo con otra geometría.

    La fricción usa I_predicted (la inercia respecto al pivote, constante del
    CAD), de modo que los valores derivados quedan desacoplados del ruido de
    periodo de cada corrida:
    - viscosa b   = 2·α·I_predicted (envolvente de decaimiento: b/(2I) = α).
    - Coulomb F_c = δ_c · m · g · d / 2 (escalón de Coulomb por medio ciclo).
    """
    # Import local: la geometría solo se lee del URDF cuando se necesita.
    from furuta.sim.pendulum_geometry import (
        PENDULUM_COM_M,
        PENDULUM_I_COM_SWING_KG_M2,
        PENDULUM_MASS_KG,
    )

    omega_n_small = fit.get("omega_n_small_amp_rad_s", fit["omega_n_rad_s"])
    decay = float(fit["decay_constant_s_inv"])  # = α = b/(2I)

    inertia_predicted = (
        PENDULUM_MASS_KG * PENDULUM_COM_M ** 2 + PENDULUM_I_COM_SWING_KG_M2
    )
    inertia_measured = (
        PENDULUM_MASS_KG * gravity * PENDULUM_COM_M / (omega_n_small ** 2)
    )
    viscous = 2.0 * decay * inertia_predicted

    delta_c = float(fit.get("delta_c_per_half_cycle_rad", 0.0))
    coulomb = delta_c * PENDULUM_MASS_KG * gravity * PENDULUM_COM_M / 2.0

    return {
        "viscous_friction_N_m_s": viscous,
        "coulomb_friction_N_m": coulomb,
        "inertia_predicted_kg_m2": inertia_predicted,
        "inertia_measured_kg_m2": inertia_measured,
    }


# ---------------------------------------------------------------------------
# Agregación de varias corridas de oscilación libre
# ---------------------------------------------------------------------------

# Claves para las que se reporta mediana + desviación estándar entre
# corridas, y se toma la mediana como valor canónico. Las demás claves se
# toman de la corrida mediana por legibilidad (no tiene sentido promediar
# "n_extrema_used", etc.).
_AGGREGATE_KEYS: tuple[str, ...] = (
    "period_s",
    "small_amp_period_s",
    "omega_d_rad_s",
    "omega_n_rad_s",
    "omega_n_small_amp_rad_s",
    "zeta",
    "decay_constant_s_inv",
    "r_per_half_cycle",
    "delta_c_per_half_cycle_rad",
)


def aggregate_free_swing_fits(fits: list[dict]) -> dict:
    """Combina N ajustes de oscilación libre en un único diccionario canónico.

    Devuelve un dict compatible en esquema con la salida de una sola corrida
    de `fit_free_swing` (para que los consumidores sigan funcionando). El
    valor canónico de cada clave agregada es la mediana entre corridas; las
    listas por corrida y las desviaciones estándar se exponen bajo
    `*_all_runs` / `*_stdev` para diagnóstico.
    """
    if not fits:
        raise ValueError("aggregate_free_swing_fits called with no fits")
    if len(fits) == 1:
        out = dict(fits[0])
        out["n_runs"] = 1
        return out

    # Se elige como "donante" de los escalares de diagnóstico no agregados
    # (rest_angle_rad, initial_amplitude_rad, RMSE de envolvente, etc.) la
    # corrida cuyo periodo está más cerca de la mediana. Después, las claves
    # agregadas se sobrescriben con las medianas.
    periods = np.asarray([f["period_s"] for f in fits], dtype=float)
    median_idx = int(np.argmin(np.abs(periods - float(np.median(periods)))))
    out = dict(fits[median_idx])

    for k in _AGGREGATE_KEYS:
        values = np.asarray([f[k] for f in fits], dtype=float)
        out[k] = float(np.median(values))
        out[f"{k}_stdev"] = float(np.std(values, ddof=0))
        out[f"{k}_all_runs"] = [float(v) for v in values]

    out["n_runs"] = len(fits)
    out["median_run_index"] = median_idx
    return out


# ---------------------------------------------------------------------------
# Verificación / validación
# ---------------------------------------------------------------------------

@dataclass
class Warning_:
    """Un mensaje de validación de sysid. severity es 'info' | 'warn' | 'error'."""

    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def validate_free_swing(fit: dict, derived: dict) -> list[Warning_]:
    warnings: list[Warning_] = []
    period = float(fit["period_s"])
    zeta = float(fit["zeta"])
    inertia_measured = float(derived["inertia_measured_kg_m2"])
    inertia_predicted = float(derived["inertia_predicted_kg_m2"])

    # Rango esperado del periodo para este péndulo: 0.3 a 0.7 s.
    if not (0.3 <= period <= 0.7):
        warnings.append(Warning_(
            "warn",
            f"period_s = {period:.3f}s is outside the expected 0.3-0.7s range. "
            "Wildly different means something is off with the geometry, the release, "
            "or the motor arm being free to wiggle. Did you hold the arm firmly?",
        ))
    # ζ ≤ 0: no decae (error). ζ ≥ 0.1: demasiado amortiguado para ser un
    # péndulo ligero (advertencia).
    if zeta <= 0:
        warnings.append(Warning_(
            "error",
            f"zeta = {zeta:.4f} is non-positive. The recording does not show a "
            "decaying envelope; the pendulum may not be swinging freely."
        ))
    elif zeta >= 0.1:
        warnings.append(Warning_(
            "warn",
            f"zeta = {zeta:.4f} is high (>0.1). Expected light damping (<0.1) — "
            "check that the motor arm is held still and the bearings spin freely."
        ))

    # Inercia medida frente a la predicha por el CAD: se señala la divergencia
    # como indicio de que el URDF está desactualizado (cambió la geometría del
    # péndulo) o de que la grabación no fue una oscilación libre limpia.
    #   error relativo = |I_medida − I_CAD| / I_CAD
    rel_err = abs(inertia_measured - inertia_predicted) / inertia_predicted
    if rel_err > 0.10:
        warnings.append(Warning_(
            "warn",
            f"I_measured = {inertia_measured:.3e} kg·m² differs from CAD "
            f"I_predicted = {inertia_predicted:.3e} by {rel_err*100:.1f}% "
            "(>10%). Either the pendulum geometry has changed and "
            "model/model.urdf is stale, or the recording is contaminated "
            "(motor arm not held firmly, magnet brushing the boss, etc.)."
        ))

    return warnings


# ---------------------------------------------------------------------------
# Sugerencia de frecuencia de control (de reference/control-rate.md del sitio
# de documentación del proyecto original)
# ---------------------------------------------------------------------------

def suggest_control_rate(pendulum_period_s: float, motor_rise_time_s: float) -> dict:
    """Devuelve una ventana sugerida de frecuencia de control y un punto de
    operación recomendado.

    Ventana: [5 * f_n, 3 * BW_motor], donde f_n = 1/period_s y el ancho de
    banda efectivo del motor se aproxima como BW_motor ≈ 1/rise_time_95 — la
    regla práctica de `reference/control-rate.md` del sitio original (los 64
    ms de subida medidos en el robot corresponden a ~16 Hz de ancho de banda
    efectivo, validados empíricamente contra el punto de operación de 35 Hz).
    La frecuencia recomendada se sesga hacia el borde inferior;
    max_action_delta_rad se elige de modo que el slew ≤ BW_motor × 0.2.
    """
    f_n = 1.0 / pendulum_period_s
    bw_motor = (1.0 / motor_rise_time_s) if motor_rise_time_s > 0 else float("nan")
    lower = 5.0 * f_n          # por debajo: se pierde la dinámica del péndulo
    upper = 3.0 * bw_motor     # por encima: el motor no alcanza a responder
    if not math.isfinite(upper) or upper <= lower:
        # Ventana vacía: no hay frecuencia que cumpla ambos límites.
        return {
            "f_n_hz": f_n,
            "bw_motor_hz": bw_motor,
            "window_lower_hz": lower,
            "window_upper_hz": upper,
            "feasible": False,
            "recommended_rate_hz": None,
            "recommended_max_action_delta_rad": None,
            "recommended_slew_rad_s": None,
        }
    # Sesgo hacia el borde inferior: lower + 0.4 × ancho de la ventana.
    recommended = lower + 0.4 * (upper - lower)
    slew_cap = bw_motor * 0.2
    delta = min(0.10, slew_cap / recommended) if recommended > 0 else 0.10
    return {
        "f_n_hz": f_n,
        "bw_motor_hz": bw_motor,
        "window_lower_hz": lower,
        "window_upper_hz": upper,
        "feasible": True,
        "recommended_rate_hz": recommended,
        "recommended_max_action_delta_rad": delta,
        "recommended_slew_rad_s": delta * recommended,
    }


# ---------------------------------------------------------------------------
# Salida JSON
# ---------------------------------------------------------------------------

def write_params_json(
    path: str,
    *,
    pendulum_payload: dict | None = None,
    run_meta: dict | None = None,
) -> dict:
    """Escribe/combina el sysid_params_*.json de cada robot que produce el wizard.

    Conserva el bloque `pendulum` de nivel superior que lee `pendulum_env.py`.
    `run_meta` es solo informativo (marca de tiempo, archivos de origen,
    versión del ajustador); el código posterior lo ignora.
    """
    doc: dict = {}
    # Si el archivo ya existe se lee y se actualiza (p. ej. conserva la
    # sección `motor_step`) en lugar de sobrescribirlo entero.
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    if pendulum_payload is not None:
        doc["pendulum"] = pendulum_payload
    if run_meta is not None:
        doc["run_meta"] = run_meta
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    return doc


def source_file_for_json(recording_path: str, out_json_path: str) -> str:
    """Ruta de una grabación, expresada relativa al directorio del JSON.

    Mantiene el JSON portable entre máquinas / clones del repositorio.
    """
    out_dir = Path(out_json_path).resolve().parent
    rec_path = Path(recording_path).resolve()
    return os.path.relpath(rec_path, out_dir)


def format_warnings(warnings: Iterable[Warning_]) -> str:
    """Muestra las advertencias como un bloque indentado legible."""
    return "\n".join(f"  {w}" for w in warnings)
