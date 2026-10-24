"""Entrenamiento con SB3: fábricas de entornos, γ, evaluación y una corrida mínima."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from furuta.train.evaluate import evaluate_model, make_deploy_env_fn
from furuta.train.make_env import make_eval_env_fn, make_train_env_fn
from furuta.train.sac_sb3 import TrainConfig, default_gamma, train


class _PoliticaCero:
    """Imita la interfaz de un modelo de SB3: siempre acción 0."""

    def predict(self, obs, deterministic: bool = True):
        return np.array([0.0], dtype=np.float32), None


class _EntornoFalso:
    """Reproduce una trayectoria de phi conocida (phi = π es péndulo arriba)."""

    def __init__(self, phis: list[float]) -> None:
        self._phis, self._i = phis, 0

    def reset(self, seed: int | None = None):
        self._i = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        phi = self._phis[self._i]
        self._i += 1
        done = self._i >= len(self._phis)
        return np.zeros(1, dtype=np.float32), -1.0, False, done, {"phi": phi}


def _pasos(phi: float, n: int) -> list[float]:
    return [phi] * n


def test_la_metrica_de_recaptura_detecta_el_giro_tras_subir() -> None:
    arriba, abajo = math.pi, 0.0              # phi = π: arriba; phi = 0: colgando
    # Cuelga 50 pasos, sube (θ≈0) 10 pasos, se va hasta abajo 20 pasos y vuelve a subir.
    con_giro = _pasos(abajo, 50) + _pasos(arriba, 10) + _pasos(abajo, 20) + _pasos(arriba, 120)
    # Sube y se queda.
    sin_giro = _pasos(abajo, 50) + _pasos(arriba, 150)
    for phis, esperado in ((con_giro, 100.0), (sin_giro, 0.0)):
        res = evaluate_model(_PoliticaCero(), n_episodes=2, env_fn=lambda p=phis: _EntornoFalso(p))
        assert res["recaptura_%"] == esperado
        assert res["equilibrado_%"] == 100.0
        assert math.isclose(res["tiempo_arriba_s"], 50 / 50.0)    # paso 50 a 50 Hz = 1.0 s


def test_si_nunca_sube_no_hay_tiempo_arriba_ni_recaptura() -> None:
    colgado = _pasos(0.0, 200)
    res = evaluate_model(_PoliticaCero(), n_episodes=2, env_fn=lambda: _EntornoFalso(colgado))
    assert math.isnan(res["tiempo_arriba_s"])
    assert res["recaptura_%"] == 0.0 and res["equilibrado_%"] == 0.0


def test_el_entorno_de_despliegue_tiene_dr_y_retardo() -> None:
    base = make_deploy_env_fn()().unwrapped
    assert base.domain_randomization is True
    assert base._dr_action_delay_steps_range == (1, 1)
    res = evaluate_model(_PoliticaCero(), n_episodes=2, env_fn=make_deploy_env_fn())
    assert math.isfinite(res["retorno_medio"]) and res["pasos_medios"] == 400


def test_el_entorno_de_evaluacion_es_determinista_y_sin_sesgo() -> None:
    env = make_eval_env_fn()()
    assert isinstance(env, Monitor)
    base = env.unwrapped
    assert base.domain_randomization is False
    assert base._dr_theta_bias_max_rad == 0.0


def test_el_entorno_de_evaluacion_ignora_un_intento_de_activar_la_dr() -> None:
    base = make_eval_env_fn(domain_randomization=True, dr_theta_bias_max_rad=0.05)().unwrapped
    assert base.domain_randomization is False
    assert base._dr_theta_bias_max_rad == 0.0


def test_el_entorno_de_entrenamiento_pasa_los_kwargs() -> None:
    env = make_train_env_fn(
        domain_randomization=True, dr_action_delay_steps_range=(0, 1)
    )()
    assert isinstance(env, Monitor)
    assert env.unwrapped.domain_randomization is True
    assert env.unwrapped._dr_action_delay_steps_range == (0, 1)


def test_gamma_conserva_el_horizonte_en_segundos() -> None:
    assert math.isclose(default_gamma(35.0), 0.99)
    g50 = default_gamma(50.0)
    assert math.isclose(g50, 0.99 ** 0.7)
    assert 0.9929 < g50 < 0.9931
    # Mismo horizonte efectivo en segundos: 1/(1-γ) pasos a su frecuencia.
    assert math.isclose(1 / (1 - 0.99) / 35.0, 1 / (1 - g50) / 50.0, rel_tol=0.01)


def test_evaluar_la_politica_cero() -> None:
    res = evaluate_model(_PoliticaCero(), n_episodes=3, seed=0)
    assert res["episodios"] == 3
    assert -4500 < res["retorno_medio"] < -3700      # colgado ≈ −9.87 por paso × 400
    assert res["pasos_medios"] == 400
    assert res["equilibrado_%"] == 0.0               # colgado no es equilibrado
    assert res["tope_duro_%"] == 0.0
    assert res["recaptura_%"] == 0.0                 # nunca llega arriba
    assert math.isnan(res["tiempo_arriba_s"])


def test_corrida_minima_y_reanudacion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run1 = Path(tmp) / "etapa1"
        final = train(TrainConfig(
            run_dir=run1, total_steps=300, eval_freq=200, n_eval_episodes=1,
            device="cpu", net_arch=(16, 16),
        ))
        assert final.exists() and (run1 / "monitor.csv").exists()
        assert (run1 / "best_model.zip").exists()
        cfg = json.loads((run1 / "config.json").read_text())
        assert math.isclose(cfg["gamma_efectivo"], default_gamma())
        assert cfg["domain_randomization"] is False

        # El modelo guardado se carga y se puede evaluar.
        model = SAC.load(final, device="cpu")
        res = evaluate_model(model, n_episodes=1, seed=0)
        assert res["pasos_medios"] == 400 and math.isfinite(res["retorno_medio"])

        # Etapa siguiente: retoma los pesos y ya con DR y retardo.
        run2 = Path(tmp) / "etapa2"
        final2 = train(TrainConfig(
            run_dir=run2, total_steps=200, eval_freq=200, n_eval_episodes=1,
            device="cpu", domain_randomization=True, dr_delay_steps_range=(0, 1),
            dr_lag_tau_range_s=(0.0, 0.02), resume=final,
        ))
        assert final2.exists()
        cfg2 = json.loads((run2 / "config.json").read_text())
        assert cfg2["domain_randomization"] is True and cfg2["resume"] == str(final)


if __name__ == "__main__":
    test_la_metrica_de_recaptura_detecta_el_giro_tras_subir()
    test_si_nunca_sube_no_hay_tiempo_arriba_ni_recaptura()
    test_el_entorno_de_despliegue_tiene_dr_y_retardo()
    test_el_entorno_de_evaluacion_es_determinista_y_sin_sesgo()
    test_el_entorno_de_evaluacion_ignora_un_intento_de_activar_la_dr()
    test_el_entorno_de_entrenamiento_pasa_los_kwargs()
    test_gamma_conserva_el_horizonte_en_segundos()
    test_evaluar_la_politica_cero()
    test_corrida_minima_y_reanudacion()
    print("OK")
