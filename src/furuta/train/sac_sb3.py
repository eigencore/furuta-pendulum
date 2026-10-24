"""Entrenamiento con SAC de Stable-Baselines3 (referencia para nuestro propio SAC).

Hiperparámetros del original (policy/train_sac.py). Cada corrida escribe en
`runs/<nombre>/`: config.json, monitor.csv, best_model.zip, final_model.zip.

Uso (etapa 1 del currículo: sin DR, para aprender el swing-up):
    uv run python -m furuta.train.sac_sb3 --run-name etapa1 --total-steps 100000

Etapa con DR, retomando la anterior:
    uv run python -m furuta.train.sac_sb3 --run-name etapa2 --total-steps 100000 \\
        --domain-randomization --dr-delay-max 1 --dr-lag-tau-max 0.020 \\
        --resume runs/etapa1/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from furuta.sim.pendulum_env import (
    CONTROL_FREQ_HZ,
    DR_ACTION_DELAY_STEPS_RANGE,
    DR_ACTION_LAG_TAU_RANGE_S,
)
from furuta.train.make_env import make_eval_env_fn, make_train_env_fn

# El autor calibró γ = 0.99 a 35 Hz. Al subir la frecuencia de control se
# ajusta γ para que el horizonte efectivo (en segundos) no cambie.
GAMMA_REFERENCE = 0.99
GAMMA_REFERENCE_HZ = 35.0


def default_gamma(control_freq_hz: float = CONTROL_FREQ_HZ) -> float:
    """γ = 0.99^(35/f): a 50 Hz sale ≈ 0.993."""
    return GAMMA_REFERENCE ** (GAMMA_REFERENCE_HZ / control_freq_hz)


@dataclass(frozen=True)
class TrainConfig:
    run_dir: Path
    total_steps: int = 100_000
    seed: int = 0
    device: str = "auto"
    net_arch: tuple[int, ...] | None = None     # None → el de SB3 (256, 256)
    gamma: float | None = None                  # None → default_gamma()
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    domain_randomization: bool = False
    dr_delay_steps_range: tuple[int, int] = DR_ACTION_DELAY_STEPS_RANGE
    dr_lag_tau_range_s: tuple[float, float] = DR_ACTION_LAG_TAU_RANGE_S
    resume: Path | None = None


def eval_transport(cfg: TrainConfig) -> tuple[int, float]:
    """Retardo (ticks) y lag τ (s) con los que se evalúa: el punto medio de la DR.

    El entorno de evaluación es determinista pero NO a retardo cero: si todo el
    entrenamiento tiene ≥ 1 tick de retardo, elegir el mejor modelo sin él lo
    escogería con una prueba que los episodios de entrenamiento nunca vieron.
    """
    if not cfg.domain_randomization:
        return 0, 0.0
    delay = int(round(sum(cfg.dr_delay_steps_range) / 2))
    lag = sum(cfg.dr_lag_tau_range_s) / 2.0
    return delay, lag


def train(cfg: TrainConfig) -> Path:
    """Entrena y devuelve la ruta de `final_model.zip`."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    gamma = cfg.gamma if cfg.gamma is not None else default_gamma()

    env_kwargs = {
        "domain_randomization": cfg.domain_randomization,
        "dr_action_delay_steps_range": cfg.dr_delay_steps_range,
        "dr_action_lag_tau_range_s": cfg.dr_lag_tau_range_s,
    }
    train_env = DummyVecEnv([make_train_env_fn(
        monitor_path=cfg.run_dir / "monitor.csv", **env_kwargs
    )])
    eval_delay, eval_lag = eval_transport(cfg)
    eval_env = DummyVecEnv([make_eval_env_fn(
        action_delay_steps=eval_delay, action_lag_tau_s=eval_lag
    )])

    if cfg.resume is not None:
        # Retoma pesos y entropía; el buffer de repetición empieza vacío.
        model = SAC.load(cfg.resume, env=train_env, device=cfg.device)
        model.gamma = gamma
        model.set_random_seed(cfg.seed)
    else:
        model = SAC(
            "MlpPolicy", train_env,
            learning_rate=3e-4, buffer_size=200_000, batch_size=256,
            tau=0.005, gamma=gamma, train_freq=1, gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=list(cfg.net_arch)) if cfg.net_arch else None,
            seed=cfg.seed, device=cfg.device, verbose=1,
        )

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "gamma_efectivo": gamma,
        "eval_delay_steps": eval_delay,
        "eval_lag_tau_s": eval_lag,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "dispositivo_real": str(model.device),
    }
    (cfg.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(cfg.run_dir),
        log_path=str(cfg.run_dir / "eval"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.total_steps, callback=eval_callback)
    print(f"entrenamiento: {time.perf_counter() - t0:.0f} s "
          f"({cfg.total_steps / (time.perf_counter() - t0):.0f} pasos/s)")

    final = cfg.run_dir / "final_model.zip"
    model.save(final)
    return final


def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=time.strftime("sac_sb3_%Y-%m-%d_%H%M%S"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--net-arch", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=None, help="capas ocultas, p. ej. '128,128' (SB3: 256,256)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="por defecto 0.99^(35/f_control)")
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--domain-randomization", action="store_true")
    ap.add_argument("--dr-delay-min", type=int, default=DR_ACTION_DELAY_STEPS_RANGE[0])
    ap.add_argument("--dr-delay-max", type=int, default=DR_ACTION_DELAY_STEPS_RANGE[1])
    ap.add_argument("--dr-lag-tau-min", type=float, default=DR_ACTION_LAG_TAU_RANGE_S[0])
    ap.add_argument("--dr-lag-tau-max", type=float, default=DR_ACTION_LAG_TAU_RANGE_S[1])
    ap.add_argument("--resume", type=Path, default=None, help="modelo .zip de la etapa anterior")
    a = ap.parse_args()
    return TrainConfig(
        run_dir=a.out_dir / a.run_name, total_steps=a.total_steps, seed=a.seed,
        device=a.device, net_arch=a.net_arch, gamma=a.gamma, eval_freq=a.eval_freq,
        domain_randomization=a.domain_randomization,
        dr_delay_steps_range=(a.dr_delay_min, a.dr_delay_max),
        dr_lag_tau_range_s=(a.dr_lag_tau_min, a.dr_lag_tau_max),
        resume=a.resume,
    )


if __name__ == "__main__":
    train(_parse_args())
