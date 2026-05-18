"""Training entry point.

Usage:
    python train.py --config configs/minigrid_vw_icm.yaml
    python train.py --config configs/minigrid_standard_icm.yaml --seed 0
    python train.py --config configs/minigrid_vw_icm.yaml --total_steps 1000000
"""

import argparse
import logging
import os
import time
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import make_env
from icm import make_encoder, StandardICM, VWICM
from algorithms.ppo import PPO, ActorCritic


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return cfg


def build_icm(cfg: dict, encoder, action_dim: int, is_discrete: bool):
    icm_cfg = cfg["icm"]
    icm_type = icm_cfg["type"]
    kwargs = dict(
        encoder=encoder,
        action_dim=action_dim,
        is_discrete=is_discrete,
        forward_loss_coeff=icm_cfg.get("forward_loss_coeff", 0.2),
        inverse_loss_coeff=icm_cfg.get("inverse_loss_coeff", 0.8),
    )
    if icm_type == "vw":
        return VWICM(
            **kwargs,
            ema_alpha=icm_cfg.get("ema_alpha", 0.01),
            warmup_steps=icm_cfg.get("warmup_steps", 1000),
        )
    elif icm_type == "standard":
        return StandardICM(**kwargs)
    else:
        raise ValueError(f"Unknown ICM type: {icm_type}")


def setup_file_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.env is not None:
        overrides["env"] = args.env
    if args.log_dir is not None:
        overrides["logging.log_dir"] = args.log_dir

    cfg = load_config(args.config, overrides)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    icm_type = cfg["icm"]["type"]
    print(f"Device: {device} | ICM: {icm_type} | Env: {cfg['env']}")

    env, obs_type = make_env(cfg["env"], seed=seed, max_steps=cfg.get("max_steps", None))
    obs_shape = env.observation_space.shape
    is_discrete = cfg.get("action_space", "discrete") == "discrete"
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    enc_cfg = cfg["encoder"]
    encoder = make_encoder(obs_shape, enc_cfg["feature_dim"], enc_cfg.get("type", "auto"))
    icm = build_icm(cfg, encoder, action_dim, is_discrete)
    ac = ActorCritic(encoder, action_dim, is_discrete)

    ppo_cfg = cfg["ppo"]
    agent = PPO(
        actor_critic=ac,
        icm=icm,
        lr=ppo_cfg.get("lr", 2.5e-4),
        n_steps=ppo_cfg.get("n_steps", 128),
        n_epochs=ppo_cfg.get("n_epochs", 4),
        batch_size=ppo_cfg.get("batch_size", 256),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        vf_coeff=ppo_cfg.get("vf_coeff", 0.5),
        ent_coeff=ppo_cfg.get("ent_coeff", 0.01),
        intrinsic_coeff=cfg["icm"].get("intrinsic_coeff", 0.01),
        normalize_intrinsic=cfg["icm"].get("normalize_intrinsic", True),
        device=device,
    )

    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "runs/default")
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_file_logger(log_dir)
    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    total_steps = cfg.get("total_steps", 2_000_000)
    n_steps = ppo_cfg.get("n_steps", 128)
    save_freq = log_cfg.get("save_freq", 100_000)
    log_freq = log_cfg.get("log_freq", 10)

    base_intrinsic_coeff = cfg["icm"].get("intrinsic_coeff", 0.01)
    decay_start_frac = cfg["icm"].get("intrinsic_decay_start", None)
    decay_start_step = int(decay_start_frac * total_steps) if decay_start_frac is not None else None

    n_updates = total_steps // n_steps
    steps_done = 0
    ep_returns = []
    first_success_step = None

    # Log header once
    is_vw = icm_type == "vw"
    header = "steps     | success% | ep_ret  | pg_loss | icm_loss | intr_rew | coeff"
    if is_vw:
        header += " | anisotropy | vt_step | weight_max"
    logger.info(f"=== Training start: {icm_type.upper()}-ICM | {cfg['env']} ===")
    logger.info(header)

    print(f"Training for {total_steps:,} steps ({n_updates} updates) → {log_dir}/train.log")
    t0 = time.time()

    for update in tqdm(range(n_updates), desc="Updates"):
        # Intrinsic coeff decay: linearly anneal to 0 after decay_start_step
        if decay_start_step is not None and steps_done >= decay_start_step:
            progress = (steps_done - decay_start_step) / max(total_steps - decay_start_step, 1)
            agent.intrinsic_coeff = base_intrinsic_coeff * max(0.0, 1.0 - progress)
        else:
            agent.intrinsic_coeff = base_intrinsic_coeff

        rollout = agent.collect_rollout(env, episode_returns=ep_returns)
        update_info = agent.update(rollout)
        steps_done += n_steps

        if update % log_freq == 0:
            # TensorBoard
            writer.add_scalar("train/intrinsic_coeff", agent.intrinsic_coeff, steps_done)
            writer.add_scalar("train/pg_loss", update_info["pg_loss"], steps_done)
            writer.add_scalar("train/vf_loss", update_info["vf_loss"], steps_done)
            writer.add_scalar("train/icm_loss", update_info["icm_loss"], steps_done)
            writer.add_scalar("train/ent_loss", update_info["ent_loss"], steps_done)
            writer.add_scalar("train/intr_reward", update_info.get("intr_reward_mean", 0), steps_done)

            recent = ep_returns[-20:] if ep_returns else []
            ep_ret_mean   = sum(recent) / len(recent) if recent else float("nan")
            success_rate  = sum(r > 0 for r in recent) / len(recent) if recent else float("nan")

            if recent:
                writer.add_scalar("train/ep_return_mean", ep_ret_mean, steps_done)
                writer.add_scalar("train/ep_success_rate", success_rate, steps_done)
                if first_success_step is None and any(r > 0 for r in ep_returns):
                    first_success_step = steps_done
                    logger.info(f"*** FIRST SUCCESS at step {steps_done:,} ***")

            # File log
            line = (
                f"{steps_done:>9,} | {success_rate*100:>7.1f}% | {ep_ret_mean:>7.3f} | "
                f"{update_info['pg_loss']:>7.4f} | {update_info['icm_loss']:>8.4f} | "
                f"{update_info.get('intr_reward_mean', 0):>8.5f} | {agent.intrinsic_coeff:.5f}"
            )

            if is_vw and hasattr(icm, "variance_tracker"):
                vt = icm.variance_tracker
                aniso = vt.anisotropy
                vt_step = vt.step.item()
                w_max = vt.weights.max().item()
                in_warmup = vt_step < vt.warmup_steps

                writer.add_scalar("vw_icm/anisotropy", aniso, steps_done)
                writer.add_scalar("vw_icm/weight_max", w_max, steps_done)
                writer.add_scalar("vw_icm/tracker_step", vt_step, steps_done)

                warmup_tag = " [WARMUP]" if in_warmup else ""
                line += f" | {aniso:>10.4f} | {vt_step:>7} | {w_max:>10.4f}{warmup_tag}"

            logger.info(line)

        if steps_done % save_freq < n_steps:
            agent.save(os.path.join(log_dir, f"ckpt_{steps_done}.pt"))

    agent.save(os.path.join(log_dir, "final.pt"))
    elapsed = time.time() - t0

    # Final summary to log
    recent = ep_returns[-50:] if ep_returns else []
    final_success = sum(r > 0 for r in recent) / len(recent) if recent else 0
    logger.info("=" * 80)
    first_str = f"step {first_success_step:,}" if first_success_step else "never"
    logger.info(f"Training done in {elapsed:.0f}s | final success rate (last 50 ep): {final_success*100:.1f}% | first success: {first_str}")
    if is_vw and hasattr(icm, "variance_tracker"):
        vt = icm.variance_tracker
        w = vt.weights
        top5 = w.topk(5).values.tolist()
        logger.info(f"VW-ICM final anisotropy={vt.anisotropy:.4f} | top-5 weights={[f'{v:.3f}' for v in top5]}")
        logger.info(f"Variance tracker steps={vt.step.item()} | warmup_steps={vt.warmup_steps}")

    print(f"Done in {elapsed:.0f}s. Log: {log_dir}/train.log")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
