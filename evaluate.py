"""Evaluation and comparison between Standard ICM and VW-ICM.

Usage:
    # Compare two checkpoints
    python evaluate.py \\
        --checkpoint_std runs/standard_icm/final.pt \\
        --checkpoint_vw  runs/vw_icm/final.pt \\
        --config_std configs/minigrid_standard_icm.yaml \\
        --config_vw  configs/minigrid_vw_icm.yaml \\
        --n_episodes 20

    # Evaluate a single checkpoint
    python evaluate.py --checkpoint runs/vw_icm/final.pt --config configs/minigrid_vw_icm.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from envs import make_env
from icm import make_encoder, StandardICM, VWICM
from algorithms.ppo import PPO, ActorCritic


def load_agent(checkpoint_path: str, config_path: str, device: str, eval_seed: int = 0):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Use a different seed from training to avoid testing only layouts seen early in training
    env, _ = make_env(cfg["env"], seed=eval_seed, max_steps=cfg.get("max_steps", None))
    obs_shape = env.observation_space.shape
    is_discrete = cfg.get("action_space", "discrete") == "discrete"
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    enc_cfg = cfg["encoder"]
    encoder = make_encoder(obs_shape, enc_cfg["feature_dim"], enc_cfg.get("type", "auto"))

    icm_cfg = cfg["icm"]
    icm_kwargs = dict(
        encoder=encoder,
        action_dim=action_dim,
        is_discrete=is_discrete,
        forward_loss_coeff=icm_cfg.get("forward_loss_coeff", 0.2),
        inverse_loss_coeff=icm_cfg.get("inverse_loss_coeff", 0.8),
    )
    if icm_cfg["type"] == "vw":
        icm = VWICM(**icm_kwargs,
                     ema_alpha=icm_cfg.get("ema_alpha", 0.01),
                     warmup_steps=icm_cfg.get("warmup_steps", 1000))
    else:
        icm = StandardICM(**icm_kwargs)

    ac = ActorCritic(encoder, action_dim, is_discrete)
    ppo_cfg = cfg["ppo"]
    agent = PPO(
        actor_critic=ac, icm=icm,
        lr=ppo_cfg.get("lr", 2.5e-4),
        n_steps=ppo_cfg.get("n_steps", 128),
        device=device,
    )
    agent.load(checkpoint_path)
    return agent, env, cfg


def run_episodes(agent, env, n_episodes: int, device: str):
    returns, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist, _ = agent.ac(obs_t)
                # Greedy action selection for deterministic, stable evaluation
                if agent.ac.is_discrete:
                    action = dist.probs.argmax(-1).squeeze(0)
                else:
                    action = dist.mean.squeeze(0)
            act = int(action.item()) if agent.ac.is_discrete else action.cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
    return np.array(returns), np.array(lengths)


def print_stats(name: str, returns: np.ndarray, lengths: np.ndarray):
    success = returns > 0
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Episodes   : {len(returns)}")
    print(f"  Return     : {returns.mean():.3f} ± {returns.std():.3f}")
    print(f"  Success    : {success.mean() * 100:.1f}%")
    print(f"  Ep. Length : {lengths.mean():.1f} ± {lengths.std():.1f}")
    if success.any():
        print(f"  Return (success only): {returns[success].mean():.3f} ± {returns[success].std():.3f}")


def plot_comparison(
    returns_std, returns_vw,
    label_std="Standard ICM", label_vw="VW-ICM",
    save_path="comparison.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar([label_std, label_vw], [returns_std.mean(), returns_vw.mean()],
           yerr=[returns_std.std(), returns_vw.std()], capsize=5, color=["steelblue", "tomato"])
    ax.set_title("Mean Episode Return")
    ax.set_ylabel("Return")

    ax = axes[1]
    ax.hist(returns_std, alpha=0.6, label=label_std, color="steelblue", bins=15)
    ax.hist(returns_vw, alpha=0.6, label=label_vw, color="tomato", bins=15)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Return")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    # Comparison mode
    parser.add_argument("--checkpoint_std", default=None)
    parser.add_argument("--checkpoint_vw", default=None)
    parser.add_argument("--config_std", default=None)
    parser.add_argument("--config_vw", default=None)
    # Single eval mode
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)

    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=100)
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint_std and args.checkpoint_vw:
        agent_std, env_std, _ = load_agent(args.checkpoint_std, args.config_std, device, args.eval_seed)
        agent_vw, env_vw, _ = load_agent(args.checkpoint_vw, args.config_vw, device, args.eval_seed)

        returns_std, lengths_std = run_episodes(agent_std, env_std, args.n_episodes, device)
        returns_vw, lengths_vw = run_episodes(agent_vw, env_vw, args.n_episodes, device)

        print_stats("Standard ICM", returns_std, lengths_std)
        print_stats("VW-ICM", returns_vw, lengths_vw)

        delta = returns_vw.mean() - returns_std.mean()
        print(f"\n  VW-ICM improvement: {delta:+.3f} return")

        plot_comparison(returns_std, returns_vw, save_path=args.output)
        env_std.close()
        env_vw.close()

    elif args.checkpoint and args.config:
        agent, env, cfg = load_agent(args.checkpoint, args.config, device, args.eval_seed)
        returns, lengths = run_episodes(agent, env, args.n_episodes, device)
        label = f"{cfg['icm']['type'].upper()}-ICM ({cfg['env']})"
        print_stats(label, returns, lengths)
        env.close()

    else:
        print("Provide --checkpoint_std/--checkpoint_vw or --checkpoint/--config")


if __name__ == "__main__":
    main()
