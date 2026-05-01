# evaluate_policy.py pyright: reportAttributeAccessIssue=false import argparse
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO

from env import ReachEnv


def load_training_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def evaluate(
    model_path: str,
    scene_path: str,
    num_episodes: int,
    max_steps: int,
    success_threshold: float,
    seed: int,
    deterministic: bool,
):
    mujoco_model = mujoco.MjModel.from_xml_path(scene_path)

    env = ReachEnv(mujoco_model, seed=seed)
    env.max_steps = max_steps
    env.success_threshold = success_threshold

    policy = PPO.load(model_path, device="cpu")

    final_distances = []
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)

        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not done and not truncated:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_len += 1

        final_distance = float(info["distance"])
        is_success = bool(info["is_success"])

        final_distances.append(final_distance)
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        successes.append(is_success)

        print(
            f"Episode {ep + 1:03d}/{num_episodes} | "
            f"success={is_success} | "
            f"final_distance={final_distance:.4f} | "
            f"length={ep_len} | "
            f"reward={ep_reward:.2f}"
        )

    final_distances = np.array(final_distances, dtype=np.float64)
    episode_rewards = np.array(episode_rewards, dtype=np.float64)
    episode_lengths = np.array(episode_lengths, dtype=np.float64)
    successes = np.array(successes, dtype=np.float64)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Model: {model_path}")
    print(f"Scene: {scene_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Success threshold: {success_threshold}")
    print(f"Deterministic: {deterministic}")

    print("\nSuccess")
    print("-" * 80)
    print(f"Success rate: {successes.mean():.3f}")
    print(f"Success count: {int(successes.sum())}/{num_episodes}")

    print("\nFinal distance")
    print("-" * 80)
    print(f"Mean:   {final_distances.mean():.4f}")
    print(f"Median: {np.median(final_distances):.4f}")
    print(f"Std:    {final_distances.std():.4f}")
    print(f"Min:    {final_distances.min():.4f}")
    print(f"Max:    {final_distances.max():.4f}")
    print(
        "Percentiles [10, 25, 50, 75, 90]:",
        np.percentile(final_distances, [10, 25, 50, 75, 90]),
    )

    print("\nEpisode reward")
    print("-" * 80)
    print(f"Mean:   {episode_rewards.mean():.2f}")
    print(f"Median: {np.median(episode_rewards):.2f}")
    print(f"Std:    {episode_rewards.std():.2f}")
    print(f"Min:    {episode_rewards.min():.2f}")
    print(f"Max:    {episode_rewards.max():.2f}")

    print("\nEpisode length")
    print("-" * 80)
    print(f"Mean:   {episode_lengths.mean():.2f}")
    print(f"Median: {np.median(episode_lengths):.2f}")
    print(f"Std:    {episode_lengths.std():.2f}")
    print(f"Min:    {episode_lengths.min():.0f}")
    print(f"Max:    {episode_lengths.max():.0f}")

    return {
        "success_rate": float(successes.mean()),
        "success_count": int(successes.sum()),
        "num_episodes": int(num_episodes),
        "final_distance_mean": float(final_distances.mean()),
        "final_distance_median": float(np.median(final_distances)),
        "final_distance_std": float(final_distances.std()),
        "final_distance_min": float(final_distances.min()),
        "final_distance_max": float(final_distances.max()),
        "final_distance_percentiles": {
            "10": float(np.percentile(final_distances, 10)),
            "25": float(np.percentile(final_distances, 25)),
            "50": float(np.percentile(final_distances, 50)),
            "75": float(np.percentile(final_distances, 75)),
            "90": float(np.percentile(final_distances, 90)),
        },
        "episode_reward_mean": float(episode_rewards.mean()),
        "episode_reward_median": float(np.median(episode_rewards)),
        "episode_length_mean": float(episode_lengths.mean()),
        "episode_length_median": float(np.median(episode_lengths)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO .zip model.",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/scene.xml",
        help="Path to MuJoCo scene XML.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config JSON. Used to read max_steps and success_threshold.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max episode steps. Overrides config if provided.",
    )

    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Success threshold in meters. Overrides config if provided.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Evaluation seed.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic actions.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary JSON.",
    )

    args = parser.parse_args()

    cfg = load_training_config(args.config)

    config_max_steps = cfg.get("env", {}).get("max_steps", 500)
    config_success_threshold = cfg.get("env", {}).get("success_threshold", 0.05)

    max_steps = args.max_steps if args.max_steps is not None else config_max_steps
    success_threshold = (
        args.success_threshold
        if args.success_threshold is not None
        else config_success_threshold
    )

    summary = evaluate(
        model_path=args.model,
        scene_path=args.scene,
        num_episodes=args.episodes,
        max_steps=max_steps,
        success_threshold=success_threshold,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\nSaved evaluation summary to: {save_path}")


if __name__ == "__main__":
    main()
