"""Animation visualization for GPPO training progress."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.ndimage import uniform_filter1d

from src.common.paths import ANIMATIONS_DIR, DEFAULT_RESULTS_PATH, ensure_output_dirs


class TrainingAnimator:
    """Create animated training visualizations."""

    @staticmethod
    def create_learning_animation(json_path: str, output_path: str = None, fps: int = 10, gif_workers: int | None = None):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        episodes = np.arange(len(results["episode_rewards"]))
        rewards = np.asarray(results["episode_rewards"], dtype=float)
        costs = np.asarray(results["episode_costs"], dtype=float)
        validity = np.array(results["valid_deployments"]) * 100
        finite_costs = np.where(np.isfinite(costs), costs, np.nan)
        finite_cost_values = finite_costs[np.isfinite(finite_costs)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")
        fig.suptitle("GPPO Training Progress (Episode 0)", fontsize=14, fontweight="bold")

        line_reward, = axes[0].plot([], [], linewidth=2.5, color="#FF6B6B", label="Reward")
        line_cost, = axes[1].plot([], [], linewidth=2.5, color="#4ECDC4", label="Cost")
        line_validity, = axes[2].plot([], [], linewidth=2.5, color="#45B7D1", label="Validity %")

        axes[0].set_xlim(0, len(episodes))
        axes[0].set_ylim(min(rewards) - 1, max(rewards) + 1)
        axes[0].set_xlabel("Episode", fontweight="bold")
        axes[0].set_ylabel("Cumulative Reward", fontweight="bold")
        axes[0].set_title("Learning Progress: Reward", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].set_xlim(0, len(episodes))
        if finite_cost_values.size:
            cost_min = float(finite_cost_values.min())
            cost_max = float(finite_cost_values.max())
            padding = max((cost_max - cost_min) * 0.1, 1.0)
            axes[1].set_ylim(cost_min - padding, cost_max + padding)
        else:
            axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Episode", fontweight="bold")
        axes[1].set_ylabel("Deployment Cost", fontweight="bold")
        axes[1].set_title("Resource Efficiency: Cost", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].set_xlim(0, len(episodes))
        axes[2].set_ylim(0, 110)
        axes[2].set_xlabel("Episode", fontweight="bold")
        axes[2].set_ylabel("Valid Deployments (%)", fontweight="bold")
        axes[2].set_title("Reliability: Validity", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        def animate(frame):
            end = frame + 1
            line_reward.set_data(episodes[:end], rewards[:end])
            axes[0].fill_between(episodes[:end], rewards[:end], alpha=0.3, color="#FF6B6B")
            axes[0].set_title(f"Reward at Ep {frame}: {rewards[frame]:.3f}", fontweight="bold")

            line_cost.set_data(episodes[:end], finite_costs[:end])
            axes[1].fill_between(episodes[:end], finite_costs[:end], alpha=0.3, color="#4ECDC4")
            cost_label = f"{costs[frame]:.2f}" if np.isfinite(costs[frame]) else "inf"
            axes[1].set_title(f"Cost at Ep {frame}: {cost_label}", fontweight="bold")

            line_validity.set_data(episodes[:end], validity[:end])
            axes[2].fill_between(episodes[:end], validity[:end], alpha=0.3, color="#45B7D1")
            axes[2].set_title(f"Validity at Ep {frame}: {validity[frame]:.1f}%", fontweight="bold")

            fig.suptitle(f"GPPO Training Progress (Episode {frame}/{len(episodes) - 1})", fontsize=14, fontweight="bold")
            return line_reward, line_cost, line_validity

        anim = FuncAnimation(fig, animate, frames=len(episodes), interval=1000 // fps, blit=True, repeat=True)

        if output_path:
            print(f"Saving animation to {output_path}...")
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
            anim.save(output_path, writer=writer)
            print(f"✓ Animation saved: {output_path}")

        return fig, anim

    @staticmethod
    def create_reward_landscape(json_path: str, output_path: str = None, window_size: int = 10):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        rewards = np.array(results["episode_rewards"])
        episodes = np.arange(len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode="nearest")

        fig = plt.figure(figsize=(14, 5), facecolor="white")

        ax1 = fig.add_subplot(121)
        ax1.scatter(episodes, rewards, alpha=0.5, s=30, color="#FF6B6B", label="Raw")
        ax1.plot(episodes, smoothed, linewidth=3, color="#FF0000", label=f"Smoothed (window={window_size})")
        ax1.fill_between(episodes, rewards, smoothed, alpha=0.2, color="#FF6B6B")
        ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Reward", fontsize=11, fontweight="bold")
        ax1.set_title("Raw vs Smoothed Reward Curves", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        ax2 = fig.add_subplot(122)
        improvement = np.diff(smoothed)
        colors = ["#FF6B6B" if x < 0 else "#45B7D1" for x in improvement]
        ax2.bar(episodes[1:], improvement, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Reward Change (Δ)", fontsize=11, fontweight="bold")
        ax2.set_title("Episode-to-Episode Improvement", fontsize=12, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.suptitle("Reward Landscape Analysis", fontsize=13, fontweight="bold", y=1.00)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {output_path}")

        return fig, (ax1, ax2)


class NetworkStateAnimator:
    """Animate network state changes during training."""

    @staticmethod
    def create_network_state_animation(output_path: str = None, fps: int = 5):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
        fig.suptitle("Network State Evolution (Frame 0)", fontsize=14, fontweight="bold")

        frames = 30

        def animate(frame):
            progress = frame / frames
            for ax in axes:
                ax.clear()

            es_util = [0.3 + progress * 0.4, 0.4 + progress * 0.3, 0.2 + progress * 0.5]
            es_names = ["ES0", "ES1", "ES2"]
            colors_es = ["#FFB3B3" if u < 0.5 else "#FF6B6B" for u in es_util]
            axes[0].bar(es_names, es_util, color=colors_es, edgecolor="black", linewidth=2)
            axes[0].set_ylabel("Utilization %", fontsize=11, fontweight="bold")
            axes[0].set_title("Edge Server Utilization", fontsize=12, fontweight="bold")
            axes[0].set_ylim([0, 1])

            rc_util = [0.2 + progress * 0.3, 0.3 + progress * 0.4]
            rc_names = ["RC0", "RC1"]
            colors_rc = ["#B3E5FC" if u < 0.5 else "#45B7D1" for u in rc_util]
            axes[1].bar(rc_names, rc_util, color=colors_rc, edgecolor="black", linewidth=2)
            axes[1].set_ylabel("Utilization %", fontsize=11, fontweight="bold")
            axes[1].set_title("Regional Cloud Utilization", fontsize=12, fontweight="bold")
            axes[1].set_ylim([0, 1])

            rh_satisfaction = np.ones(8) * (0.5 + progress * 0.4)
            rh_names = [f"RH{i}" for i in range(8)]
            colors_rh = ["#FFB3B3" if s < 0.7 else "#45B7D1" for s in rh_satisfaction]
            axes[2].bar(rh_names, rh_satisfaction, color=colors_rh, edgecolor="black", linewidth=1.5)
            axes[2].set_ylabel("SLA Satisfaction", fontsize=11, fontweight="bold")
            axes[2].set_title("Radio Head SLA Satisfaction", fontsize=12, fontweight="bold")
            axes[2].set_ylim([0, 1])

            fig.suptitle(f"Network State Evolution (Frame {frame}/{frames})", fontsize=14, fontweight="bold")

        anim = FuncAnimation(fig, animate, frames=frames, interval=1000 // fps, repeat=True)

        if output_path:
            print("Creating network state animation...")
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
            anim.save(output_path, writer=writer)
            print(f"✓ Animation saved: {output_path}")

        return fig, anim


def create_all_animations(results_path: Path = DEFAULT_RESULTS_PATH, gif_workers: int | None = None) -> None:
    ensure_output_dirs()
    success = True

    print("\n" + "=" * 70)
    print("GPPO Animation Suite - Creating Visualizations")
    print("=" * 70 + "\n")

    if results_path.exists():
        print("[1/3] Creating Learning Animation...")
        try:
            TrainingAnimator.create_learning_animation(
                str(results_path),
                output_path=str(ANIMATIONS_DIR / "01_learning_progress.mp4"),
                fps=10,
                gif_workers=gif_workers,
            )
            print("     ✓ Saved: animations/01_learning_progress.mp4\n")
        except Exception as exc:
            success = False
            print(f"     ✗ Error: {exc}\n")

        print("[2/3] Creating Reward Landscape...")
        try:
            TrainingAnimator.create_reward_landscape(
                str(results_path),
                output_path=str(ANIMATIONS_DIR / "02_reward_landscape.png"),
                window_size=10,
            )
            print("     ✓ Saved: animations/02_reward_landscape.png\n")
        except Exception as exc:
            success = False
            print(f"     ✗ Error: {exc}\n")
    else:
        print("[1/3] No training results found. Run training first!")
        print("[2/3] Skipping reward landscape...\n")

    print("[3/3] Creating Network State Animation...")
    try:
        NetworkStateAnimator.create_network_state_animation(
            output_path=str(ANIMATIONS_DIR / "03_network_state.mp4"),
            fps=5,
        )
        print("     ✓ Saved: animations/03_network_state.mp4\n")
    except Exception as exc:
        success = False
        print(f"     ✗ Error: {exc}\n")

    print("=" * 70)
    if success:
        print("✓ All Animations Created Successfully!")
    else:
        print("Animation generation completed with errors.")
    print("=" * 70)
