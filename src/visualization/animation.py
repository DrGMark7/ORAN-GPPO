"""Animation visualization for GPPO training progress."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.common.paths import ANIMATIONS_DIR, DEFAULT_RESULTS_PATH, EPISODE_TRACES_DIR, ensure_output_dirs
from src.core import get_topology_spec


class TrainingAnimator:
    """Create animated training visualizations."""

    @staticmethod
    def _save_animation_with_tqdm(anim, output_path: str, writer, total_frames: int, description: str) -> None:
        with tqdm(total=total_frames, desc=description, unit="frame") as progress:
            last_frame = {"value": -1}

            def progress_callback(frame_idx: int, _frame_total: int) -> None:
                increment = frame_idx - last_frame["value"]
                if increment > 0:
                    progress.update(increment)
                last_frame["value"] = frame_idx

            anim.save(output_path, writer=writer, progress_callback=progress_callback)
            remaining = total_frames - progress.n
            if remaining > 0:
                progress.update(remaining)

    @staticmethod
    def create_learning_animation(
        json_path: str,
        output_path: str = None,
        fps: int = 10,
        gif_workers: int | None = None,
        max_frames: int = 100,
    ):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        raw_episode_count = len(results["episode_rewards"])
        if raw_episode_count == 0:
            raise ValueError("No episode data available for learning animation.")
        frame_indices = np.arange(raw_episode_count, dtype=int)
        if max_frames and raw_episode_count > max_frames:
            frame_indices = np.unique(np.linspace(0, raw_episode_count - 1, num=max_frames, dtype=int))

        episodes = frame_indices
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
        point_reward, = axes[0].plot([], [], marker="o", markersize=7, color="#7a1f1f")
        point_cost, = axes[1].plot([], [], marker="o", markersize=7, color="#124f47")
        point_validity, = axes[2].plot([], [], marker="o", markersize=7, color="#1d4e89")

        axes[0].set_xlim(0, raw_episode_count)
        axes[0].set_ylim(min(rewards) - 1, max(rewards) + 1)
        axes[0].set_xlabel("Episode", fontweight="bold")
        axes[0].set_ylabel("Cumulative Reward", fontweight="bold")
        axes[0].set_title("Learning Progress: Reward", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].set_xlim(0, raw_episode_count)
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

        axes[2].set_xlim(0, raw_episode_count)
        axes[2].set_ylim(0, 110)
        axes[2].set_xlabel("Episode", fontweight="bold")
        axes[2].set_ylabel("Valid Deployments (%)", fontweight="bold")
        axes[2].set_title("Reliability: Validity", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        def animate(frame):
            source_idx = int(frame_indices[frame])
            line_reward.set_data(episodes[: frame + 1], rewards[frame_indices[: frame + 1]])
            point_reward.set_data([source_idx], [rewards[source_idx]])
            axes[0].set_title(f"Reward at Ep {source_idx}: {rewards[source_idx]:.3f}", fontweight="bold")

            line_cost.set_data(episodes[: frame + 1], finite_costs[frame_indices[: frame + 1]])
            point_cost.set_data([source_idx], [finite_costs[source_idx]])
            cost_label = f"{costs[source_idx]:.2f}" if np.isfinite(costs[source_idx]) else "inf"
            axes[1].set_title(f"Cost at Ep {source_idx}: {cost_label}", fontweight="bold")

            line_validity.set_data(episodes[: frame + 1], validity[frame_indices[: frame + 1]])
            point_validity.set_data([source_idx], [validity[source_idx]])
            axes[2].set_title(f"Validity at Ep {source_idx}: {validity[source_idx]:.1f}%", fontweight="bold")

            fig.suptitle(
                f"GPPO Training Progress (Frame {frame + 1}/{len(frame_indices)} | Episode {source_idx + 1}/{raw_episode_count})",
                fontsize=14,
                fontweight="bold",
            )
            return line_reward, line_cost, line_validity, point_reward, point_cost, point_validity

        anim = FuncAnimation(fig, animate, frames=len(frame_indices), interval=1000 // fps, blit=True, repeat=True)

        if output_path:
            print(f"Saving animation to {output_path}...")
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
            TrainingAnimator._save_animation_with_tqdm(
                anim,
                output_path,
                writer,
                total_frames=len(frame_indices),
                description="Encoding learning animation",
            )
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
            TrainingAnimator._save_animation_with_tqdm(
                anim,
                output_path,
                writer,
                total_frames=frames,
                description="Encoding network-state animation",
            )
            print(f"✓ Animation saved: {output_path}")

        return fig, anim


class EpisodeTraceAnimator:
    """Animate one evaluated episode using per-time-slot trace export."""

    SPLIT_COLORS = ["#f94144", "#f3722c", "#f9c74f", "#90be6d"]

    @staticmethod
    def _build_positions(graph: nx.Graph):
        pos = {}
        rh_nodes = sorted([node for node in graph.nodes if node.startswith("RH")], key=lambda node: int(node[2:]))
        es_nodes = sorted([node for node in graph.nodes if node.startswith("ES")], key=lambda node: int(node[2:]))
        rc_nodes = sorted([node for node in graph.nodes if node.startswith("RC")], key=lambda node: int(node[2:]))
        rh_positions = np.linspace(0.05, 0.95, max(len(rh_nodes), 2))[:len(rh_nodes)]
        es_positions = np.linspace(0.15, 0.85, max(len(es_nodes), 2))[:len(es_nodes)]
        rc_positions = np.linspace(0.25, 0.75, max(len(rc_nodes), 2))[:len(rc_nodes)]
        for x, node in zip(rh_positions, rh_nodes):
            pos[node] = (x, 0.05)
        for x, node in zip(es_positions, es_nodes):
            pos[node] = (x, 0.5)
        for x, node in zip(rc_positions, rc_nodes):
            pos[node] = (x, 0.95)
        return pos

    @classmethod
    def create_episode_animation(cls, trace_path: str, output_path: str | None = None, fps: int = 4):
        with open(trace_path, "r", encoding="utf-8") as file_obj:
            trace = json.load(file_obj)

        topology_id = trace["topology_id"]
        slots = trace["slots"]
        summary = trace["summary"]
        spec = get_topology_spec(topology_id)
        graph = spec.build_graph()
        pos = cls._build_positions(graph)
        rh_nodes = sorted([node for node in graph.nodes if node.startswith("RH")], key=lambda node: int(node[2:]))
        es_nodes = sorted([node for node in graph.nodes if node.startswith("ES")], key=lambda node: int(node[2:]))
        rc_nodes = sorted([node for node in graph.nodes if node.startswith("RC")], key=lambda node: int(node[2:]))

        fig, (ax_graph, ax_summary) = plt.subplots(
            1,
            2,
            figsize=(18, 8),
            facecolor="white",
            gridspec_kw={"width_ratios": [2.2, 1.0]},
        )

        def draw_frame(frame_idx: int):
            ax_graph.clear()
            ax_summary.clear()
            slot = slots[frame_idx]

            ax_graph.set_facecolor("#fbfaf6")
            base_edge_colors = []
            base_edge_widths = []
            for u, v, _ in graph.edges(data=True):
                if (u.startswith("RH") and v.startswith("RC")) or (u.startswith("RC") and v.startswith("RH")):
                    base_edge_colors.append("#d97706")
                    base_edge_widths.append(2.6)
                elif (u.startswith("ES") and v.startswith("RC")) or (u.startswith("RC") and v.startswith("ES")):
                    base_edge_colors.append("#5dade2")
                    base_edge_widths.append(2.0)
                else:
                    base_edge_colors.append("#7f8c8d")
                    base_edge_widths.append(1.5)
            nx.draw_networkx_edges(graph, pos, ax=ax_graph, edge_color=base_edge_colors, width=base_edge_widths, alpha=0.55)
            nx.draw_networkx_nodes(graph, pos, nodelist=rh_nodes, node_color="#e76f51", node_size=900, ax=ax_graph, linewidths=1.2, edgecolors="#3d2c2a")
            nx.draw_networkx_nodes(graph, pos, nodelist=es_nodes, node_color="#2a9d8f", node_size=1300, ax=ax_graph, linewidths=1.2, edgecolors="#193d39")
            nx.draw_networkx_nodes(graph, pos, nodelist=rc_nodes, node_color="#457b9d", node_size=1500, ax=ax_graph, linewidths=1.2, edgecolors="#22313f")
            nx.draw_networkx_labels(graph, pos, labels={node: node for node in graph.nodes}, font_size=9, font_weight="bold", ax=ax_graph)

            decision_edges = []
            decision_colors = []
            splits = slot["split_vector"]
            es_choices = slot["es_choice_vector"]
            rc_choices = slot["rc_choice_vector"]
            for rh_idx in range(len(splits)):
                rh_node = f"RH{rh_idx}"
                es_node = f"ES{int(es_choices[rh_idx])}"
                rc_node = f"RC{int(rc_choices[rh_idx])}"
                split_idx = int(splits[rh_idx])
                color = cls.SPLIT_COLORS[split_idx]
                if split_idx == 3 and graph.has_edge(rh_node, rc_node):
                    decision_edges.append((rh_node, rc_node))
                    decision_colors.append(color)
                else:
                    if graph.has_edge(rh_node, es_node):
                        decision_edges.append((rh_node, es_node))
                        decision_colors.append(color)
                    if graph.has_edge(es_node, rc_node):
                        decision_edges.append((es_node, rc_node))
                        decision_colors.append(color)
                x, y = pos[rh_node]
                ax_graph.text(
                    x,
                    y - 0.08,
                    f"S{split_idx + 1}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
            for (u, v), color in zip(decision_edges, decision_colors):
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax_graph, edge_color=color, width=4.2, alpha=0.95)

            if slot["valid_deployment"]:
                valid_text = (
                    "structurally valid with soft violations"
                    if slot["invalid_reasons"]
                    else "valid"
                )
            else:
                valid_text = "invalid"
            cost_text = slot["deployment_cost"]
            ax_graph.set_title(
                f"Single Time-Slot Decision Snapshot | {topology_id} | slot {slot['time_slot']}/{trace['episode_length_time_slots']} | {valid_text} | cost={cost_text}",
                fontsize=13,
                fontweight="bold",
                pad=16,
            )
            ax_graph.axis("off")

            ax_summary.axis("off")
            if slot["valid_deployment"]:
                reason_label = "Soft violations" if slot["invalid_reasons"] else "Soft violations"
                reason_text = ", ".join(slot["invalid_reasons"]) if slot["invalid_reasons"] else "none"
            else:
                reason_label = "Invalid reasons"
                reason_text = ", ".join(slot["invalid_reasons"]) if slot["invalid_reasons"] else "none"

            summary_lines = [
                "Paper-Aligned Episode Summary",
                f"Benchmark: {trace['benchmark']}",
                f"Pool: {trace['topology_pool_name']}",
                f"Constraint mode: {trace['constraint_mode']}",
                f"Average cost / slot: {summary['average_cost_per_slot']:.3f}",
                f"Average reward / slot: {summary['average_reward_per_slot']:.3f}",
                f"Valid slots: {summary['valid_slot_percentage']:.1f}%",
                f"Total split changes: {int(summary['total_split_changes'])}",
                f"Total ES changes: {int(summary['total_es_changes'])}",
                f"Total RC changes: {int(summary['total_rc_changes'])}",
                f"Total reconfig changes: {int(summary['total_reconfiguration_changes'])}",
                f"Total reconfig cost: {summary['total_reconfiguration_cost']:.3f}",
                "",
                f"Current slot processing cost: {slot['processing_cost']:.3f}",
                f"Current slot routing cost: {slot['routing_cost']:.3f}",
                f"Current slot reconfig cost: {slot['reconfiguration_cost']:.3f}",
                f"Current slot SLA penalty: {slot['sla_penalty']:.3f}",
                f"Current slot changes: split={slot['split_changes']} es={slot['es_changes']} rc={slot['rc_changes']}",
                f"{reason_label}: {reason_text}",
            ]
            ax_summary.text(
                0.02,
                0.98,
                "\n".join(summary_lines),
                transform=ax_summary.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.suptitle(
                "GPPO Episode Evolution Across Paper Time Slots",
                fontsize=15,
                fontweight="bold",
                y=0.98,
            )

        anim = FuncAnimation(fig, draw_frame, frames=len(slots), interval=1000 // fps, repeat=True)
        if output_path:
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
            TrainingAnimator._save_animation_with_tqdm(
                anim,
                output_path,
                writer,
                total_frames=len(slots),
                description="Encoding episode animation",
            )
            print(f"✓ Episode animation saved: {output_path}")
        return fig, anim


def _find_default_trace_path() -> Path | None:
    trace_paths = list(EPISODE_TRACES_DIR.glob("*.json"))
    return max(trace_paths, key=lambda path: path.stat().st_mtime) if trace_paths else None


def create_all_animations(
    results_path: Path = DEFAULT_RESULTS_PATH,
    gif_workers: int | None = None,
    episode_trace_path: Path | None = None,
) -> None:
    ensure_output_dirs()
    success = True
    selected_trace_path = episode_trace_path if episode_trace_path is not None else _find_default_trace_path()

    print("\n" + "=" * 70)
    print("GPPO Animation Suite - Creating Visualizations")
    print("=" * 70 + "\n")

    if results_path.exists():
        print("[1/4] Creating Learning Animation...")
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

        print("[2/4] Creating Reward Landscape...")
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
        print("[1/4] No training results found. Run training first!")
        print("[2/4] Skipping reward landscape...\n")

    print("[3/4] Creating Paper-Aligned Episode Animation...")
    if selected_trace_path is not None and selected_trace_path.exists():
        try:
            EpisodeTraceAnimator.create_episode_animation(
                str(selected_trace_path),
                output_path=str(ANIMATIONS_DIR / "03_episode_evolution.mp4"),
                fps=4,
            )
            print("     ✓ Saved: animations/03_episode_evolution.mp4\n")
        except Exception as exc:
            success = False
            print(f"     ✗ Error: {exc}\n")
    else:
        print("     No episode trace found. Run evaluation or training with evaluation first.\n")

    print("[4/4] Creating Network State Animation...")
    try:
        NetworkStateAnimator.create_network_state_animation(
            output_path=str(ANIMATIONS_DIR / "04_network_state.mp4"),
            fps=5,
        )
        print("     ✓ Saved: animations/04_network_state.mp4\n")
    except Exception as exc:
        success = False
        print(f"     ✗ Error: {exc}\n")

    print("=" * 70)
    if success:
        print("✓ All Animations Created Successfully!")
    else:
        print("Animation generation completed with errors.")
    print("=" * 70)
