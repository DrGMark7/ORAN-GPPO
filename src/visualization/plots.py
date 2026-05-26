"""Visualization tools for GPPO framework."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from src.common.paths import DEFAULT_CHECKPOINT_PATH
from src.core import GNNFeatureExtractor, ORANGraphBuilder, PPOAgent, SimplifiedORANEnv, get_topology_spec


class NetworkTopologyVisualizer:
    """Visualize a random O-RAN topology and optional GPPO inference."""

    def __init__(self, num_rhs: int = 8, num_ess: int = 3, num_rcs: int = 2, seed: int = 42, benchmark: Optional[str] = None):
        self.num_rhs = num_rhs
        self.num_ess = num_ess
        self.num_rcs = num_rcs
        self.seed = seed
        self.benchmark = benchmark or ("large" if num_rhs > 8 else "small")

    def create_environment(
        self,
        topology_pool_name: str = "train",
        topology_id: Optional[str] = None,
    ) -> SimplifiedORANEnv:
        if topology_id is not None:
            spec = get_topology_spec(topology_id)
            benchmark = spec.benchmark
            num_rhs = spec.num_rhs
            num_ess = spec.num_ess
            num_rcs = spec.num_rcs
        else:
            benchmark = self.benchmark
            num_rhs = self.num_rhs
            num_ess = self.num_ess
            num_rcs = self.num_rcs

        env = SimplifiedORANEnv(
            num_rhs=num_rhs,
            num_ess=num_ess,
            num_rcs=num_rcs,
            max_steps=1,
            benchmark=benchmark,
            topology_pool_name=topology_pool_name,
            topology_selection_mode="fixed",
            topology_id=topology_id,
        )
        env.reset(
            seed=self.seed,
            options={
                "topology_pool_name": topology_pool_name,
                "topology_selection_mode": "fixed",
                "topology_id": topology_id,
            },
        )
        return env

    def create_graph(self, topology_pool_name: str = "train", topology_id: Optional[str] = None) -> nx.Graph:
        return self.create_environment(topology_pool_name=topology_pool_name, topology_id=topology_id).topology.copy()

    @staticmethod
    def infer_topology_from_checkpoint(checkpoint_path: str) -> Optional[Tuple[int, int, int]]:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None

        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            metadata = payload.get("metadata", {})
            if {"num_rhs", "num_ess", "num_rcs"} <= set(metadata):
                return int(metadata["num_rhs"]), int(metadata["num_ess"]), int(metadata["num_rcs"])
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        split_shape = state_dict["split_head.bias"].shape[0]
        es_shape = state_dict["es_head.bias"].shape[0]
        rc_shape = state_dict["rc_head.bias"].shape[0]

        for num_rhs in range(1, 257):
            if split_shape % num_rhs != 0 or es_shape % num_rhs != 0 or rc_shape % num_rhs != 0:
                continue
            num_splits = split_shape // num_rhs
            num_ess = es_shape // num_rhs
            num_rcs = rc_shape // num_rhs
            if num_splits == 4 and num_ess > 0 and num_rcs > 0:
                return num_rhs, num_ess, num_rcs
        return None

    def _build_positions(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        pos: Dict[str, Tuple[float, float]] = {}
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

    def _load_agent_and_run_inference(
        self,
        env: SimplifiedORANEnv,
        checkpoint_path: Optional[str],
        device: str = "cpu",
    ) -> Optional[Dict[str, object]]:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None

        adjacency, edge_features, _ = env._get_adjacency_info()
        graph_builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs)
        graph = graph_builder.build_graph(
            env.rh_demands,
            env.rh_latencies,
            env.es_remaining,
            env.rc_remaining,
            adjacency,
            edge_features,
        ).to(device)

        gnn = GNNFeatureExtractor(input_dim=6, hidden_dim=64, output_dim=128).to(device)
        agent = PPOAgent(
            feature_dim=128,
            num_rhs=env.num_rhs,
            num_splits=4,
            num_ess=env.num_ess,
            num_rcs=env.num_rcs,
            device=device,
        )

        payload = torch.load(checkpoint_path, map_location=device)
        agent.load(checkpoint_path)
        if isinstance(payload, dict) and payload.get("gnn_state_dict") is not None:
            gnn.load_state_dict(payload["gnn_state_dict"])
            print("Loaded trained GNN weights from checkpoint for visualization.")
        agent.policy.eval()
        gnn.eval()

        with torch.no_grad():
            features = gnn(graph)
            action_mask = env.get_action_mask()
            action, _, _, _ = agent.select_action_sequential(
                features.squeeze(0),
                action_mask,
                env.get_conditional_rc_mask,
                deterministic=True,
            )

        _, _, _, _, info = env.step(action)
        splits, es_choices, rc_choices = env._split_action(action)
        return {
            "action": action,
            "splits": splits,
            "es_choices": es_choices,
            "rc_choices": rc_choices,
            "info": info,
        }

    def draw_topology(
        self,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = str(DEFAULT_CHECKPOINT_PATH),
        figsize: Tuple[int, int] = (16, 11),
        device: str = "cpu",
        topology_pool_name: str = "train",
        topology_id: Optional[str] = None,
    ):
        env = self.create_environment(topology_pool_name=topology_pool_name, topology_id=topology_id)
        graph = env.topology.copy()
        pos = self._build_positions(graph)
        inference = self._load_agent_and_run_inference(env, checkpoint_path, device=device)

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("#fbfaf6")

        base_edge_colors = []
        base_edge_widths = []
        for u, v, _ in graph.edges(data=True):
            if u.startswith("RH") and v.startswith("RC") or u.startswith("RC") and v.startswith("RH"):
                base_edge_colors.append("#d97706")
                base_edge_widths.append(2.8)
            elif u.startswith("ES") and v.startswith("RC") or u.startswith("RC") and v.startswith("ES"):
                base_edge_colors.append("#5dade2")
                base_edge_widths.append(2.2)
            else:
                base_edge_colors.append("#7f8c8d")
                base_edge_widths.append(1.8)

        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=base_edge_colors, width=base_edge_widths, alpha=0.65)

        rh_nodes = sorted([node for node in graph.nodes if node.startswith("RH")], key=lambda node: int(node[2:]))
        es_nodes = sorted([node for node in graph.nodes if node.startswith("ES")], key=lambda node: int(node[2:]))
        rc_nodes = sorted([node for node in graph.nodes if node.startswith("RC")], key=lambda node: int(node[2:]))

        nx.draw_networkx_nodes(graph, pos, nodelist=rh_nodes, node_color="#e76f51", node_size=1000, ax=ax, linewidths=1.5, edgecolors="#3d2c2a")
        nx.draw_networkx_nodes(graph, pos, nodelist=es_nodes, node_color="#2a9d8f", node_size=1400, ax=ax, linewidths=1.5, edgecolors="#193d39")
        nx.draw_networkx_nodes(graph, pos, nodelist=rc_nodes, node_color="#457b9d", node_size=1700, ax=ax, linewidths=1.5, edgecolors="#22313f")

        labels = {}
        for node in rh_nodes:
            rh_idx = int(node[2:])
            labels[node] = f"{node}\n{env.rh_demands[rh_idx]:.0f} Mbps\n{env.rh_latencies[rh_idx]:.1f} ms"
        for node in es_nodes:
            es_idx = int(node[2:])
            labels[node] = f"{node}\ncap {env.es_remaining[es_idx]:.1f}"
        for node in rc_nodes:
            rc_idx = int(node[2:])
            labels[node] = f"{node}\ncap {env.rc_remaining[rc_idx]:.1f}"

        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            edge_labels[(u, v)] = f"d={data['delay']:.2f}\nbw={data['bandwidth']:.1f}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, rotate=False, ax=ax, bbox={"alpha": 0.75, "color": "white", "pad": 0.15})

        decision_lines: List[Tuple[str, str]] = []
        decision_colors: List[str] = []
        if inference is not None:
            split_palette = ["#f94144", "#f3722c", "#f9c74f", "#90be6d"]
            splits = inference["splits"]
            es_choices = inference["es_choices"]
            rc_choices = inference["rc_choices"]

            for rh_idx in range(env.num_rhs):
                rh_node = f"RH{rh_idx}"
                es_node = f"ES{int(es_choices[rh_idx])}"
                rc_node = f"RC{int(rc_choices[rh_idx])}"
                split_idx = int(splits[rh_idx])
                color = split_palette[split_idx]

                if split_idx == 3 and graph.has_edge(rh_node, rc_node):
                    decision_lines.append((rh_node, rc_node))
                    decision_colors.append(color)
                else:
                    if graph.has_edge(rh_node, es_node):
                        decision_lines.append((rh_node, es_node))
                        decision_colors.append(color)
                    if graph.has_edge(es_node, rc_node):
                        decision_lines.append((es_node, rc_node))
                        decision_colors.append(color)

                x, y = pos[rh_node]
                ax.text(x, y - 0.08, f"S{split_idx + 1} -> ES{int(es_choices[rh_idx])}/RC{int(rc_choices[rh_idx])}", ha="center", va="top", fontsize=8, color=color, fontweight="bold")

            for (u, v), color in zip(decision_lines, decision_colors):
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax, edge_color=color, width=4.5, alpha=0.95)

            info = inference["info"]
            title = (
                f"Single Time-Slot Decision Snapshot | {env.topology_id} | "
                f"valid={info['valid_deployment']} cost={info['deployment_cost']:.2f} "
                f"failed_links={info['failed_links']}"
            )
        else:
            title = f"Single Time-Slot Decision Snapshot | {env.topology_id} | No checkpoint inference"

        legend_handles = [
            mpatches.Patch(color="#e76f51", label="Radio Head"),
            mpatches.Patch(color="#2a9d8f", label="Edge Server"),
            mpatches.Patch(color="#457b9d", label="Regional Cloud"),
            mpatches.Patch(color="#d97706", label="Direct RH-RC Link"),
        ]
        if inference is not None:
            legend_handles.extend(
                [
                    mpatches.Patch(color="#f94144", label="Split 1 path"),
                    mpatches.Patch(color="#f3722c", label="Split 2 path"),
                    mpatches.Patch(color="#f9c74f", label="Split 3 path"),
                    mpatches.Patch(color="#90be6d", label="Split 4 path"),
                ]
            )

        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.95)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, ax


class TrainingVisualization:
    @staticmethod
    def plot_training_curves(json_path: str, save_path: str = None, figsize: Tuple = (15, 5)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
        episodes = np.arange(len(results["episode_rewards"]))
        rewards = np.asarray(results["episode_rewards"], dtype=float)
        costs = np.asarray(results["episode_costs"], dtype=float)
        validity_rate = np.asarray(results["valid_deployments"], dtype=float) * 100.0

        finite_costs = np.where(np.isfinite(costs), costs, np.nan)
        window = min(10, max(len(episodes), 1))
        reward_ma = uniform_filter1d(rewards, size=window, mode="nearest")
        cost_fill = np.where(np.isfinite(costs), costs, np.nanmedian(finite_costs) if np.isfinite(finite_costs).any() else 0.0)
        cost_ma = uniform_filter1d(cost_fill, size=window, mode="nearest")

        axes[0].plot(episodes, rewards, linewidth=2.5, color="#e76f51", label="Reward")
        axes[0].fill_between(episodes, rewards, alpha=0.25, color="#e76f51")
        axes[0].plot(episodes, reward_ma, linewidth=2.0, color="#7a1f1f", linestyle="--", label=f"MA({window})")
        axes[0].set_xlabel("Episode", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Cumulative Reward", fontsize=11, fontweight="bold")
        axes[0].set_title("Learning Progress", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(episodes, finite_costs, linewidth=2.5, color="#2a9d8f", label="Cost")
        axes[1].fill_between(episodes, finite_costs, alpha=0.25, color="#2a9d8f")
        axes[1].plot(episodes, cost_ma, linewidth=2.0, color="#124f47", linestyle="--", label=f"MA({window})")
        axes[1].set_xlabel("Episode", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Deployment Cost", fontsize=11, fontweight="bold")
        axes[1].set_title("Cost on Valid Episodes", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(episodes, validity_rate, linewidth=2.5, color="#457b9d", label="Validity")
        axes[2].fill_between(episodes, validity_rate, alpha=0.25, color="#457b9d")
        axes[2].set_xlabel("Episode", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Valid Deployments (%)", fontsize=11, fontweight="bold")
        axes[2].set_title("Constraint Satisfaction", fontsize=12, fontweight="bold")
        axes[2].set_ylim([0, 110])
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        plt.suptitle("GPPO Training Progress", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, axes

    @staticmethod
    def plot_split_usage(json_path: str, save_path: str = None, figsize: Tuple = (14, 5)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        episode_split_usage = results.get("episode_split_usage", [])
        topology_ids = results.get("episode_topology_ids", [])
        evaluation = results.get("evaluation", {})
        split_labels = [f"S{i}" for i in range(1, 5)]

        training_totals = {label: 0 for label in split_labels}
        for episode_usage in episode_split_usage:
            for key, value in episode_usage.items():
                training_totals[key] += int(value)

        topology_totals: Dict[str, Dict[str, int]] = {}
        for topology_id, episode_usage in zip(topology_ids, episode_split_usage):
            topology_totals.setdefault(topology_id, {label: 0 for label in split_labels})
            for key, value in episode_usage.items():
                topology_totals[topology_id][key] += int(value)

        eval_by_topology: Dict[str, Dict[str, float]] = {}
        for pool_summary in evaluation.values():
            for topology_summary in pool_summary.values():
                for topology_id, per_topology in topology_summary.get("per_topology", {}).items():
                    eval_by_topology[topology_id] = {
                        key: float(per_topology.get("split_distribution", {}).get(key, 0.0))
                        for key in split_labels
                    }

        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
        split_colors = ["#f94144", "#f3722c", "#f9c74f", "#90be6d"]

        training_values = np.asarray([training_totals[key] for key in split_labels], dtype=float)
        training_freq = training_values / max(training_values.sum(), 1.0)
        axes[0].bar(split_labels, training_freq, color=split_colors, edgecolor="black", linewidth=1.5)
        axes[0].set_ylim([0, 1])
        axes[0].set_ylabel("Frequency", fontsize=11, fontweight="bold")
        axes[0].set_title("Training Split Usage", fontsize=12, fontweight="bold")
        axes[0].grid(True, axis="y", alpha=0.3)

        if topology_totals:
            topology_names = list(topology_totals.keys())
            x = np.arange(len(topology_names))
            bottom = np.zeros(len(topology_names), dtype=float)
            for split_idx, split_label in enumerate(split_labels):
                values = np.asarray([topology_totals[name][split_label] for name in topology_names], dtype=float)
                total = np.asarray([sum(topology_totals[name].values()) for name in topology_names], dtype=float)
                freq = np.divide(values, np.maximum(total, 1.0))
                axes[1].bar(x, freq, bottom=bottom, color=split_colors[split_idx], edgecolor="black", linewidth=1.0, label=split_label)
                bottom += freq
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(topology_names, rotation=25, ha="right")
            axes[1].set_ylim([0, 1])
            axes[1].set_title("Training Usage by Topology", fontsize=12, fontweight="bold")
            axes[1].grid(True, axis="y", alpha=0.3)
            axes[1].legend(loc="best")

        if eval_by_topology:
            topology_names = list(eval_by_topology.keys())
            x = np.arange(len(topology_names))
            bottom = np.zeros(len(topology_names), dtype=float)
            for split_idx, split_label in enumerate(split_labels):
                values = np.asarray([eval_by_topology[name][split_label] for name in topology_names], dtype=float)
                axes[2].bar(x, values, bottom=bottom, color=split_colors[split_idx], edgecolor="black", linewidth=1.0, label=split_label)
                bottom += values
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(topology_names, rotation=25, ha="right")
            axes[2].set_ylim([0, 1])
            axes[2].set_title("Evaluation Usage by Topology", fontsize=12, fontweight="bold")
            axes[2].grid(True, axis="y", alpha=0.3)
            axes[2].legend(loc="best")

        plt.suptitle("Policy-Derived Split Usage", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, axes

    @staticmethod
    def plot_statistics(json_path: str, save_path: str = None, figsize: Tuple = (14, 5)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
        quarters = 4
        n_episodes = len(results["episode_rewards"])
        phase_size = max(n_episodes // quarters, 1)
        phases = []
        phase_rewards = []
        phase_costs = []

        for i in range(quarters):
            start = i * phase_size
            end = n_episodes if i == quarters - 1 else min((i + 1) * phase_size, n_episodes)
            if start >= n_episodes:
                break
            phases.append(f"Phase {i + 1}\n(Ep {start + 1}-{end})")
            phase_rewards.append(float(np.mean(results["episode_rewards"][start:end])))
            costs = np.asarray(results["episode_costs"][start:end], dtype=float)
            finite_costs = costs[np.isfinite(costs)]
            phase_costs.append(float(np.mean(finite_costs)) if finite_costs.size else np.nan)

        colors_reward = ["#e76f51", "#ef8a62", "#f4a261", "#f6bd60"][:len(phases)]
        colors_cost = ["#2a9d8f", "#42b3a6", "#63c4b8", "#84d5ca"][:len(phases)]

        axes[0].bar(phases, phase_rewards, color=colors_reward, edgecolor="black", linewidth=1.5)
        axes[0].set_ylabel("Average Reward", fontsize=11, fontweight="bold")
        axes[0].set_title("Reward by Training Phase", fontsize=12, fontweight="bold")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(phases, phase_costs, color=colors_cost, edgecolor="black", linewidth=1.5)
        axes[1].set_ylabel("Average Cost", fontsize=11, fontweight="bold")
        axes[1].set_title("Finite Valid Cost by Phase", fontsize=12, fontweight="bold")
        axes[1].grid(True, axis="y", alpha=0.3)

        plt.suptitle("Training Phase Analysis", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, axes

    @staticmethod
    def plot_cost_breakdown_by_phase(json_path: str, save_path: str = None, figsize: Tuple = (14, 6)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        breakdowns = results.get("episode_cost_breakdowns", [])
        n_episodes = len(breakdowns)
        quarters = 4
        phase_size = max(n_episodes // quarters, 1)
        phases = []
        labels = ["processing_cost", "routing_cost", "reconfiguration_cost", "sla_penalty"]
        phase_values = {label: [] for label in labels}

        for i in range(quarters):
            start = i * phase_size
            end = n_episodes if i == quarters - 1 else min((i + 1) * phase_size, n_episodes)
            if start >= n_episodes:
                break
            phases.append(f"Phase {i + 1}\n(Ep {start + 1}-{end})")
            phase_slice = breakdowns[start:end]
            for label in labels:
                phase_values[label].append(float(np.mean([episode[label] for episode in phase_slice])))

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        x = np.arange(len(phases))
        bottom = np.zeros(len(phases), dtype=float)
        colors = ["#e76f51", "#2a9d8f", "#e9c46a", "#457b9d"]
        for color, label in zip(colors, labels):
            values = np.asarray(phase_values[label], dtype=float)
            ax.bar(x, values, bottom=bottom, color=color, edgecolor="black", linewidth=1.0, label=label)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel("Average Cost", fontsize=11, fontweight="bold")
        ax.set_title("Cost Breakdown by Training Phase", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, ax

    @staticmethod
    def plot_evaluation_topology_summary(json_path: str, save_path: str = None, figsize: Tuple = (15, 6)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        evaluation = results.get("evaluation", {})
        topology_rows = []
        for pool_name, pool_summary in evaluation.items():
            for _, topology_summary in pool_summary.items():
                for topology_id, per_topology in topology_summary.get("per_topology", {}).items():
                    topology_rows.append({
                        "pool": pool_name,
                        "topology_id": topology_id,
                        "mean_reward": float(per_topology.get("mean_reward", 0.0)),
                        "mean_cost": float(per_topology.get("mean_cost", 0.0)),
                        "validity_rate": float(per_topology.get("validity_rate", 0.0)),
                        "sla_penalty": float(per_topology.get("sla_penalty", 0.0)),
                    })

        fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor="white")
        if not topology_rows:
            for ax in axes:
                ax.text(0.5, 0.5, "No evaluation data", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            return fig, axes

        topology_names = [row["topology_id"] for row in topology_rows]
        colors = ["#e76f51" if row["pool"] == "train_pool" else "#457b9d" for row in topology_rows]
        x = np.arange(len(topology_rows))
        metric_specs = [
            ("mean_reward", "Reward"),
            ("mean_cost", "Cost"),
            ("validity_rate", "Valid Rate"),
            ("sla_penalty", "SLA Penalty"),
        ]

        for ax, (metric_key, title) in zip(axes, metric_specs):
            values = [row[metric_key] for row in topology_rows]
            ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(topology_names, rotation=25, ha="right")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            if metric_key == "validity_rate":
                ax.set_ylim([0, 1.05])

        plt.suptitle("Train vs Test Topology Evaluation Summary", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, axes


class ActionSpaceVisualizer:
    @staticmethod
    def plot_action_space(num_rhs=8, num_ess=3, num_rcs=2, save_path: str = None, figsize: Tuple = (12, 8)):
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        splits = 4
        total_actions = splits * num_ess * num_rcs * num_rhs
        actions_per_rh = splits * num_ess * num_rcs
        y_pos = 0
        colors = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f"]

        for rh_idx in range(num_rhs):
            ax.text(-0.5, y_pos + 0.5, f"RH{rh_idx}", fontsize=10, fontweight="bold", ha="right", va="center")
            for split_idx in range(splits):
                x_pos = split_idx * (num_ess * num_rcs)
                rect = mpatches.Rectangle((x_pos, y_pos), num_ess * num_rcs, 1, linewidth=1.5, edgecolor="black", facecolor=colors[split_idx], alpha=0.8)
                ax.add_patch(rect)
            y_pos += 2

        ax.set_xlim(-1, total_actions // num_rhs)
        ax.set_ylim(-1, y_pos + 1)
        ax.set_xlabel(f"Action Index (0-{actions_per_rh - 1})", fontsize=11, fontweight="bold")
        ax.set_title(f"Action Space: {num_rhs} RHs x {actions_per_rh} actions = {total_actions}", fontsize=12, fontweight="bold")
        ax.set_yticks(np.arange(0.5, y_pos, 2))
        ax.set_yticklabels([f"RH{i}" for i in range(num_rhs)])
        ax.grid(True, axis="x", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, ax


class CostBreakdownVisualizer:
    @staticmethod
    def plot_cost_components(json_path: str, save_path: str = None, figsize: Tuple = (12, 6)):
        with open(json_path, "r", encoding="utf-8") as file_obj:
            results = json.load(file_obj)

        training_summary = results.get("training_summary", {})
        cost_breakdown = training_summary.get("avg_cost_breakdown_per_episode", {})
        labels = ["processing_cost", "routing_cost", "reconfiguration_cost", "sla_penalty"]
        values = [float(cost_breakdown.get(label, 0.0)) for label in labels]
        colors = ["#e76f51", "#2a9d8f", "#e9c46a", "#457b9d"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="white")
        bars = ax1.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("Average Cost Per Episode", fontsize=11, fontweight="bold")
        ax1.set_title("Observed Cost Breakdown", fontsize=12, fontweight="bold")
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels, rotation=25, ha="right")
        ax1.grid(True, axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontweight="bold")

        ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 1.5})
        ax2.set_title("Relative Observed Cost Distribution", fontsize=12, fontweight="bold")

        plt.suptitle("Policy-Derived Cost Analysis", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, (ax1, ax2)


class PerformanceComparison:
    @staticmethod
    def plot_baseline_comparison(methods: List[str], rewards: List[float], costs: List[float], validity: List[float], save_path: str = None, figsize: Tuple = (14, 5)):
        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
        x = np.arange(len(methods))
        width = 0.6

        colors_reward = ["#d9d9d9" if "GPPO" not in method else "#e76f51" for method in methods]
        colors_cost = ["#d9d9d9" if "GPPO" not in method else "#2a9d8f" for method in methods]
        colors_valid = ["#d9d9d9" if "GPPO" not in method else "#457b9d" for method in methods]

        axes[0].bar(x, rewards, width, color=colors_reward, edgecolor="black", linewidth=1.5)
        axes[0].set_ylabel("Mean Reward", fontsize=11, fontweight="bold")
        axes[0].set_title("Reward Comparison", fontsize=12, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha="right")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(x, costs, width, color=colors_cost, edgecolor="black", linewidth=1.5)
        axes[1].set_ylabel("Deployment Cost", fontsize=11, fontweight="bold")
        axes[1].set_title("Cost Comparison", fontsize=12, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=45, ha="right")
        axes[1].grid(True, axis="y", alpha=0.3)

        axes[2].bar(x, np.asarray(validity) * 100.0, width, color=colors_valid, edgecolor="black", linewidth=1.5)
        axes[2].set_ylabel("Validity Rate (%)", fontsize=11, fontweight="bold")
        axes[2].set_title("Deployment Validity", fontsize=12, fontweight="bold")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(methods, rotation=45, ha="right")
        axes[2].set_ylim([0, 110])
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.suptitle("GPPO vs Baselines", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")

        return fig, axes
