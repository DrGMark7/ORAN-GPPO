"""Small demo for the paper-aligned GPPO components."""

import torch

from src.core import GNNFeatureExtractor, ORANGraphBuilder, PPOAgent, SimplifiedORANEnv


def demo_environment() -> None:
    print("\n" + "=" * 70)
    print("DEMO 1: O-RAN Environment")
    print("=" * 70)

    env = SimplifiedORANEnv(num_rhs=8, num_ess=3, num_rcs=2, max_steps=20)
    state, _ = env.reset(seed=42)
    action_mask = env.get_action_mask()

    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Split mask shape: {action_mask['split'].shape}")
    print(f"ES mask shape: {action_mask['es'].shape}")
    print(f"RC mask shape: {action_mask['rc'].shape}")

    sample_action = env.action_space.sample()
    _, reward, _, _, info = env.step(sample_action)
    print(f"Sample reward: {reward:.4f}")
    print(f"Valid deployment: {info['valid_deployment']}")
    print(f"Deployment cost: {info['deployment_cost']}")


def demo_gnn() -> None:
    print("\n" + "=" * 70)
    print("DEMO 2: GNN Feature Extractor")
    print("=" * 70)

    env = SimplifiedORANEnv(num_rhs=8, num_ess=3, num_rcs=2, max_steps=1)
    env.reset(seed=7)
    adjacency, edge_features, _ = env._get_adjacency_info()

    builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs)
    graph = builder.build_graph(
        env.rh_demands,
        env.rh_latencies,
        env.es_remaining,
        env.rc_remaining,
        adjacency,
        edge_features,
    )

    gnn = GNNFeatureExtractor(input_dim=6, hidden_dim=64, output_dim=128)
    with torch.no_grad():
        embedding = gnn(graph)

    print(f"Graph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.num_edges}")
    print(f"Embedding shape: {embedding.shape}")


def demo_ppo_agent() -> None:
    print("\n" + "=" * 70)
    print("DEMO 3: PPO Agent")
    print("=" * 70)

    env = SimplifiedORANEnv(num_rhs=8, num_ess=3, num_rcs=2, max_steps=10)
    env.reset(seed=123)

    agent = PPOAgent(
        feature_dim=128,
        num_rhs=env.num_rhs,
        num_splits=4,
        num_ess=env.num_ess,
        num_rcs=env.num_rcs,
    )
    features = torch.randn(128)
    action_mask = env.get_action_mask()
    action, log_prob, value, _ = agent.select_action_sequential(
        features,
        action_mask,
        env.get_conditional_rc_mask,
    )

    print(f"Action vector length: {len(action)}")
    print(f"Action head sample: {action[:6]}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"Value estimate: {value:.4f}")


def demo_integration() -> None:
    print("\n" + "=" * 70)
    print("DEMO 4: Full Integration")
    print("=" * 70)

    env = SimplifiedORANEnv(num_rhs=8, num_ess=3, num_rcs=2, max_steps=5)
    env.reset(seed=99)
    gnn = GNNFeatureExtractor(input_dim=6, hidden_dim=64, output_dim=128)
    builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs)
    agent = PPOAgent(
        feature_dim=128,
        num_rhs=env.num_rhs,
        num_splits=4,
        num_ess=env.num_ess,
        num_rcs=env.num_rcs,
    )

    for step in range(5):
        adjacency, edge_features, _ = env._get_adjacency_info()
        graph = builder.build_graph(
            env.rh_demands,
            env.rh_latencies,
            env.es_remaining,
            env.rc_remaining,
            adjacency,
            edge_features,
        )

        with torch.no_grad():
            features = gnn(graph)

        action, _, _, _ = agent.select_action_sequential(
            features.squeeze(0),
            env.get_action_mask(),
            env.get_conditional_rc_mask,
        )
        _, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step + 1}: reward={reward:.4f}, "
            f"valid={info['valid_deployment']}, cost={info['deployment_cost']}"
        )

        if terminated or truncated:
            break


def run_demo() -> None:
    demo_environment()
    demo_gnn()
    demo_ppo_agent()
    demo_integration()
