from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class MaskedPPOPolicy(nn.Module):
    """Factorized PPO policy for split, ES, and RC decisions."""

    def __init__(
        self,
        feature_dim: int,
        num_rhs: int,
        num_splits: int,
        num_ess: int,
        num_rcs: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_rhs = num_rhs
        self.num_splits = num_splits
        self.num_ess = num_ess
        self.num_rcs = num_rcs

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.split_head = nn.Linear(hidden_dim, num_rhs * num_splits)
        self.es_head = nn.Linear(hidden_dim, num_rhs * num_ess)
        self.rc_head = nn.Linear(hidden_dim, num_rhs * num_rcs)
        self.value_net = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        split_logits = self.split_head(hidden).view(-1, self.num_rhs, self.num_splits)
        es_logits = self.es_head(hidden).view(-1, self.num_rhs, self.num_ess)
        rc_logits = self.rc_head(hidden).view(-1, self.num_rhs, self.num_rcs)
        value = self.value_net(hidden)
        return split_logits, es_logits, rc_logits, value

    def get_distributions(
        self,
        features: torch.Tensor,
        action_mask: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Categorical, torch.distributions.Categorical, torch.Tensor]:
        split_logits, es_logits, rc_logits, value = self.forward(features)

        if action_mask is not None:
            split_logits = split_logits.masked_fill(~action_mask["split"].bool(), float("-inf"))
            es_logits = es_logits.masked_fill(~action_mask["es"].bool(), float("-inf"))
            rc_logits = rc_logits.masked_fill(~action_mask["rc"].bool(), float("-inf"))

        split_dist = torch.distributions.Categorical(logits=split_logits)
        es_dist = torch.distributions.Categorical(logits=es_logits)
        rc_dist = torch.distributions.Categorical(logits=rc_logits)
        return split_dist, es_dist, rc_dist, value


class PPOAgent:
    """Masked PPO agent for the paper's 3N action vector."""

    def __init__(
        self,
        feature_dim: int,
        num_rhs: int,
        num_splits: int,
        num_ess: int,
        num_rcs: int,
        lr: float = 1e-4,
        gamma: float = 0.98,
        gae_lambda: float = 0.97,
        clip_ratio: float = 0.3,
        device: str = "cpu",
    ):
        self.num_rhs = num_rhs
        self.num_splits = num_splits
        self.num_ess = num_ess
        self.num_rcs = num_rcs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.device = torch.device(device)
        self.feature_extractor = None

        self.policy = MaskedPPOPolicy(
            feature_dim=feature_dim,
            num_rhs=num_rhs,
            num_splits=num_splits,
            num_ess=num_ess,
            num_rcs=num_rcs,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.lr = lr

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

    def attach_feature_extractor(self, feature_extractor: nn.Module) -> None:
        self.feature_extractor = feature_extractor
        params = list(self.policy.parameters()) + list(feature_extractor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def _mask_to_tensors(self, action_mask: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.bool, device=self.device)
            for key, value in action_mask.items()
        }

    def select_action(
        self,
        features: torch.Tensor,
        action_mask: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)

            mask_tensors = {
                key: tensor.unsqueeze(0)
                for key, tensor in self._mask_to_tensors(action_mask).items()
            }
            split_dist, es_dist, rc_dist, value = self.policy.get_distributions(features, mask_tensors)

            if deterministic:
                split_action = split_dist.probs.argmax(dim=-1)
                es_action = es_dist.probs.argmax(dim=-1)
                rc_action = rc_dist.probs.argmax(dim=-1)
            else:
                split_action = split_dist.sample()
                es_action = es_dist.sample()
                rc_action = rc_dist.sample()

            total_log_prob = (
                split_dist.log_prob(split_action).sum(dim=-1) +
                es_dist.log_prob(es_action).sum(dim=-1) +
                rc_dist.log_prob(rc_action).sum(dim=-1)
            )

            action = torch.cat([split_action, es_action, rc_action], dim=-1).squeeze(0).cpu().numpy()
            return action.astype(int), float(total_log_prob.item()), float(value.item())

    def select_action_sequential(
        self,
        features: torch.Tensor,
        action_mask: Dict[str, np.ndarray],
        rc_mask_fn,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, np.ndarray]]:
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)

            split_logits, es_logits, rc_logits, value = self.policy.forward(features)
            base_masks = self._mask_to_tensors(action_mask)
            split_mask = base_masks["split"].unsqueeze(0)
            es_mask = base_masks["es"].unsqueeze(0)

            split_dist = torch.distributions.Categorical(
                logits=split_logits.masked_fill(~split_mask, float("-inf"))
            )
            es_dist = torch.distributions.Categorical(
                logits=es_logits.masked_fill(~es_mask, float("-inf"))
            )

            if deterministic:
                split_action = split_dist.probs.argmax(dim=-1)
                es_action = es_dist.probs.argmax(dim=-1)
            else:
                split_action = split_dist.sample()
                es_action = es_dist.sample()

            rc_mask_np = rc_mask_fn(
                split_action.squeeze(0).cpu().numpy(),
                es_action.squeeze(0).cpu().numpy(),
            )
            rc_mask = torch.as_tensor(rc_mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)
            rc_dist = torch.distributions.Categorical(
                logits=rc_logits.masked_fill(~rc_mask, float("-inf"))
            )

            if deterministic:
                rc_action = rc_dist.probs.argmax(dim=-1)
            else:
                rc_action = rc_dist.sample()

            total_log_prob = (
                split_dist.log_prob(split_action).sum(dim=-1) +
                es_dist.log_prob(es_action).sum(dim=-1) +
                rc_dist.log_prob(rc_action).sum(dim=-1)
            )

            action = torch.cat([split_action, es_action, rc_action], dim=-1).squeeze(0).cpu().numpy().astype(int)
            used_masks = {
                "split": action_mask["split"].copy(),
                "es": action_mask["es"].copy(),
                "rc": rc_mask_np.copy(),
            }
            return action, float(total_log_prob.item()), float(value.item()), used_masks

    def store_transition(
        self,
        state,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: Dict[str, np.ndarray],
    ) -> None:
        self.states.append(state.cpu() if hasattr(state, "cpu") else state)
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.action_masks.append({key: value.copy() for key, value in action_mask.items()})

    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            next_value = 0.0 if t == len(self.rewards) - 1 or self.dones[t] else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - float(self.dones[t]))
            advantages[t] = gae
            returns[t] = gae + self.values[t]

        return advantages, returns

    def update(self, batch_size: int = 128, epochs: int = 3) -> None:
        if not self.states:
            return

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions = torch.as_tensor(np.stack(self.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        split_masks = torch.as_tensor(
            np.stack([mask["split"] for mask in self.action_masks]),
            dtype=torch.bool,
            device=self.device,
        )
        es_masks = torch.as_tensor(
            np.stack([mask["es"] for mask in self.action_masks]),
            dtype=torch.bool,
            device=self.device,
        )
        rc_masks = torch.as_tensor(
            np.stack([mask["rc"] for mask in self.action_masks]),
            dtype=torch.bool,
            device=self.device,
        )

        for _ in range(epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)

            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                if self.feature_extractor is not None:
                    batch_graphs = [self.states[idx] for idx in batch_indices]
                    batch_graph = Batch.from_data_list(batch_graphs).to(self.device)
                    batch_states = self.feature_extractor(batch_graph)
                else:
                    batch_states = torch.stack([self.states[idx] for idx in batch_indices]).to(self.device)
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_masks = {
                    "split": split_masks[batch_indices],
                    "es": es_masks[batch_indices],
                    "rc": rc_masks[batch_indices],
                }

                split_dist, es_dist, rc_dist, values = self.policy.get_distributions(batch_states, batch_masks)
                values = values.squeeze(-1)

                split_actions = batch_actions[:, :self.num_rhs]
                es_actions = batch_actions[:, self.num_rhs:2 * self.num_rhs]
                rc_actions = batch_actions[:, 2 * self.num_rhs:]

                new_log_probs = (
                    split_dist.log_prob(split_actions).sum(dim=-1) +
                    es_dist.log_prob(es_actions).sum(dim=-1) +
                    rc_dist.log_prob(rc_actions).sum(dim=-1)
                )

                entropy = (
                    split_dist.entropy().sum(dim=-1) +
                    es_dist.entropy().sum(dim=-1) +
                    rc_dist.entropy().sum(dim=-1)
                ).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + 0.5 * value_loss - 1e-6 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

    def save(self, path: str, metadata: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "state_dict": self.policy.state_dict(),
            "gnn_state_dict": self.feature_extractor.state_dict() if self.feature_extractor is not None else None,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        self.policy.load_state_dict(state_dict)
