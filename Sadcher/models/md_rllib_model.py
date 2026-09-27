"""RLlib ModelV2 wrapper around the fixed-size MD enhanced IL policy."""
from __future__ import annotations
import copy
from dataclasses import fields
import torch
import torch.nn as nn
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from models.md_enhanced_policy import MDEnhancedPolicyConfig, MDEnhancedSchedulerNetwork
from models.md_policy import MDPolicyInputs, _incoming_mean
from models.md_training_enhancements import MDStateValueHead
from reinforcement_learning.md_joint_action import autoregressive_action_masks


class MDAutoregressiveActionDistribution(TorchDistributionWrapper):
    """Fixed robot-order categorical distribution over legal MD joint actions."""

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.context = model._last_action_context
        self.robot_count = int(self.context["action_mask"].shape[1])
        self.action_count = int(self.context["action_mask"].shape[2])
        self.logits = self.inputs.reshape(-1, self.robot_count, self.action_count)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return int(sum(action_space.nvec))

    def _masked_logits(self, actions):
        masks = autoregressive_action_masks(self.context, actions)
        return self.logits.masked_fill(~masks, -1.0e9)

    def _draw(self, deterministic):
        batch_size = self.logits.shape[0]
        actions = torch.full(
            (batch_size, self.robot_count),
            -1,
            dtype=torch.long,
            device=self.logits.device,
        )
        for robot_index in range(self.robot_count):
            logits = self._masked_logits(actions)[:, robot_index]
            categorical = torch.distributions.Categorical(logits=logits)
            choice = torch.argmax(logits, dim=-1) if deterministic else categorical.sample()
            actions[:, robot_index] = choice
        self.last_sample = actions
        return actions

    def sample(self):
        return self._draw(False)

    def deterministic_sample(self):
        return self._draw(True)

    def logp(self, actions):
        actions = actions.long().reshape(-1, self.robot_count)
        logits = self._masked_logits(actions)
        return torch.stack(
            [
                torch.distributions.Categorical(logits=logits[:, index]).log_prob(actions[:, index])
                for index in range(self.robot_count)
            ],
            dim=1,
        ).sum(dim=1)

    def entropy(self):
        actions = self.last_sample
        if actions is None:
            actions = self.deterministic_sample()
        logits = self._masked_logits(actions)
        return torch.stack(
            [torch.distributions.Categorical(logits=logits[:, index]).entropy() for index in range(self.robot_count)],
            dim=1,
        ).sum(dim=1)

    def kl(self, other: ActionDistribution):
        if not isinstance(other, MDAutoregressiveActionDistribution):
            raise TypeError("KL requires another MD autoregressive distribution")
        actions = self.last_sample
        if actions is None:
            actions = self.deterministic_sample()
        left = self._masked_logits(actions)
        right = other._masked_logits(actions)
        return torch.stack(
            [
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Categorical(logits=left[:, index]),
                    torch.distributions.Categorical(logits=right[:, index]),
                )
                for index in range(self.robot_count)
            ],
            dim=1,
        ).sum(dim=1)



class MDIndependentActionDistribution(TorchDistributionWrapper):
    """Non-autoregressive: each robot samples independently from the base action mask.

    Contract identical to MDAutoregressiveActionDistribution:
      - inputs: (B, R * A) reshaped to (B, R, A) logits
      - action_mask: (B, R, A) from observation; robot-task feasibility
      - sample()/deterministic_sample() returns (B, R) int64
      - logp/entropy/kl return (B,)

    Difference: the mask is fixed (base_mask), NOT conditioned on prior robots.
    Conflict resolution (two robots picking same task) is delegated to the
    deploy-side LearnedConstrainedDecoder via bipartite matching. During RL
    training the environment already reports illegal_assignments; the policy
    learns via reward shaping to avoid conflicts.

    This entirely bypasses the CPU-side per-batch masking loop in
    autoregressive_action_masks(), which was the real throughput bottleneck.
    """

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.context = model._last_action_context
        self.robot_count = int(self.context["action_mask"].shape[1])
        self.action_count = int(self.context["action_mask"].shape[2])
        self.logits = self.inputs.reshape(-1, self.robot_count, self.action_count)
        # Static per-robot mask lives on GPU; broadcast across sample steps.
        base_mask = self.context["action_mask"].bool()
        self._masked = self.logits.masked_fill(~base_mask, -1.0e9)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return int(sum(action_space.nvec))

    def _draw(self, deterministic):
        # One GPU-parallel batched sample across all robots.
        if deterministic:
            actions = torch.argmax(self._masked, dim=-1)
        else:
            flat_logits = self._masked.reshape(-1, self.action_count)
            actions = torch.distributions.Categorical(logits=flat_logits).sample()
            actions = actions.reshape(-1, self.robot_count)
        self.last_sample = actions
        return actions

    def sample(self):
        return self._draw(False)

    def deterministic_sample(self):
        return self._draw(True)

    def logp(self, actions):
        actions = actions.long().reshape(-1, self.robot_count)
        flat_logits = self._masked.reshape(-1, self.action_count)
        flat_actions = actions.reshape(-1)
        return (
            torch.distributions.Categorical(logits=flat_logits)
            .log_prob(flat_actions)
            .reshape(-1, self.robot_count)
            .sum(dim=1)
        )

    def entropy(self):
        flat_logits = self._masked.reshape(-1, self.action_count)
        return (
            torch.distributions.Categorical(logits=flat_logits)
            .entropy()
            .reshape(-1, self.robot_count)
            .sum(dim=1)
        )

    def kl(self, other: ActionDistribution):
        if not isinstance(other, MDIndependentActionDistribution):
            raise TypeError("KL requires another MD independent distribution")
        left = self._masked.reshape(-1, self.action_count)
        right = other._masked.reshape(-1, self.action_count)
        return (
            torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(logits=left),
                torch.distributions.Categorical(logits=right),
            )
            .reshape(-1, self.robot_count)
            .sum(dim=1)
        )


class MDRLlibModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = dict(model_config.get("custom_model_config", {}))
        cfg.update(kwargs)
        checkpoint = cfg.get("il_checkpoint")
        if not checkpoint:
            raise ValueError("MDRLlibModel requires an IL checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        spec = payload["model_spec"]
        allowed = {field.name for field in fields(MDEnhancedPolicyConfig)}
        state_dict = payload["state_dict"]
        md_values = {key: value for key, value in payload["md_policy_config"].items() if key in allowed}
        md_values["use_signed_opportunity_prior"] = "opportunity_raw_magnitudes" in state_dict
        md_config = MDEnhancedPolicyConfig(**md_values)
        self.net = MDEnhancedSchedulerNetwork(**spec, md_config=md_config)
        incompatible = self.net.load_state_dict(state_dict, strict=False)
        allowed_missing = {"opportunity_raw_magnitudes", "opportunity_projection.weight"}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise ValueError(f"incompatible IL checkpoint: missing={sorted(unexpected_missing)}, unexpected={incompatible.unexpected_keys}")
        self.reference_net = copy.deepcopy(self.net).eval()
        for parameter in self.reference_net.parameters():
            parameter.requires_grad_(False)
        if cfg.get("freeze_encoders", True):
            for module in self._encoder_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.kl_weight = float(cfg.get("kl_weight", 0.0))
        self.use_graph_value_head = bool(cfg.get("use_graph_value_head", False))
        if self.use_graph_value_head:
            hidden_dim = 64
            self.value_robot_projection = nn.Sequential(
                nn.Linear(spec["embed_dim"] + 4, hidden_dim), nn.Tanh()
            )
            self.value_task_projection = nn.Sequential(
                nn.Linear(3 * spec["embed_dim"] + 8, hidden_dim), nn.Tanh()
            )
            self.value_head = MDStateValueHead(hidden_dim=hidden_dim)
        else:
            self.value_head = nn.Sequential(
                nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1)
            )
        self._last_value = None
        self._last_kl = None
        self._last_action_context = None
        self._last_policy_logits = None
        self._last_reference_logits = None

    def _encoder_modules(self):
        return (self.net.robot_embedding, self.net.task_embedding, self.net.robot_GATN, self.net.task_GATN, self.net.robot_transformer_encoder, self.net.task_transformer_encoder)

    @staticmethod
    def _md_inputs(obs):
        return MDPolicyInputs(
            robot_metadata=obs["robot_metadata"].float(), task_metadata=obs["task_metadata"].float(),
            pair_metadata=obs["pair_metadata"].float(), task_is_transport=obs["task_is_transport"].bool(),
            typed_adjacency=obs["typed_adjacency"].float(), downstream_task_index=obs["downstream_task_index"].long(),
            hard_feasibility_mask=obs["action_mask"][..., 1:].bool(), opportunity_context=obs["opportunity_context"].float(),
        )

    @staticmethod
    def _action_logits(scores):
        idle = torch.zeros((*scores.shape[:2], 1), device=scores.device, dtype=scores.dtype)
        return torch.cat((idle, scores), dim=-1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        robot, task, adjacency = obs["robot_features"].float(), obs["task_features"].float(), obs["task_adjacency"].float()
        md_inputs = self._md_inputs(obs)
        logits = self._action_logits(self.net(robot, task, adjacency, md_inputs=md_inputs))
        with torch.no_grad():
            reference = self._action_logits(self.reference_net(robot, task, adjacency, md_inputs=md_inputs))
        self._last_action_context = obs
        self._last_policy_logits = logits
        self._last_reference_logits = reference
        if self.use_graph_value_head:
            encoding = self.net.encode(robot, task, adjacency)
            task_tokens = encoding.task_tokens
            typed_adjacency = obs["typed_adjacency"].float()
            robot_context = self.value_robot_projection(
                torch.cat((encoding.robot_tokens, obs["robot_metadata"].float()), dim=-1)
            )
            task_context = self.value_task_projection(
                torch.cat(
                    (
                        task_tokens,
                        _incoming_mean(typed_adjacency[:, 0], task_tokens),
                        _incoming_mean(typed_adjacency[:, 1], task_tokens),
                        obs["task_metadata"].float(),
                    ),
                    dim=-1,
                )
            )
            self._last_value = self.value_head(
                robot_context, task_context, obs["task_is_transport"].bool()
            )
        else:
            pooled = torch.cat((robot.mean(1), task.mean(1)), dim=-1)
            self._last_value = self.value_head(pooled).squeeze(-1)
        return logits.reshape(logits.shape[0], -1), state

    def value_function(self):
        if self._last_value is None: raise RuntimeError("forward must run before value_function")
        return self._last_value

    def custom_loss(self, policy_loss, loss_inputs):
        if self._last_policy_logits is None: raise RuntimeError("forward must run before custom_loss")
        actions = loss_inputs[SampleBatch.ACTIONS].long().reshape(
            -1, self._last_policy_logits.shape[1]
        )
        masks = autoregressive_action_masks(self._last_action_context, actions)
        policy = self._last_policy_logits.masked_fill(~masks, -1.0e9)
        reference = self._last_reference_logits.masked_fill(~masks, -1.0e9)
        self._last_kl = torch.stack(
            [
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Categorical(logits=reference[:, index]),
                    torch.distributions.Categorical(logits=policy[:, index]),
                )
                for index in range(policy.shape[1])
            ],
            dim=1,
        ).sum(dim=1).mean()
        if isinstance(policy_loss, list):
            return [loss + self.kl_weight * self._last_kl for loss in policy_loss]
        return policy_loss + self.kl_weight * self._last_kl

    def metrics(self):
        return {"il_reference_kl": float(self._last_kl.detach().cpu()) if self._last_kl is not None else 0.0}


__all__ = [
    "MDAutoregressiveActionDistribution",
    "MDRLlibModel",
    "autoregressive_action_masks",
]
