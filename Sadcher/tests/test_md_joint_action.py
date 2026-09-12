import numpy as np
import pytest
import torch

from models.md_rllib_model import (
    MDRLlibModel,
    MDAutoregressiveActionDistribution,
    autoregressive_action_masks,
)
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from models.md_enhanced_policy import MDEnhancedSchedulerNetwork, MDEnhancedPolicyConfig
from models.md_policy import MDPolicyInputs
from reinforcement_learning.md_gym_environment import MDGymEnvironment
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
    TransportRobot,
    TransportTask,
)
from simulation_environment.hard_feasibility import TaskStatus


def _joint_domain() -> SchedulingDomain:
    return SchedulingDomain.create(
        config=MaterialDeliveryConfig(enabled=True),
        tasks=(
            ProcessTask(1, (0, 0), 2, (True, True)),
            TransportTask(2, (0, 0), (0, 0), 1, 1, 1),
            ProcessTask(3, (0, 0), 1, (True, False)),
        ),
        material_edges=((2, 3),),
        robots=(
            ProcessRobot(0, (0, 0), (True, False)),
            ProcessRobot(1, (0, 0), (False, True)),
            TransportRobot(2, (0, 0), True, 2, 1, loaded_speed=1),
            TransportRobot(3, (0, 0), True, 2, 1, loaded_speed=1),
        ),
    )


def _context(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}


class _DistributionModel:
    def __init__(self, context):
        self._last_action_context = context


def _distribution(observation, logits=None):
    context = _context(observation)
    robot_count, action_count = context["action_mask"].shape[1:]
    if logits is None:
        logits = torch.zeros(1, robot_count * action_count, requires_grad=True)
    return MDAutoregressiveActionDistribution(logits, _DistributionModel(context)), logits


IL_CHECKPOINT = "runs/md_offline_il_pilot_2026-08-27/training/best_checkpoint.pt"


def _real_model_and_observation(device="cpu", *, use_graph_value_head=False):
    env = MDGymEnvironment(
        lambda: generate_md_instance(MDGeneratorConfig(seed=5101200)).domain
    )
    observation, _ = env.reset()
    batch = {
        key: torch.as_tensor(value).unsqueeze(0).to(device)
        for key, value in observation.items()
    }
    model = MDRLlibModel(
        env.observation_space,
        env.action_space,
        65,
        {
            "custom_model_config": {
                "il_checkpoint": IL_CHECKPOINT,
                "freeze_encoders": True,
                "kl_weight": 0.02,
                "use_graph_value_head": use_graph_value_head,
            }
        },
        "test_model",
    ).to(device)
    return env, batch, model


def test_transport_task_is_singleton_in_every_sample():
    env = MDGymEnvironment(_joint_domain)
    observation, _ = env.reset()
    distribution, _ = _distribution(observation)

    for _ in range(500):
        action = distribution.sample()[0]
        assert int((action == 2).sum()) <= 1


def test_process_coalition_is_expressible_and_executes_as_sampled():
    env = MDGymEnvironment(_joint_domain)
    observation, _ = env.reset()
    logits = torch.full((1, 16), -20.0)
    logits[0, 1] = 20.0
    logits[0, 4] = 20.0
    distribution, _ = _distribution(observation, logits)

    action = distribution.deterministic_sample()[0]
    assert action.tolist()[:2] == [1, 1]
    _next, _reward, _terminated, _truncated, info = env.step(action.numpy())

    assert info["executed_action"] == action.tolist()
    assert info["illegal_assignments"] == 0
    assert env.sim.task_state(1).status is TaskStatus.IN_PROGRESS
    assert env.sim.task_state(1).assigned_robot_ids == {0, 1}


def test_random_joint_actions_are_executed_without_rejection():
    env = MDGymEnvironment(_joint_domain)
    for _ in range(100):
        observation, _ = env.reset()
        distribution, _ = _distribution(observation, torch.randn(1, 16))
        action = distribution.sample()[0]
        _next, _reward, _terminated, _truncated, info = env.step(action.numpy())
        assert info["illegal_assignments"] == 0
        assert info["executed_action"] == action.tolist()


def test_prefix_masks_allow_complete_coalition_and_forbid_duplicate_transport():
    env = MDGymEnvironment(_joint_domain)
    observation, _ = env.reset()
    context = _context(observation)
    actions = torch.tensor([[1, 1, 2, 2]])

    masks = autoregressive_action_masks(context, actions)

    assert masks.shape == (1, 4, 4)
    assert masks[0, 0, 1]
    assert masks[0, 1, 1]
    assert masks[0, 2, 2]
    assert not masks[0, 3, 2]


def test_distribution_sample_logp_shapes_and_gradients_are_finite():
    env = MDGymEnvironment(_joint_domain)
    observation, _ = env.reset()
    batch = {key: torch.as_tensor(value).repeat(7, *([1] * value.ndim)) for key, value in observation.items()}
    logits = torch.randn(7, 16, requires_grad=True)
    distribution = MDAutoregressiveActionDistribution(logits, _DistributionModel(batch))

    actions = distribution.sample()
    logp = distribution.logp(actions)
    entropy = distribution.entropy()
    loss = -(logp + 0.01 * entropy).mean()
    loss.backward()

    assert actions.shape == (7, 4)
    assert logp.shape == (7,)
    assert entropy.shape == (7,)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.allclose(logp, distribution.sampled_action_logp())


def test_deterministic_stochastic_and_state_restore_are_stable():
    _env, batch, model = _real_model_and_observation()
    output, _ = model({"obs": batch}, [], None)
    deterministic = MDAutoregressiveActionDistribution(output, model).deterministic_sample()
    stochastic = MDAutoregressiveActionDistribution(output, model).sample()

    restored = _real_model_and_observation()[2]
    restored.load_state_dict(model.state_dict())
    restored_output, _ = restored({"obs": batch}, [], None)
    restored_action = MDAutoregressiveActionDistribution(
        restored_output, restored
    ).deterministic_sample()

    assert deterministic.shape == stochastic.shape == (1, 5)
    assert torch.equal(deterministic, restored_action)
    assert torch.allclose(output, restored_output)


def test_loaded_policy_logits_equal_original_il_network():
    _env, batch, model = _real_model_and_observation()
    payload = torch.load(IL_CHECKPOINT, map_location="cpu", weights_only=True)
    config_values = dict(payload["md_policy_config"])
    config_values["use_signed_opportunity_prior"] = False
    original = MDEnhancedSchedulerNetwork(
        **payload["model_spec"], md_config=MDEnhancedPolicyConfig(**config_values)
    )
    incompatible = original.load_state_dict(payload["state_dict"], strict=False)
    assert set(incompatible.missing_keys) == {
        "opportunity_raw_magnitudes",
        "opportunity_projection.weight",
    }
    assert not incompatible.unexpected_keys
    md_inputs = MDPolicyInputs(
        robot_metadata=batch["robot_metadata"].float(),
        task_metadata=batch["task_metadata"].float(),
        pair_metadata=batch["pair_metadata"].float(),
        task_is_transport=batch["task_is_transport"].bool(),
        typed_adjacency=batch["typed_adjacency"].float(),
        downstream_task_index=batch["downstream_task_index"].long(),
        hard_feasibility_mask=batch["action_mask"][..., 1:].bool(),
        opportunity_context=batch["opportunity_context"].float(),
    )
    with torch.no_grad():
        expected = original(
            batch["robot_features"].float(),
            batch["task_features"].float(),
            batch["task_adjacency"].float(),
            md_inputs=md_inputs,
        )
        actual = model.net(
            batch["robot_features"].float(),
            batch["task_features"].float(),
            batch["task_adjacency"].float(),
            md_inputs=md_inputs,
        )
    assert torch.equal(expected, actual)


def test_frozen_encoders_do_not_change_after_optimizer_step():
    _env, batch, model = _real_model_and_observation()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    output, _ = model({"obs": batch}, [], None)
    loss = output.square().mean() + model.value_function().square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert frozen
    for name, before in frozen.items():
        assert torch.equal(before, dict(model.named_parameters())[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_forward_backward_has_no_nan():
    _env, batch, model = _real_model_and_observation("cuda:0")
    output, _ = model({"obs": batch}, [], None)
    distribution = MDAutoregressiveActionDistribution(output, model)
    actions = distribution.sample()
    loss = -distribution.logp(actions).mean() + model.value_function().square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_value_head_gpu_forward_backward_has_no_nan():
    _env, batch, model = _real_model_and_observation(
        "cuda:0", use_graph_value_head=True
    )
    output, _ = model({"obs": batch}, [], None)
    distribution = MDAutoregressiveActionDistribution(output, model)
    actions = distribution.sample()
    loss = -distribution.logp(actions).mean() + model.value_function().square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert torch.isfinite(output).all()
    assert torch.isfinite(model.value_function()).all()
    assert torch.isfinite(loss)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
