import torch

from baselines.exact_online_action_oracle import enumerate_complete_online_actions
from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from data_generation.md_residual_generation import SnapshotCandidate, residual_state
from models.md_online_features import build_md_policy_inputs_from_simulator
from models.md_prefix_autoregressive import MDPrefixAutoregressivePolicy
from reinforcement_learning.md_joint_action import (
    autoregressive_action_masks,
    oracle_aligned_action_is_legal,
    oracle_aligned_action_masks,
)
from simulation_environment.md_discrete_simulator import MDDiscreteSimulator


def _observation(seed: int = 76201):
    domain = generate_md_instance(MDGeneratorConfig(seed=seed)).domain
    simulator = MDDiscreteSimulator(domain)
    robot, task, _adjacency, md = build_md_policy_inputs_from_simulator(simulator)
    hard = md.hard_feasibility_mask
    return domain, {
        "robot_features": robot.float(),
        "task_features": task.float(),
        "robot_metadata": md.robot_metadata.float(),
        "task_metadata": md.task_metadata.float(),
        "pair_metadata": md.pair_metadata.float(),
        "task_is_transport": md.task_is_transport.bool(),
        "typed_adjacency": md.typed_adjacency.float(),
        "opportunity_context": md.opportunity_context.float(),
        "process_covered_skills": torch.zeros(
            1, hard.shape[2], 3, dtype=torch.float32
        ),
        "action_mask": torch.cat((~hard.any(dim=-1, keepdim=True), hard), dim=-1),
    }


def test_oracle_aligned_mask_covers_exact_nonempty_actions():
    domain, observation = _observation()
    snapshot = SnapshotCandidate("prefix-test", 76201, "initial", domain, MDDiscreteSimulator(domain), {})
    actions = enumerate_complete_online_actions(domain, residual_state(snapshot))
    robot_ids = sorted(robot.robot_id for robot in domain.robots)
    task_index = {
        task_id: index + 1
        for index, task_id in enumerate(sorted(task.task_id for task in domain.tasks))
    }

    vectors = []
    for action in actions:
        assignment = dict(action.assignments)
        vectors.append([task_index[assignment[r]] if r in assignment else 0 for r in robot_ids])
    assert vectors
    assert all(
        oracle_aligned_action_is_legal(observation, torch.tensor([vector]))
        for vector in vectors
    )


def test_legacy_mask_still_forces_a_task_when_one_is_available():
    _domain, observation = _observation()
    partial = torch.full(
        (1, observation["robot_features"].shape[1]), -1, dtype=torch.long
    )
    legacy = autoregressive_action_masks(observation, partial)[0, 0]
    oracle = oracle_aligned_action_masks(observation, partial)[0, 0]
    assert bool(legacy[0]) is False
    assert bool(oracle[0]) is True
    legacy_without_coverage = autoregressive_action_masks(
        {key: value for key, value in observation.items() if key != "process_covered_skills"},
        partial,
    )
    assert torch.equal(legacy, legacy_without_coverage[0, 0])


def test_prefix_logits_change_after_different_prefixes_and_zero_ablation_does_not():
    _domain, observation = _observation()
    torch.manual_seed(4)
    prefix = MDPrefixAutoregressivePolicy(dropout=0.0, use_prefix=True).eval()
    zero = MDPrefixAutoregressivePolicy(dropout=0.0, use_prefix=False).eval()
    prefix_encoding = prefix.encode(observation)
    zero_encoding = zero.encode(observation)
    prefix_state = prefix.initial_prefix_state(prefix_encoding)
    zero_state = zero.initial_prefix_state(zero_encoding)
    partial = torch.full((1, observation["robot_features"].shape[1]), -1, dtype=torch.long)
    first_mask = oracle_aligned_action_masks(observation, partial)[0, 0]
    task_choice = int(torch.nonzero(first_mask[1:], as_tuple=False)[0].item()) + 1

    idle_prefix = prefix.advance(prefix_encoding, prefix_state, 0, torch.tensor([0]))
    task_prefix = prefix.advance(prefix_encoding, prefix_state, 0, torch.tensor([task_choice]))
    idle_zero = zero.advance(zero_encoding, zero_state, 0, torch.tensor([0]))
    task_zero = zero.advance(zero_encoding, zero_state, 0, torch.tensor([task_choice]))

    idle_logits = prefix.step(prefix_encoding, idle_prefix, 1)
    task_logits = prefix.step(prefix_encoding, task_prefix, 1)
    assert not torch.allclose(idle_logits, task_logits, atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        zero.step(zero_encoding, idle_zero, 1),
        zero.step(zero_encoding, task_zero, 1),
        atol=1e-7,
        rtol=1e-7,
    )


def test_log_prob_gradients_and_beam_decode_are_finite_and_legal():
    _domain, observation = _observation(76202)
    torch.manual_seed(7)
    model = MDPrefixAutoregressivePolicy(dropout=0.0)
    candidates = model.beam_candidates(observation, beam_width=16)
    assert candidates.actions.shape == (16, observation["robot_features"].shape[1])
    assert candidates.log_probabilities.shape == (16,)
    assert bool(
        (candidates.log_probabilities[:-1] >= candidates.log_probabilities[1:]).all()
    )
    assert all(
        oracle_aligned_action_is_legal(observation, row.unsqueeze(0))
        for row in candidates.actions
    )
    decoded = model.beam_decode(observation, beam_width=16)
    assert decoded.actions.shape == (1, observation["robot_features"].shape[1])
    assert int(decoded.actions.gt(0).sum()) > 0
    assert oracle_aligned_action_is_legal(observation, decoded.actions)

    log_probability = model.log_prob(observation, decoded.actions)
    loss = -log_probability.mean()
    loss.backward()
    assert torch.isfinite(log_probability).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_oracle_mask_rejects_a_redundant_process_coalition():
    _domain, observation = _observation(77016)
    # Robot 2 already covers every required skill of process task 9, so adding
    # robot 1 to the same coalition is legal physically but is not one of the
    # exact oracle's minimal complete actions.
    redundant = torch.tensor([[0, 9, 9, 11, 10]])
    assert not oracle_aligned_action_is_legal(observation, redundant)
