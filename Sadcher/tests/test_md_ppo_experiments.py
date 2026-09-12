import pytest
import torch

from data_generation.md_instance_generator import MDGeneratorConfig, generate_md_instance
from models.md_rllib_model import MDRLlibModel
from reinforcement_learning.md_gym_environment import MDGymEnvironment
from reinforcement_learning.md_ray_ppo import (
    MDPPOExperimentConfig,
    extract_learner_diagnostics,
    is_better_validation,
    phase_a_experiment_configs,
    reward_tracker_for_experiment,
)


def test_phase_a_configs_match_preregistered_diagnostic_matrix():
    configs = phase_a_experiment_configs()

    assert tuple(configs) == ("A0", "A1", "A2", "A3", "A4")
    assert configs["A0"] == MDPPOExperimentConfig()
    assert configs["A1"].train_batch_size == 2048
    assert configs["A1"].num_epochs == 5
    assert configs["A1"].minibatch_size == 256
    assert configs["A1"].il_kl_weight == pytest.approx(0.005)
    assert configs["A2"].il_kl_weight == 0.0
    assert configs["A3"].starvation_penalty == 0.0
    assert configs["A4"].use_graph_value_head


def test_ppo_experiment_config_rejects_invalid_batch_geometry():
    with pytest.raises(ValueError, match="multiple"):
        MDPPOExperimentConfig(train_batch_size=1000, minibatch_size=256)


def test_reward_tracker_respects_starvation_ablation():
    current = reward_tracker_for_experiment(100.0, MDPPOExperimentConfig())
    ablated = reward_tracker_for_experiment(
        100.0, MDPPOExperimentConfig(starvation_penalty=0.0)
    )

    current_reward = current.step(time=1, completed_tasks=0, starvation={"1": 10})
    ablated_reward = ablated.step(time=1, completed_tasks=0, starvation={"1": 10})

    assert current_reward == pytest.approx(-0.21)
    assert ablated_reward == pytest.approx(-0.01)


def test_graph_value_head_uses_typed_task_relations():
    torch.manual_seed(7)
    checkpoint = "runs/md_offline_il_pilot_2026-08-27/training/best_checkpoint.pt"
    env = MDGymEnvironment(
        lambda: generate_md_instance(MDGeneratorConfig(seed=5101100)).domain
    )
    observation, _ = env.reset()
    batch = {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
    model = MDRLlibModel(
        env.observation_space,
        env.action_space,
        65,
        {"custom_model_config": {
            "il_checkpoint": checkpoint,
            "freeze_encoders": True,
            "use_graph_value_head": True,
        }},
        "graph_value_test",
    ).eval()

    model({"obs": batch}, [], None)
    with_relations = model.value_function().detach().clone()
    without_relations = dict(batch)
    without_relations["typed_adjacency"] = torch.zeros_like(batch["typed_adjacency"])
    model({"obs": without_relations}, [], None)

    assert with_relations.shape == (1,)
    assert torch.isfinite(with_relations).all()
    assert not torch.equal(with_relations, model.value_function().detach())


def test_legacy_learner_diagnostics_are_extracted_with_stable_names():
    result = {"info": {"learner": {"default_policy": {"learner_stats": {
        "kl": 0.012,
        "entropy": 1.5,
        "vf_loss": 0.7,
        "vf_explained_var": 0.25,
        "policy_loss": -0.03,
        "cur_kl_coeff": 0.4,
        "il_reference_kl": 0.006,
    }}}}}

    assert extract_learner_diagnostics(result) == {
        "ppo_kl": 0.012,
        "entropy": 1.5,
        "value_loss": 0.7,
        "value_explained_variance": 0.25,
        "policy_loss": -0.03,
        "ppo_kl_coefficient": 0.4,
        "il_reference_kl": 0.006,
    }


def test_model_metrics_are_extracted_from_legacy_learner_model_stats():
    result = {"info": {"learner": {"default_policy": {
        "learner_stats": {},
        "model": {"il_reference_kl": 0.004},
    }}}}

    assert extract_learner_diagnostics(result)["il_reference_kl"] == pytest.approx(
        0.004
    )


def test_validation_selection_enforces_safety_before_makespan():
    best = {
        "success_rate": 1.0,
        "illegal_assignments": 0,
        "mean_makespan": 100.0,
        "mean_material_starvation": 10.0,
        "p95_latency_seconds": 0.01,
    }
    unsafe_but_fast = dict(best, mean_makespan=90.0, illegal_assignments=1)
    safe_and_faster = dict(best, mean_makespan=99.0)
    starvation_tie_break = dict(best, mean_material_starvation=9.0)

    assert not is_better_validation(unsafe_but_fast, best, il_success_rate=1.0)
    assert is_better_validation(safe_and_faster, best, il_success_rate=1.0)
    assert is_better_validation(starvation_tie_break, best, il_success_rate=1.0)
