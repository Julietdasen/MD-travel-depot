import unittest

from experiments.md_context_ablation_relational_data import (
    RELATIONAL_FAMILIES,
    build_unconditioned_relational_candidates,
)
from experiments.md_terminal_tail_training import (
    build_terminal_tail_training_pairs,
    relabel_terminal_state,
    terminal_action_values,
)


class MDTerminalTailTrainingTests(unittest.TestCase):
    def test_zero_terminal_weight_preserves_original_oracle(self):
        state = build_unconditioned_relational_candidates(
            candidates_per_family=2, seed=7303
        )[0].before
        values = terminal_action_values(state, terminal_weight=0.0)
        relabeled = relabel_terminal_state(state, terminal_weight=0.0)
        self.assertEqual(values, state.action_values)
        self.assertEqual(relabeled.oracle_action, state.oracle_action)

    def test_terminal_tail_selection_is_deterministic_and_positive_regret(self):
        kwargs = {
            "candidate_pool_per_family": 120,
            "regular_pairs_per_family": 8,
            "terminal_pairs_per_family": 2,
            "regular_seed": 3030,
            "terminal_seed": 7303,
            "terminal_weight": 4.0,
        }
        first_pairs, first_manifest = build_terminal_tail_training_pairs(**kwargs)
        second_pairs, second_manifest = build_terminal_tail_training_pairs(**kwargs)
        self.assertEqual(first_pairs, second_pairs)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(len(first_pairs), 10 * len(RELATIONAL_FAMILIES))
        self.assertEqual(first_manifest["terminal_pair_fraction"], 0.2)
        for family in RELATIONAL_FAMILIES:
            row = first_manifest["family_counts"][family]
            self.assertEqual(row["regular_pairs"], 8)
            self.assertEqual(row["terminal_pairs"], 2)
            self.assertGreater(row["minimum_selected_terminal_regret"], 0.0)
        terminal_pairs = [
            pair for pair in first_pairs if pair.pair_id.startswith("terminal-tail-")
        ]
        self.assertEqual(len(terminal_pairs), 2 * len(RELATIONAL_FAMILIES))

    def test_invalid_terminal_weight_is_rejected(self):
        state = build_unconditioned_relational_candidates(
            candidates_per_family=2, seed=7303
        )[0].before
        with self.assertRaisesRegex(ValueError, "terminal_weight"):
            terminal_action_values(state, terminal_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
