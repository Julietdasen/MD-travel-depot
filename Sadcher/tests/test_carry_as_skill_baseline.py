import unittest

from experiments.protocol import DatasetSplit
from schedulers.carry_as_skill import (
    project_carry_as_skill_domain,
    run_carry_as_skill_baseline,
)
from simulation_environment.domain_model import ProcessRobot, ProcessTask
from tests.fixtures.md_diagnostic_scenarios import UNLOCK


class CarryAsSkillBaselineTests(unittest.TestCase):
    def test_projection_explicitly_removes_material_unlock_semantics(self):
        projection = project_carry_as_skill_domain(UNLOCK.domain)

        self.assertFalse(projection.domain.config.enabled)
        self.assertEqual(projection.domain.material_edges, ())
        self.assertEqual(projection.ignored_material_edges, ((2, 1),))
        self.assertTrue(all(isinstance(task, ProcessTask) for task in projection.domain.tasks))
        self.assertTrue(all(isinstance(robot, ProcessRobot) for robot in projection.domain.robots))
        self.assertEqual(projection.carry_skill_index, 1)

    def test_diagnostic_run_exposes_downstream_start_before_carry_completion(self):
        result = run_carry_as_skill_baseline(
            UNLOCK.domain,
            exit_location=UNLOCK.exit_location,
            run_id="carry-as-skill-unlock",
            instance_id=UNLOCK.name,
            seed=UNLOCK.seed,
            split=DatasetSplit.TEST,
            max_steps=30,
        )

        self.assertTrue(result.success)
        records = {record["task_id"]: record for record in result.process_execution_records}
        self.assertLess(records[1]["started_at"], records[2]["completed_at"])
        self.assertFalse(result.metadata["uses_material_ready_semantics"])
        self.assertEqual(result.metadata["ignored_material_edges"], ((2, 1),))


if __name__ == "__main__":
    unittest.main()
