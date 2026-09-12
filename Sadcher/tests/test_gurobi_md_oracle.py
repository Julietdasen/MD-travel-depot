import unittest
from types import SimpleNamespace
from unittest.mock import patch

import baselines.gurobi_md_oracle as oracle
from baselines.gurobi_md_oracle import (
    GurobiOracleStatus,
    OracleAction,
    build_gurobi_md_model,
    replay_oracle_actions,
    solve_gurobi_md_oracle,
)
from data_generation.md_instance_generator import (
    MDGeneratorConfig,
    generate_md_instance,
)
from experiments.protocol import DatasetSplit, FailureReason
from simulation_environment.domain_model import (
    MaterialDeliveryConfig,
    ProcessRobot,
    ProcessTask,
    SchedulingDomain,
)
from tests.fixtures.md_diagnostic_scenarios import LATE_MATERIAL


class _Symbol:
    def __init__(self, expression):
        self.expression = expression

    @staticmethod
    def _render(value):
        if isinstance(value, _Symbol):
            return value.expression
        return repr(value)

    def _combine(self, operator, other, *, reverse=False):
        left = self._render(other) if reverse else self.expression
        right = self.expression if reverse else self._render(other)
        return _Symbol(f"({left} {operator} {right})")


    def __add__(self, other):
        return self._combine("+", other)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine("-", other)

    def __rsub__(self, other):
        return self._combine("-", other, reverse=True)

    def __mul__(self, other):
        return self._combine("*", other)

    __rmul__ = __mul__

    def __eq__(self, other):
        return self._combine("==", other)

    def __ge__(self, other):
        return self._combine(">=", other)


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.Params = SimpleNamespace()
        self.constraints = []
        self.variables = []
        self.objective = None

    def addVar(self, **kwargs):
        self.variables.append(dict(kwargs))
        return _Symbol(kwargs["name"])

    def addConstr(self, constraint, name=None):
        self.constraints.append((name, constraint.expression))

    def setObjective(self, objective, sense):
        self.objective = (objective, sense)


def _fake_gurobi():
    return SimpleNamespace(
        Model=_FakeModel,
        GRB=SimpleNamespace(BINARY="binary", INTEGER="integer", MINIMIZE="minimize"),
        quicksum=lambda values: sum(values, _Symbol("0")),
    )


class GurobiMDOracleTests(unittest.TestCase):
    def test_license_free_model_spec_covers_md_constraints_and_exit_objective(self):
        model = build_gurobi_md_model(
            LATE_MATERIAL.domain,
            exit_location=LATE_MATERIAL.exit_location,
        )

        self.assertEqual(model.normal_precedence, ((2, 1),))
        self.assertEqual(model.material_precedence, ((3, 1),))
        self.assertEqual(model.objective, "minimize_latest_robot_exit_return")
        self.assertEqual(model.transport_eligible_robots[3], (1,))
        self.assertEqual(model.process_skill_coverage[1][0], (0,))
        self.assertGreater(model.big_m, 0)
        process_task = next(task for task in LATE_MATERIAL.domain.tasks if task.task_id == 1)
        process_robot = next(robot for robot in LATE_MATERIAL.domain.robots if robot.robot_id == 0)
        self.assertEqual(oracle._assignment_travel(process_robot.location, process_task, 1.0), 0)

    def test_license_free_symbolic_backend_materializes_constraint_families(self):
        formulation = build_gurobi_md_model(
            LATE_MATERIAL.domain,
            exit_location=LATE_MATERIAL.exit_location,
        )

        model, _, _, _ = oracle._materialize(formulation, _fake_gurobi())
        constraints = {
            name: expression
            for name, expression in model.constraints
            if name
        }
        names = set(constraints)
        unnamed = tuple(
            expression
            for name, expression in model.constraints
            if name is None
        )

        self.assertIn("transport_one_robot_3", names)
        self.assertIn("skill_1_0", names)
        self.assertTrue(any(name.startswith("skill_owner_one_") for name in names))
        self.assertTrue(any(name.startswith("minimal_coalition_") for name in names))
        self.assertTrue(any(name.startswith("unique_skill_") for name in names))
        integer_time_variables = [
            variable
            for variable in model.variables
            if str(variable["name"]).startswith((
                "start_", "completion_", "all_tasks_complete",
                "longest_exit_return", "latest_exit_return",
            ))
        ]
        self.assertTrue(integer_time_variables)
        self.assertTrue(all(variable["vtype"] == "integer" for variable in integer_time_variables))
        self.assertIn("duration_3", names)
        self.assertIn("precedence_2_1", names)
        self.assertIn("precedence_3_1", names)
        self.assertTrue(any(name.startswith("initial_travel_") for name in names))
        self.assertTrue(any(name.startswith("last_return_") for name in names))
        self.assertEqual(unnamed, ())
        self.assertEqual(
            constraints["precedence_2_1"], "(start_1 >= completion_2)"
        )
        self.assertEqual(
            constraints["precedence_3_1"], "(start_1 >= completion_3)"
        )
        duration = constraints["duration_3"]
        for token in ("completion_3", "start_3", "a_1_3", "=="):
            self.assertIn(token, duration)
        last_return = constraints["last_return_1_3"]
        for token in ("longest_exit_return", "last_1_3", "4"):
            self.assertIn(token, last_return)
        terminal = constraints["canonical_terminal_makespan"]
        for token in (
            "latest_exit_return",
            "all_tasks_complete",
            "longest_exit_return",
        ):
            self.assertIn(token, terminal)
        for name in ("sequence_0_1_2", "sequence_0_2_1"):
            for token in (
                "before_0_1_2",
                "a_0_1",
                "a_0_2",
                str(formulation.big_m),
            ):
                self.assertIn(token, constraints[name])
        self.assertIsNotNone(model.objective)

    def test_empty_skill_process_task_keeps_single_robot_constraint(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(ProcessTask(1, (0, 0), 1, (False,)),),
            robots=(ProcessRobot(0, (0, 0), (False,), speed=1),),
        )
        formulation = build_gurobi_md_model(domain)

        model, _, _, _ = oracle._materialize(formulation, _fake_gurobi())
        names = {name for name, _ in model.constraints if name is not None}

        self.assertIn("empty_skill_coalition_1", names)
        self.assertFalse(any(name.startswith("minimal_coalition_") for name in names))

    def test_replay_orders_redundant_process_members_before_minimal_cover(self):
        domain = generate_md_instance(
            MDGeneratorConfig(
                seed=8,
                task_count=6,
                transport_ratio=1 / 3,
                precedence_density=0.5,
                critical_path_length=2,
                process_robot_count=2,
                transport_robot_count=1,
                skill_count=2,
            )
        ).domain
        formulation = build_gurobi_md_model(domain)
        schedule = (
            oracle.OracleScheduleEntry(0, 2, 235.0, 253.0),
            oracle.OracleScheduleEntry(1, 2, 235.0, 253.0),
        )

        actions = oracle._build_replay_actions(formulation, schedule)

        self.assertEqual(
            tuple(action.robot_id for action in actions),
            (1, 0),
        )

    def test_replay_rejects_an_unreplayable_redundant_member(self):
        domain = SchedulingDomain.create(
            config=MaterialDeliveryConfig(enabled=True),
            tasks=(ProcessTask(1, (0, 0), 2, (True,)),),
            robots=(
                ProcessRobot(0, (0, 0), (True,), speed=1),
                ProcessRobot(1, (0, 0), (True,), speed=1),
            ),
        )
        result = replay_oracle_actions(
            domain,
            (
                OracleAction(0, 0, 1, 0.0, 0.0),
                OracleAction(1, 1, 1, 0.0, 0.0),
            ),
            run_id="redundant",
            instance_id="redundant",
            seed=1,
            split=DatasetSplit.TEST,
            max_steps=10,
        )

        self.assertFalse(result.success)
        self.assertIs(result.failure_reason, FailureReason.SCHEDULER_FAILURE)
        metadata = result.to_dict()["metadata"]
        self.assertEqual(metadata["pending_action_count"], 1)
        self.assertEqual(metadata["replay_error"]["reason_code"], "task_not_pending")

    def test_missing_gurobi_reports_an_auditable_unavailable_result(self):
        with patch(
            "baselines.gurobi_md_oracle.importlib.import_module",
            side_effect=ImportError,
        ):
            result = solve_gurobi_md_oracle(
                LATE_MATERIAL.domain,
                exit_location=LATE_MATERIAL.exit_location,
                time_limit_seconds=1,
            )

        self.assertIs(result.status, GurobiOracleStatus.UNAVAILABLE)
        self.assertIsNone(result.feasible)
        self.assertIsNone(result.objective)
        self.assertEqual(result.action_order, ())
        self.assertIn("gurobipy", result.message)

    def test_timeout_without_incumbent_keeps_feasibility_unknown(self):
        fake_model = SimpleNamespace(
            Params=SimpleNamespace(),
            Status=2,
            SolCount=0,
            optimize=lambda: None,
        )
        fake_gp = SimpleNamespace(
            GRB=SimpleNamespace(INFEASIBLE=1, TIME_LIMIT=2, OPTIMAL=3),
            GurobiError=RuntimeError,
        )
        with (
            patch(
                "baselines.gurobi_md_oracle._materialize",
                return_value=(fake_model, {}, {}, {}),
            ),
            patch(
                "baselines.gurobi_md_oracle.importlib.import_module",
                return_value=fake_gp,
            ),
        ):
            result = solve_gurobi_md_oracle(
                LATE_MATERIAL.domain,
                exit_location=LATE_MATERIAL.exit_location,
                time_limit_seconds=1,
            )

        self.assertIs(result.status, GurobiOracleStatus.TIMEOUT)
        self.assertTrue(result.timeout)
        self.assertIsNone(result.feasible)

    def test_non_gurobi_model_errors_are_not_hidden_as_unavailable(self):
        fake_gp = SimpleNamespace(
            GRB=SimpleNamespace(INFEASIBLE=1, TIME_LIMIT=2, OPTIMAL=3),
            GurobiError=ValueError,
        )
        with (
            patch(
                "baselines.gurobi_md_oracle._materialize",
                side_effect=RuntimeError("broken formulation"),
            ),
            patch(
                "baselines.gurobi_md_oracle.importlib.import_module",
                return_value=fake_gp,
            ),
        ):
            result = solve_gurobi_md_oracle(LATE_MATERIAL.domain)

        self.assertIs(result.status, GurobiOracleStatus.ERROR)
        self.assertIn("broken formulation", result.message)

    def test_timed_actions_replay_through_the_canonical_simulator(self):
        actions = (
            OracleAction(0, 0, 2, 0.0, 0.0),
            OracleAction(1, 1, 3, 0.0, 0.0),
            OracleAction(2, 0, 1, 7.0, 7.0),
        )

        result = replay_oracle_actions(
            LATE_MATERIAL.domain,
            actions,
            exit_location=LATE_MATERIAL.exit_location,
            run_id="oracle-replay",
            instance_id=LATE_MATERIAL.name,
            seed=LATE_MATERIAL.seed,
            split=DatasetSplit.TEST,
            max_steps=30,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.method, "gurobi_md_oracle_replay")
        self.assertEqual(len(result.transport_execution_records), 1)
        self.assertEqual(len(result.process_execution_records), 2)



if __name__ == "__main__":
    unittest.main()
