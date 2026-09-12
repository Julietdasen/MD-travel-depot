# MD-SADCHER++ Domain Model

Status: design agreed; pending validation against the official dataset.

## Scope

The first milestone is an environment-level material-delivery rollout: data model, simulator state machine, centralized hard feasibility, baseline metrics, deterministic scenarios, and Greedy baselines. It does not change the neural network, GAT, checkpoints, or decoder.

## Entities

Real tasks have two mutually exclusive types:

- `PROCESS`: uses the original SADCHER skill requirements and dynamic skill coalitions.
- `TRANSPORT`: moves material from pickup to the downstream process location.

Real robots have two mutually exclusive types:

- `PROCESS_ROBOT`: process capabilities and coalition participation; never executes transport.
- `TRANSPORT_ROBOT`: transport capability, capacity, unloaded speed, and loaded speed; never joins process skill coalitions.

The simulator may retain `START`, `IDLE`, and `EXIT` sentinels from original SADCHER. They are not a third real task type and are excluded from domain-task metrics and the material graph.

## Relations

Each TransportTask unlocks exactly one ProcessTask. MVP is strict one-to-one: a ProcessTask has at most one material predecessor. Transport tasks have no normal predecessors; material is available at `t=0`. Process tasks may have zero or one material predecessor and multiple normal process predecessors.

The serialized source of truth is:

```yaml
material_edges:
  - [transport_task_id, process_task_id]
```

Runtime objects derive both endpoint fields and validate consistency. A transport delivery location must equal its downstream process location.

## Transport state machine

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

Task start is the beginning of loading; completion is the end of unloading. A TransportRobot is busy and non-preemptive from assignment through completion. Execution records preserve empty travel, loading, loaded transport, unloading, service time, and occupied time.

The discrete simulator uses `ceil(distance / speed)` for movement stages. The theoretical duration is empty travel + loading + loaded travel + unloading.

## Readiness and feasibility

Process readiness requires `PENDING`, all normal predecessors done, and its material predecessor done when present. Transport readiness requires `PENDING` and no normal predecessor.

All schedulers use a centralized `is_assignment_feasible(robot, task)` returning a typed `FeasibilityResult`. Assignment feasibility is separate from process coalition start feasibility: a ProcessRobot may contribute one uncovered skill; the coalition starts only after all requirements are covered.

MVP forbids transport coalitions, capacity aggregation, synchronization waiting, and preemption. A load that no single TransportRobot can carry makes the instance infeasible; no hidden fallback is allowed.

## Metrics and data integration

Metrics include makespan, material starvation, robot utilization, success, and stable failure reason codes. Process and transport execution records are structured event records.

Legacy `Q/R/T_e/T_t/task_locations` instances remain process-only with material delivery disabled. The official dataset may require direct changes to loading and related data structures, but all inputs must converge to this runtime model and preserve the state-machine semantics. Legacy `T_t` remains available for legacy solvers; MD runtime computes coordinate-based travel time until the dataset integration decision is made.

Future cooperative transport uses fixed `SOLO`, `SMALL_COALITION`, and `LARGE_COALITION` modes, first with rule-based selection and TransportRobot-only coalitions.
