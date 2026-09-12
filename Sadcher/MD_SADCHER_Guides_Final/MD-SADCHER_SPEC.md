# MD-SADCHER++ MVP Specification

Status: local project specification, synthesized from the grill decisions and the accepted ADRs. This specification is maintained in the repository and is not published to an issue tracker.

## Problem Statement

SADCHER currently models heterogeneous process scheduling, but it does not express material delivery as a first-class dependency. A process may therefore appear executable even when its input material has not reached its location. Transport capability is also different from process skill capability, yet the legacy representation does not distinguish transport robots from process robots.

The project needs a minimal material-delivery (MD) runtime that can be validated independently of the neural policy. The first implementation must establish unambiguous task/robot types, transport timing, readiness, hard feasibility, metrics, and deterministic scenarios while preserving the behavior of existing SADCHER experiments and checkpoints.

## Solution

Add an environment-level MD domain model and discrete-time simulator with two mutually exclusive real task types (`PROCESS`, `TRANSPORT`) and two mutually exclusive real robot types (`PROCESS_ROBOT`, `TRANSPORT_ROBOT`). A transport task carries one material load from an explicit pickup location to an explicit delivery location and unlocks exactly one downstream process task.

Use a centralized hard-feasibility service for every scheduler and baseline. Model transport with the explicit state machine:

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

Validate the semantics with deterministic hand-crafted scenarios, structured execution records, basic metrics, and Greedy baselines. Keep neural networks, GAT inputs, checkpoints, and decoder behavior unchanged in this milestone. Treat the uploaded optimal `8t3r3s` dataset as a legacy process-only benchmark and regression fixture; do not invent MD fields for it.

## User Stories

1. As a scheduling researcher, I want to declare whether a real task is `PROCESS` or `TRANSPORT`, so that material movement cannot be confused with process execution.
2. As a scheduling researcher, I want process tasks to retain SADCHER skill requirements, so that existing process-coalition semantics remain comparable.
3. As a scheduling researcher, I want transport tasks to store pickup and delivery locations explicitly, so that transport timing and material arrival are observable.
4. As an environment user, I want each transport task to unlock exactly one process task, so that material precedence has a deterministic interpretation in the MVP.
5. As an environment user, I want each process task to have at most one material predecessor, so that the first material model does not hide an aggregation or inventory policy.
6. As a scheduling researcher, I want transport tasks to have no normal predecessors and material to be available at time zero, so that transport readiness is independent of an unimplemented inventory subsystem.
7. As a simulator user, I want transport robots and process robots to be mutually exclusive, so that a process-only robot cannot silently execute transport and vice versa.
8. As a process scheduler, I want process robots to form dynamic skill coalitions, so that a process starts only when all required skills are covered.
9. As a transport scheduler, I want a transport assignment to be atomic and single-robot in the MVP, so that infeasible loads are reported instead of receiving a hidden coalition fallback.
10. As a transport scheduler, I want capacity, transport capability, unloaded speed, and loaded speed checked explicitly, so that a robot is assigned only when it can complete the load.
11. As a simulator user, I want a transport robot to remain busy and non-preemptive from assignment through unloading, so that occupancy and contention have deterministic semantics.
12. As a metrics consumer, I want task start to mean loading start and completion to mean unloading completion, so that reported schedules reflect the physical service boundary.
13. As a metrics consumer, I want structured phase timestamps and durations, so that empty travel, loading, loaded travel, unloading, service, and occupied time can be audited.
14. As a scheduler author, I want one typed hard-feasibility result with stable reason codes, so that all policies make the same legality decision.
15. As an instance validator, I want static invalid instances to be rejected before simulation, so that impossible capacity, type, graph, and location definitions are visible.
16. As a simulator user, I want infeasible, deadlocked, and timed-out runs distinguished, so that a missing makespan is not misreported as a normal result.
17. As a process owner, I want a downstream process to become material-ready only after its transport completes, so that transport completion is the unlock event.
18. As a baseline researcher, I want process skill-coverage Greedy and transport Distance, ETA, and Unlock Greedy policies, so that MD semantics can be compared without a learned policy.
19. As a reproducibility researcher, I want fixed tie-breaking and deterministic hand-crafted scenarios, so that failures and metric differences can be reproduced on CPU.
20. As a legacy SADCHER user, I want old `Q/R/T_e/T_t/task_locations/precedence_constraints` experiments to keep running, so that the MD rollout does not invalidate existing baselines.
21. As a legacy dataset user, I want the official `8t3r3s` optimal dataset to load through the existing process-only path, so that its schedules remain regression evidence.
22. As a data integration maintainer, I want to change external loaders and data structures after official dataset inspection, so that the runtime model is not frozen to an incidental file layout.
23. As a runtime maintainer, I want the domain model and MD state machine to remain stable across loader changes, so that downstream metrics and tests retain their meaning.
24. As a future transport researcher, I want reserved `SOLO`, `SMALL_COALITION`, and `LARGE_COALITION` modes, so that later cooperative transport can be added without redefining the MVP semantics.
25. As a future policy researcher, I want cooperative transport mode selection to start rule-based, so that learned mode selection is evaluated only after the solo semantics are trusted.

## Implementation Decisions

- The runtime domain contains `ProcessTask`, `TransportTask`, `ProcessRobot`, and `TransportRobot`; `START`, `IDLE`, and `EXIT` remain optional legacy sentinels and are excluded from real-task and material metrics.
- Task IDs are continuous and stable within an instance. Normal process edges and material edges are typed separately. The material relation is the single source of truth and must agree with both endpoint fields.
- A `TransportTask` stores `pickup_location`, `delivery_location`, downstream process ID, load requirement, loading duration, unloading duration, and transport mode. The MVP mode is `SOLO`.
- A `ProcessTask` stores normal predecessors, an optional `material_predecessor` transport ID, location, duration, and the original skill requirement vector.
- A `TransportRobot` stores transport capability, capacity, current location, unloaded speed, loaded speed, and runtime phase/occupancy state. A `ProcessRobot` stores the original capability vector and coalition state. Hybrid execution is forbidden.
- Transport readiness is `PENDING` with no normal predecessor. Process readiness requires all normal predecessors and its material predecessor, if any, to be complete.
- Assignment is centralized through `is_assignment_feasible(robot, task) -> FeasibilityResult`. Stable reason codes cover task state, readiness, robot type, capability, capacity, occupancy, invalid graph, invalid location, and static infeasibility.
- Process assignment contribution is distinct from process coalition start feasibility. Robots may contribute uncovered skills, but the coalition cannot start until every required skill is covered.
- Transport assignment is single-robot and atomic. Transport coalition, capacity aggregation, synchronization waiting, relay, multi-trip, and preemption are prohibited in the MVP.
- The discrete simulator advances in integer timesteps. Movement duration is `ceil(distance / speed)` for the applicable unloaded or loaded phase. Loading and unloading use task-defined durations.
- Transport execution records preserve assignment, pickup arrival, loading start/completion, delivery arrival, completion, and phase durations. Process records preserve coalition membership, waiting, start, and completion.
- Metrics expose makespan, per-process material starvation, robot utilization, success, failure reason, and structured execution records. Makespan is valid only for successful completion; failure reports carry an explicit reason.
- Baselines share the hard-feasibility service and deterministic tie-breaking. The process baseline retains original skill-coverage Greedy; transport baselines are Distance, ETA, and Unlock policies.
- Legacy instances are interpreted as process-only with material delivery disabled. The legacy `T_t` matrix remains available to legacy solvers; MD movement initially uses coordinates and phase speeds.
- The uploaded official dataset is recorded as validated legacy data: 231,634 problem/solution pairs, all `8t3r3s`, with 2,084,545 indexed decision samples. It has no MD transport fields and is not converted.
- No neural network, GAT, checkpoint, decoder, or legacy feature dimension changes are part of this specification.

## Testing Decisions

- Tests assert externally observable environment behavior: state transitions, readiness, feasibility reasons, metrics, records, and legacy schedule compatibility. They do not assert private helper structure or incidental class layout.
- The primary seam is the environment/simulator boundary: construct an instance, apply assignments through the scheduler-facing interface, step discrete time, and inspect the terminal result and records.
- Runtime domain validation tests cover continuous IDs, typed edges, location consistency, acyclicity, numeric ranges, and explicit static infeasibility.
- Feasibility tests cover robot type mismatch, unavailable skills, insufficient single-robot capacity, occupied robots, unready tasks, and valid assignments, including stable reason codes.
- State-machine tests cover every transport phase, phase timing, non-preemption, location updates, task start/completion boundaries, and downstream unlock after unloading.
- Metrics tests cover makespan, material starvation, utilization, success, failure reason, and structured process/transport execution records.
- Deterministic scenario tests cover transport unlock, late material, capacity failure, type mismatch, location mismatch, process coalition, baseline policy differences, and legacy regression.
- Legacy regression tests reuse existing SADCHER simulator, Greedy, and dataset-loader seams. Fixed fixtures preserve schedule/makespan/feasible assignment behavior and the matching checkpoint remains CPU-loadable.
- The official dataset validation remains a read-only data contract test: paired filenames, JSON shape, Q/R semantics, Euclidean `T_t`, acyclic precedence, valid solution references, and loader indexing count.
- All MVP tests must run deterministically on CPU without neural-policy or GPU dependencies.

## Out of Scope

- Neural network, GAT, decoder, checkpoint, or learned feature changes.
- Transport coalitions, capacity aggregation, synchronization waiting, relay, multi-trip, preemption, or dynamic transport reassignment.
- Inventory, material production state, material consumption accounting, batching, or multiple material predecessors.
- Hybrid robots that execute both process and transport.
- Non-coordinate travel maps, battery, charging, low-level motion control, collision avoidance, or continuous-time simulation.
- Learned transport mode selection or learned cooperative transport before solo transport is validated.
- Converting the official legacy optimal dataset into synthetic MD records.

## Further Notes

The official dataset confirms that the current external benchmark is legacy process-only. Loader changes are permitted when later datasets provide MD fields, but every loader must converge to this runtime contract. English and Chinese versions of this specification and the related decisions are maintained together. This local spec is the implementation authority for the MVP together with the accepted ADRs and dataset validation record.
