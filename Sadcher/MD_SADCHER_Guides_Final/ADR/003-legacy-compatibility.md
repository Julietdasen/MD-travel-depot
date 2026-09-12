# ADR-003: Legacy Compatibility and Official Data Integration

- Status: accepted
- Scope: MD-SADCHER++ MVP

## Decision

Legacy `Q/R/T_e/T_t/task_locations/precedence_constraints` instances remain process-only instances with `material_delivery.enabled=false`. Internal classes and loading code may change, but legacy experiments must remain runnable, with regression fixtures for schedule, makespan, and feasible assignments.

After the official dataset arrives, the loader and related data structures may be changed directly; a permanent adapter file is not required. Every external format must still converge to the stable runtime domain model and preserve MD state-machine, readiness, feasibility, and metric semantics.

Legacy `T_t` remains available to old MILP, Greedy, visualization, and dataset consumers. MD runtime initially computes movement from coordinates and phase speeds. Whether an official travel matrix can be reused is deferred until dataset inspection.

## Rationale

This protects baseline comparability without prematurely freezing an external schema. The stable contract is domain semantics, not an incidental file layout.
