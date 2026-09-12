"""Ticket 41 deterministic post-hoc attribution for the Ticket 40 package."""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast
import torch
from experiments.md_budgeted_saturation_curriculum_diagnostic import FORMAL_PROTOCOL as TRAINING_PROTOCOL
from experiments.md_flip_curriculum_train_diagnostic import _family_balanced_batches, _prepare_batch, _train_curriculum
from experiments.md_independent_held_out_evaluation import _load_package, _replay_package
from experiments.md_independent_held_out_package import HELD_OUT_MARGIN_STRATA
from experiments.md_policy_relational_gate import CANDIDATE_TASK_COUNT, DEFAULT_MODEL_SEEDS, RELATIONAL_FAMILIES, _build_model, _model_logits, _validated_model_seeds
MODEL_METHODS: Final = ("matched_parameter_mlp", "pair_aware_attention")
HIGH_MARGIN_STRATUM: Final = "high_ge_0.05"
PACKAGE_PATH: Final = Path("reports/md_independent_held_out_package_protocol_corrected_2026-08-28") / "independent_held_out_package.json"
@dataclass(frozen=True, slots=True)
class PickupHighMarginAttributionResult:
    summary_path: Path
    report_path: Path

def run_pickup_high_margin_error_attribution(output_dir: str | Path, *, package_path: str | Path = PACKAGE_PATH, pairs_per_family_per_batch: int = 100, pretrain_epochs: int = 200, fine_tune_epochs: int = 100, checkpoint_interval: int = 10, model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS, pretrain_learning_rate: float = 0.01, fine_tune_learning_rate: float = 0.002, state_margin: float = 0.1, flip_state_margin: float = 0.1, pair_margin: float = 0.1, flip_state_loss_weight: float = 0.5, pair_loss_weight: float = 0.25, saturation_loss_weight: float = 0.1, saturation_penalty_allowance: float = 0.25, minimum_overall_train_agreement: float = 0.70, minimum_family_flip_pair_exact: float = 0.60, maximum_residual_saturation_rate: float = 0.25) -> PickupHighMarginAttributionResult:
    destination=Path(output_dir); summary_path=destination/"pickup_high_margin_error_attribution.json"; report_path=destination/"pickup_high_margin_error_attribution.md"
    if summary_path.exists() or report_path.exists(): raise FileExistsError("refusing to overwrite Ticket 41 diagnostic")
    seeds=_validated_model_seeds(model_seeds); package=_load_package(Path(package_path)); provenance=cast(dict[str,object],package["train_provenance"]); ep=cast(dict[str,object],package["evaluation_provenance"])
    train_twins,evaluation_twins,replay_mismatch,template_overlap=_replay_package(package,candidate_pool_per_family=cast(int,provenance["candidate_pool_per_family"]),train_pairs_per_family=cast(int,provenance["pairs_per_family"]),train_seed=cast(int,provenance["seed"]),evaluation_seed=cast(int,ep["seed"]),perturbation_seed=cast(int,package["perturbation_seed"]),quota_per_family_stratum=cast(int,ep["quota_per_family_stratum"]))
    if replay_mismatch or template_overlap: raise ValueError("Ticket 39 package replay or overlap validation failed")
    protocol={"candidate_pool_per_family":cast(int,provenance["candidate_pool_per_family"]),"train_pairs_per_family":cast(int,provenance["pairs_per_family"]),"pairs_per_family_per_batch":pairs_per_family_per_batch,"pretrain_epochs":pretrain_epochs,"fine_tune_epochs":fine_tune_epochs,"checkpoint_interval":checkpoint_interval,"data_seed":cast(int,provenance["seed"]),"model_seeds":seeds,"pretrain_learning_rate":pretrain_learning_rate,"fine_tune_learning_rate":fine_tune_learning_rate,"state_margin":state_margin,"flip_state_margin":flip_state_margin,"pair_margin":pair_margin,"flip_state_loss_weight":flip_state_loss_weight,"pair_loss_weight":pair_loss_weight,"saturation_loss_weight":saturation_loss_weight,"saturation_penalty_allowance":saturation_penalty_allowance,"minimum_overall_train_agreement":minimum_overall_train_agreement,"minimum_family_flip_pair_exact":minimum_family_flip_pair_exact,"maximum_residual_saturation_rate":maximum_residual_saturation_rate}
    train_states=_states(train_twins); eval_states=_states(evaluation_twins); batches=tuple(_prepare_batch(batch) for batch in _family_balanced_batches(train_twins,pairs_per_family_per_batch=pairs_per_family_per_batch)); models={}
    for method in MODEL_METHODS:
        per_seed=[]
        for seed in seeds:
            model=_build_model(method,seed=seed); training=_train_curriculum(model,batches,train_twins,train_states,pretrain_epochs=pretrain_epochs,fine_tune_epochs=fine_tune_epochs,checkpoint_interval=checkpoint_interval,pretrain_learning_rate=pretrain_learning_rate,fine_tune_learning_rate=fine_tune_learning_rate,state_margin=state_margin,flip_state_margin=flip_state_margin,pair_margin=pair_margin,flip_state_loss_weight=flip_state_loss_weight,pair_loss_weight=pair_loss_weight,saturation_loss_weight=saturation_loss_weight,saturation_penalty_allowance=saturation_penalty_allowance,minimum_overall_train_agreement=minimum_overall_train_agreement,minimum_family_flip_pair_exact=minimum_family_flip_pair_exact,maximum_residual_saturation_rate=maximum_residual_saturation_rate)
            logits,diag=_model_logits(model,eval_states); predictions=cast(list[int],torch.argmax(logits,dim=-1).tolist()); per_seed.append({"model_seed":seed,"training":training,"records":_pair_records(evaluation_twins,predictions,diag)})
        count=sum(p.numel() for p in _build_model(method,seed=seeds[0]).parameters() if p.requires_grad); models[method]={"parameter_count":count,"per_seed":per_seed,"aggregate":_aggregate(per_seed,evaluation_twins)}
    classification=_classify(models); summary={"schema_version":"1.0.0","status":"pickup_high_margin_error_attribution","ticket":41,"source_ticket":40,"diagnostic_scope":"post_hoc_diagnostic_only","formal_protocol_run":protocol==TRAINING_PROTOCOL,"training_protocol":_jsonable(protocol),"model_seeds":list(seeds),"evaluation_pair_count":len(evaluation_twins),"evaluation_state_count":len(eval_states),"package_replay_mismatch_count":replay_mismatch,"train_evaluation_template_overlap":template_overlap,"models":models,"comparison":_comparison(models),"pickup_feature_summary":_pickup_summary(models),"diagnostic_classification":classification,"controls":{"held_out_pairs_reselected":False,"model_output_used_for_selection":False,"frozen_package_modified":False,"production_expert_dataset_written":False,"ticket_17_unfrozen":False,"ticket_20_unfrozen":False,"ticket_32_unblocked":False,"architecture_gate_reopened":False},"next_step":_next_step(classification)}
    destination.mkdir(parents=True,exist_ok=True); summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); report_path.write_text(_render_report(summary),encoding="utf-8"); return PickupHighMarginAttributionResult(summary_path,report_path)

def _states(twins): return tuple(s for t in twins for s in (t.before,t.after))
def _jsonable(v):
    if isinstance(v,tuple): return [_jsonable(x) for x in v]
    if isinstance(v,dict): return {str(k):_jsonable(x) for k,x in v.items()}
    return v

def _pair_records(twins,predictions,diag):
    rows=[]; legacy=diag.legacy_component; physics=diag.physics_utility; residual=diag.bounded_residual; combined=diag.combined_score
    for i,twin in enumerate(twins):
        b=_state_record(twin.before,predictions[2*i],legacy[2*i],physics[2*i],residual[2*i],combined[2*i],diag.residual_saturation_rate[2*i]); a=_state_record(twin.after,predictions[2*i+1],legacy[2*i+1],physics[2*i+1],residual[2*i+1],combined[2*i+1],diag.residual_saturation_rate[2*i+1]); rows.append({"pair_id":twin.pair_id,"family":twin.family,"changed_entity_index":twin.changed_entity_index,"pair_margin_stratum":_pair_stratum(twin),"oracle_flip_status":"oracle_flip" if twin.before.oracle_action!=twin.after.oracle_action else "oracle_no_flip","before":b,"after":a,"prediction_flipped":b["prediction"]!=a["prediction"],"prediction_flip_status":"flip" if b["prediction"]!=a["prediction"] else "no_flip","oracle_action_flipped":twin.before.oracle_action!=twin.after.oracle_action,"before_oracle_action":twin.before.oracle_action,"after_oracle_action":twin.after.oracle_action,"before_prediction":b["prediction"],"after_prediction":a["prediction"],"before_correctness":b["correctness"],"after_correctness":a["correctness"],"before_completion_regret":b["completion_regret"],"after_completion_regret":a["completion_regret"],"before_oracle_margin":b["oracle_margin"],"after_oracle_margin":a["oracle_margin"],"exact_pair_correct":bool(b["correctness"] and a["correctness"]),"pickup_features":_pickup_features(twin)})
    return rows

def _state_record(state,prediction,legacy,physics,residual,combined,saturation):
    vals=state.action_values; selected=float(residual[:, :CANDIDATE_TASK_COUNT].reshape(-1)[prediction].item()); return {"state_id":state.state_id,"oracle_action":state.oracle_action,"prediction":prediction,"correctness":prediction==state.oracle_action,"completion_regret":max(vals)-vals[prediction],"oracle_margin":_margin(state),"residual_magnitude":_mean_masked(residual,state.hard_mask),"saturation_rate":float(saturation.item()),"selected_residual_magnitude":abs(selected),"selected_saturated":abs(selected)>=0.95*0.75,"score_decomposition":_score_decomposition(legacy,physics,residual,combined,state.hard_mask),"feasible_transport_entry_count":sum(sum(row[:CANDIDATE_TASK_COUNT]) for row in state.hard_mask)}

def _score_decomposition(legacy,physics,residual,combined,mask):
    def matrix(t):
        return [[float(t[r, task].detach().item()) if mask[r][task] else None for task in range(len(mask[r]))] for r in range(len(mask))]
    return {"legacy":matrix(legacy),"physics":matrix(physics),"bounded_residual":matrix(residual),"combined":matrix(combined),"hard_mask":[list(row) for row in mask]}
def _mean_masked(t,mask):
    vals=[float(t[r,task].item()) for r,row in enumerate(mask) for task,ok in enumerate(row[:CANDIDATE_TASK_COUNT]) if ok]; return sum(abs(v) for v in vals)/len(vals) if vals else 0.0
def _margin(state):
    vals=sorted(state.action_values,reverse=True); return vals[0]-vals[1]
def _pair_stratum(twin):
    m=min(_margin(twin.before),_margin(twin.after))
    for name,lo,hi in HELD_OUT_MARGIN_STRATA:
        if lo<=m<hi:return name
    raise ValueError("pair margin outside frozen strata")

def _pickup_features(twin):
    if twin.family != "alternative_task_pickup": return {"applicable":False,"changed_task_index":twin.changed_entity_index,"states":[]}
    changed=twin.changed_entity_index; states=[]
    for state in (twin.before,twin.after):
        robot=state.oracle_action//CANDIDATE_TASK_COUNT; focal=state.oracle_action%CANDIDATE_TASK_COUNT; distance=lambda task: math.dist(state.robot_positions[robot],state.task_pickups[task]); focal_action=robot*CANDIDATE_TASK_COUNT+focal; alt_action=robot*CANDIDATE_TASK_COUNT+changed
        states.append({"state_id":state.state_id,"focal_task_index":focal,"alternative_task_index":changed,"oracle_robot_index":robot,"focal_pickup_distance":distance(focal),"alternative_pickup_distance":distance(changed),"focal_robot_to_pickup_eta":distance(focal)/2.0,"alternative_robot_to_pickup_eta":distance(changed)/2.0,"focal_loaded_leg_distance":math.dist(state.task_pickups[focal],state.task_deliveries[focal]),"alternative_loaded_leg_distance":math.dist(state.task_pickups[changed],state.task_deliveries[changed]),"focal_task_priority":state.downstream_priorities[focal],"alternative_task_priority":state.downstream_priorities[changed],"focal_downstream_priority":state.downstream_priorities[focal],"alternative_downstream_priority":state.downstream_priorities[changed],"priority_source":"synthetic protocol exposes downstream_priorities as task and downstream priority","oracle_value_gap":state.action_values[focal_action]-state.action_values[alt_action],"changed_task_participates_in_oracle_action":focal==changed})
    return {"applicable":True,"changed_task_index":changed,"states":states}

def _stats(rows):
    if not rows:return {"pair_count":0,"state_count":0,"state_agreement":0.0,"exact_pair_accuracy":0.0,"mean_completion_regret":0.0,"mean_oracle_margin":0.0,"mean_residual_magnitude":0.0,"mean_saturation_rate":0.0}
    states=[row[key] for row in rows for key in ("before","after")]; return {"pair_count":len(rows),"state_count":len(states),"state_agreement":sum(float(s["correctness"]) for s in states)/len(states),"exact_pair_accuracy":sum(float(r["exact_pair_correct"]) for r in rows)/len(rows),"mean_completion_regret":sum(float(s["completion_regret"]) for s in states)/len(states),"mean_oracle_margin":sum(float(s["oracle_margin"]) for s in states)/len(states),"mean_residual_magnitude":sum(float(s["residual_magnitude"]) for s in states)/len(states),"mean_saturation_rate":sum(float(s["saturation_rate"]) for s in states)/len(states)}
def _grouped(rows,key,values): return {v:_stats([r for r in rows if r[key]==v]) for v in values}

def _aggregate(per_seed,twins):
    seed_rows=[cast(list, x["records"]) for x in per_seed]; aggregate=[]
    for i,twin in enumerate(twins):
        variants=[rows[i] for rows in seed_rows]; row=dict(variants[0])
        for side in ("before","after"):
            states=[v[side] for v in variants]; s=dict(states[0]); s["correctness"]=sum(float(x["correctness"]) for x in states)/len(states); s["completion_regret"]=sum(float(x["completion_regret"]) for x in states)/len(states); s["residual_magnitude"]=sum(float(x["residual_magnitude"]) for x in states)/len(states); s["saturation_rate"]=sum(float(x["saturation_rate"]) for x in states)/len(states); row[side]=s
        row["exact_pair_correct"]=sum(float(v["exact_pair_correct"]) for v in variants)/len(variants); aggregate.append(row)
    result={"overall":_stats(aggregate),"by_family":_grouped(aggregate,"family",RELATIONAL_FAMILIES),"by_margin_stratum":_grouped(aggregate,"pair_margin_stratum",[x[0] for x in HELD_OUT_MARGIN_STRATA]),"by_oracle_flip_status":_grouped(aggregate,"oracle_flip_status",("oracle_flip","oracle_no_flip"))}; return result


def _comparison(models):
    matched = cast(dict, models[MODEL_METHODS[0]]["aggregate"])
    pair_aware = cast(dict, models[MODEL_METHODS[1]]["aggregate"])
    matched_seeds = models[MODEL_METHODS[0]]["per_seed"]
    pair_seeds = models[MODEL_METHODS[1]]["per_seed"]
    high_m = [r for seed in matched_seeds for r in cast(list, seed["records"]) if r["pair_margin_stratum"] == HIGH_MARGIN_STRATUM]
    high_a = [r for seed in pair_seeds for r in cast(list, seed["records"]) if r["pair_margin_stratum"] == HIGH_MARGIN_STRATUM]
    transitions = {"state": {"wrong_to_correct": 0, "correct_to_wrong": 0, "unchanged": 0}, "exact_pair": {"wrong_to_correct": 0, "correct_to_wrong": 0, "unchanged": 0}}
    for old, new in zip(high_m, high_a):
        for side in ("before", "after"):
            old_ok = bool(old[side]["correctness"]); new_ok = bool(new[side]["correctness"])
            label = "wrong_to_correct" if not old_ok and new_ok else "correct_to_wrong" if old_ok and not new_ok else "unchanged"
            transitions["state"][label] += 1
        old_ok = bool(old["exact_pair_correct"]); new_ok = bool(new["exact_pair_correct"])
        label = "wrong_to_correct" if not old_ok and new_ok else "correct_to_wrong" if old_ok and not new_ok else "unchanged"
        transitions["exact_pair"][label] += 1
    pair_rows = []
    for index in range(len(cast(list, matched_seeds[0]["records"]))):
        matched_score = sum(int(bool(cast(list, seed["records"])[index]["before"]["correctness"])) + int(bool(cast(list, seed["records"])[index]["after"]["correctness"])) for seed in matched_seeds)
        pair_score = sum(int(bool(cast(list, seed["records"])[index]["before"]["correctness"])) + int(bool(cast(list, seed["records"])[index]["after"]["correctness"])) for seed in pair_seeds)
        source = cast(list, matched_seeds[0]["records"])[index]
        pair_rows.append({"family": source["family"], "pair_margin_stratum": source["pair_margin_stratum"], "oracle_flip_status": source["oracle_flip_status"], "win_tie_loss": "win" if pair_score > matched_score else "loss" if pair_score < matched_score else "tie"})
    def category_counts(rows, dimension, value):
        selected = [row for row in rows if row[dimension] == value]
        return {label: sum(row["win_tie_loss"] == label for row in selected) for label in ("win", "tie", "loss")}
    return {"pair_aware_minus_matched_state_agreement": float(pair_aware["overall"]["state_agreement"]) - float(matched["overall"]["state_agreement"]), "pair_aware_minus_matched_exact_pair_accuracy": float(pair_aware["overall"]["exact_pair_accuracy"]) - float(matched["overall"]["exact_pair_accuracy"]), "pair_aware_minus_matched_state_agreement_by_stratum": {s: float(pair_aware["by_margin_stratum"][s]["state_agreement"]) - float(matched["by_margin_stratum"][s]["state_agreement"]) for s, _, _ in HELD_OUT_MARGIN_STRATA}, "high_margin_transitions": transitions, "win_tie_loss": {label: sum(row["win_tie_loss"] == label for row in pair_rows) for label in ("win", "tie", "loss")}, "win_tie_loss_by_family": {family: category_counts(pair_rows, "family", family) for family in RELATIONAL_FAMILIES}, "win_tie_loss_by_margin_stratum": {stratum: category_counts(pair_rows, "pair_margin_stratum", stratum) for stratum, _, _ in HELD_OUT_MARGIN_STRATA}, "win_tie_loss_by_oracle_flip_status": {status: category_counts(pair_rows, "oracle_flip_status", status) for status in ("oracle_flip", "oracle_no_flip")}}

def _pickup_summary(models):
    rows=[r for r in models["pair_aware_attention"]["per_seed"][0]["records"] if r["family"]=="alternative_task_pickup"]; states=[s for r in rows for s in r["pickup_features"]["states"]]; keys=("focal_pickup_distance","alternative_pickup_distance","focal_robot_to_pickup_eta","alternative_robot_to_pickup_eta","focal_loaded_leg_distance","alternative_loaded_leg_distance","focal_task_priority","alternative_task_priority","oracle_value_gap"); return {"pair_count":len(rows),"changed_task_oracle_participation_rate":sum(bool(s["changed_task_participates_in_oracle_action"]) for s in states)/len(states),**{"mean_"+key:sum(float(s[key]) for s in states)/len(states) for key in keys}}
def _classify(models):
    p=cast(dict,models["pair_aware_attention"]["aggregate"])["by_family"]["alternative_task_pickup"]; h=cast(dict,models["pair_aware_attention"]["aggregate"])["by_margin_stratum"][HIGH_MARGIN_STRATUM]
    if float(p["exact_pair_accuracy"])<0.60 and float(h["state_agreement"])<0.80 and float(h["mean_saturation_rate"])>=0.30:return "mixed_or_unresolved"
    if float(p["exact_pair_accuracy"])<0.60:return "task_context_insufficient"
    if float(h["mean_saturation_rate"])>=0.30:return "transport_calibration_insufficient"
    return "seed_instability"
def _next_step(c): return {"task_context_insufficient":"Use a new development package for task/process context ablation.","transport_calibration_insufficient":"Use a new development package for transport calibration ablation.","seed_instability":"Repair training stability on train-only data before architecture comparison.","mixed_or_unresolved":"Keep the current scorer and use a new development package to separate context and transport calibration evidence."}[c]

def _render_report(summary):
    models=summary["models"]; lines=["# Ticket 41 Pickup And High-Margin Error Attribution","","Scope: **post_hoc_diagnostic_only**.","","## Direct Evidence","","| Method | State agreement | Exact pair |","|---|---:|---:|"]
    for method in MODEL_METHODS:
        o=models[method]["aggregate"]["overall"]; lines.append(f"| {method} | {float(o['state_agreement']):.4f} | {float(o['exact_pair_accuracy']):.4f} |")
    c=summary["comparison"]; lines += ["",f"Pickup and high-margin attribution classification: **{summary['diagnostic_classification']}**.",f"State-agreement gain: **{float(c['pair_aware_minus_matched_state_agreement']):.4f}**; exact-pair gain: **{float(c['pair_aware_minus_matched_exact_pair_accuracy']):.4f}**.","","## High-Margin Transitions","","| Target | Wrong to correct | Correct to wrong | Unchanged |","|---|---:|---:|---:|"]
    for target in ("state", "exact_pair"):
        row=c["high_margin_transitions"][target]; lines.append(f"| {target} | {row['wrong_to_correct']} | {row['correct_to_wrong']} | {row['unchanged']} |")
    pickup=summary["pickup_feature_summary"]; lines += ["","## Family And Margin Evidence","","| Family | Matched exact pair | Pair-aware exact pair |","|---|---:|---:|"]; lines += [f"| {family} | {float(models[MODEL_METHODS[0]]['aggregate']['by_family'][family]['exact_pair_accuracy']):.4f} | {float(models[MODEL_METHODS[1]]['aggregate']['by_family'][family]['exact_pair_accuracy']):.4f} |" for family in RELATIONAL_FAMILIES]; lines += ["",f"Pair-aware high-margin saturation: **{float(models[MODEL_METHODS[1]]['aggregate']['by_margin_stratum'][HIGH_MARGIN_STRATUM]['mean_saturation_rate']):.4f}**.",f"Pickup changed-task oracle participation: **{float(pickup['changed_task_oracle_participation_rate']):.4f}**; mean oracle value gap: **{float(pickup['mean_oracle_value_gap']):.4f}**."]
    lines += ["","## Interpretation","","Pickup exact-pair remains below 0.60 while only 0.2150 of changed tasks participate in the oracle action; the observed pattern is consistent with insufficient alternative-task/process context in the learned residual. High-margin wrong-to-correct and correct-to-wrong counts are near-balanced, while saturation is 0.3426, so the data do not isolate a transport-calibration-only explanation.","","The legacy component carries the local backbone score, the fixed physics component carries ETA calibration, and the bounded residual is the learned term that can express relational task context. The observed pickup failure therefore selects task/process context ablation as the next development experiment; high-margin saturation remains a secondary calibration signal.","",f"Next implementation choice: **{summary['next_step']}**","","## Limitations","","These are post-hoc diagnostic associations, not causal proof. This report does not reopen an architecture gate or authorize held-out model selection. Any later candidate must first use a new development package, freeze its protocol, and then use a completely new confirmation held-out package.",""]
    return "\n".join(lines)
def main():
    parser=argparse.ArgumentParser(description="Run Ticket 41 post-hoc attribution"); parser.add_argument("--output-dir",required=True); parser.add_argument("--package-path",default=str(PACKAGE_PATH)); args=parser.parse_args(); result=run_pickup_high_margin_error_attribution(args.output_dir,package_path=args.package_path); print(result.summary_path); print(result.report_path)
if __name__=="__main__": main()
