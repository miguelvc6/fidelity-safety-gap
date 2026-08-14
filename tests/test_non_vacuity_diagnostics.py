from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.constraint_checkers import EvidenceState
from modules.repair_eval import RepairSample, evaluate_global_repair_samples
from modules.reranker_eval import _evidence_preservation_details


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(facts: dict[int, dict[int, set[int]]]) -> EvidenceState:
    return EvidenceState(
        facts_by_entity=facts,
        predicates_present={1: {10}},
        assume_complete=True,
        missing_edits=set(),
        focus_subject=1,
        focus_predicate=10,
        focus_object=2,
        other_subject=0,
        other_predicate=0,
        other_object=0,
    )


def test_evidence_details_detect_focus_preservation_and_non_vacuous_primary_fix() -> None:
    pre = _state({1: {10: {2}}})
    post = _state({1: {10: {2}}})

    details = _evidence_preservation_details(
        pre_state=pre,
        post_state=post,
        candidate_slots=[0, 0, 0, 0, 0, 0],
        placeholder_map={},
        primary_satisfied=1,
        pre_global_satisfied_fraction=0.5,
        post_global_satisfied_fraction=0.5,
    )

    assert details["pre_focus_present"] == 1
    assert details["post_focus_present"] == 1
    assert details["focus_preserved"] == 1
    assert details["focus_deleted"] == 0
    assert details["candidate_deletes_focus"] == 0
    assert details["non_vacuous_primary_fix"] == 1
    assert details["vacuous_satisfaction_improvement"] == 0


def test_evidence_details_detect_focus_deletion_and_vacuous_improvement() -> None:
    pre = _state({1: {10: {2}}})
    post = _state({1: {10: set()}})

    details = _evidence_preservation_details(
        pre_state=pre,
        post_state=post,
        candidate_slots=[0, 0, 0, 1, 10, 2],
        placeholder_map={},
        primary_satisfied=1,
        pre_global_satisfied_fraction=0.5,
        post_global_satisfied_fraction=0.75,
    )

    assert details["pre_focus_present"] == 1
    assert details["post_focus_present"] == 0
    assert details["focus_preserved"] == 0
    assert details["focus_deleted"] == 1
    assert details["candidate_deletes_focus"] == 1
    assert details["non_vacuous_primary_fix"] == 0
    assert details["vacuous_satisfaction_improvement"] == 1


class _FakeEvaluator:
    def __init__(self, details: list[dict[str, object]]) -> None:
        self.details = details
        self.index = 0

    def evaluate_full(self, *args, **kwargs):
        del args, kwargs
        item = self.details[self.index]
        self.index += 1
        return item


def _detail(*, focus_deleted: int, non_vacuous: int) -> dict[str, object]:
    return {
        "local_constraint_ids": [101],
        "primary_factor_index": 0,
        "pre_checkable": [True],
        "pre_satisfied": [0],
        "post_checkable": [True],
        "post_satisfied": [1],
        "primary_satisfied": 1,
        "global_satisfied_fraction": 1.0,
        "secondary_regressions": 0,
        "secondary_improvements": 0,
        "secondary_regressions_denom": 0,
        "secondary_improvements_denom": 0,
        "srr": 0.0,
        "sir": 0.0,
        "add_count": 0,
        "del_count": focus_deleted,
        "pre_global_satisfied_fraction": 0.0,
        "post_global_satisfied_fraction": 1.0,
        "pre_focus_present": 1,
        "post_focus_present": 1 - focus_deleted,
        "focus_preserved": 1 - focus_deleted,
        "focus_deleted": focus_deleted,
        "candidate_deletes_focus": focus_deleted,
        "non_vacuous_primary_fix": non_vacuous,
        "vacuous_satisfaction_improvement": focus_deleted,
    }


def _paper_instance(
    *,
    pfr: int = 1,
    local_num: int = 1,
    local_denom: int = 1,
    base_deleted: int = 0,
    deletes_base: int = 0,
    eppf: int = 1,
    vacuous: int = 0,
    disruption: int = 1,
) -> dict[str, object]:
    return {
        "events": {
            "pfr": {"numerator": pfr, "denominator": 1},
            "local_satisfaction": {"numerator": local_num, "denominator": local_denom},
            "delta_local_satisfaction": {"numerator": 1, "denominator": 1},
            "sir": {"numerator": 0, "denominator": 1},
            "srr": {"numerator": 0, "denominator": 1},
            "disruption": {"numerator": disruption, "denominator": 1},
            "base_deletion_rate": {"numerator": base_deleted, "denominator": 1},
            "deletes_base_action_rate": {"numerator": deletes_base, "denominator": 1},
            "eppf": {"numerator": eppf, "denominator": 1},
            "vacuous_improvement": {"numerator": vacuous, "denominator": 1},
        },
        "post_base_present": 1 - base_deleted,
        "resolved_add": None,
        "resolved_del": (1, 10, 2) if base_deleted else None,
    }


def test_global_aggregation_includes_evidence_preservation_rates() -> None:
    samples = [
        RepairSample("single", predicted={"add": None, "del": None}, gold={"add": None, "del": None}),
        RepairSample("single", predicted={"add": None, "del": (1, 10, 2)}, gold={"add": None, "del": None}),
    ]
    metrics = evaluate_global_repair_samples(
        samples=samples,
        rows=[object(), object()],
        evaluator=_FakeEvaluator([_detail(focus_deleted=0, non_vacuous=1), _detail(focus_deleted=1, non_vacuous=0)]),
        none_class=0,
    )

    paper = metrics["paper_metrics"]
    assert paper["base_deletion_rate"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert paper["deletes_base_action_rate"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert paper["eppf"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert paper["vacuous_improvement"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert len(metrics["per_instance"]) == 2


def test_candidate_oracle_aggregate_tracks_non_vacuous_safe_gap() -> None:
    oracle = _load_script("analyze_candidate_oracle")
    agg = oracle.Aggregate()
    safe_delete = _paper_instance(
        base_deleted=1,
        deletes_base=1,
        eppf=0,
        vacuous=1,
    )
    selected = _paper_instance(pfr=0, local_num=1, local_denom=2, eppf=0)

    agg.add(
        candidate_count=2,
        oracle=safe_delete,
        selected=selected,
        oracle_safe=True,
        selected_safe=False,
        oracle_non_vacuous_safe=False,
        selected_non_vacuous_safe=False,
    )

    result = agg.to_dict()
    assert result["oracle_safe_available_rate"] == 1.0
    assert result["oracle_non_vacuous_safe_available_rate"] == 0.0
    assert result["selected_non_vacuous_safe_rate"] == 0.0
    assert result["oracle_paper_metrics"]["base_deletion_rate"]["value"] == 1.0
    assert result["oracle_paper_metrics"]["vacuous_improvement"]["value"] == 1.0


def test_deletion_degeneracy_aggregate_exact_and_metric_equivalent_cases() -> None:
    deletion = _load_script("analyze_deletion_degeneracy")
    agg = deletion.Aggregate()
    h1_slots = [0, 0, 0, 1, 10, 2]
    same = _paper_instance(
        base_deleted=1,
        deletes_base=1,
        eppf=0,
        vacuous=1,
    )
    agg.add(g0_slots=h1_slots, h1_slots=h1_slots, g0=same, h1=same)
    agg.add(g0_slots=[0, 0, 0, 3, 10, 4], h1_slots=h1_slots, g0=same, h1=same)

    result = agg.to_dict()
    assert result["prediction_exact_match_rate"] == 0.5
    assert result["metric_equivalent_rate"] == 1.0
    assert result["g0_paper_metrics"]["base_deletion_rate"]["value"] == 1.0
    assert result["dfb_paper_metrics"]["deletes_base_action_rate"]["value"] == 1.0
    assert result["dfb_paper_metrics"]["eppf"]["value"] == 0.0
