from __future__ import annotations

import json

from modules.constraint_checkers import (
    ConstraintInstance,
    EvidenceState,
    is_checkable_conflict_with,
    is_satisfied_conflict_with,
)
from modules.identity_audit import (
    ADD,
    DELETE,
    ROLE_NAMES,
    SemanticEncoder,
    canonical_term,
    grounding_reason,
    input_topology,
    parse_raw_record,
    role_alias_mask,
    roundtrip_slots,
    semantic_id,
    semantic_id_to_term,
)


def _entity(qid: str) -> str:
    return f"<http://www.wikidata.org/entity/{qid}>"


def _property(pid: str) -> str:
    return f"<http://www.wikidata.org/prop/direct/{pid}>"


def _base_fields() -> list[str]:
    return [
        "<http://www.wikidata.org/entity/statement/P1-ABC>",
        "<http://www.wikidata.org/revision/1>",
        _entity("Q1"),
        _property("P1"),
        _entity("Q2"),
        _entity("Q3"),
        _property("P1"),
        _entity("Q2"),
        "->",
    ]


def test_semantic_ids_are_reversible_and_do_not_alias_rare_terms() -> None:
    terms = (
        canonical_term(_entity("Q987654321")),
        canonical_term(_entity("Q987654322")),
        canonical_term('"1"^^<http://www.w3.org/2001/XMLSchema#integer>'),
        canonical_term('"1"'),
    )
    ids = [semantic_id(term) for term in terms]
    assert len(set(ids)) == len(ids)
    assert [semantic_id_to_term(value) for value in ids] == list(terms)
    assert semantic_id(canonical_term("_:x", blank_scope="row-a")) != semantic_id(
        canonical_term("_:x", blank_scope="row-b")
    )


def test_semantic_encoder_normalizes_true_aliases_only() -> None:
    encoder = SemanticEncoder()
    assert encoder.encode("Q7") == encoder.encode(_entity("Q7"))
    assert encoder.encode("P31") == encoder.encode(_property("P31"))
    assert encoder.encode("Q7") != encoder.encode("Q8")


def test_parser_retains_real_description_owners_and_typed_literals() -> None:
    fields = _base_fields()
    fields.extend([_entity("Q1"), _property("P2"), '"7"^^<http://www.w3.org/2001/XMLSchema#integer>', ADD])
    # Deliberately put Q3's description in the first description position.
    fields.extend(
        [
            json.dumps(
                {
                    "type": "entity",
                    "id": _entity("Q3"),
                    "facts": {_property("P9"): [_entity("Q10")]},
                }
            ),
            "",
            "",
        ]
    )
    record = parse_raw_record(fields, source_key="fixture:1")
    assert record.compatible
    assert canonical_term(_entity("Q3")) in record.facts_by_owner
    assert record.additions[0].object == canonical_term(
        '"7"^^<http://www.w3.org/2001/XMLSchema#integer>'
    )


def test_partial_noop_and_multi_operation_accounting() -> None:
    noop = parse_raw_record([*_base_fields(), "", "", ""], source_key="fixture:noop")
    assert noop.compatible
    assert noop.six_slots() == ("", "", "", "", "", "")

    partial_fields = _base_fields() + [_entity("Q1"), _property("P2"), "", ADD, "", "", ""]
    partial = parse_raw_record(partial_fields, source_key="fixture:partial")
    assert not partial.compatible
    assert "incomplete_operation" in partial.exclusion_reason

    multi_fields = _base_fields()
    multi_fields += [_entity("Q1"), _property("P2"), _entity("Q4"), ADD]
    multi_fields += [_entity("Q1"), _property("P2"), _entity("Q5"), ADD]
    multi_fields += ["", "", ""]
    multi = parse_raw_record(multi_fields, source_key="fixture:multi")
    assert not multi.compatible
    assert "multiple_additions" in multi.exclusion_reason


def test_roundtrip_supports_true_role_aliases_and_rejects_absent_roles() -> None:
    roles = tuple(canonical_term(value) for value in (_entity("Q1"), _property("P1"), _entity("Q2"), "", "", ""))
    slots = (roles[0], roles[1], roles[2], "", "", "")
    permitted = [set(range(6)) for _ in range(6)]
    constants = [set() for _ in range(6)]
    covered, resolved = roundtrip_slots(
        slots,
        roles,
        role_indices_by_slot=permitted,
        constants_by_slot=constants,
    )
    assert covered
    assert tuple(resolved) == slots
    assert role_alias_mask(canonical_term(_entity("Q9")), roles) == 0


def test_no_copy_is_subset_of_copy_and_copy_has_no_gold_expansion() -> None:
    target = canonical_term(_entity("Q9"))
    fixed, copied, reason = grounding_reason(
        term=target,
        role_mask=0,
        permitted_role_indices=set(range(6)),
        constant_terms=set(),
        input_terms={target},
        primary_parameter_terms=set(),
        local_parameter_terms=set(),
        canonical_to_encoder_id={},
    )
    assert not fixed and copied
    assert reason == "missing_role_present_local_neighbor"
    fixed, copied, _ = grounding_reason(
        term=canonical_term(_entity("Q10")),
        role_mask=0,
        permitted_role_indices=set(range(6)),
        constant_terms=set(),
        input_terms={target},
        primary_parameter_terms=set(),
        local_parameter_terms=set(),
        canonical_to_encoder_id={},
    )
    assert not fixed and not copied


def test_topology_is_identity_preserving_but_feature_threshold_independent() -> None:
    fields = _base_fields()
    fields.extend([_entity("Q1"), _property("P2"), _entity("Q4"), DELETE, "", "", ""])
    record = parse_raw_record(fields, source_key="fixture:topology")
    all_known = {
        term: index + 1
        for index, term in enumerate(
            (record.roles[0], record.roles[1], record.roles[2], record.roles[3])
        )
        if term
    }
    rare_unknown = {record.roles[1]: 1}
    topology_known = input_topology(
        record,
        factor_ids=[record.constraint_id],
        parameter_pairs=[],
        canonical_to_encoder_id=all_known,
    )
    topology_unknown = input_topology(
        record,
        factor_ids=[record.constraint_id],
        parameter_pairs=[],
        canonical_to_encoder_id=rare_unknown,
    )
    assert topology_known["nodes"] == topology_unknown["nodes"]
    assert topology_known["edges"] == topology_unknown["edges"]
    assert topology_known["distinct_semantic_terms"] == topology_unknown["distinct_semantic_terms"]
    assert topology_known["unknown_feature_nodes"] < topology_unknown["unknown_feature_nodes"]


def test_injective_renaming_preserves_validator_outcome() -> None:
    def evaluate(rename: dict[int, int]) -> tuple[bool, bool]:
        subject, constrained, conflict, obj = (rename[value] for value in (1, 2, 3, 4))
        state = EvidenceState(
            facts_by_entity={subject: {constrained: {obj}, conflict: {obj}}},
            predicates_present={subject: {constrained, conflict}},
            assume_complete=True,
            missing_edits=set(),
            focus_subject=subject,
            focus_predicate=constrained,
            focus_object=obj,
            other_subject=subject,
            other_predicate=conflict,
            other_object=obj,
        )
        constraint = ConstraintInstance(
            constraint_id=rename[5],
            constraint_type="conflictWith",
            constraint_type_id=rename[6],
            constrained_property=constrained,
            required_properties=set(),
            allowed_items=set(),
            allowed_classes=set(),
            relation_predicates=[],
            inverse_properties=[],
            conflict_properties={conflict},
        )
        p_local = {constrained, conflict}
        return (
            is_checkable_conflict_with(state, constraint, p_local),
            is_satisfied_conflict_with(state, constraint, p_local),
        )

    identity = {value: value for value in range(1, 7)}
    renamed = {value: 10_000 + 17 * value for value in range(1, 7)}
    assert evaluate(identity) == evaluate(renamed) == (True, False)
