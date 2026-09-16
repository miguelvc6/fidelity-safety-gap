from __future__ import annotations

from copy import deepcopy

import pytest

from modules.identity_audit import ROLE_NAMES
from modules.verified_vocab_followup import (
    ACTION_CATEGORIES,
    VerifiedVocabularyBuilder,
    action_features,
    assert_v1_subset_v0,
    capped_target_allocation,
    edit_coverage,
    js_divergence,
    largest_remainder_allocation,
)


ROLE_IDS = {name: 100 + index for index, name in enumerate(ROLE_NAMES)}


def row(**updates):
    value = {
        "split": "train",
        "v3_F": True,
        "R": True,
        "subject": "wd:Q1",
        "predicate": "wd:P1",
        "object": "wd:Q2",
        "other_subject": "wd:Q3",
        "other_predicate": "wd:P2",
        "other_object": "wd:Q4",
        "add_subject": "wd:Q1",
        "add_predicate": "wd:P9",
        "add_object": "wd:Q9",
        "del_subject": "",
        "del_predicate": "",
        "del_object": "",
        "target_role_masks": [1, 0, 0, 0, 0, 0],
        "target_encoder_ids": [10, 19, 29, 0, 0, 0],
    }
    value.update(updates)
    return value


def build_vocab(rows):
    builder = VerifiedVocabularyBuilder(ROLE_IDS, input_encoder_size=1000)
    accepted = [builder.add_row(value) for value in rows]
    return builder, builder.build(), accepted


def test_verified_vocab_uses_only_train_and_fixed_f() -> None:
    builder, vocab, accepted = build_vocab(
        [
            row(),
            row(split="val", add_object="wd:QVAL", target_encoder_ids=[10, 19, 31, 0, 0, 0]),
            row(split="test", add_object="wd:QTEST", target_encoder_ids=[10, 19, 32, 0, 0, 0]),
            row(v3_F=False, add_object="wd:QNONF", target_encoder_ids=[10, 19, 33, 0, 0, 0]),
        ]
    )
    assert accepted == [True, False, False, False]
    assert builder.fit_rows == 1
    assert "wd:Q9" in vocab["entity_terms"]
    assert not {"wd:QVAL", "wd:QTEST", "wd:QNONF"} & set(vocab["entity_terms"])


def test_validation_and_test_targets_cannot_expand_v1() -> None:
    _, baseline, _ = build_vocab([row()])
    _, with_heldout, _ = build_vocab(
        [
            row(),
            row(split="val", add_predicate="wd:P777", target_encoder_ids=[10, 777, 29, 0, 0, 0]),
            row(split="test", add_predicate="wd:P888", target_encoder_ids=[10, 888, 29, 0, 0, 0]),
        ]
    )
    assert with_heldout["fingerprint"] == baseline["fingerprint"]
    assert with_heldout["predicate_terms"] == baseline["predicate_terms"]


def test_semantic_role_aliases_precede_constants_consistently() -> None:
    aliased = row(
        object="wd:Q1",
        add_subject="wd:Q1",
        target_role_masks=[(1 << 0) | (1 << 2), 0, 0, 0, 0, 0],
        target_encoder_ids=[777, 19, 29, 0, 0, 0],
    )
    builder, vocab, _ = build_vocab([aliased])
    assert builder.raw_support[(0, "role", "subject")] == 1
    assert "wd:Q1" not in vocab["entity_terms"]
    details = edit_coverage(aliased, vocab)
    assert details["covered"]
    assert details["references"][0]["kind"] == "role"
    assert details["references"][0]["reference"] == "subject"


def test_every_claimed_v1_edit_roundtrips_exactly() -> None:
    training = row()
    _, vocab, _ = build_vocab([training])
    assert edit_coverage(training, vocab)["covered"]
    heldout_same_terms = row(split="test")
    assert edit_coverage(heldout_same_terms, vocab)["covered"]
    heldout_new_term = row(
        split="test", add_object="wd:Q404", target_encoder_ids=[10, 19, 404, 0, 0, 0]
    )
    assert not edit_coverage(heldout_new_term, vocab)["covered"]


def test_v1_coverage_is_deterministic_under_training_row_order() -> None:
    rows = [
        row(),
        row(add_predicate="wd:P8", target_encoder_ids=[10, 18, 29, 0, 0, 0]),
    ]
    _, first, _ = build_vocab(rows)
    _, second, _ = build_vocab(list(reversed(rows)))
    assert first["fingerprint"] == second["fingerprint"]
    assert edit_coverage(row(), first) == edit_coverage(row(), second)


def test_v1_coverage_never_uses_neural_unk_as_identity() -> None:
    training = row()
    _, vocab, _ = build_vocab([training])
    heldout = row(split="test", add_object="wd:QDIFFERENT")
    first = deepcopy(heldout)
    second = deepcopy(heldout)
    first["target_encoder_ids"] = [10, 19, 0, 0, 0, 0]
    second["target_encoder_ids"] = [10, 19, 0, 0, 0, 0]
    assert first["add_object"] != training["add_object"]
    assert not edit_coverage(first, vocab)["covered"]
    assert edit_coverage(first, vocab) == edit_coverage(second, vocab)


def test_validator_f_flag_is_not_recomputed_from_decoder_fields() -> None:
    non_f = row(v3_F=False, v3_pre_outcome="violated", v3_post_outcome="satisfied")
    builder, vocab, accepted = build_vocab([non_f])
    assert accepted == [False]
    assert builder.fit_rows == 0
    assert vocab["entity_terms"] == []
    assert vocab["predicate_terms"] == []


def test_action_categories_are_mutually_exclusive_and_reconcile() -> None:
    fixtures = [
        row(),
        row(
            add_subject="",
            add_predicate="",
            add_object="",
            del_subject="wd:Q1",
            del_predicate="wd:P1",
            del_object="wd:Q2",
        ),
        row(
            add_subject="wd:Q1",
            add_predicate="wd:P1",
            add_object="wd:Q8",
            del_subject="wd:Q1",
            del_predicate="wd:P1",
            del_object="wd:Q2",
        ),
        row(
            add_subject="wd:Q3",
            add_predicate="wd:P2",
            add_object="wd:Q8",
            del_subject="wd:Q1",
            del_predicate="wd:P1",
            del_object="wd:Q2",
        ),
    ]
    categories = [action_features(value)["category"] for value in fixtures]
    assert tuple(categories) == ACTION_CATEGORIES
    assert len(categories) == sum(categories.count(category) for category in ACTION_CATEGORIES)
    assert action_features(fixtures[1])["base_deleting"]
    assert not action_features(fixtures[2])["base_ultimately_preserved"]


def test_v1_support_counts_equal_raw_training_target_occurrences() -> None:
    builder, _, _ = build_vocab([row(), row()])
    assert builder.raw_support[(0, "role", "subject")] == 2
    assert builder.raw_support[(1, "constant", "wd:P9")] == 2
    assert builder.raw_support[(2, "constant", "wd:Q9")] == 2
    assert builder.raw_support[(3, "none", "NONE")] == 2
    assert sum(builder.raw_support.values()) == 12


def test_v1_subset_and_budget_helpers_create_no_selected_row_list() -> None:
    _, v1, _ = build_vocab([row()])
    v0 = deepcopy(v1)
    v0["entity_terms"] = sorted(set(v0["entity_terms"]) | {"wd:Qextra"})
    assert_v1_subset_v0(v1, v0)
    with pytest.raises(AssertionError):
        assert_v1_subset_v0(v0, v1)

    allocation = largest_remainder_allocation({"a": 8, "b": 2}, 5)
    assert allocation == {"a": 4, "b": 1}
    capped = capped_target_allocation({"a": 4, "b": 1}, {"a": 3, "b": 4}, 5)
    assert sum(capped.values()) == 5
    assert all(isinstance(value, int) for value in capped.values())
    assert set(capped) == {"a", "b"}
    assert js_divergence({"a": 1}, {"a": 1}) == 0.0
