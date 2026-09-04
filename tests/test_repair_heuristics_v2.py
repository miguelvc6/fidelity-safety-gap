from __future__ import annotations

from modules.repair_eval import ConstraintRepairHeuristics, ViolationContext


TOKENS = {
    "<http://www.wikidata.org/entity/P31>": 31,
    "<http://www.wikidata.org/entity/P279>": 279,
    "<http://www.wikidata.org/entity/P2305>": 2305,
    "<http://www.wikidata.org/entity/P2306>": 2306,
    "<http://www.wikidata.org/entity/P2308>": 2308,
    "<http://www.wikidata.org/entity/P2309>": 2309,
    "<http://www.wikidata.org/entity/Q5>": 5,
    "<http://www.wikidata.org/entity/Q7>": 7,
    "<http://www.wikidata.org/entity/Q21503252>": 1001,
    "<http://www.wikidata.org/entity/Q21514624>": 1002,
    "<http://www.wikidata.org/entity/Q30208840>": 1003,
}


class Encoder:
    def encode(self, token, *, add_new=False):
        assert not add_new
        return TOKENS.get(token, 0)


def context(family: str, parameters: list[tuple[int, int]]) -> ViolationContext:
    return ViolationContext(
        constraint_type=family,
        constraint_id=900,
        subject=1,
        predicate=10,
        object=5,
        other_subject=0,
        other_predicate=0,
        other_object=0,
        constraint_predicates=tuple(predicate for predicate, _ in parameters),
        constraint_objects=tuple(value for _, value in parameters),
    )


def heuristics() -> ConstraintRepairHeuristics:
    return ConstraintRepairHeuristics(
        encoder=Encoder(),
        placeholder_ids={"subject": 101, "predicate": 110, "object": 105},
        none_class=0,
    )


def test_type_candidates_map_selector_items_to_relation_predicates() -> None:
    instance = context("type", [(2308, 7), (2309, 1002)])
    additions = heuristics().candidates_for(instance).add
    assert additions
    assert {predicate for pattern in additions for predicate in pattern.predicates or ()} == {279}
    assert all(2309 not in (pattern.predicates or ()) for pattern in additions)


def test_inverse_and_required_candidates_use_p2306_and_optional_p2305() -> None:
    inverse = heuristics().candidates_for(context("inverse", [(2306, 20)])).add
    assert any(pattern.predicates == frozenset({20}) for pattern in inverse)

    required = heuristics().candidates_for(
        context("itemRequiresStatement", [(2306, 20), (2305, 7)])
    ).add
    assert any(
        pattern.predicates == frozenset({20}) and pattern.objects == frozenset({7})
        for pattern in required
    )
