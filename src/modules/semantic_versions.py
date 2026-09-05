"""Version identifiers for validator-adjacent semantic contracts."""

EFFECTIVE_HIERARCHY_POLICY_VERSION = 1
CANDIDATE_OBJECTIVE_VERSION = 2


def semantic_contract_versions() -> dict[str, int]:
    return {
        "effective_hierarchy_policy_version": EFFECTIVE_HIERARCHY_POLICY_VERSION,
        "candidate_objective_version": CANDIDATE_OBJECTIVE_VERSION,
    }
