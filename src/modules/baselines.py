# src/modules/baselines.py
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from modules.model_store import baseline_dir
from modules.models import BaseGraphModel

logger = logging.getLogger(__name__)

BASELINE_NAMES: Tuple[str, ...] = (
    "DeleteFocusBaseline",
    "AddMirrorBaseline",
    "ConstraintDefinitionMajorityBaseline",
    "ConstraintFamilyMajorityBaseline",
)


# ----------------------------
# Utilities
# ----------------------------


def _extract_focus_triple(d: Data) -> Tuple[int, int, int]:
    """
    Extract (s, p, o) of the *violating* triple from a PyG Data object.

    Required fields (any ONE of these is sufficient):
      - d.focus_triple: Tensor [3] or Sequence[int]
      - d.violation_triple: Tensor [3] or Sequence[int]
      - (d.focus_s, d.focus_p, d.focus_o) as ints
      - (d.viol_s, d.viol_p, d.viol_o) as ints

    Raises ValueError with a helpful message if not found.
    """
    for attr in ("focus_triple", "violation_triple"):
        if hasattr(d, attr):
            t = getattr(d, attr)
            if isinstance(t, torch.Tensor):
                t = t.tolist()
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                raise ValueError(f"{attr} must be length-3, got: {t}")
            return int(t[0]), int(t[1]), int(t[2])

    combos = [
        ("focus_s", "focus_p", "focus_o"),
        ("viol_s", "viol_p", "viol_o"),
    ]
    for s_attr, p_attr, o_attr in combos:
        if all(hasattr(d, x) for x in (s_attr, p_attr, o_attr)):
            return int(getattr(d, s_attr)), int(getattr(d, p_attr)), int(getattr(d, o_attr))

    raise ValueError(
        "Cannot extract violating triple. Provide one of: "
        "focus_triple / violation_triple (len-3), or focus_s/focus_p/focus_o, or viol_s/viol_p/viol_o."
    )


def _extract_shape_id(d: Data) -> Optional[int]:
    """
    Extract a stable ID for the constraint type (for majority baseline).
    Accepted names: shape_id, constraint_shape_id.
    Returns None if missing.
    """
    for name in ("shape_id", "constraint_shape_id"):
        if hasattr(d, name):
            return int(getattr(d, name))
    return None


def _extract_constraint_type(d: Data) -> Optional[str]:
    """
    (Optional) Extract a coarse constraint type string such as 'inverse', 'symmetric', etc.
    Accepted names: constraint_type, constraint_kind, shape_kind.
    Returns None if missing.
    """
    for name in ("constraint_type", "constraint_kind", "shape_kind"):
        if hasattr(d, name):
            return str(getattr(d, name))
    return None


# ----------------------------
# Baseline API
# ----------------------------


class Baseline(ABC):
    """
    Stateless (or lightly stateful) baseline API.
    Predicts indices per head in shape (6,) for one example, or (B,6) for a list.
    """

    def __init__(
        self,
        num_graph_nodes: int,
        default_add_class: int = 0,
        default_del_class: int = 0,
        *,
        placeholders: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.num_graph_nodes = int(num_graph_nodes)
        self.default_add_class = int(default_add_class)
        self.default_del_class = int(default_del_class)
        # Cache of placeholder ids such as "subject" or "object" so that
        # predictions align with the encoded training targets written by
        # ``02_dataframe_builder``.
        self._placeholders: dict[str, int] = {}
        if placeholders:
            for key, value in placeholders.items():
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring non-integer placeholder %s=%r", key, value)
                    continue
                if idx < 0 or idx >= self.num_graph_nodes:
                    logger.debug(
                        "Ignoring placeholder %s=%d (out of bounds for %d nodes)",
                        key,
                        idx,
                        self.num_graph_nodes,
                    )
                    continue
                self._placeholders[key] = idx

        # simple sanity
        for v in (self.default_add_class, self.default_del_class):
            if not (0 <= v < self.num_graph_nodes):
                raise ValueError(f"default class {v} must be in [0, {self.num_graph_nodes})")

    def fit(self, train_data: Optional[Iterable[Data]] = None) -> None:
        """Optional: some baselines learn priors from training data."""
        return

    @abstractmethod
    def predict_one(self, d: Data) -> torch.Tensor:
        """
        Returns Tensor of shape (6,) with indices for
        [add_s, add_p, add_o, del_s, del_p, del_o].
        """
        ...

    def predict_batch(self, batch: Sequence[Data]) -> torch.Tensor:
        with torch.no_grad():
            preds = [self.predict_one(d) for d in batch]
        return torch.stack(preds)

    def _placeholder(self, name: str, fallback: int) -> int:
        """Return placeholder id *name* if known, otherwise *fallback*."""
        return self._placeholders.get(name, fallback)


# ----------------------------
# (1) Delete-Focus Baseline (DFB)
# ----------------------------


class DeleteFocusBaseline(Baseline):
    """
    For any violation, remove the violation's focus triple.
    Is a strong floor for “single value / conflict with” styles.
    'Add' heads default to a no-op class (configurable).
    """

    def predict_one(self, d: Data) -> torch.Tensor:
        s, p, o = _extract_focus_triple(d)
        return torch.tensor(
            [
                self.default_add_class,
                self.default_add_class,
                self.default_add_class,
                self._placeholder("subject", s),
                self._placeholder("predicate", p),
                self._placeholder("object", o),
            ],
            dtype=torch.long,
        )


# ----------------------------
# (2) Add-Mirror Baseline (AMB)
# ----------------------------


class AddMirrorBaseline(Baseline):
    """
    For inverse/symmetric constraints, add the mirror triple.
    - If predicate is symmetric: add (o, p, s)
    - Else if predicate has a known inverse p_inv: add (o, p_inv, s)
    - Else (unknown): fall back to (o, p, s) which is correct for symmetric
      and a heuristic for inverse when mapping is missing.

    By default, only fires if the sample is marked as inverse/symmetric via
    `constraint_type` / `shape_kind`. Set `only_if_marked=False` to always attempt.
    """

    def __init__(
        self,
        num_graph_nodes: int,
        inverse_predicates: Optional[Dict[int, int]] = None,
        symmetric_predicates: Optional[Iterable[int]] = None,
        *,
        only_if_marked: bool = True,
        default_add_class: int = 0,
        default_del_class: int = 0,
        placeholders: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__(
            num_graph_nodes,
            default_add_class,
            default_del_class,
            placeholders=placeholders,
        )
        self.inverse_predicates = dict(inverse_predicates or {})
        self.symmetric_predicates = set(int(x) for x in (symmetric_predicates or []))
        self.only_if_marked = bool(only_if_marked)

    def predict_one(self, d: Data) -> torch.Tensor:
        s, p, o = _extract_focus_triple(d)
        kind = _extract_constraint_type(d)

        should_fire = not self.only_if_marked or (kind in {"inverse", "symmetric"})
        if should_fire:
            if p in self.symmetric_predicates:
                p_m = p
            elif p in self.inverse_predicates:
                p_m = self.inverse_predicates[p]
            else:
                # Unknown: heuristic fallback
                p_m = p

            add_s = self._placeholder("object", o)
            add_p = self._placeholder("predicate", p_m) if p_m == p else p_m
            add_o = self._placeholder("subject", s)
        else:
            add_s = add_p = add_o = self.default_add_class

        return torch.tensor(
            [add_s, add_p, add_o, self.default_del_class, self.default_del_class, self.default_del_class],
            dtype=torch.long,
        )


# ----------------------------
# Majority baselines
# ----------------------------


class _PerKeyMajorityBaseline(Baseline):
    """
    Learns modal repair slots from training data for keys extracted by subclasses.

    The majority is computed per output head. This mirrors the historical
    slot-argmax evaluation setup used by the learned models.
    """

    def __init__(
        self,
        num_graph_nodes: int,
        *,
        default_add_class: int = 0,
        default_del_class: int = 0,
        placeholders: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__(
            num_graph_nodes,
            default_add_class,
            default_del_class,
            placeholders=placeholders,
        )
        self.key_to_majority: Dict[Any, List[int]] = {}
        self.global_majority: List[int] = [self.default_add_class] * 3 + [self.default_del_class] * 3

    @abstractmethod
    def _extract_key(self, d: Data) -> Any | None:
        ...

    @property
    def _display_name(self) -> str:
        return type(self).__name__

    def fit(self, train_data: Optional[Iterable[Data]] = None) -> None:
        if train_data is None:
            raise ValueError(f"{self._display_name}.fit(train_data=...) requires training data.")

        global_counters = [Counter() for _ in range(6)]
        key_counters: Dict[Any, List[Counter]] = defaultdict(lambda: [Counter() for _ in range(6)])

        n_used = 0
        for d in train_data:
            key = self._extract_key(d)
            y = getattr(d, "y", None)
            if key is None or y is None:
                continue
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().view(-1).tolist()
            if not (isinstance(y, (list, tuple)) and len(y) == 6):
                continue

            n_used += 1
            for i, cls in enumerate(y):
                cls = int(cls)
                if 0 <= cls < self.num_graph_nodes:
                    global_counters[i][cls] += 1
                    key_counters[key][i][cls] += 1

        if n_used == 0:
            return

        def argmax_counter(c: Counter, fallback: int) -> int:
            if not c:
                return fallback
            return max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        fallbacks = [self.default_add_class] * 3 + [self.default_del_class] * 3
        self.global_majority = [argmax_counter(global_counters[i], fallbacks[i]) for i in range(6)]

        for key, head_counters in key_counters.items():
            self.key_to_majority[key] = [
                argmax_counter(head_counters[i], self.global_majority[i]) for i in range(6)
            ]

    def predict_one(self, d: Data) -> torch.Tensor:
        key = self._extract_key(d)
        if key is not None and key in self.key_to_majority:
            y = self.key_to_majority[key]
        else:
            y = self.global_majority
        return torch.tensor(y, dtype=torch.long)


class ConstraintDefinitionMajorityBaseline(_PerKeyMajorityBaseline):
    """
    Learns, from training data, the modal repair per concrete constraint definition.

    This is an IID/memorization baseline: in the parquet loader, ``shape_id`` is
    the concrete ``constraint_id``.
    Requires per-sample:
      - shape_id or constraint_shape_id (int)
      - y labels present in training Data (Tensor [6])
    """

    def _extract_key(self, d: Data) -> int | None:
        return _extract_shape_id(d)


class ConstraintFamilyMajorityBaseline(_PerKeyMajorityBaseline):
    """
    Learns, from training data, the modal repair per coarse constraint family.

    This is the true shape/family majority baseline: keys are values such as
    ``inverse``, ``single``, or ``valueType``, not concrete constraint IDs.
    Requires per-sample:
      - constraint_type / constraint_kind / shape_kind (str)
      - y labels present in training Data (Tensor [6])
    """

    def _extract_key(self, d: Data) -> str | None:
        return _extract_constraint_type(d)


# ----------------------------
# Adapter: make any Baseline look like a nn.Module that returns logits
# ----------------------------


class BaselineAdapter(BaseGraphModel):
    """
    Wraps a Baseline so it exposes forward(data)->logits of shape (B, 6, num_graph_nodes),
    with one-hot-like logits: chosen index has 0.0, all others a large negative number.
    This lets you reuse the existing eval() that calls .argmax().
    """

    def __init__(self, baseline: Baseline) -> None:
        super().__init__(
            baseline.num_graph_nodes,
            num_embedding_size=1,
            num_layers=1,
            hidden=1,
            use_node_embeddings=False,
        )
        self.baseline = baseline

    def create_conv_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module: ...

    @property
    def num_graph_nodes(self) -> int:
        return self.num_input_graph_nodes

    def forward(self, data) -> torch.Tensor:
        # data is a batched PyG Data; split into a list of graph-level Data objects.
        if hasattr(data, "to_data_list"):
            dlist = data.to_data_list()
        elif isinstance(data, list):
            dlist = data
        else:
            dlist = [data]

        idx = self.baseline.predict_batch(dlist)  # (B, 6)
        assert idx.ndim == 2 and idx.shape[1] == 6, f"Baseline must return (B,6) indices, got {idx.shape}"

        B = idx.shape[0]
        device = next(self.parameters()).device
        idx = idx.to(device)
        logits = torch.full((B, 6, self.num_graph_nodes), fill_value=-1e9, device=device)
        logits.scatter_(2, idx.unsqueeze(-1), 0.0)
        return logits

    def forward_for_evaluation(self, data) -> torch.Tensor:
        """Evaluate a baseline without factor-model-only forward arguments."""

        return self.forward(data)


def _maybe_load_json(path: Optional[str]) -> Optional[Any]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_cli_baseline(
    name: str,
    *,
    num_graph_nodes: int,
    default_add_class: int,
    default_del_class: int,
    inverse_map: Dict[int, int],
    symmetric_set: Iterable[int],
    train_data: Optional[Iterable[Data]],
    fit_csm_on_train: bool,
    placeholders: Optional[Mapping[str, int]],
) -> Baseline:
    if name == "DeleteFocusBaseline":
        return DeleteFocusBaseline(
            num_graph_nodes=num_graph_nodes,
            default_add_class=default_add_class,
            default_del_class=default_del_class,
            placeholders=placeholders,
        )
    if name == "AddMirrorBaseline":
        return AddMirrorBaseline(
            num_graph_nodes=num_graph_nodes,
            inverse_predicates=inverse_map,
            symmetric_predicates=symmetric_set,
            only_if_marked=True,
            default_add_class=default_add_class,
            default_del_class=default_del_class,
            placeholders=placeholders,
        )
    if name == "ConstraintDefinitionMajorityBaseline":
        definition_majority = ConstraintDefinitionMajorityBaseline(
            num_graph_nodes=num_graph_nodes,
            default_add_class=default_add_class,
            default_del_class=default_del_class,
            placeholders=placeholders,
        )
        if fit_csm_on_train and train_data is not None:
            definition_majority.fit(train_data)
        return definition_majority
    if name == "ConstraintFamilyMajorityBaseline":
        family_majority = ConstraintFamilyMajorityBaseline(
            num_graph_nodes=num_graph_nodes,
            default_add_class=default_add_class,
            default_del_class=default_del_class,
            placeholders=placeholders,
        )
        if fit_csm_on_train and train_data is not None:
            family_majority.fit(train_data)
        return family_majority
    raise ValueError(f"Unknown baseline: {name}")


def evaluate_baselines(
    *,
    baseline_choice: str,
    dataset: str,
    encoding: str,
    num_graph_nodes: int,
    default_add_class: int,
    default_del_class: int,
    inverse_map_json: Optional[str],
    symmetric_set_json: Optional[str],
    fit_csm_on_train: bool,
    train_data: Optional[Iterable[Data]],
    device: torch.device,
    save_run: Callable[[str, BaseGraphModel], Dict[str, float]],
    results_dir: Path | None = None,
    placeholders: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    if baseline_choice != "all" and baseline_choice not in BASELINE_NAMES:
        allowed_display = ", ".join(["'all'"] + [f"'{name}'" for name in BASELINE_NAMES])
        raise ValueError(f"baseline_choice must be one of {{{allowed_display}}}; got {baseline_choice!r}")

    inverse_map_raw = _maybe_load_json(inverse_map_json) or {}
    symmetric_raw = _maybe_load_json(symmetric_set_json) or []

    inverse_map = {int(k): int(v) for k, v in inverse_map_raw.items()}
    symmetric_set = [int(x) for x in symmetric_raw]

    names = list(BASELINE_NAMES) if baseline_choice == "all" else [baseline_choice]
    aggregated: Dict[str, Dict[str, float]] = {}

    target_dir = results_dir
    if target_dir is None:
        target_dir = baseline_dir(dataset, encoding, create=True)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        baseline = _build_cli_baseline(
            name,
            num_graph_nodes=num_graph_nodes,
            default_add_class=default_add_class,
            default_del_class=default_del_class,
            inverse_map=inverse_map,
            symmetric_set=symmetric_set,
            train_data=train_data,
            fit_csm_on_train=fit_csm_on_train,
            placeholders=placeholders,
        )
        model = BaselineAdapter(baseline).to(device)
        aggregated[name] = save_run(f"baseline-{name}", model)

    if baseline_choice == "all":
        target_dir.mkdir(parents=True, exist_ok=True)
        agg_path = target_dir / "baselines_all.json"
        with agg_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, indent=4)
        logger.info("Wrote aggregate baseline results to %s", agg_path)

    return aggregated
