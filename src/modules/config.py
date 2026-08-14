import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping


def _filter_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(data) - valid_fields
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {unknown_list}")
    return {key: data[key] for key in valid_fields if key in data}


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Configuration file at {path} must contain an object at the top level.")
    return loaded


def _normalize_class_ids(value: Any) -> tuple[int, ...] | None:
    """Return ``value`` as a tuple[int, ...] if provided, otherwise ``None``."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, (list, set, frozenset)):
        return tuple(int(v) for v in value)
    if isinstance(value, (int,)):
        return (int(value),)
    if hasattr(value, "tolist"):
        return tuple(int(v) for v in value.tolist())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(int(v) for v in value)
    raise TypeError(f"Expected iterable of ints for class ids, got {type(value)!r}")


@dataclass
class DynamicReweightingConfig:
    enabled: bool = False  # Toggle dynamic per-constraint loss weighting.
    target_metrics: tuple[str, ...] = ("loss",)  # Validation metrics used to derive difficulty.
    update_frequency: str = "epoch"  # Either "epoch" (default) or "batch".
    scale: float = 1.0  # Strength of the reweighting relative to uniform weights.
    min_weight: float = 0.5  # Lower clamp for generated weights.
    max_weight: float = 3.0  # Upper clamp for generated weights.
    smoothing: float = 0.2  # Interpolation factor toward previous weights (0 = overwrite).

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DynamicReweightingConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "DynamicReweightingConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "target_metrics" in filtered:
            value = filtered["target_metrics"]
            if isinstance(value, str):
                filtered["target_metrics"] = (value,)
            else:
                filtered["target_metrics"] = tuple(str(v) for v in value)

        if "update_frequency" in filtered:
            freq = str(filtered["update_frequency"]).lower()
            if freq not in {"epoch", "batch"}:
                raise ValueError("DynamicReweightingConfig.update_frequency must be 'epoch' or 'batch'")
            filtered["update_frequency"] = freq

        for float_field in ("scale", "min_weight", "max_weight", "smoothing"):
            if float_field in filtered and filtered[float_field] is not None:
                filtered[float_field] = float(filtered[float_field])

        if "smoothing" in filtered:
            smoothing_value = filtered["smoothing"]
            if not 0.0 <= smoothing_value <= 1.0:
                raise ValueError("DynamicReweightingConfig.smoothing must be between 0 and 1 inclusive")

        if "min_weight" in filtered or "max_weight" in filtered:
            min_weight = filtered.get("min_weight", self.min_weight)
            max_weight = filtered.get("max_weight", self.max_weight)
            if max_weight < min_weight:
                raise ValueError("DynamicReweightingConfig.max_weight must be >= min_weight")

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_metrics": list(self.target_metrics),
            "update_frequency": self.update_frequency,
            "scale": self.scale,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "smoothing": self.smoothing,
        }


@dataclass
class ConstraintLossConfig:
    dynamic_reweighting: DynamicReweightingConfig = field(default_factory=DynamicReweightingConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ConstraintLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "ConstraintLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        dynamic_payload = filtered.pop("dynamic_reweighting", None)
        current.update(filtered)

        if dynamic_payload is not None:
            if isinstance(dynamic_payload, DynamicReweightingConfig):
                current["dynamic_reweighting"] = dynamic_payload
            else:
                current["dynamic_reweighting"] = self.dynamic_reweighting.updated(dynamic_payload)

        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamic_reweighting": self.dynamic_reweighting.to_dict(),
        }


@dataclass
class FixProbabilityLossConfig:
    enabled: bool = False  # Toggle the fix-aware loss term.
    initial_weight: float = 0.5  # Weight at the start (after warmup).
    final_weight: float = 0.05  # Asymptotic weight once decay finishes.
    decay_epochs: float = 40.0  # Time constant (exponential) or span (linear).
    warmup_epochs: float = 0.0  # Epochs to hold the initial weight before decay.
    schedule: str = "exponential"  # 'exponential' (default) or 'linear'.

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FixProbabilityLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "FixProbabilityLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        for key in ("initial_weight", "final_weight", "decay_epochs", "warmup_epochs"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = float(filtered[key])

        if "schedule" in filtered and filtered["schedule"] is not None:
            value = str(filtered["schedule"]).lower()
            if value not in {"exponential", "linear"}:
                raise ValueError("FixProbabilityLossConfig.schedule must be 'exponential' or 'linear'")
            filtered["schedule"] = value

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "initial_weight": self.initial_weight,
            "final_weight": self.final_weight,
            "decay_epochs": self.decay_epochs,
            "warmup_epochs": self.warmup_epochs,
            "schedule": self.schedule,
        }


@dataclass
class FactorLossConfig:
    enabled: bool = False
    weight_pre: float = 0.1
    weight_post_gold: float = 0.1
    pos_weight: float | None = None
    only_checkable: bool = True
    per_graph_reduction: str = "mean"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "FactorLossConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "FactorLossConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "weight_pre" in filtered and filtered["weight_pre"] is not None:
            filtered["weight_pre"] = float(filtered["weight_pre"])
        if "weight_post_gold" in filtered and filtered["weight_post_gold"] is not None:
            filtered["weight_post_gold"] = float(filtered["weight_post_gold"])
        if "pos_weight" in filtered and filtered["pos_weight"] is not None:
            filtered["pos_weight"] = float(filtered["pos_weight"])
        if "only_checkable" in filtered and filtered["only_checkable"] is not None:
            filtered["only_checkable"] = bool(filtered["only_checkable"])
        if "per_graph_reduction" in filtered and filtered["per_graph_reduction"] is not None:
            value = str(filtered["per_graph_reduction"]).lower()
            if value not in {"mean", "sum"}:
                raise ValueError("FactorLossConfig.per_graph_reduction must be 'mean' or 'sum'")
            filtered["per_graph_reduction"] = value

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight_pre": self.weight_pre,
            "weight_post_gold": self.weight_post_gold,
            "pos_weight": self.pos_weight,
            "only_checkable": self.only_checkable,
            "per_graph_reduction": self.per_graph_reduction,
        }


@dataclass
class ChooserConfig:
    enabled: bool = False
    topk_candidates: int = 20
    max_candidates_total: int = 80
    beta_no_regression: float = 0.5
    gamma_primary: float = 0.0
    loss_weight: float = 0.25
    loss_mode: str = "fix1"  # fix1 | primary_only | global_fix

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ChooserConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "ChooserConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "enabled" in filtered and filtered["enabled"] is not None:
            filtered["enabled"] = bool(filtered["enabled"])
        for key in ("topk_candidates", "max_candidates_total"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = int(filtered[key])
        for key in ("beta_no_regression", "gamma_primary", "loss_weight"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = float(filtered[key])
        if "loss_mode" in filtered and filtered["loss_mode"] is not None:
            value = str(filtered["loss_mode"]).lower()
            if value not in {"fix1", "primary_only", "global_fix"}:
                raise ValueError("ChooserConfig.loss_mode must be 'fix1', 'primary_only', or 'global_fix'")
            filtered["loss_mode"] = value

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "topk_candidates": self.topk_candidates,
            "max_candidates_total": self.max_candidates_total,
            "beta_no_regression": self.beta_no_regression,
            "gamma_primary": self.gamma_primary,
            "loss_weight": self.loss_weight,
            "loss_mode": self.loss_mode,
        }


@dataclass
class DirectSafetyConfig:
    enabled: bool = False
    alpha_primary: float = 1.0
    beta_secondary: float = 0.5
    topk_candidates: int = 20
    max_candidates_total: int = 80

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DirectSafetyConfig":
        instance = cls()
        return instance.updated(data or {})

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "DirectSafetyConfig":
        payload = dict(data or {})
        payload.update(overrides)
        filtered = _filter_fields(type(self), payload)

        if "enabled" in filtered and filtered["enabled"] is not None:
            filtered["enabled"] = bool(filtered["enabled"])
        for key in ("alpha_primary", "beta_secondary"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = float(filtered[key])
        for key in ("topk_candidates", "max_candidates_total"):
            if key in filtered and filtered[key] is not None:
                filtered[key] = int(filtered[key])

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        current.update(filtered)
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha_primary": self.alpha_primary,
            "beta_secondary": self.beta_secondary,
            "topk_candidates": self.topk_candidates,
            "max_candidates_total": self.max_candidates_total,
        }


@dataclass
class ModelConfig:
    dataset_variant: str = "full" 
    """Which intermediate dataset variant to consume."""
    encoding: str = "text_embedding"
    """Node feature encoding that selects graph files."""
    model: str = "GIN"
    """Identifier for the GNN architecture to instantiate."""
    min_occurrence: int = 100
    """Frequency threshold used when building the dataset."""
    num_embedding_size: int = 128  
    """Width of learned embeddings for integer node ids."""
    num_layers: int = 2  
    """Number of message-passing layers in the backbone."""
    hidden_channels: int = 128  
    """Channel size inside the message-passing stack."""
    head_hidden: int = 128
    """Hidden width shared by the prediction heads."""
    dropout: float = 0.5  
    """Dropout probability applied to head activations."""
    use_node_embeddings: bool = True  
    """Toggle between embedding integer ids or passing features through."""
    use_role_embeddings: bool = False  
    """Whether to append learned focus-role embeddings to node features."""
    role_embedding_dim: int = 8  
    """Dimensionality of each learned role embedding vector."""
    num_role_types: int = 4  
    """Number of distinct role ids expected in role_flags tensors."""
    use_edge_attributes: bool = False 
    """Whether to use edge attributes, instead of treating edges as nodes."""
    use_edge_subtraction: bool = False 
    """Whether to use edge subtraction, which requires use_edge_attributes to be True."""
    entity_class_ids: tuple[int, ...] | None = None  
    """Optional vocabulary subset for entity targets."""
    predicate_class_ids: tuple[int, ...] | None = None  
    """Optional vocabulary subset for predicate targets."""
    num_factor_types: int = 0
    """Upper bound of the stable factor-type id address space."""
    active_factor_type_ids: tuple[int, ...] | None = None
    """Stable factor ids allocated by compact executor implementations."""
    factor_type_embedding_dim: int = 8
    """Embedding dim for factor type conditioning."""
    pressure_enabled: bool = False
    """Toggle factor pressure injection during message passing."""
    pressure_type_conditioning: str = "none"
    """Condition pressure messages on factor types: none|concat|gate."""
    pressure_module_sharing: str = "per_type"
    """Share factor pressure modules: per_type|shared."""
    pressure_residual_scale: float = 0.1
    """Scale applied to degree-normalized pressure residual messages."""
    enable_policy_choice: bool = False
    """Enable policy choice head over graph embeddings."""
    policy_num_classes: int = 6
    """Number of policy classes for policy choice head."""
    constraint_representation: str = "factorized"
    """Graph representation regime: factorized or eswc_passive."""
    factor_executor_impl: str = "per_type_v1"
    """Factor executor implementation: per_type_v1, per_type_grouped_v2, or legacy_shared."""
    gold_edit_embedding_mode: str = "full"
    """Gold-edit embedding storage: full or compact."""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls().updated(data)

    @classmethod
    def from_path(cls, path: Path | None) -> "ModelConfig":
        if path is None:
            return cls()
        return cls.from_mapping(_load_mapping(path))

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "ModelConfig":
        """Update the configuration with values from ``data`` and ``overrides``."""
        data_dict = dict(data or {})
        if "hidden" in data_dict and "hidden_channels" not in data_dict:
            data_dict["hidden_channels"] = data_dict.pop("hidden")
        filtered = _filter_fields(type(self), data_dict)
        filtered.update({k: v for k, v in overrides.items() if v is not None})

        for field_name in ("entity_class_ids", "predicate_class_ids", "active_factor_type_ids"):
            if field_name in filtered:
                filtered[field_name] = _normalize_class_ids(filtered[field_name])

        if "role_embedding_dim" in filtered and filtered["role_embedding_dim"] is not None:
            filtered["role_embedding_dim"] = int(filtered["role_embedding_dim"])
        if "num_role_types" in filtered and filtered["num_role_types"] is not None:
            filtered["num_role_types"] = int(filtered["num_role_types"])
        if "use_role_embeddings" in filtered and filtered["use_role_embeddings"] is not None:
            filtered["use_role_embeddings"] = bool(filtered["use_role_embeddings"])
        if "num_factor_types" in filtered and filtered["num_factor_types"] is not None:
            filtered["num_factor_types"] = int(filtered["num_factor_types"])
        if "active_factor_type_ids" in filtered and filtered["active_factor_type_ids"] is not None:
            active_ids = tuple(int(value) for value in filtered["active_factor_type_ids"])
            if not active_ids:
                raise ValueError("active_factor_type_ids must not be empty when provided")
            if tuple(sorted(active_ids)) != active_ids:
                raise ValueError("active_factor_type_ids must be strictly increasing")
            if len(set(active_ids)) != len(active_ids):
                raise ValueError("active_factor_type_ids must not contain duplicates")
            if active_ids[0] < 0:
                raise ValueError("active_factor_type_ids must be non-negative")
            address_space = int(filtered.get("num_factor_types", self.num_factor_types))
            if address_space <= 0 or active_ids[-1] >= address_space:
                raise ValueError(
                    "active_factor_type_ids must fit inside num_factor_types stable id address space"
                )
            filtered["active_factor_type_ids"] = active_ids
        if "factor_type_embedding_dim" in filtered and filtered["factor_type_embedding_dim"] is not None:
            filtered["factor_type_embedding_dim"] = int(filtered["factor_type_embedding_dim"])
        if "pressure_enabled" in filtered and filtered["pressure_enabled"] is not None:
            filtered["pressure_enabled"] = bool(filtered["pressure_enabled"])
        if "pressure_type_conditioning" in filtered and filtered["pressure_type_conditioning"] is not None:
            value = str(filtered["pressure_type_conditioning"]).lower()
            if value not in {"none", "concat", "gate"}:
                raise ValueError("pressure_type_conditioning must be 'none', 'concat', or 'gate'")
            filtered["pressure_type_conditioning"] = value
        if "pressure_module_sharing" in filtered and filtered["pressure_module_sharing"] is not None:
            value = str(filtered["pressure_module_sharing"]).lower()
            if value not in {"per_type", "shared"}:
                raise ValueError("pressure_module_sharing must be 'per_type' or 'shared'")
            filtered["pressure_module_sharing"] = value
        if "pressure_residual_scale" in filtered and filtered["pressure_residual_scale"] is not None:
            value = float(filtered["pressure_residual_scale"])
            if value < 0.0:
                raise ValueError("pressure_residual_scale must be non-negative")
            filtered["pressure_residual_scale"] = value
        if "enable_policy_choice" in filtered and filtered["enable_policy_choice"] is not None:
            filtered["enable_policy_choice"] = bool(filtered["enable_policy_choice"])
        if "policy_num_classes" in filtered and filtered["policy_num_classes"] is not None:
            filtered["policy_num_classes"] = int(filtered["policy_num_classes"])
        if "constraint_representation" in filtered and filtered["constraint_representation"] is not None:
            value = str(filtered["constraint_representation"]).lower()
            if value not in {"factorized", "eswc_passive"}:
                raise ValueError(
                    "constraint_representation must be 'factorized' or 'eswc_passive'"
                )
            filtered["constraint_representation"] = value
        if "factor_executor_impl" in filtered and filtered["factor_executor_impl"] is not None:
            value = str(filtered["factor_executor_impl"]).lower()
            if value not in {"per_type_v1", "per_type_grouped_v2", "legacy_shared"}:
                raise ValueError(
                    "factor_executor_impl must be 'per_type_v1', 'per_type_grouped_v2', "
                    "or 'legacy_shared'"
                )
            filtered["factor_executor_impl"] = value
        if "gold_edit_embedding_mode" in filtered and filtered["gold_edit_embedding_mode"] is not None:
            value = str(filtered["gold_edit_embedding_mode"]).lower()
            if value not in {"full", "compact"}:
                raise ValueError("gold_edit_embedding_mode must be 'full' or 'compact'")
            filtered["gold_edit_embedding_mode"] = value

        if filtered.get("enable_policy_choice") and "policy_num_classes" in filtered:
            if int(filtered["policy_num_classes"]) < 6:
                raise ValueError("policy_num_classes must be >= 6 for default policy set.")

        current = asdict(self)
        current.update(filtered)
        if (
            str(current["factor_executor_impl"]).lower() == "per_type_grouped_v2"
            and current["active_factor_type_ids"] is None
        ):
            raise ValueError(
                "active_factor_type_ids must be explicit for per_type_grouped_v2"
            )
        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    seed: int = 42  # Single paper-suite seed.
    batch_size: int = 124  # Number of graphs per optimization step.
    num_epochs: int = 8  # Maximum number of training epochs.
    early_stopping_rounds: int = 2  # Patience before early stopping triggers.
    grad_clip: float | None = 0.5  # Gradient norm cap; set None to disable clipping.
    learning_rate: float = 1e-4  # Base learning rate for Adam.
    weight_decay: float = 5e-4  # L2 penalty applied through Adam weight decay.
    scheduler_factor: float = 0.5  # Multiplicative drop factor for the LR scheduler.
    scheduler_patience: int = 0  # Epochs with no improvement before lowering LR.
    num_workers: int = 0  # Worker processes used by DataLoader.
    pin_memory: bool | None = None  # Override DataLoader pin_memory behaviour (None keeps the default).
    validate_factor_labels: bool = False  # Enable strict factor label assertions per batch.
    validation_subset_size: int | None = None  # Optional cap on validation graphs per epoch.
    constraint_loss: ConstraintLossConfig = field(default_factory=ConstraintLossConfig)
    fix_probability_loss: FixProbabilityLossConfig = field(default_factory=FixProbabilityLossConfig)
    factor_loss: FactorLossConfig = field(default_factory=FactorLossConfig)
    chooser: ChooserConfig = field(default_factory=ChooserConfig)
    direct_safety: DirectSafetyConfig = field(default_factory=DirectSafetyConfig)
    policy_filter_strict: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        return cls().updated(data)

    @classmethod
    def from_path(cls, path: Path | None) -> "TrainingConfig":
        if path is None:
            return cls()
        return cls.from_mapping(_load_mapping(path))

    def updated(self, data: Mapping[str, Any] | None = None, **overrides: Any) -> "TrainingConfig":
        payload = dict(data or {})
        dynamic_fallback = payload.pop("dynamic_reweighting", None)

        filtered = _filter_fields(type(self), payload)
        filtered.update({k: v for k, v in overrides.items() if v is not None})

        current = {f.name: getattr(self, f.name) for f in fields(type(self))}
        constraint_update = filtered.pop("constraint_loss", None)
        fix_loss_update = filtered.pop("fix_probability_loss", None)
        factor_loss_update = filtered.pop("factor_loss", None)
        chooser_update = filtered.pop("chooser", None)
        direct_safety_update = filtered.pop("direct_safety", None)
        if "policy_filter_strict" in filtered and filtered["policy_filter_strict"] is not None:
            filtered["policy_filter_strict"] = bool(filtered["policy_filter_strict"])
        if "validation_subset_size" in filtered and filtered["validation_subset_size"] is not None:
            subset_size = int(filtered["validation_subset_size"])
            if subset_size <= 0:
                raise ValueError("TrainingConfig.validation_subset_size must be positive when set")
            filtered["validation_subset_size"] = subset_size
        if "seed" in filtered and filtered["seed"] is not None:
            filtered["seed"] = int(filtered["seed"])

        if dynamic_fallback is not None:
            if constraint_update is None:
                constraint_update = {"dynamic_reweighting": dynamic_fallback}
            else:
                if isinstance(constraint_update, ConstraintLossConfig):
                    constraint_update = constraint_update.to_dict()
                if isinstance(constraint_update, Mapping):
                    constraint_update = dict(constraint_update)
                    if "dynamic_reweighting" not in constraint_update:
                        constraint_update["dynamic_reweighting"] = dynamic_fallback
                else:
                    raise TypeError("constraint_loss must be mapping-compatible when combining configuration sources")

        current.update(filtered)

        if constraint_update is not None:
            if isinstance(constraint_update, ConstraintLossConfig):
                current["constraint_loss"] = constraint_update
            else:
                current["constraint_loss"] = self.constraint_loss.updated(constraint_update)

        if fix_loss_update is not None:
            if isinstance(fix_loss_update, FixProbabilityLossConfig):
                current["fix_probability_loss"] = fix_loss_update
            else:
                current["fix_probability_loss"] = self.fix_probability_loss.updated(fix_loss_update)

        if factor_loss_update is not None:
            if isinstance(factor_loss_update, FactorLossConfig):
                current["factor_loss"] = factor_loss_update
            else:
                current["factor_loss"] = self.factor_loss.updated(factor_loss_update)
        if chooser_update is not None:
            if isinstance(chooser_update, ChooserConfig):
                current["chooser"] = chooser_update
            else:
                current["chooser"] = self.chooser.updated(chooser_update)
        if direct_safety_update is not None:
            if isinstance(direct_safety_update, DirectSafetyConfig):
                current["direct_safety"] = direct_safety_update
            else:
                current["direct_safety"] = self.direct_safety.updated(direct_safety_update)

        return type(self)(**current)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraint_loss"] = self.constraint_loss.to_dict()
        payload["fix_probability_loss"] = self.fix_probability_loss.to_dict()
        payload["factor_loss"] = self.factor_loss.to_dict()
        payload["chooser"] = self.chooser.to_dict()
        payload["direct_safety"] = self.direct_safety.to_dict()
        payload["policy_filter_strict"] = self.policy_filter_strict
        return payload


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "ConstraintLossConfig",
    "DynamicReweightingConfig",
    "FixProbabilityLossConfig",
    "FactorLossConfig",
    "ChooserConfig",
    "DirectSafetyConfig",
]
