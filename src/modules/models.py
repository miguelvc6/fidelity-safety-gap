from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Callable, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, global_mean_pool

from .config import ModelConfig
from .factor_dispatch import GroupedDispatch, GroupedLinearBank, build_grouped_dispatch
from .factor_types import normalize_active_factor_type_ids


FACTOR_ROLE_PREDICATE = 0
FACTOR_ROLE_SUBJECT = 1
FACTOR_ROLE_OBJECT = 2
FACTOR_ROLE_IDS: tuple[int, int, int] = (
    FACTOR_ROLE_PREDICATE,
    FACTOR_ROLE_SUBJECT,
    FACTOR_ROLE_OBJECT,
)


class FactorScopeRuntime(NamedTuple):
    factor_node_index: torch.Tensor
    factor_node_emb: torch.Tensor
    factor_graph_index: torch.Tensor
    factor_type_ids: torch.Tensor
    factor_type_indices: torch.Tensor
    factor_dispatch: GroupedDispatch | None
    predicate_summary: torch.Tensor
    subject_summary: torch.Tensor
    object_summary: torch.Tensor
    predicate_counts: torch.Tensor
    subject_counts: torch.Tensor
    object_counts: torch.Tensor
    predicate_factor_pos: torch.Tensor
    predicate_dst_index: torch.Tensor
    subject_factor_pos: torch.Tensor
    subject_dst_index: torch.Tensor
    object_factor_pos: torch.Tensor
    object_dst_index: torch.Tensor


class FactorTypeExecutor(nn.Module):
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.pre_head = nn.Linear(state_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_mlp(inputs)
        logit = self.pre_head(state).squeeze(-1)
        return state, logit


class FactorPostEditHead(nn.Module):
    def __init__(self, state_dim: int, edit_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + edit_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1),
        )

    def forward(self, state: torch.Tensor, edit_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, edit_repr], dim=-1)).squeeze(-1)


class GroupedFactorTypeExecutor(nn.Module):
    """Independent per-type executors stored in compact packed banks."""

    def __init__(self, num_types: int, input_dim: int, state_dim: int):
        super().__init__()
        self.input_layer = GroupedLinearBank(num_types, input_dim, state_dim)
        self.state_layer = GroupedLinearBank(num_types, state_dim, state_dim)
        self.pre_head = GroupedLinearBank(num_types, state_dim, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        dispatch: GroupedDispatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_inputs = inputs.index_select(0, dispatch.order)
        state = F.relu(self.input_layer.forward_sorted(sorted_inputs, dispatch))
        state = F.relu(self.state_layer.forward_sorted(state, dispatch))
        logits = self.pre_head.forward_sorted(state, dispatch).squeeze(-1)
        return (
            state.index_select(0, dispatch.inverse_order),
            logits.index_select(0, dispatch.inverse_order),
        )

    @property
    def last_backend(self) -> str:
        return self.input_layer.last_backend


class GroupedFactorPostEditHead(nn.Module):
    """Independent per-type post-edit heads stored in compact packed banks."""

    def __init__(self, num_types: int, state_dim: int, edit_dim: int):
        super().__init__()
        self.hidden = GroupedLinearBank(num_types, state_dim + edit_dim, state_dim)
        self.output = GroupedLinearBank(num_types, state_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        edit_repr: torch.Tensor,
        dispatch: GroupedDispatch,
    ) -> torch.Tensor:
        joint = torch.cat([state, edit_repr], dim=-1).index_select(0, dispatch.order)
        hidden = F.relu(self.hidden.forward_sorted(joint, dispatch))
        logits = self.output.forward_sorted(hidden, dispatch).squeeze(-1)
        return logits.index_select(0, dispatch.inverse_order)


class GroupedPressureRole(nn.Module):
    """Independent per-type pressure MLPs stored in compact packed banks."""

    def __init__(self, num_types: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden = GroupedLinearBank(num_types, input_dim, hidden_dim)
        self.output = GroupedLinearBank(num_types, hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, dispatch: GroupedDispatch) -> torch.Tensor:
        sorted_inputs = inputs.index_select(0, dispatch.order)
        hidden = F.relu(self.hidden.forward_sorted(sorted_inputs, dispatch))
        outputs = self.output.forward_sorted(hidden, dispatch)
        return outputs.index_select(0, dispatch.inverse_order)


def _select_factor_node_indices(
    node_emb: torch.Tensor,
    data,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    factor_node_index = getattr(data, "factor_node_index", None)
    factor_node_mask = getattr(data, "is_factor_node", None)
    batch = getattr(data, "batch", None)
    if batch is None:
        return None, None

    mask = None
    if factor_node_mask is not None:
        mask = torch.as_tensor(
            factor_node_mask, dtype=torch.bool, device=node_emb.device
        ).view(-1)
        if mask.numel() != node_emb.size(0):
            mask = None

    if mask is not None and mask.any():
        factor_global_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        factor_graph_index = batch.index_select(0, factor_global_indices)

        ptr = getattr(data, "ptr", None)
        if factor_node_index is not None and ptr is not None:
            factor_node_index = torch.as_tensor(
                factor_node_index, device=node_emb.device
            ).view(-1)
            ptr = torch.as_tensor(ptr, device=node_emb.device).view(-1)
            local_indices = factor_global_indices - ptr[factor_graph_index]

            if factor_node_index.numel() == factor_global_indices.numel():
                order: list[torch.Tensor] = []
                offset = 0
                num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
                for graph_id in range(num_graphs):
                    graph_mask = factor_graph_index == graph_id
                    count = int(graph_mask.sum().item())
                    if count == 0:
                        continue
                    expected = factor_node_index[offset : offset + count]
                    offset += count
                    local_g = local_indices[graph_mask]
                    if expected.numel() != count:
                        idxs = torch.argsort(local_g)
                    else:
                        local_list = local_g.tolist()
                        positions: dict[int, deque[int]] = defaultdict(deque)
                        for pos_idx, val in enumerate(local_list):
                            positions[int(val)].append(pos_idx)

                        reorder: list[int] = []
                        missing = False
                        for expected_val in expected.tolist():
                            q = positions.get(int(expected_val))
                            if not q:
                                missing = True
                                break
                            reorder.append(q.popleft())
                        if missing:
                            idxs = torch.argsort(local_g)
                        else:
                            idxs = torch.tensor(
                                reorder,
                                device=node_emb.device,
                                dtype=torch.long,
                            )
                    factor_indices_g = torch.nonzero(graph_mask, as_tuple=False).view(-1)
                    order.append(factor_indices_g.index_select(0, idxs))
                if order:
                    perm = torch.cat(order, dim=0)
                    factor_global_indices = factor_global_indices.index_select(0, perm)
                    factor_graph_index = factor_graph_index.index_select(0, perm)
        return factor_global_indices, factor_graph_index

    if factor_node_index is not None:
        factor_node_index = torch.as_tensor(
            factor_node_index, device=node_emb.device
        ).view(-1)
        factor_graph_index = batch.index_select(0, factor_node_index)
        return factor_node_index, factor_graph_index

    return None, None


def _build_factor_scope_runtime(
    node_emb: torch.Tensor,
    data,
    *,
    factor_edge_types: tuple[int, int, int],
    num_factor_types: int,
    factor_type_id_to_compact: torch.Tensor | None = None,
    num_active_factor_types: int | None = None,
) -> FactorScopeRuntime | None:
    factor_node_index, factor_graph_index = _select_factor_node_indices(node_emb, data)
    if factor_node_index is None or factor_graph_index is None or factor_node_index.numel() == 0:
        return None

    factor_node_emb = node_emb.index_select(0, factor_node_index)
    num_factors = int(factor_node_index.numel())
    hidden_dim = int(node_emb.size(-1))

    factor_types = getattr(data, "factor_types", None)
    if factor_types is not None:
        raw_factor_types = torch.as_tensor(
            factor_types, dtype=torch.long, device=node_emb.device
        ).view(-1)
        if raw_factor_types.numel() != num_factors:
            raise ValueError(
                f"factor_types length {raw_factor_types.numel()} does not match factor count {num_factors}"
            )
        if num_factor_types <= 0:
            raise ValueError("factor_types were provided but num_factor_types is not positive.")
        invalid_mask = (raw_factor_types < 0) | (raw_factor_types >= int(num_factor_types))
        if invalid_mask.any():
            invalid_values = sorted(
                {int(v) for v in raw_factor_types[invalid_mask].detach().cpu().tolist()}
            )
            raise ValueError(
                f"factor_types contain values outside [0, {num_factor_types}): {invalid_values[:8]}"
            )
        factor_type_ids = raw_factor_types
    else:
        raise ValueError("Missing factor_types on factorized graph; per-type factor execution requires type ids.")

    if factor_type_id_to_compact is None:
        factor_type_indices = factor_type_ids
        factor_dispatch = None
    else:
        lookup = factor_type_id_to_compact.to(device=node_emb.device)
        factor_type_indices = lookup.index_select(0, factor_type_ids)
        inactive_mask = factor_type_indices < 0
        if inactive_mask.any():
            inactive_values = sorted(
                {int(v) for v in factor_type_ids[inactive_mask].detach().cpu().tolist()}
            )
            raise ValueError(
                "factor_types contain stable ids absent from active_factor_type_ids: "
                f"{inactive_values[:8]}"
            )
        active_type_count = int(
            num_active_factor_types
            if num_active_factor_types is not None
            else int(factor_type_indices.max().item()) + 1
        )
        factor_dispatch = build_grouped_dispatch(
            factor_type_indices,
            num_types=active_type_count,
        )

    edge_index = getattr(data, "edge_index", None)
    edge_type = getattr(data, "edge_type", None)
    if edge_index is None or edge_type is None:
        zero_summary = torch.zeros((num_factors, hidden_dim), device=node_emb.device, dtype=node_emb.dtype)
        zero_counts = torch.zeros((num_factors, 1), device=node_emb.device, dtype=node_emb.dtype)
        empty = torch.empty((0,), device=node_emb.device, dtype=torch.long)
        return FactorScopeRuntime(
            factor_node_index=factor_node_index,
            factor_node_emb=factor_node_emb,
            factor_graph_index=factor_graph_index,
            factor_type_ids=factor_type_ids,
            factor_type_indices=factor_type_indices,
            factor_dispatch=factor_dispatch,
            predicate_summary=zero_summary,
            subject_summary=zero_summary.clone(),
            object_summary=zero_summary.clone(),
            predicate_counts=zero_counts,
            subject_counts=zero_counts.clone(),
            object_counts=zero_counts.clone(),
            predicate_factor_pos=empty,
            predicate_dst_index=empty,
            subject_factor_pos=empty,
            subject_dst_index=empty,
            object_factor_pos=empty,
            object_dst_index=empty,
        )

    edge_index = edge_index.to(device=node_emb.device)
    edge_type = torch.as_tensor(edge_type, dtype=torch.long, device=node_emb.device).view(-1)
    if edge_index.size(1) != edge_type.numel():
        raise ValueError("edge_type length must match number of edges")

    factor_pos_map = torch.full((node_emb.size(0),), -1, dtype=torch.long, device=node_emb.device)
    factor_pos_map.index_copy_(
        0,
        factor_node_index,
        torch.arange(num_factors, device=node_emb.device, dtype=torch.long),
    )

    def _role_summary(role_edge_type: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = edge_type == int(role_edge_type)
        if not mask.any():
            return (
                torch.zeros((num_factors, hidden_dim), device=node_emb.device, dtype=node_emb.dtype),
                torch.zeros((num_factors, 1), device=node_emb.device, dtype=node_emb.dtype),
                torch.empty((0,), device=node_emb.device, dtype=torch.long),
                torch.empty((0,), device=node_emb.device, dtype=torch.long),
            )
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        factor_pos = factor_pos_map.index_select(0, src)
        valid = factor_pos >= 0
        if not valid.any():
            return (
                torch.zeros((num_factors, hidden_dim), device=node_emb.device, dtype=node_emb.dtype),
                torch.zeros((num_factors, 1), device=node_emb.device, dtype=node_emb.dtype),
                torch.empty((0,), device=node_emb.device, dtype=torch.long),
                torch.empty((0,), device=node_emb.device, dtype=torch.long),
            )
        factor_pos = factor_pos[valid]
        dst = dst[valid]
        sums = torch.zeros((num_factors, hidden_dim), device=node_emb.device, dtype=node_emb.dtype)
        sums.index_add_(0, factor_pos, node_emb.index_select(0, dst))
        counts = torch.zeros((num_factors, 1), device=node_emb.device, dtype=node_emb.dtype)
        counts.index_add_(
            0,
            factor_pos,
            torch.ones((dst.numel(), 1), device=node_emb.device, dtype=node_emb.dtype),
        )
        means = sums / counts.clamp(min=1.0)
        return means, counts, factor_pos, dst

    predicate_summary, predicate_counts, predicate_factor_pos, predicate_dst_index = _role_summary(
        factor_edge_types[0]
    )
    subject_summary, subject_counts, subject_factor_pos, subject_dst_index = _role_summary(
        factor_edge_types[1]
    )
    object_summary, object_counts, object_factor_pos, object_dst_index = _role_summary(
        factor_edge_types[2]
    )

    return FactorScopeRuntime(
        factor_node_index=factor_node_index,
        factor_node_emb=factor_node_emb,
        factor_graph_index=factor_graph_index,
        factor_type_ids=factor_type_ids,
        factor_type_indices=factor_type_indices,
        factor_dispatch=factor_dispatch,
        predicate_summary=predicate_summary,
        subject_summary=subject_summary,
        object_summary=object_summary,
        predicate_counts=predicate_counts,
        subject_counts=subject_counts,
        object_counts=object_counts,
        predicate_factor_pos=predicate_factor_pos,
        predicate_dst_index=predicate_dst_index,
        subject_factor_pos=subject_factor_pos,
        subject_dst_index=subject_dst_index,
        object_factor_pos=object_factor_pos,
        object_dst_index=object_dst_index,
    )


class BaseGraphModel(nn.Module, ABC):
    """
    Base class for GNN models to enforce atttributes and
    methods later used in evaluation script.
    """

    def __init__(
        self,
        num_input_graph_nodes: int,  # number of unique nodes in input graphs
        num_embedding_size: int,
        num_layers: int,
        hidden: int,
        head_hidden: int | None = None,
        branch_hidden: int | None = None,
        dropout: float = 0.5,
        use_node_embeddings: bool = True,
        use_role_embeddings: bool = False,
        num_role_types: int = 0,
        role_embedding_dim: int = 0,
        use_edge_attributes: bool = False,
        use_edge_subtraction: bool = False,
        entity_class_ids: Sequence[int] | torch.Tensor | None = None,
        predicate_class_ids: Sequence[int] | torch.Tensor | None = None,
        num_factor_types: int = 0,
        active_factor_type_ids: Sequence[int] | torch.Tensor | None = None,
        factor_type_embedding_dim: int = 8,
        factor_executor_impl: str = "per_type_v1",
        gold_edit_embedding_mode: str = "full",
        constraint_representation: str = "factorized",
        pressure_enabled: bool = False,
        pressure_type_conditioning: str = "none",
        pressure_module_sharing: str = "per_type",
        pressure_residual_scale: float = 0.1,
        enable_policy_choice: bool = False,
        policy_num_classes: int = 6,
    ):
        super().__init__()
        self._num_input_graph_nodes = int(num_input_graph_nodes)
        hidden_mp = int(hidden)
        head_hidden = int(hidden if head_hidden is None else head_hidden)
        branch_hidden = int(head_hidden if branch_hidden is None else branch_hidden)
        self._dropout = float(dropout)
        self._use_node_embeddings = bool(use_node_embeddings)
        self._use_role_embeddings = bool(use_role_embeddings and num_role_types > 0 and role_embedding_dim > 0)
        self._num_role_types = int(num_role_types if self._use_role_embeddings else 0)
        self._role_embedding_dim = int(role_embedding_dim if self._use_role_embeddings else 0)
        self._mask_fill_value = float(-1e9)
        self._node_embedding_dim = int(num_embedding_size)
        input_channels = self._node_embedding_dim + (self._role_embedding_dim if self._use_role_embeddings else 0)
        self.use_edge_attributes = use_edge_attributes
        self.use_edge_subtraction = use_edge_subtraction
        assert not (self.use_edge_subtraction and not self.use_edge_attributes), (
            "Edge subtraction requires use_edge_attributes to be True."
        )

        if self.use_edge_attributes:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_mp, hidden_mp),
                nn.ReLU(),
                nn.Dropout(p=self._dropout),
                nn.Linear(hidden_mp, hidden_mp),
                nn.Dropout(p=self._dropout),
            )
        else:
            self.edge_mlp = None

        ## Initialization Step
        # NOTE: I changed this so the edge attribute handling can be more consistent.
        # self.initialization = self.create_conv_layer(input_channels, hidden_mp)
        self.initialization = nn.Sequential(
            nn.Linear(input_channels, hidden_mp),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=self._dropout),
            # plain last layer
            nn.Linear(hidden_mp, hidden_mp),
            nn.Dropout(p=self._dropout),
        )

        # Node Embedding
        if self._use_node_embeddings:
            self.node_embeddings = nn.Embedding(
                self.num_input_graph_nodes, self._node_embedding_dim
            )
        else:
            self.node_embeddings = nn.Identity()

        if self._use_role_embeddings:
            self.role_embeddings = nn.Embedding(self._num_role_types, self._role_embedding_dim)
        else:
            self.role_embeddings = None

        # Aggregation Layers
        self.mp_layers = nn.ModuleList([self.create_conv_layer(hidden_mp, hidden_mp) for _ in range(num_layers - 1)])

        self.shared_projection = nn.Linear(hidden_mp, head_hidden)

        def make_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_hidden, branch_hidden),
                nn.LeakyReLU(negative_slope=0.1),
            )

        self.subject_branch = make_branch()
        self.object_branch = make_branch()
        self.predicate_branch = make_branch()


        self._num_target_ids = -1
        if predicate_class_ids is not None:
            self._num_target_ids = int(max(predicate_class_ids)) + 1
        if entity_class_ids is not None:
            self._num_target_ids = max(self._num_target_ids, int(max(entity_class_ids)) + 1)

        # If still -1, set to num_input_graph_nodes
        if self._num_target_ids == -1:
            self._num_target_ids = self.num_input_graph_nodes

        entity_ids_tensor, entity_is_full_vocab = self._prepare_class_ids(
            entity_class_ids, self.num_target_ids
        )
        predicate_ids_tensor, predicate_is_full_vocab = self._prepare_class_ids(
            predicate_class_ids, self.num_target_ids
        )
        self.register_buffer("entity_class_ids", entity_ids_tensor)
        self.register_buffer("predicate_class_ids", predicate_ids_tensor)
        self.entity_class_ids: torch.Tensor
        self.predicate_class_ids: torch.Tensor
        self._entity_full_vocab = bool(entity_is_full_vocab)
        self._predicate_full_vocab = bool(predicate_is_full_vocab)

        entity_vocab = int(entity_ids_tensor.numel())
        predicate_vocab = int(predicate_ids_tensor.numel())

        self.subject_add_head = nn.Linear(branch_hidden, entity_vocab)
        self.subject_del_head = nn.Linear(branch_hidden, entity_vocab)
        self.object_add_head = nn.Linear(branch_hidden, entity_vocab)
        self.object_del_head = nn.Linear(branch_hidden, entity_vocab)
        self.predicate_add_head = nn.Linear(branch_hidden, predicate_vocab)
        self.predicate_del_head = nn.Linear(branch_hidden, predicate_vocab)
        self._hidden_channels = hidden_mp
        self._head_hidden = head_hidden
        self._branch_hidden = branch_hidden
        self._num_layers = int(num_layers)
        self._chooser_enabled = False
        self._candidate_id_embeddings: nn.Embedding | None = None
        self._chooser_head: nn.Sequential | None = None
        self._num_factor_types = int(num_factor_types)
        self._factor_type_embedding_dim = int(factor_type_embedding_dim)
        self._factor_executor_impl = str(factor_executor_impl).lower()
        if self._factor_executor_impl not in {
            "per_type_v1",
            "per_type_grouped_v2",
            "legacy_shared",
        }:
            raise ValueError(
                "factor_executor_impl must be 'per_type_v1', 'per_type_grouped_v2', "
                "or 'legacy_shared'"
            )
        normalized_active_factor_type_ids = normalize_active_factor_type_ids(
            active_factor_type_ids,
            num_factor_types=self._num_factor_types,
            require_explicit=self._factor_executor_impl == "per_type_grouped_v2",
        )
        if (
            self._factor_executor_impl != "per_type_grouped_v2"
            and normalized_active_factor_type_ids
            != tuple(range(max(self._num_factor_types, 0)))
        ):
            raise ValueError(
                f"{self._factor_executor_impl} only supports the dense legacy "
                "factor-type layout."
            )
        self._num_active_factor_types = len(normalized_active_factor_type_ids)
        compact_to_stable = torch.tensor(
            normalized_active_factor_type_ids,
            dtype=torch.long,
        )
        stable_to_compact = torch.full(
            (max(self._num_factor_types, 0),),
            -1,
            dtype=torch.long,
        )
        if compact_to_stable.numel():
            stable_to_compact.index_copy_(
                0,
                compact_to_stable,
                torch.arange(compact_to_stable.numel(), dtype=torch.long),
            )
        persistent_type_mapping = (
            active_factor_type_ids is not None
            and self._factor_executor_impl == "per_type_grouped_v2"
        )
        self.register_buffer(
            "factor_type_ids_compact_to_stable",
            compact_to_stable,
            persistent=persistent_type_mapping,
        )
        self.register_buffer(
            "factor_type_id_to_compact",
            stable_to_compact,
            persistent=persistent_type_mapping,
        )
        self._gold_edit_embedding_mode = str(gold_edit_embedding_mode).lower()
        if self._gold_edit_embedding_mode not in {"full", "compact"}:
            raise ValueError("gold_edit_embedding_mode must be 'full' or 'compact'")
        self._constraint_representation = str(constraint_representation).lower()
        if self._constraint_representation not in {"factorized", "eswc_passive"}:
            raise ValueError("constraint_representation must be 'factorized' or 'eswc_passive'")
        self._pressure_type_conditioning = str(pressure_type_conditioning).lower()
        self._pressure_module_sharing = str(pressure_module_sharing).lower()
        if self._pressure_module_sharing not in {"per_type", "shared"}:
            raise ValueError("pressure_module_sharing must be 'per_type' or 'shared'")
        self._pressure_residual_scale = float(pressure_residual_scale)
        self._factor_state_dim = head_hidden
        self._factor_scope_feature_dim = (hidden_mp * 4) + 3
        self._num_factor_executor_modules = max(1, self._num_active_factor_types)
        gold_edit_class_ids = torch.unique(
            torch.cat(
                [
                    entity_ids_tensor,
                    predicate_ids_tensor,
                    torch.tensor([0], dtype=torch.long),
                ]
            ),
            sorted=True,
        )
        gold_edit_id_to_compact = torch.full(
            (self.num_target_ids,),
            -1,
            dtype=torch.long,
        )
        gold_edit_id_to_compact.index_copy_(
            0,
            gold_edit_class_ids,
            torch.arange(gold_edit_class_ids.numel(), dtype=torch.long),
        )
        persistent_gold_mapping = self._gold_edit_embedding_mode == "compact"
        self.register_buffer(
            "gold_edit_class_ids",
            gold_edit_class_ids,
            persistent=persistent_gold_mapping,
        )
        self.register_buffer(
            "gold_edit_id_to_compact",
            gold_edit_id_to_compact,
            persistent=persistent_gold_mapping,
        )
        if self._factor_executor_impl == "legacy_shared":
            if self._num_factor_types > 0 and self._factor_type_embedding_dim > 0:
                self.factor_type_embeddings = nn.Embedding(
                    self._num_factor_types, self._factor_type_embedding_dim
                )
                factor_input_dim = head_hidden + self._factor_type_embedding_dim
            else:
                self.factor_type_embeddings = None
                factor_input_dim = head_hidden
            self.factor_pre_head = nn.Sequential(
                nn.Linear(factor_input_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
            self._factor_executors = None
            self._factor_post_heads = None
            self._gold_edit_embeddings = None
        elif self._factor_executor_impl == "per_type_grouped_v2":
            self.factor_type_embeddings = None
            self.factor_pre_head = None
            self._factor_executors = GroupedFactorTypeExecutor(
                self._num_factor_executor_modules,
                self._factor_scope_feature_dim,
                self._factor_state_dim,
            )
            self._factor_post_heads = GroupedFactorPostEditHead(
                self._num_factor_executor_modules,
                self._factor_state_dim,
                self._factor_state_dim,
            )
            gold_embedding_rows = (
                int(gold_edit_class_ids.numel())
                if self._gold_edit_embedding_mode == "compact"
                else self.num_target_ids
            )
            self._gold_edit_embeddings = nn.Embedding(
                gold_embedding_rows,
                self._factor_state_dim,
            )
        else:
            self.factor_type_embeddings = None
            self.factor_pre_head = None
            self._factor_executors = nn.ModuleList(
                [
                    FactorTypeExecutor(self._factor_scope_feature_dim, self._factor_state_dim)
                    for _ in range(self._num_factor_executor_modules)
                ]
            )
            self._factor_post_heads = nn.ModuleList(
                [
                    FactorPostEditHead(self._factor_state_dim, self._factor_state_dim)
                    for _ in range(self._num_factor_executor_modules)
                ]
            )
            gold_embedding_rows = (
                int(gold_edit_class_ids.numel())
                if self._gold_edit_embedding_mode == "compact"
                else self.num_target_ids
            )
            self._gold_edit_embeddings = nn.Embedding(
                gold_embedding_rows,
                self._factor_state_dim,
            )
        self._policy_enabled = bool(enable_policy_choice)
        self._policy_num_classes = int(policy_num_classes)
        if self._policy_enabled:
            if self._policy_num_classes <= 0:
                raise ValueError("policy_num_classes must be positive when policy choice is enabled.")
            self.policy_head = nn.Linear(head_hidden, self._policy_num_classes)
        else:
            self.policy_head = None

    @property
    def num_input_graph_nodes(self) -> int:
        return int(self._num_input_graph_nodes)
    
    @property
    def num_target_ids(self) -> int:
        return int(self._num_target_ids)

    @property
    def hidden_channels(self) -> int:
        return self._hidden_channels

    @property
    def head_hidden(self) -> int:
        return self._head_hidden

    @property
    def branch_hidden(self) -> int:
        return self._branch_hidden

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def role_embedding_dim(self) -> int:
        return self._role_embedding_dim

    @property
    def chooser_enabled(self) -> bool:
        return bool(self._chooser_enabled)

    @property
    def factor_dispatch_backend(self) -> str:
        executors = self._factor_executors
        if isinstance(executors, GroupedFactorTypeExecutor):
            return executors.last_backend
        if isinstance(executors, nn.ModuleList):
            return "module_loop"
        if self._factor_executor_impl == "legacy_shared":
            return "shared_linear"
        return "disabled"

    def enable_chooser(self) -> None:
        if self._chooser_enabled:
            return
        device = next(self.parameters()).device
        candidate_emb = nn.Embedding(self.num_target_ids, self._head_hidden).to(device)
        chooser_head = nn.Sequential(
            nn.Linear(self._head_hidden * 2, self._head_hidden),
            nn.ReLU(),
            nn.Linear(self._head_hidden, 1),
        ).to(device)
        self._candidate_id_embeddings = candidate_emb
        self._chooser_head = chooser_head
        self._chooser_enabled = True

    def score_candidates(self, graph_emb: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if not self._chooser_enabled or self._candidate_id_embeddings is None or self._chooser_head is None:
            raise RuntimeError("Chooser head not enabled; call enable_chooser() before scoring candidates.")
        if candidates.dim() == 1:
            candidates = candidates.view(1, -1)
        if candidates.size(-1) != 6:
            raise ValueError(f"Expected candidate slots shape (*,6), got {tuple(candidates.shape)}")
        candidates = candidates.to(device=graph_emb.device, dtype=torch.long)
        candidate_emb = self._candidate_id_embeddings(
            candidates.clamp(min=0, max=self.num_target_ids - 1)
        )
        candidate_repr = candidate_emb.mean(dim=1)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.view(1, -1)
        graph_expand = graph_emb.expand(candidate_repr.size(0), -1)
        joint = torch.cat([graph_expand, candidate_repr], dim=-1)
        scores = self._chooser_head(joint).squeeze(-1)
        return scores

    def score_candidates_packed(
        self,
        graph_emb: torch.Tensor,
        candidates: torch.Tensor,
        candidate_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a packed candidate tensor where each row belongs to one graph index.
        """
        if not self._chooser_enabled or self._candidate_id_embeddings is None or self._chooser_head is None:
            raise RuntimeError("Chooser head not enabled; call enable_chooser() before scoring candidates.")
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.view(1, -1)
        if graph_emb.dim() != 2:
            raise ValueError(f"Expected graph_emb shape (B,H), got {tuple(graph_emb.shape)}")
        if candidates.dim() == 1:
            candidates = candidates.view(1, -1)
        if candidates.dim() != 2 or candidates.size(-1) != 6:
            raise ValueError(f"Expected candidate slots shape (N,6), got {tuple(candidates.shape)}")
        if candidate_graph_index.dim() != 1:
            candidate_graph_index = candidate_graph_index.view(-1)
        if candidate_graph_index.numel() != candidates.size(0):
            raise ValueError(
                "candidate_graph_index length must match number of packed candidates "
                f"({candidate_graph_index.numel()} vs {candidates.size(0)})"
            )

        candidates = candidates.to(device=graph_emb.device, dtype=torch.long)
        candidate_graph_index = candidate_graph_index.to(device=graph_emb.device, dtype=torch.long)
        if candidate_graph_index.numel() > 0:
            if int(candidate_graph_index.min().item()) < 0 or int(candidate_graph_index.max().item()) >= graph_emb.size(0):
                raise ValueError("candidate_graph_index contains out-of-range graph ids.")

        candidate_emb = self._candidate_id_embeddings(
            candidates.clamp(min=0, max=self.num_target_ids - 1)
        )
        candidate_repr = candidate_emb.mean(dim=1)
        graph_expand = graph_emb.index_select(0, candidate_graph_index)
        joint = torch.cat([graph_expand, candidate_repr], dim=-1)
        return self._chooser_head(joint).squeeze(-1)

    def _factor_edge_types(self) -> tuple[int, int, int]:
        return (
            getattr(self, "EDGE_FACTOR_TO_LOCAL_PREDICATE", 4),
            getattr(self, "EDGE_FACTOR_TO_LOCAL_SUBJECT", 5),
            getattr(self, "EDGE_FACTOR_TO_LOCAL_OBJECT", 6),
        )

    def _build_factor_scope_runtime(self, node_emb: torch.Tensor, data) -> FactorScopeRuntime | None:
        compact_execution = self._factor_executor_impl == "per_type_grouped_v2"
        return _build_factor_scope_runtime(
            node_emb,
            data,
            factor_edge_types=self._factor_edge_types(),
            num_factor_types=self._num_factor_types,
            factor_type_id_to_compact=(
                self.factor_type_id_to_compact if compact_execution else None
            ),
            num_active_factor_types=(
                self._num_active_factor_types if compact_execution else None
            ),
        )

    def _factor_scope_features(self, runtime: FactorScopeRuntime) -> torch.Tensor:
        counts = torch.cat(
            [
                torch.log1p(runtime.predicate_counts),
                torch.log1p(runtime.subject_counts),
                torch.log1p(runtime.object_counts),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                runtime.factor_node_emb,
                runtime.predicate_summary,
                runtime.subject_summary,
                runtime.object_summary,
                counts,
            ],
            dim=-1,
        )

    def _run_per_type_factor_executors(
        self,
        runtime: FactorScopeRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._factor_executors is None:
            raise RuntimeError("Per-type factor executors are not initialized.")
        inputs = self._factor_scope_features(runtime)
        if isinstance(self._factor_executors, GroupedFactorTypeExecutor):
            if runtime.factor_dispatch is None:
                raise RuntimeError("Grouped factor executor requires compact dispatch metadata.")
            return self._factor_executors(inputs, runtime.factor_dispatch)
        states = inputs.new_zeros((inputs.size(0), self._factor_state_dim))
        logits = inputs.new_zeros((inputs.size(0),))
        for type_id in torch.unique(runtime.factor_type_ids).tolist():
            mask = runtime.factor_type_ids == int(type_id)
            if not mask.any():
                continue
            state, logit = self._factor_executors[int(type_id)](inputs[mask])
            states[mask] = state.to(dtype=states.dtype)
            logits[mask] = logit.to(dtype=logits.dtype)
        return states, logits

    def _run_per_type_post_heads(
        self,
        factor_states: torch.Tensor,
        runtime: FactorScopeRuntime,
        edit_repr: torch.Tensor,
    ) -> torch.Tensor:
        if self._factor_post_heads is None:
            raise RuntimeError("Per-type factor post heads are not initialized.")
        if isinstance(self._factor_post_heads, GroupedFactorPostEditHead):
            if runtime.factor_dispatch is None:
                raise RuntimeError("Grouped factor post head requires compact dispatch metadata.")
            return self._factor_post_heads(
                factor_states,
                edit_repr,
                runtime.factor_dispatch,
            )
        logits = factor_states.new_zeros((factor_states.size(0),))
        factor_type_ids = runtime.factor_type_ids
        for type_id in torch.unique(factor_type_ids).tolist():
            mask = factor_type_ids == int(type_id)
            if not mask.any():
                continue
            logits[mask] = self._factor_post_heads[int(type_id)](
                factor_states[mask], edit_repr[mask]
            ).to(dtype=logits.dtype)
        return logits

    def _gold_edit_representation(
        self,
        targets: torch.Tensor,
        factor_graph_index: torch.Tensor,
    ) -> torch.Tensor | None:
        if self._gold_edit_embeddings is None:
            return None
        if targets.dim() == 1:
            targets = targets.view(1, -1)
        if targets.dim() != 2 or targets.size(-1) != 6:
            raise ValueError(f"Expected gold targets shape (B,6), got {tuple(targets.shape)}")
        target_ids = targets.to(dtype=torch.long, device=factor_graph_index.device)
        if self._gold_edit_embedding_mode == "compact":
            invalid = (target_ids < 0) | (target_ids >= self.num_target_ids)
            if invalid.any():
                values = sorted(
                    {int(v) for v in target_ids[invalid].detach().cpu().tolist()}
                )
                raise ValueError(
                    f"Gold edit target ids outside [0, {self.num_target_ids}): {values[:8]}"
                )
            compact_ids = self.gold_edit_id_to_compact.index_select(
                0,
                target_ids.view(-1),
            ).view_as(target_ids)
            unreachable = compact_ids < 0
            if unreachable.any():
                values = sorted(
                    {int(v) for v in target_ids[unreachable].detach().cpu().tolist()}
                )
                raise ValueError(
                    f"Gold edit targets absent from compact target vocabulary: {values[:8]}"
                )
            embedding_ids = compact_ids
        else:
            # Preserve the legacy full-vocabulary behavior for old configs and
            # checkpoints. Compact mode is deliberately strict because its
            # sparse lookup cannot represent arbitrary stable ids.
            embedding_ids = target_ids.clamp(min=0, max=self.num_target_ids - 1)
        edit_emb = self._gold_edit_embeddings(embedding_ids).mean(dim=1)
        return edit_emb.index_select(0, factor_graph_index)

    def _compute_factor_outputs(
        self,
        node_emb: torch.Tensor,
        data,
        *,
        include_post_gold: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        factor_mask_pre = None
        factor_checkable = getattr(data, "factor_checkable_pre", None)
        if factor_checkable is not None:
            factor_mask_pre = torch.as_tensor(
                factor_checkable, dtype=torch.bool, device=node_emb.device
            ).view(-1)

        if self._constraint_representation != "factorized":
            return {
                "factor_logits_pre": None,
                "factor_logits_post_gold": None,
                "factor_mask_pre": factor_mask_pre,
                "factor_graph_index": None,
            }

        factor_node_emb, factor_graph_index = _select_factor_nodes(node_emb, data)
        factor_logits_pre = None
        factor_logits_post_gold = None

        if factor_node_emb is None or factor_node_emb.numel() == 0 or factor_graph_index is None:
            return {
                "factor_logits_pre": factor_logits_pre,
                "factor_logits_post_gold": factor_logits_post_gold,
                "factor_mask_pre": factor_mask_pre,
                "factor_graph_index": factor_graph_index,
            }

        if self._factor_executor_impl == "legacy_shared":
            factor_types = getattr(data, "factor_types", None)
            factor_hidden = F.relu(self.shared_projection(factor_node_emb))
            factor_hidden = F.dropout(factor_hidden, p=self._dropout, training=self.training)
            factor_type_embeddings = self.factor_type_embeddings
            if factor_type_embeddings is not None:
                if factor_types is not None:
                    factor_types_tensor = torch.as_tensor(
                        factor_types, dtype=torch.long, device=node_emb.device
                    ).view(-1)
                    invalid_mask = (factor_types_tensor < 0) | (factor_types_tensor >= self._num_factor_types)
                    if invalid_mask.any():
                        invalid_values = sorted({int(v) for v in factor_types_tensor[invalid_mask].tolist()})
                        raise ValueError(
                            f"factor_types contain values outside [0, {self._num_factor_types}): {invalid_values[:8]}"
                        )
                    factor_type_emb = factor_type_embeddings(factor_types_tensor)
                else:
                    raise ValueError(
                        "Missing factor_types on factorized graph; factor type conditioning requires dense type ids."
                    )
                factor_hidden = torch.cat([factor_hidden, factor_type_emb], dim=-1)
            if self.factor_pre_head is None:
                raise RuntimeError("Legacy factor_pre_head is not initialized.")
            factor_logits_pre = self.factor_pre_head(factor_hidden).squeeze(-1)
            return {
                "factor_logits_pre": factor_logits_pre,
                "factor_logits_post_gold": factor_logits_post_gold,
                "factor_mask_pre": factor_mask_pre,
                "factor_graph_index": factor_graph_index,
            }

        runtime = self._build_factor_scope_runtime(node_emb, data)
        if runtime is None:
            return {
                "factor_logits_pre": factor_logits_pre,
                "factor_logits_post_gold": factor_logits_post_gold,
                "factor_mask_pre": factor_mask_pre,
                "factor_graph_index": factor_graph_index,
            }

        factor_states, factor_logits_pre = self._run_per_type_factor_executors(runtime)
        targets = getattr(data, "y", None) if include_post_gold else None
        if targets is not None:
            gold_edit_repr = self._gold_edit_representation(
                torch.as_tensor(targets, device=node_emb.device),
                runtime.factor_graph_index,
            )
            if gold_edit_repr is not None:
                factor_logits_post_gold = self._run_per_type_post_heads(
                    factor_states,
                    runtime,
                    gold_edit_repr,
                )

        return {
            "factor_logits_pre": factor_logits_pre,
            "factor_logits_post_gold": factor_logits_post_gold,
            "factor_mask_pre": factor_mask_pre,
            "factor_graph_index": runtime.factor_graph_index,
        }

    @abstractmethod
    def create_conv_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Return a torch_geometric convolution layer."""
        raise NotImplementedError

    def forward(self, data, *, include_factor_post_gold: bool = True):
        # Extract data
        x, batch = data.x, data.batch
        if self.use_edge_attributes:
            edge_index = data.edge_index_non_flattened
            edge_attr = data.edge_attr_non_flattened
            # Preserve local node indices because edge attributes refer to those
            # indices; isolated nodes are therefore retained in this representation.
        else:
            edge_index = data.edge_index
            edge_attr = None

        if self._use_node_embeddings:
            if x.dtype not in (torch.long, torch.int64, torch.int32):
                x = x.long()
            x = self.node_embeddings(x)
        else:
            if not torch.is_floating_point(x):
                x = x.float()
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

        # Preppend role embeddings for subject/predicate/object roles
        role_embeddings = self.role_embeddings
        if self._use_role_embeddings and role_embeddings is not None:
            role_flags = getattr(data, "role_flags", None)
            if role_flags is None:
                role_flags = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                role_flags = role_flags.view(-1)
                if role_flags.size(0) != x.size(0):
                    raise ValueError(
                        f"role_flags length {role_flags.size(0)} does not match node feature rows {x.size(0)}"
                    )
                if role_flags.dtype != torch.long:
                    role_flags = role_flags.to(dtype=torch.long)
                if role_flags.device != x.device:
                    role_flags = role_flags.to(device=x.device)
            role_features = role_embeddings(role_flags)
            x = torch.cat([x, role_features], dim=-1)

        # Forward pass through the model
        # x = self.initialization(x, edge_index)
        x = self.initialization(x)
        for conv in self.mp_layers:
            x = F.dropout(x, p=self._dropout, training=self.training)
            if self.use_edge_attributes:
                edge_mlp = self.edge_mlp
                assert edge_mlp is not None
                edge_features = edge_mlp(x[edge_attr])
                if self.use_edge_subtraction:
                    # The optional inverse-message variant encodes reverse edges by
                    # negating their predicate features. Paper configurations disable it.
                    inverse_edge_indices = edge_index.flip(0)
                    inverse_edge_features = -edge_features
                    edge_index_forward = torch.cat([edge_index, inverse_edge_indices], dim=1)
                    edge_features_forward = torch.cat(
                        [edge_features, inverse_edge_features], dim=0
                    )
                else:
                    edge_index_forward = edge_index
                    edge_features_forward = edge_features
                x = conv(x, edge_index_forward, edge_attr=edge_features_forward)
            else:
                x = conv(x, edge_index)
        node_emb = x
        graph_emb = global_mean_pool(x, batch)

        ## Classification Head
        shared = F.leaky_relu(self.shared_projection(graph_emb), negative_slope=0.1)
        shared = F.dropout(shared, p=self._dropout, training=self.training)

        # Role-specificic branches for subject, object, predicate predictions
        subject_features = self.subject_branch(shared)
        subject_features = F.dropout(subject_features, p=self._dropout, training=self.training)

        object_features = self.object_branch(shared)
        object_features = F.dropout(object_features, p=self._dropout, training=self.training)

        predicate_features = self.predicate_branch(shared)
        predicate_features = F.dropout(predicate_features, p=self._dropout, training=self.training)

        policy_logits = None
        if self._policy_enabled and self.policy_head is not None:
            policy_logits = self.policy_head(shared)

        # Entity logits exclude predicate IDs
        # Here we expand to prediction size (filling excluded with negative values)
        # Materialize the fixed global target vocabulary expected by the training
        # and evaluation interfaces, filling excluded IDs with negative values.
        y_add_s = self._expand_entity_logits(self.subject_add_head(subject_features))
        y_del_s = self._expand_entity_logits(self.subject_del_head(subject_features))

        y_add_o = self._expand_entity_logits(self.object_add_head(object_features))
        y_del_o = self._expand_entity_logits(self.object_del_head(object_features))

        # Predicate logits exclude entity IDs
        y_add_p = self._expand_predicate_logits(self.predicate_add_head(predicate_features))
        y_del_p = self._expand_predicate_logits(self.predicate_del_head(predicate_features))

        prediction = torch.stack([y_add_s, y_add_p, y_add_o, y_del_s, y_del_p, y_del_o], dim=1)
        assert prediction.shape == (
            graph_emb.shape[0],
            6,
            self.num_target_ids,
        ), f"Expected {(graph_emb.shape[0], 6, self.num_target_ids)}, got {prediction.shape}"
        factor_outputs = self._compute_factor_outputs(
            node_emb,
            data,
            include_post_gold=include_factor_post_gold,
        )

        return {
            "edit_logits": prediction,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "factor_logits_pre": factor_outputs["factor_logits_pre"],
            "factor_logits_post_gold": factor_outputs["factor_logits_post_gold"],
            "factor_mask_pre": factor_outputs["factor_mask_pre"],
            "factor_graph_index": factor_outputs["factor_graph_index"],
            "policy_logits": policy_logits,
        }

    def forward_for_evaluation(self, data):
        """Run inference without the training-only gold-edit factor branch."""

        return self.forward(data, include_factor_post_gold=False)

    @staticmethod
    def _prepare_class_ids(
        class_ids: Sequence[int] | torch.Tensor | None,
        num_total_ids: int,
    ) -> tuple[torch.Tensor, bool]:
        if class_ids is None:
            tensor = torch.arange(num_total_ids, dtype=torch.long)
            return tensor, True
        if isinstance(class_ids, torch.Tensor):
            tensor = class_ids.to(dtype=torch.long)
        else:
            tensor = torch.tensor([int(idx) for idx in class_ids], dtype=torch.long)

        if tensor.numel() == 0:
            tensor = torch.tensor([0], dtype=torch.long)

        tensor = torch.unique(tensor, sorted=True)
        full_vocab = tensor.numel() == num_total_ids and torch.equal(
            tensor, torch.arange(num_total_ids, dtype=torch.long)
        )
        return tensor, full_vocab

    def _expand_entity_logits(self, logits: torch.Tensor) -> torch.Tensor:
        class_ids = self.entity_class_ids
        return self._expand_logits(logits, class_ids, self._entity_full_vocab)

    def _expand_predicate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        class_ids = self.predicate_class_ids
        return self._expand_logits(logits, class_ids, self._predicate_full_vocab)

    def _expand_logits(
        self, logits: torch.Tensor, class_ids: torch.Tensor, full_vocab: bool
    ) -> torch.Tensor:
        if full_vocab:
            return logits
        batch_size = logits.size(0)
        expanded = logits.new_full((batch_size, self.num_target_ids), self._mask_fill_value)
        scatter_index = class_ids.unsqueeze(0).expand(batch_size, -1)
        expanded.scatter_(1, scatter_index, logits)
        return expanded


class RepairGAT(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        if self.use_edge_attributes:
            # Edge features are dynamically produced by edge_mlp with size=in_channels.
            # Disable self-loops in this mode because there is no predicate label for synthetic self-edges.
            return GATConv(
                in_channels,
                out_channels,
                heads=1,
                concat=True,
                edge_dim=in_channels,
                add_self_loops=False,
            )
        return GATConv(in_channels, out_channels, heads=1, concat=True)


class RepairGIN(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        if self.use_edge_attributes:
            # adds the edge attributes to node featrues, then uses ReLU on top.
            return GINEConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                    Linear(out_channels, out_channels),
                )
            )
        else:
            return GINConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                    Linear(out_channels, out_channels),
                )
            )


class RepairGINFactorPressure(BaseGraphModel):
    EDGE_FACTOR_TO_LOCAL_PREDICATE = 4
    EDGE_FACTOR_TO_LOCAL_SUBJECT = 5
    EDGE_FACTOR_TO_LOCAL_OBJECT = 6

    def __init__(
        self,
        *args,
        pressure_enabled: bool = False,
        pressure_type_conditioning: str = "none",
        pressure_module_sharing: str = "per_type",
        pressure_residual_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            *args,
            pressure_type_conditioning=pressure_type_conditioning,
            pressure_module_sharing=pressure_module_sharing,
            pressure_residual_scale=pressure_residual_scale,
            **kwargs,
        )
        self._pressure_enabled = bool(pressure_enabled)
        self._pressure_type_conditioning = str(pressure_type_conditioning).lower()
        self._pressure_module_sharing = str(pressure_module_sharing).lower()
        self._pressure_residual_scale = float(pressure_residual_scale)
        if self._pressure_type_conditioning not in {"none", "concat", "gate"}:
            raise ValueError(
                "pressure_type_conditioning must be 'none', 'concat', or 'gate'"
            )
        if self._pressure_module_sharing not in {"per_type", "shared"}:
            raise ValueError("pressure_module_sharing must be 'per_type' or 'shared'")
        self._pressure_role_dim = 8
        self._pressure_role_embeddings = nn.Embedding(3, self._pressure_role_dim)
        self._pressure_violation_head = nn.Linear(self.hidden_channels, 1)
        self._pressure_type_dim = (
            self._factor_type_embedding_dim if self._num_factor_types > 0 and self._factor_type_embedding_dim > 0 else 0
        )
        base_pressure_in = self.hidden_channels * 2 + self._pressure_role_dim + 1
        if self._pressure_type_conditioning == "concat" and self._pressure_type_dim > 0:
            pressure_in = base_pressure_in + self._pressure_type_dim
        else:
            pressure_in = base_pressure_in
        self._pressure_mlp = nn.Sequential(
            nn.Linear(pressure_in, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        if self._pressure_type_conditioning == "gate" and self._pressure_type_dim > 0:
            self._pressure_type_gate = nn.Linear(self._pressure_type_dim, self.hidden_channels)
        else:
            self._pressure_type_gate = None
        if self._factor_executor_impl in {"per_type_v1", "per_type_grouped_v2"}:
            pressure_modules_per_role = (
                self._num_factor_executor_modules
                if self._pressure_module_sharing == "per_type"
                else 1
            )
            pressure_input_dim = self._factor_state_dim + self.hidden_channels
            if (
                self._factor_executor_impl == "per_type_grouped_v2"
                and self._pressure_module_sharing == "per_type"
            ):
                self._pressure_role_modules = nn.ModuleDict(
                    {
                        str(role_id): GroupedPressureRole(
                            pressure_modules_per_role,
                            pressure_input_dim,
                            self.hidden_channels,
                        )
                        for role_id in FACTOR_ROLE_IDS
                    }
                )
            else:
                self._pressure_role_modules = nn.ModuleDict(
                    {
                        str(role_id): nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(pressure_input_dim, self.hidden_channels),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_channels, self.hidden_channels),
                                )
                                for _ in range(pressure_modules_per_role)
                            ]
                        )
                        for role_id in FACTOR_ROLE_IDS
                    }
                )
        else:
            self._pressure_role_modules = None

    def create_conv_layer(self, in_channels: int, out_channels: int):
        if self.use_edge_attributes:
            return GINEConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                    Linear(out_channels, out_channels),
                )
            )
        return GINConv(
            Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                BN(out_channels),
                Linear(out_channels, out_channels),
            )
        )

    def _apply_pressure(self, x: torch.Tensor, data) -> torch.Tensor:
        if not self._pressure_enabled or self._constraint_representation != "factorized":
            return x
        if self._factor_executor_impl in {"per_type_v1", "per_type_grouped_v2"}:
            runtime = self._build_factor_scope_runtime(x, data)
            if runtime is None:
                return x
            factor_states, _ = self._run_per_type_factor_executors(runtime)
            aggregated = torch.zeros_like(x)

            def _apply_role(
                role_id: int,
                factor_pos: torch.Tensor,
                dst_index: torch.Tensor,
            ) -> None:
                if factor_pos.numel() == 0 or dst_index.numel() == 0:
                    return
                role_modules = self._pressure_role_modules
                if role_modules is None:
                    return
                role_key = str(role_id)
                compact_type_ids = runtime.factor_type_indices.index_select(0, factor_pos)
                dst_emb = x.index_select(0, dst_index)
                factor_state = factor_states.index_select(0, factor_pos)
                message_input = torch.cat([factor_state, dst_emb], dim=-1)
                role_module = role_modules[role_key]
                if self._pressure_module_sharing == "shared":
                    if not isinstance(role_module, nn.ModuleList):
                        raise RuntimeError("Shared pressure role module has an invalid layout.")
                    per_edge_messages = role_module[0](message_input)
                elif isinstance(role_module, GroupedPressureRole):
                    edge_dispatch = build_grouped_dispatch(
                        compact_type_ids,
                        num_types=self._num_active_factor_types,
                    )
                    per_edge_messages = role_module(message_input, edge_dispatch)
                else:
                    per_edge_messages = x.new_zeros(
                        (dst_index.numel(), self.hidden_channels)
                    )
                    stable_type_ids = runtime.factor_type_ids.index_select(0, factor_pos)
                    for type_id in torch.unique(stable_type_ids).tolist():
                        mask = stable_type_ids == int(type_id)
                        if not mask.any():
                            continue
                        per_edge_messages[mask] = role_module[int(type_id)](
                            message_input[mask]
                        )
                aggregated.index_add_(0, dst_index, per_edge_messages)

            _apply_role(FACTOR_ROLE_PREDICATE, runtime.predicate_factor_pos, runtime.predicate_dst_index)
            _apply_role(FACTOR_ROLE_SUBJECT, runtime.subject_factor_pos, runtime.subject_dst_index)
            _apply_role(FACTOR_ROLE_OBJECT, runtime.object_factor_pos, runtime.object_dst_index)
            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            for dst_index in (
                runtime.predicate_dst_index,
                runtime.subject_dst_index,
                runtime.object_dst_index,
            ):
                if dst_index.numel() == 0:
                    continue
                degree.index_add_(0, dst_index, torch.ones(dst_index.size(0), device=x.device, dtype=x.dtype))
            aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(-1)
            return x + (self._pressure_residual_scale * aggregated)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            return x
        edge_index = edge_index.to(device=x.device)
        edge_type = getattr(data, "edge_type", None)

        role_emb = None
        type_emb = None
        if edge_type is not None:
            edge_type = edge_type.to(device=x.device)
            mask = (
                (edge_type == self.EDGE_FACTOR_TO_LOCAL_PREDICATE)
                | (edge_type == self.EDGE_FACTOR_TO_LOCAL_SUBJECT)
                | (edge_type == self.EDGE_FACTOR_TO_LOCAL_OBJECT)
            )
            if mask.any():
                src = edge_index[0, mask]
                dst = edge_index[1, mask]
                if src.numel() == 0:
                    return x

                role_map = torch.full_like(edge_type, 0)
                role_map[edge_type == self.EDGE_FACTOR_TO_LOCAL_PREDICATE] = 0
                role_map[edge_type == self.EDGE_FACTOR_TO_LOCAL_SUBJECT] = 1
                role_map[edge_type == self.EDGE_FACTOR_TO_LOCAL_OBJECT] = 2
                role_ids = role_map[mask]
                role_emb = self._pressure_role_embeddings(role_ids)
            else:
                edge_type = None

        if edge_type is None:
            factor_node_index = getattr(data, "factor_node_index", None)
            factor_node_mask = getattr(data, "is_factor_node", None)
            if factor_node_index is not None:
                factor_nodes = torch.as_tensor(
                    factor_node_index, device=x.device, dtype=torch.long
                ).view(-1)
            elif factor_node_mask is not None:
                factor_node_mask = torch.as_tensor(
                    factor_node_mask, device=x.device, dtype=torch.bool
                ).view(-1)
                if factor_node_mask.numel() != x.size(0):
                    return x
                factor_nodes = torch.nonzero(factor_node_mask, as_tuple=False).view(-1)
            else:
                return x

            if factor_nodes.numel() == 0:
                return x

            src = edge_index[0]
            dst = edge_index[1]
            mask = torch.isin(src, factor_nodes)
            if not mask.any():
                return x
            src = src[mask]
            dst = dst[mask]
            role_emb = torch.zeros(
                src.size(0),
                self._pressure_role_dim,
                device=x.device,
                dtype=x.dtype,
            )

        if self._pressure_type_conditioning != "none" and self.factor_type_embeddings is not None:
            factor_types = getattr(data, "factor_types", None)
            factor_node_index = getattr(data, "factor_node_index", None)
            if factor_types is not None and factor_node_index is not None:
                factor_types_tensor = torch.as_tensor(
                    factor_types, dtype=torch.long, device=x.device
                ).view(-1)
                factor_node_index = torch.as_tensor(
                    factor_node_index, dtype=torch.long, device=x.device
                ).view(-1)
                if factor_types_tensor.numel() == factor_node_index.numel() and factor_node_index.numel() > 0:
                    invalid_mask = (factor_types_tensor < 0) | (factor_types_tensor >= self._num_factor_types)
                    if invalid_mask.any():
                        invalid_values = sorted({int(v) for v in factor_types_tensor[invalid_mask].tolist()})
                        raise ValueError(
                            f"factor_types contain values outside [0, {self._num_factor_types}): {invalid_values[:8]}"
                        )
                    type_ids = torch.full(
                        (x.size(0),),
                        -1,
                        device=x.device,
                        dtype=torch.long,
                    )
                    type_ids.index_copy_(0, factor_node_index, factor_types_tensor)
                    src_type_ids = type_ids.index_select(0, src)
                    if (src_type_ids < 0).any():
                        raise ValueError("Missing factor type ids on pressure edges.")
                    type_emb = self.factor_type_embeddings(src_type_ids)

        h_c = x.index_select(0, src)
        h_v = x.index_select(0, dst)
        viol = torch.sigmoid(self._pressure_violation_head(h_c))

        message_input = torch.cat([h_c, h_v, role_emb, viol], dim=-1)
        if self._pressure_type_conditioning == "concat" and type_emb is not None:
            message_input = torch.cat([message_input, type_emb], dim=-1)
        messages = self._pressure_mlp(message_input)
        if self._pressure_type_conditioning == "gate" and type_emb is not None:
            gate = torch.sigmoid(self._pressure_type_gate(type_emb))
            messages = messages * gate
        aggregated = torch.zeros_like(x).index_add(0, dst, messages)
        degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
        aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(-1)
        x = x + (self._pressure_residual_scale * aggregated)
        return x

    def forward(self, data, *, include_factor_post_gold: bool = True):
        x, batch = data.x, data.batch
        if self.use_edge_attributes:
            edge_index = data.edge_index_non_flattened
            edge_attr = data.edge_attr_non_flattened
        else:
            edge_index = data.edge_index
            edge_attr = None

        if self._use_node_embeddings:
            if x.dtype not in (torch.long, torch.int64, torch.int32):
                x = x.long()
            x = self.node_embeddings(x)
        else:
            if not torch.is_floating_point(x):
                x = x.float()
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

        role_embeddings = self.role_embeddings
        if self._use_role_embeddings and role_embeddings is not None:
            role_flags = getattr(data, "role_flags", None)
            if role_flags is None:
                role_flags = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                role_flags = role_flags.view(-1)
                if role_flags.size(0) != x.size(0):
                    raise ValueError(
                        f"role_flags length {role_flags.size(0)} does not match node feature rows {x.size(0)}"
                    )
                if role_flags.dtype != torch.long:
                    role_flags = role_flags.to(dtype=torch.long)
                if role_flags.device != x.device:
                    role_flags = role_flags.to(device=x.device)
            role_features = role_embeddings(role_flags)
            x = torch.cat([x, role_features], dim=-1)

        x = self.initialization(x)
        for conv in self.mp_layers:
            x = F.dropout(x, p=self._dropout, training=self.training)
            if self.use_edge_attributes:
                edge_mlp = self.edge_mlp
                assert edge_mlp is not None
                edge_features = edge_mlp(x[edge_attr])
                if self.use_edge_subtraction:
                    inverse_edge_indices = edge_index.flip(0)
                    inverse_edge_features = -edge_features
                    edge_index_forward = torch.cat([edge_index, inverse_edge_indices], dim=1)
                    edge_features_forward = torch.cat(
                        [edge_features, inverse_edge_features], dim=0
                    )
                else:
                    edge_index_forward = edge_index
                    edge_features_forward = edge_features
                x = conv(x, edge_index_forward, edge_attr=edge_features_forward)
            else:
                x = conv(x, edge_index)
            x = self._apply_pressure(x, data)

        node_emb = x
        graph_emb = global_mean_pool(x, batch)

        shared = F.leaky_relu(self.shared_projection(graph_emb), negative_slope=0.1)
        shared = F.dropout(shared, p=self._dropout, training=self.training)

        subject_features = self.subject_branch(shared)
        subject_features = F.dropout(subject_features, p=self._dropout, training=self.training)

        object_features = self.object_branch(shared)
        object_features = F.dropout(object_features, p=self._dropout, training=self.training)

        predicate_features = self.predicate_branch(shared)
        predicate_features = F.dropout(predicate_features, p=self._dropout, training=self.training)

        policy_logits = None
        if self._policy_enabled and self.policy_head is not None:
            policy_logits = self.policy_head(shared)

        y_add_s = self._expand_entity_logits(self.subject_add_head(subject_features))
        y_del_s = self._expand_entity_logits(self.subject_del_head(subject_features))

        y_add_o = self._expand_entity_logits(self.object_add_head(object_features))
        y_del_o = self._expand_entity_logits(self.object_del_head(object_features))

        y_add_p = self._expand_predicate_logits(self.predicate_add_head(predicate_features))
        y_del_p = self._expand_predicate_logits(self.predicate_del_head(predicate_features))

        prediction = torch.stack([y_add_s, y_add_p, y_add_o, y_del_s, y_del_p, y_del_o], dim=1)
        assert prediction.shape == (
            graph_emb.shape[0],
            6,
            self.num_target_ids,
        ), f"Expected {(graph_emb.shape[0], 6, self.num_target_ids)}, got {prediction.shape}"

        factor_outputs = self._compute_factor_outputs(
            node_emb,
            data,
            include_post_gold=include_factor_post_gold,
        )

        return {
            "edit_logits": prediction,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "factor_logits_pre": factor_outputs["factor_logits_pre"],
            "factor_logits_post_gold": factor_outputs["factor_logits_post_gold"],
            "factor_mask_pre": factor_outputs["factor_mask_pre"],
            "factor_graph_index": factor_outputs["factor_graph_index"],
            "policy_logits": policy_logits,
        }


def _select_factor_nodes(
    node_emb: torch.Tensor,
    data,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    factor_node_index, factor_graph_index = _select_factor_node_indices(node_emb, data)
    if factor_node_index is None or factor_graph_index is None:
        return None, None
    return node_emb.index_select(0, factor_node_index), factor_graph_index


class RepairGCN(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        assert not self.use_edge_attributes, "GCNConv does not support edge attributes."
        return GCNConv(in_channels, out_channels, improved=True)


MODEL_REGISTRY: dict[str, Callable[..., BaseGraphModel]] = {
    "GAT": RepairGAT,
    "GIN": RepairGIN,
    "GIN_PRESSURE": RepairGINFactorPressure,
    "GCN": RepairGCN,
}


def build_model(model_name: str, num_input_graph_nodes: int, config: ModelConfig) -> BaseGraphModel:
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc

    return model_cls(
        num_input_graph_nodes=num_input_graph_nodes,
        num_embedding_size=config.num_embedding_size,
        num_layers=config.num_layers,
        hidden=config.hidden_channels,
        head_hidden=config.head_hidden,
        dropout=config.dropout,
        use_node_embeddings=config.use_node_embeddings,
        use_role_embeddings=config.use_role_embeddings,
        num_role_types=config.num_role_types,
        role_embedding_dim=config.role_embedding_dim,
        use_edge_attributes=config.use_edge_attributes,
        use_edge_subtraction=config.use_edge_subtraction,
        entity_class_ids=config.entity_class_ids,
        predicate_class_ids=config.predicate_class_ids,
        num_factor_types=config.num_factor_types,
        active_factor_type_ids=getattr(config, "active_factor_type_ids", None),
        factor_type_embedding_dim=config.factor_type_embedding_dim,
        factor_executor_impl=getattr(config, "factor_executor_impl", "per_type_v1"),
        gold_edit_embedding_mode=getattr(config, "gold_edit_embedding_mode", "full"),
        constraint_representation=getattr(config, "constraint_representation", "factorized"),
        pressure_enabled=config.pressure_enabled,
        pressure_type_conditioning=getattr(config, "pressure_type_conditioning", "none"),
        pressure_module_sharing=getattr(config, "pressure_module_sharing", "per_type"),
        pressure_residual_scale=getattr(config, "pressure_residual_scale", 0.1),
        enable_policy_choice=getattr(config, "enable_policy_choice", False),
        policy_num_classes=getattr(config, "policy_num_classes", 6),
    )


__all__ = [
    "BaseGraphModel",
    "FactorPostEditHead",
    "FactorTypeExecutor",
    "GroupedFactorPostEditHead",
    "GroupedFactorTypeExecutor",
    "GroupedPressureRole",
    "RepairGAT",
    "RepairGIN",
    "RepairGINFactorPressure",
    "RepairGCN",
    "build_model",
    "MODEL_REGISTRY",
]
