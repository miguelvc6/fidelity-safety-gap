"""Packed type-conditioned linear layers with grouped-CUDA dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GroupedDispatch:
    """Permutation and jagged-group metadata for compact type indices."""

    order: torch.Tensor
    inverse_order: torch.Tensor
    compact_type_ids_sorted: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor


def build_grouped_dispatch(
    compact_type_ids: torch.Tensor,
    *,
    num_types: int,
) -> GroupedDispatch:
    compact_type_ids = compact_type_ids.to(dtype=torch.long).view(-1)
    if compact_type_ids.numel() == 0:
        empty_long = torch.empty((0,), dtype=torch.long, device=compact_type_ids.device)
        return GroupedDispatch(
            order=empty_long,
            inverse_order=empty_long.clone(),
            compact_type_ids_sorted=empty_long.clone(),
            counts=torch.zeros((num_types,), dtype=torch.long, device=compact_type_ids.device),
            offsets=torch.zeros((num_types,), dtype=torch.int32, device=compact_type_ids.device),
        )
    invalid = (compact_type_ids < 0) | (compact_type_ids >= int(num_types))
    if invalid.any():
        values = sorted({int(value) for value in compact_type_ids[invalid].detach().cpu().tolist()})
        raise ValueError(f"Compact factor type ids outside [0, {num_types}): {values[:8]}")
    order = torch.argsort(compact_type_ids, stable=True)
    sorted_ids = compact_type_ids.index_select(0, order)
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(
        0,
        order,
        torch.arange(order.numel(), dtype=torch.long, device=order.device),
    )
    counts = torch.bincount(sorted_ids, minlength=int(num_types))
    offsets = counts.cumsum(0).to(dtype=torch.int32)
    return GroupedDispatch(
        order=order,
        inverse_order=inverse_order,
        compact_type_ids_sorted=sorted_ids,
        counts=counts,
        offsets=offsets,
    )


class GroupedLinearBank(nn.Module):
    """A bank of independent linear layers evaluated as a ragged grouped GEMM."""

    def __init__(self, num_types: int, input_dim: int, output_dim: int):
        super().__init__()
        self.num_types = int(num_types)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.empty(self.num_types, self.output_dim, self.input_dim))
        self.bias = nn.Parameter(torch.empty(self.num_types, self.output_dim))
        self.reset_parameters()
        self.last_backend = "uninitialized"

    def reset_parameters(self) -> None:
        for type_index in range(self.num_types):
            nn.init.kaiming_uniform_(self.weight[type_index], a=math.sqrt(5))
            fan_in = self.input_dim
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[type_index], -bound, bound)

    @staticmethod
    def _bf16_cuda_supported(inputs: torch.Tensor) -> bool:
        if not inputs.is_cuda or not hasattr(F, "grouped_mm"):
            return False
        major, _minor = torch.cuda.get_device_capability(inputs.device)
        return major >= 8 and (
            inputs.dtype == torch.bfloat16 or torch.is_autocast_enabled("cuda")
        )

    def _grouped_cuda_supported(self, inputs: torch.Tensor) -> bool:
        return self.output_dim % 8 == 0 and self._bf16_cuda_supported(inputs)

    def forward_sorted(
        self,
        inputs: torch.Tensor,
        dispatch: GroupedDispatch,
    ) -> torch.Tensor:
        if inputs.dim() != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(
                f"Expected grouped linear inputs (*,{self.input_dim}), got {tuple(inputs.shape)}"
            )
        if inputs.size(0) != dispatch.compact_type_ids_sorted.numel():
            raise ValueError("Grouped dispatch length does not match linear input rows.")
        if inputs.numel() == 0:
            return inputs.new_empty((0, self.output_dim))

        if self.output_dim == 1:
            use_bf16 = self._bf16_cuda_supported(inputs)
            compute_inputs = inputs.to(dtype=torch.bfloat16) if use_bf16 else inputs
            selected_weight = self.weight[:, 0, :].index_select(
                0,
                dispatch.compact_type_ids_sorted,
            ).to(dtype=compute_inputs.dtype)
            selected_bias = self.bias[:, 0].index_select(
                0,
                dispatch.compact_type_ids_sorted,
            ).to(dtype=compute_inputs.dtype)
            self.last_backend = "vectorized_dot_bf16" if use_bf16 else "vectorized_dot"
            return (
                (compute_inputs * selected_weight).sum(dim=-1, keepdim=True)
                + selected_bias.unsqueeze(-1)
            )

        if self._grouped_cuda_supported(inputs):
            alignment = 8
            padded_input_dim = ((self.input_dim + alignment - 1) // alignment) * alignment
            pad = padded_input_dim - self.input_dim
            grouped_inputs = F.pad(inputs, (0, pad)).to(dtype=torch.bfloat16)
            grouped_weight = F.pad(self.weight, (0, pad)).transpose(1, 2).to(dtype=torch.bfloat16)
            outputs = F.grouped_mm(
                grouped_inputs,
                grouped_weight,
                offs=dispatch.offsets,
            )
            grouped_bias = torch.repeat_interleave(
                self.bias.to(dtype=outputs.dtype),
                dispatch.counts,
                dim=0,
            )
            self.last_backend = "grouped_mm_bf16"
            return outputs + grouped_bias

        outputs = inputs.new_empty((inputs.size(0), self.output_dim))
        start = 0
        offsets = dispatch.offsets.detach().cpu().tolist()
        for type_index, end in enumerate(offsets):
            end = int(end)
            if end > start:
                outputs[start:end] = F.linear(
                    inputs[start:end],
                    self.weight[type_index],
                    self.bias[type_index],
                )
            start = end
        self.last_backend = "segmented_linear"
        return outputs


__all__ = [
    "GroupedDispatch",
    "GroupedLinearBank",
    "build_grouped_dispatch",
]
