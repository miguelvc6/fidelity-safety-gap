

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from modules.config import ModelConfig
from modules.models import BaseGraphModel, build_model


@dataclass
class RerankerConfig:
    candidate_embedding_dim: int = 64
    candidate_hidden_dim: int = 128
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RerankerConfig":
        payload = dict(data or {})
        return cls(
            candidate_embedding_dim=int(payload.get("candidate_embedding_dim", cls.candidate_embedding_dim)),
            candidate_hidden_dim=int(payload.get("candidate_hidden_dim", cls.candidate_hidden_dim)),
            dropout=float(payload.get("dropout", cls.dropout)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_embedding_dim": self.candidate_embedding_dim,
            "candidate_hidden_dim": self.candidate_hidden_dim,
            "dropout": self.dropout,
        }


class CandidateReranker(nn.Module):
    """Score candidate edits conditioned on a graph embedding."""

    def __init__(
        self,
        graph_encoder: BaseGraphModel,
        *,
        candidate_embedding_dim: int = 64,
        candidate_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = graph_encoder
        self.num_target_ids = graph_encoder.num_target_ids
        self.graph_embedding_dim = graph_encoder.hidden_channels

        self.candidate_embeddings = nn.Embedding(self.num_target_ids, candidate_embedding_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_embedding_dim * 6, candidate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.graph_embedding_dim + candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(candidate_hidden_dim, 1),
        )

    def encode_graphs(self, data) -> torch.Tensor:
        outputs = self.graph_encoder(data)
        graph_emb = outputs.get("graph_emb")
        if graph_emb is None:
            raise ValueError("graph_encoder must return 'graph_emb' in its outputs")
        return graph_emb

    def score_candidates(self, graph_emb: torch.Tensor, candidate_slots: torch.Tensor) -> torch.Tensor:
        if candidate_slots.dim() != 2 or candidate_slots.size(-1) != 6:
            raise ValueError(
                f"candidate_slots must be shaped (num_candidates, 6); got {tuple(candidate_slots.shape)}"
            )
        candidate_slots = candidate_slots.to(device=graph_emb.device)
        if candidate_slots.dtype != torch.long:
            candidate_slots = candidate_slots.to(dtype=torch.long)
        cand_embed = self.candidate_embeddings(candidate_slots)
        cand_embed = cand_embed.view(candidate_slots.size(0), -1)
        cand_features = self.candidate_mlp(cand_embed)
        if graph_emb.dim() == 1:
            graph_features = graph_emb.view(1, -1).expand(cand_features.size(0), -1)
        else:
            graph_features = graph_emb.expand(cand_features.size(0), -1)
        combined = torch.cat([graph_features, cand_features], dim=-1)
        return self.scorer(combined).squeeze(-1)

    def score_candidates_packed(
        self,
        graph_emb: torch.Tensor,
        candidate_slots: torch.Tensor,
        candidate_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score packed candidate rows for a batch without per-graph GPU calls."""
        if graph_emb.dim() != 2:
            raise ValueError(f"graph_emb must be shaped (batch, hidden); got {tuple(graph_emb.shape)}")
        if candidate_slots.dim() != 2 or candidate_slots.size(-1) != 6:
            raise ValueError(
                f"candidate_slots must be shaped (num_candidates, 6); got {tuple(candidate_slots.shape)}"
            )
        candidate_slots = candidate_slots.to(device=graph_emb.device, dtype=torch.long)
        candidate_graph_index = candidate_graph_index.to(
            device=graph_emb.device, dtype=torch.long
        ).view(-1)
        if candidate_graph_index.numel() != candidate_slots.size(0):
            raise ValueError("candidate_graph_index length must match candidate_slots rows.")
        if candidate_graph_index.numel() and (
            int(candidate_graph_index.min().item()) < 0
            or int(candidate_graph_index.max().item()) >= graph_emb.size(0)
        ):
            raise ValueError("candidate_graph_index contains out-of-range graph ids.")

        cand_embed = self.candidate_embeddings(candidate_slots)
        cand_features = self.candidate_mlp(cand_embed.view(candidate_slots.size(0), -1))
        graph_features = graph_emb.index_select(0, candidate_graph_index)
        combined = torch.cat([graph_features, cand_features], dim=-1)
        return self.scorer(combined).squeeze(-1)


def build_reranker(
    *,
    num_input_graph_nodes: int,
    model_cfg: ModelConfig,
    reranker_cfg: RerankerConfig,
) -> CandidateReranker:
    graph_encoder = build_model(model_cfg.model, num_input_graph_nodes, model_cfg)
    return CandidateReranker(
        graph_encoder,
        candidate_embedding_dim=reranker_cfg.candidate_embedding_dim,
        candidate_hidden_dim=reranker_cfg.candidate_hidden_dim,
        dropout=reranker_cfg.dropout,
    )


__all__ = ["CandidateReranker", "RerankerConfig", "build_reranker"]
