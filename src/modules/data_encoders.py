import logging
import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

ROLE_NONE = 0
ROLE_SUBJECT = 1  # 0b001
ROLE_PREDICATE = 2  # 0b010
ROLE_OBJECT = 4  # 0b100
NUM_ROLE_TYPES = 8  # Support all bitwise combinations of the three focus roles.
ROLE_NAMES: tuple[str, ...] = (
    "none",
    "subject",
    "predicate",
    "subject+predicate",
    "object",
    "subject+object",
    "predicate+object",
    "subject+predicate+object",
)


class RoleFeatureSpec(NamedTuple):
    enabled: bool
    num_types: int
    dtype: torch.dtype


class ArtifactWriteResult(NamedTuple):
    path: Path
    bytes_written: int
    checksum: str


class GraphArtifactInfo(NamedTuple):
    path: Path
    format: str


def _torch_load_trusted(path: Path) -> Any:
    """Load trusted local torch artifacts with PyTorch 2.6+ compatibility."""
    return torch.load(path, map_location="cpu", weights_only=False)


SCALAR_FEATURES: tuple[str, ...] = (
    "constraint_id",
    "subject",
    "predicate",
    "object",
    "other_subject",
    "other_predicate",
    "other_object",
    "add_subject",
    "add_predicate",
    "add_object",
    "del_subject",
    "del_predicate",
    "del_object",
)

SEQUENCE_FEATURES: tuple[str, ...] = (
    "constraint_predicates",
    "constraint_objects",
    "subject_predicates",
    "subject_objects",
    "object_predicates",
    "object_objects",
    "other_entity_predicates",
    "other_entity_objects",
)


class GlobalIntEncoder:
    """Bidirectional mapping **string <-> integer** shared across the script.

    The encoder assigns a unique integer to every distinct value (URI, literal,
    placeholder token, …) it encounters. Value "" is reserved for *0*.
    """

    def __init__(self) -> None:
        self._encoding: dict[str, int] = {"": 0}
        self._decoding: dict[int, str] = {0: ""}
        self._frozen = False
        self._filtered_ids: set[int] = set()
        # maps from `global_id` to a a contiguous global_id, with filtered ids removed!
        self._global_id_to_unfiltered_global_id: dict[int, int] = {}

    def encode(self, value: str | None, add_new: bool | None = None) -> int:
        """Return an integer identifier for *value*.

        If *value* has not been seen before, a new id is created.
        """
        if add_new is None:
            add_new = not self._frozen
        if value is None:
            value = ""
        value = str(value)
        if value not in self._encoding:
            if not add_new:
                return 0
            self._encoding[value] = len(self._encoding)
            self._decoding[len(self._decoding)] = value

        return self._encoding[value]

    def decode(self, id: int | None, use_filtered_id_mapping: bool = False) -> str | None:
        """Decode *value* back to its original string (if known)."""
        if id is None:
            raise Exception("Input value is None.")
        if use_filtered_id_mapping:
            unfiltered_global_id_to_global_id = {v: k for k, v in self._global_id_to_unfiltered_global_id.items()}
            assert id in unfiltered_global_id_to_global_id, f"Value {id} not found in unfiltered global IDs."
            id = unfiltered_global_id_to_global_id.get(id, id)
        decoded_value = self._decoding.get(id, None)
        if decoded_value is not None:
            decoded_value = decoded_value.replace(">", "").replace("<", "")
        return decoded_value

    def filter_ids(self, ids: Iterable[int]) -> None:
        assert "unknown" in self._encoding, "The 'unknown' id must be present before filtering."
        self._filtered_ids.update(ids)

        self._global_id_to_unfiltered_global_id = {}
        for global_id in range(len(self._decoding)):
            if global_id in self._filtered_ids:
                continue
            local_id = len(self._global_id_to_unfiltered_global_id)
            self._global_id_to_unfiltered_global_id[global_id] = local_id

    def get_unfiltered_global_id(self, global_id: int):
        if len(self._filtered_ids) == 0 and len(self._global_id_to_unfiltered_global_id) == 0:
            return global_id
        if global_id in self._filtered_ids:
            global_id = self._encoding["unknown"]
        return self._global_id_to_unfiltered_global_id[global_id]

    def save(self, file: str | Path) -> None:
        """Persist the current vocabulary to *file*."""
        with open(file, "wb") as fp:
            pickle.dump(
                (
                    self._encoding,
                    self._decoding,
                    self._filtered_ids,
                    self._global_id_to_unfiltered_global_id,
                ),
                fp,
            )

    def load(self, file: str | Path) -> None:
        """Load vocabulary from *file*."""
        with open(file, "rb") as fp:
            (
                self._encoding,
                self._decoding,
                self._filtered_ids,
                self._global_id_to_unfiltered_global_id,
            ) = pickle.load(fp)
        # in alignment with previous store and load method
        self._frozen = False

    def freeze(self) -> None:
        """Stop expanding the vocabulary when encoding new values."""
        self._frozen = True


def dataset_variant_name(dataset: str, min_occurrence: int) -> str:
    """Return the dataset name variant for the given *min_occurrence*."""
    if "_minocc" in dataset:
        return dataset
    if min_occurrence <= 1:
        return dataset
    return f"{dataset}_minocc{min_occurrence}"


def base_dataset_name(dataset: str) -> str:
    """Strip any *_minocc<digits> suffix from dataset."""
    if "_minocc" not in dataset:
        return dataset
    prefix, suffix = dataset.rsplit("_minocc", 1)
    if suffix.isdigit():
        return prefix
    return dataset


def graph_dataset_filename(
    split: str,
    encoding: str,
    *,
    constraint_representation: str = "factorized",
) -> str:
    """Return the persisted graph filename for the given split/encoding/representation."""
    representation = str(constraint_representation).lower()
    if representation == "factorized":
        return f"{split}_graph-{encoding}.pkl"
    if representation == "eswc_passive":
        return f"{split}_graph_repr-eswc_passive-{encoding}.pkl"
    raise ValueError(
        "constraint_representation must be 'factorized' or 'eswc_passive', "
        f"got {constraint_representation!r}"
    )


def discover_min_occurrence(dataset: str) -> int:
    """Auto-detect the lowest *min_occurrence* for the given *dataset*."""
    interim_root = Path("data/interim")
    if not interim_root.exists():
        raise FileNotFoundError(f"Interim directory not found: {interim_root}")

    prefix = f"{dataset}_minocc"
    candidates: list[int] = []

    for candidate in interim_root.iterdir():
        if not candidate.is_dir():
            continue
        if candidate.name == dataset:
            if any(candidate.glob("df_*.parquet")):
                candidates.append(1)
            continue
        if not candidate.name.startswith(prefix):
            continue
        suffix = candidate.name[len(prefix) :]
        if not suffix.isdigit():
            continue
        if not any(candidate.glob("df_*.parquet")):
            continue
        candidates.append(int(suffix))

    if not candidates:
        raise FileNotFoundError(
            f"No parquet datasets found for dataset='{dataset}' under {interim_root}. "
            "Provide --min-occurrence explicitly."
        )

    min_occurrence = min(candidates)
    return min_occurrence


class GlobalToLocalNodeMap:
    """Maps global node IDs to local node IDs and stores node attributes."""

    def __init__(self, literal_id: Any = None) -> None:
        self.LITERAL_ID = literal_id if literal_id is not None else "LITERAL"

        self.global_to_local: dict[int, int] = {}
        self.local_to_global: dict[int, int] = {}

        self.local_attributes: list[Any] = []
        self.local_names: list[str] = []

    def _is_literal_attribute(self, attr: Any) -> bool:
        if isinstance(attr, np.ndarray):
            return False
        return attr == self.LITERAL_ID

    @staticmethod
    def _attributes_equal(lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
            return np.array_equal(lhs, rhs) or np.allclose(lhs, rhs)
        if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            return False
        return lhs == rhs

    def store(
        self,
        global_node_id: int,
        node_attributes: Any,
        name: str | None = None,
        force_create: bool = False,
    ) -> int:
        """Encode a global node ID to a local node ID."""
        if global_node_id in self.global_to_local:
            # Literal text is not always available, so all occurrences of a
            # literal global ID use the stable literal sentinel attribute.
            existing_attr = self.local_attributes[self.global_to_local[global_node_id]]
            if self._is_literal_attribute(node_attributes) or self._is_literal_attribute(existing_attr):
                node_attributes = self.LITERAL_ID
                self.local_attributes[self.global_to_local[global_node_id]] = self.LITERAL_ID
                existing_attr = self.LITERAL_ID
            elif not self._attributes_equal(node_attributes, existing_attr):
                logging.debug(
                    "Node attributes mismatch for global node ID %s; reusing existing attributes.",
                    global_node_id,
                )
                node_attributes = existing_attr

            if not force_create:
                return self.global_to_local[global_node_id]
        return self.store_with_new_id(global_node_id, node_attributes, name)

    def store_with_new_id(self, global_node_id: int, node_attributes: Any, name: str | None = None) -> int:
        """Store a new global node ID and return a new local node ID."""
        assert global_node_id != 0, "Global node ID must not be zero."
        if name is None:
            name = f"node_{global_node_id}"
        local_node_id = len(self.local_attributes)
        self.local_attributes.append(node_attributes)
        self.local_names.append(name)

        self.local_to_global[local_node_id] = global_node_id
        if global_node_id not in self.global_to_local:
            self.global_to_local[global_node_id] = local_node_id
        return local_node_id

    def get_global_id_with_attributes(self, local_node_id: int) -> tuple[int, Any]:
        """Decode a local node ID to a global node ID."""
        if local_node_id not in self.local_to_global:
            raise ValueError(f"Local node ID {local_node_id} does not exist.")
        return self.local_to_global[local_node_id], self.local_attributes[local_node_id]

    def get_name(self, local_node_id: int) -> str:
        """Get the name of a local node ID."""
        if local_node_id < 0 or local_node_id >= len(self.local_names):
            raise ValueError(f"Local node ID {local_node_id} is out of bounds.")
        return self.local_names[local_node_id]


class WikidataCacheEntry(NamedTuple):
    text: str
    embedding: np.ndarray


class PrecomputedWikidataCache:
    """
    Loads a parquet cache with pre-computed text labels and embeddings for
    Wikidata entities and literals.
    """

    def __init__(self, cache_path: Path):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Precomputed Wikidata cache not found at {cache_path}. Run 04_wikidata_retriever.py first."
            )

        dataframe = pd.read_parquet(cache_path)

        self._id_map: dict[int, WikidataCacheEntry] = {}
        self._literal_map: dict[str, WikidataCacheEntry] = {}

        # Load entries
        for row in dataframe.itertuples(index=False):
            embedding = np.asarray(row.embedding, dtype=np.float32)
            entry = WikidataCacheEntry(text=row.text, embedding=embedding)
            if row.global_id is not None and not pd.isna(row.global_id):
                self._id_map[int(row.global_id)] = entry
            if row.kind == "literal":
                self._literal_map[row.key] = entry

        if not self._id_map:
            raise ValueError(f"Empty Wikidata cache at {cache_path}")

        # Determine fallback entry
        fallback_id = next((gid for gid, entry in self._id_map.items() if entry.text == "unknown"), None)
        if fallback_id is None:
            fallback_id = next(iter(self._id_map))
        self._fallback_id = fallback_id
        self._fallback_entry = self._id_map[self._fallback_id]

        self._embed_dim = int(len(self._fallback_entry.embedding))

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def fallback_id(self) -> int:
        return self._fallback_id

    def _convert_embedding(self, embedding: np.ndarray, dtype: np.dtype | None) -> np.ndarray:
        """Convert the embedding to the desired dtype."""
        if dtype is None:
            return embedding
        if embedding.dtype == dtype:
            return embedding
        return embedding.astype(dtype)

    def get_text_for_id(self, global_id: int, fallback_id: int | None = None) -> str | None:
        """Get the text label for a given global ID."""
        entry = self._id_map.get(global_id)
        if entry is None and fallback_id is not None:
            entry = self._id_map.get(fallback_id)
        if entry is None:
            return None
        return entry.text

    def get_embedding_for_id(
        self,
        global_id: int,
        dtype: np.dtype | None = None,
        fallback_id: int | None = None,
    ) -> np.ndarray:
        """Get the embedding for a given global ID."""
        entry = self._id_map.get(global_id)
        if entry is None and fallback_id is not None:
            entry = self._id_map.get(fallback_id)
        if entry is None:
            entry = self._fallback_entry
        return self._convert_embedding(entry.embedding, dtype)

    def get_embedding_for_literal(
        self,
        literal: str,
        dtype: np.dtype | None = None,
        fallback_id: int | None = None,
    ) -> np.ndarray:
        """Get the embedding for a given literal string."""
        entry = self._literal_map.get(literal)
        if entry is None:
            return self.get_embedding_for_id(fallback_id or self._fallback_id, dtype)
        return self._convert_embedding(entry.embedding, dtype)


def iter_stream(path: str | Path) -> Iterator[Data]:
    """Yield pickled ``Data`` objects from ``path`` one at a time."""
    with open(Path(path), "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_stream(path: str | Path) -> list[Data]:
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out


def dump_in_shards(
    objs: Iterable[Data],
    base_path: Path,
    shard_size: int,
    use_torch_save: bool,
    atomic_write: bool = True,
    start_shard: int = 0,
) -> tuple[int, int, list[ArtifactWriteResult]]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[Data] = []
    shard = max(0, int(start_shard))
    total = 0
    artifacts: list[ArtifactWriteResult] = []
    written_shards = 0
    for obj in objs:
        buf.append(obj)
        if len(buf) == shard_size:
            artifacts.append(_write_shard(buf, base_path, shard, use_torch_save, atomic_write=atomic_write))
            total += len(buf)
            buf.clear()
            shard += 1
            written_shards += 1
    if buf:
        artifacts.append(_write_shard(buf, base_path, shard, use_torch_save, atomic_write=atomic_write))
        total += len(buf)
        buf.clear()
        shard += 1
        written_shards += 1
    logging.info(
        "Wrote %s graph objects across %s newly written shard(s) using base %s",
        total,
        written_shards,
        base_path,
    )
    return total, written_shards, artifacts


def _write_shard(
    buf: list[Data],
    base_path: Path,
    shard: int,
    use_torch: bool,
    atomic_write: bool = True,
) -> ArtifactWriteResult:
    shard_path = base_path.with_name(
        f"{base_path.stem}-shard{shard:03d}{base_path.suffix if not use_torch else '.pt'}"
    )
    destination = shard_path.with_suffix(shard_path.suffix + ".tmp") if atomic_write else shard_path
    if use_torch:
        torch.save(buf, destination)
    else:
        with open(destination, "wb") as f:
            pickle.dump(buf, f, protocol=5)
    if atomic_write:
        destination.replace(shard_path)
    digest = _compute_prefix_sha256(shard_path)
    size_bytes = shard_path.stat().st_size
    logging.debug("Wrote shard %s", shard_path)
    return ArtifactWriteResult(path=shard_path, bytes_written=size_bytes, checksum=digest)


def dump_stream(
    objs: Iterable[Data],
    path: str | Path,
    protocol: int = 5,
    atomic_write: bool = True,
) -> tuple[int, ArtifactWriteResult]:
    """Stream ``objs`` to ``path`` using pickle one object at a time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_suffix(path.suffix + ".tmp") if atomic_write else path
    count = 0
    with open(destination, "wb") as f:
        for obj in objs:
            pickle.dump(obj, f, protocol=protocol)
            count += 1
    if atomic_write:
        destination.replace(path)
    digest = _compute_prefix_sha256(path)
    size_bytes = path.stat().st_size
    logging.info("Wrote %s graph objects to %s", count, path)
    return count, ArtifactWriteResult(path=path, bytes_written=size_bytes, checksum=digest)


def _compute_prefix_sha256(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_bytes: int = 16 * 1024 * 1024,
) -> str:
    hasher = sha256()
    consumed = 0
    with path.open("rb") as handle:
        while consumed < max_bytes:
            remaining = max_bytes - consumed
            block = handle.read(min(chunk_size, remaining))
            if not block:
                break
            hasher.update(block)
            consumed += len(block)
    return hasher.hexdigest()


class GraphStreamDataset(IterableDataset):
    """Lazily yield ``Data`` objects from a pickle stream."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            yield from self._iterate_stream(0, 1)
        else:
            yield from self._iterate_stream(worker.id, worker.num_workers)

    def _iterate_stream(self, worker_id: int, num_workers: int):
        with self.path.open("rb") as f:
            index = 0
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break

                if index % num_workers == worker_id:
                    # Keep a stable global graph index so downstream losses can align
                    # streamed graphs with sidecar context/row arrays.
                    setattr(obj, "context_index", index)
                    yield obj
                index += 1

    def __len__(self):
        raise TypeError("GraphStreamDataset does not support len()")


class ShardedGraphStreamDataset(GraphStreamDataset):
    """Lazily iterate graphs stored as a sequence of shard files."""

    def __init__(self, shard_paths: Iterable[Path]):
        shard_list = [Path(p) for p in shard_paths]
        if not shard_list:
            raise ValueError("ShardedGraphStreamDataset requires at least one shard path.")
        super().__init__(shard_list[0])
        self.shard_paths = shard_list

    def _iterate_stream(self, worker_id: int, num_workers: int):
        index = 0
        for shard_path in self.shard_paths:
            if shard_path.suffix == ".pt":
                shard_objects = _torch_load_trusted(shard_path)
            else:
                with shard_path.open("rb") as f:
                    shard_objects = pickle.load(f)
            if not isinstance(shard_objects, list):
                raise TypeError(f"Expected list payload in shard {shard_path}, got {type(shard_objects)!r}")
            for obj in shard_objects:
                if index % num_workers == worker_id:
                    # Keep a stable global graph index across shards.
                    setattr(obj, "context_index", index)
                    yield obj
                index += 1
            del shard_objects


def discover_graph_artifacts(path: Path) -> list[GraphArtifactInfo]:
    """
    Discover graph artifacts for ``path``.

    Returns a list with one entry for a monolithic ``.pkl`` file, or multiple
    entries for sharded ``-shardNNN.(pkl|pt)`` files.
    """
    path = Path(path)
    artifacts: list[GraphArtifactInfo] = []
    if path.exists():
        artifacts.append(GraphArtifactInfo(path=path, format="stream"))
        return artifacts

    shard_candidates: list[Path] = []
    shard_candidates.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pkl")))
    shard_candidates.extend(sorted(path.parent.glob(f"{path.stem}-shard*.pt")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in shard_candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)

    for shard_path in sorted(deduped):
        fmt = "shard_torch" if shard_path.suffix == ".pt" else "shard_pickle"
        artifacts.append(GraphArtifactInfo(path=shard_path, format=fmt))
    return artifacts


def _peek_first_graph(dataset: list[Data] | GraphStreamDataset) -> Data | None:
    if isinstance(dataset, list):
        return dataset[0] if dataset else None

    iterator = iter(dataset)
    try:
        return next(iterator)
    except StopIteration:
        return None
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _infer_role_feature_spec(role_flags: Any) -> RoleFeatureSpec:
    if role_flags is None:
        return RoleFeatureSpec(False, 0, torch.long)
    if isinstance(role_flags, RoleFeatureSpec):
        return role_flags
    if isinstance(role_flags, torch.Tensor):
        role_tensor = role_flags
    else:
        role_tensor = torch.as_tensor(role_flags)
    if role_tensor.numel() == 0:
        dtype = role_tensor.dtype if hasattr(role_tensor, "dtype") and role_tensor.dtype is not None else torch.long
        return RoleFeatureSpec(False, 0, dtype)
    role_tensor = role_tensor.view(-1)
    if torch.is_floating_point(role_tensor):
        role_tensor = role_tensor.to(dtype=torch.long)
    dtype = role_tensor.dtype
    max_value = int(role_tensor.max().item())
    if max_value < 0:
        return RoleFeatureSpec(False, 0, dtype)
    num_types = max_value + 1
    if num_types < NUM_ROLE_TYPES:
        num_types = NUM_ROLE_TYPES
    return RoleFeatureSpec(True, num_types, dtype)


def infer_node_feature_spec(
    *datasets: list[Data] | GraphStreamDataset,
) -> tuple[bool, int, torch.dtype, RoleFeatureSpec]:
    """Return a tuple ``(use_node_embeddings, feature_dim, dtype, role_spec)``."""

    for dataset in datasets:
        if dataset is None:
            continue
        sample = _peek_first_graph(dataset)
        if sample is None:
            continue
        x = getattr(sample, "x", None)
        if x is None:
            continue
        role_spec = _infer_role_feature_spec(getattr(sample, "role_flags", None))

        if torch.is_floating_point(x):
            feature_dim = int(x.shape[-1] if x.dim() > 1 else 1)
            return False, feature_dim, x.dtype, role_spec

        dtype = x.dtype
        if dtype in (torch.long, torch.int64, torch.int32):
            feature_dim = int(x.shape[-1] if x.dim() > 1 else 1)
            return True, feature_dim, dtype, role_spec

        raise TypeError(
            f"Unsupported node feature dtype for sample graph: {dtype}. Expected floating point or integer tensor."
        )

    raise ValueError("Unable to infer node feature specification; datasets appear to be empty.")
