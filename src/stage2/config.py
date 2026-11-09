"""Pipeline configuration: topology, feature layout, protocol groups and run options."""
import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

OTHER_GROUP = "OTHER"

DEFAULT_MODEL = {
    "type": "reference",      # "reference" (built-in) or "python" (your own model)
    "factory": None,          # "module:function(cfg, checkpoint, data)" returning a model, for type "python"
    "checkpoints": None,      # one checkpoint per seed; trained from scratch if absent
    "node_dim": 32,
    "gru_dim": 32,
    "edge_hidden": 64,
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 64,
    "huber_delta": 1.0,
    "patience": 10,
    "detach_target": False,
}

DEFAULT_DIAGNOSIS = {
    "node_shapley": True,     # 2^|V| evaluations per alert; the faithfulness check
    "topk_features": 3,
    "ig_steps": 32,
    "mask_steps": 150,
    "mask_lr": 0.1,
    "mask_size_weight": 0.05,
    "mask_entropy_weight": 0.05,
    "onset_eta_grid": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "batch_size": 512,
    "common_cause_rules": ["sv_smpsynch"],
    "time_source_name": "TIME_SOURCE",
}


@dataclass
class EdgeSpec:
    name: str
    src: str
    dst: str


@dataclass
class Config:
    """Everything the pipeline needs to know about the graph and its features.

    The order of ``devices``, ``edges``, ``node_features`` and ``edge_features``
    must match the axes of ``node_x`` and ``edge_x`` in the data file.
    """
    devices: List[str]
    edges: List[EdgeSpec]
    node_features: List[str]
    edge_features: List[str]
    protocol_groups: Dict[str, Dict[str, List[str]]]
    window_seconds: float = 5.0
    seq_len: int = 3
    edge_owner: str = "dst"
    node_streams: Dict[str, Dict[str, str]] = field(default_factory=dict)
    mechanism_tags: Dict[str, List[str]] = field(default_factory=dict)
    split: Optional[Dict[str, int]] = None
    dynamic_classes: List[str] = field(default_factory=lambda: ["D2", "D3", "G2", "G4"])
    dnp3_endpoints: Dict[str, str] = field(default_factory=dict)
    goose_publishers: Optional[List[str]] = None
    edge_owners: Optional[List[str]] = None   # per edge; overrides edge_owner
    model: Dict = field(default_factory=dict)
    diagnosis: Dict = field(default_factory=dict)
    source_path: Optional[str] = None

    @classmethod
    def load(cls, path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        raw["edges"] = [EdgeSpec(**e) for e in raw["edges"]]
        cfg = cls(**raw)
        cfg.source_path = str(Path(path).resolve())
        return cfg

    def to_dict(self) -> Dict:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "source_path"}
        out["edges"] = [vars(e) for e in self.edges]
        return out

    def __post_init__(self):
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("device names must be unique")
        for e in self.edges:
            for end in (e.src, e.dst):
                if end not in self.devices:
                    raise ValueError(f"edge {e.name!r} refers to unknown device {end!r}")
        if self.edge_owner not in ("src", "dst"):
            raise ValueError("edge_owner must be 'src' or 'dst'")
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.model = {**DEFAULT_MODEL, **(self.model or {})}
        self.diagnosis = {**DEFAULT_DIAGNOSIS, **(self.diagnosis or {})}
        self._groups = self._resolve_groups()

    # ------------------------------------------------------------------ shape
    @property
    def V(self) -> int:
        return len(self.devices)

    @property
    def E(self) -> int:
        return len(self.edges)

    @property
    def dn(self) -> int:
        return len(self.node_features)

    @property
    def de(self) -> int:
        return len(self.edge_features)

    def device_index(self, name: str) -> int:
        return self.devices.index(name)

    def edge_index(self) -> np.ndarray:
        idx = {d: i for i, d in enumerate(self.devices)}
        return np.array([[idx[e.src] for e in self.edges],
                         [idx[e.dst] for e in self.edges]], dtype=np.int64)

    def edge_owner_index(self, e: int) -> int:
        """Node that owns edge ``e`` when nodes are the Shapley players."""
        if self.edge_owners:
            return self.device_index(self.edge_owners[e])
        end = self.edges[e].dst if self.edge_owner == "dst" else self.edges[e].src
        return self.device_index(end)

    def incident_edges(self, device: str) -> List[int]:
        return [i for i, e in enumerate(self.edges) if device in (e.src, e.dst)]

    # ----------------------------------------------------------------- groups
    def _resolve_groups(self) -> Dict[str, Tuple[List[int], List[int]]]:
        assigned = {"node": {}, "edge": {}}
        names = {"node": self.node_features, "edge": self.edge_features}
        for group, spec in self.protocol_groups.items():
            for kind in ("node", "edge"):
                for pattern in spec.get(kind, []):
                    hits = [i for i, n in enumerate(names[kind]) if fnmatch.fnmatchcase(n, pattern)]
                    if not hits:
                        raise ValueError(f"group {group!r}: {pattern!r} matches no {kind} feature")
                    for i in hits:
                        prev = assigned[kind].get(i, group)
                        if prev != group:
                            raise ValueError(f"{kind} feature {names[kind][i]!r} is in groups "
                                             f"{prev!r} and {group!r}")
                        assigned[kind][i] = group
        groups: Dict[str, Tuple[List[int], List[int]]] = {g: ([], []) for g in self.protocol_groups}
        for i in range(self.dn):
            groups.setdefault(assigned["node"].get(i, OTHER_GROUP), ([], []))[0].append(i)
        for i in range(self.de):
            groups.setdefault(assigned["edge"].get(i, OTHER_GROUP), ([], []))[1].append(i)
        return {g: cols for g, cols in groups.items() if cols[0] or cols[1]}

    @property
    def groups(self) -> Dict[str, Tuple[List[int], List[int]]]:
        """Protocol group -> (node columns, edge columns). Unassigned columns form OTHER."""
        return self._groups

    @property
    def protocol_names(self) -> List[str]:
        return [g for g in self._groups if g != OTHER_GROUP]

    def group_kind(self, group: str) -> str:
        node_cols, edge_cols = self._groups[group]
        if node_cols and edge_cols:
            return "mixed"
        return "node" if node_cols else "edge"

    def feature_name(self, kind: str, col: int) -> str:
        return (self.node_features if kind == "node" else self.edge_features)[col]

    def feature_tag(self, kind: str, col: int) -> Optional[str]:
        """Mechanism tag of a feature: the first tag (in config order) whose pattern matches."""
        name = self.feature_name(kind, col)
        for tag, patterns in self.mechanism_tags.items():
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                return tag
        return None

    # ------------------------------------------------------------------ flows
    def node_stream(self, device: str, group: str) -> str:
        return self.node_streams.get(device, {}).get(group, f"{device}/{group}")

    def flow_names(self) -> List[str]:
        names = [e.name for e in self.edges]
        for device in (self.goose_publishers or self.devices):
            for group in self.protocol_names:
                if self.group_kind(group) != "edge":
                    names.append(self.node_stream(device, group))
        return sorted(set(names))
