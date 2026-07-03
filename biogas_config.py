"""
biogas_config.py — Domain model for CBG/CNG (Compressed Biogas) plant layout.

Encodes:
  - PROCESS_STAGES: the standard unit-operations found in an organic-waste-to-CBG
    plant, each with a typical footprint (editable in the UI) and a hazard class.
  - DEFAULT_FLOW_EDGES: the material/gas flow sequence between stages, weighted
    by how strongly co-location matters. Gas-transfer and high-pressure lines
    (digester->holder->upgrading->compression->storage) get high weights because
    pipe run length there drives pressure drop, compression energy, and capital
    cost directly. Support links (e.g. control room to compression) get low
    weights — nice to have close, not process-critical.
  - DEFAULT_SAFETY_RULES: minimum separation distances between hazard-classified
    equipment, enforced as HARD constraints by the solver (never violated in a
    returned layout).

IMPORTANT — the footprints, flow weights, and especially the safety clearance
distances below are illustrative starting defaults based on general biogas/CBG
plant layout practice. They are NOT a substitute for a certified process safety
review. Before using any layout this tool produces for actual construction,
verify separation distances against the standards that apply in your
jurisdiction (e.g., in India: OISD and PESO requirements for compressed gas
facilities; elsewhere: NFPA 30/58 or your local fire/explosives authority) and
have the layout reviewed by a qualified process safety engineer. Every number
here is exposed as an editable field in the app for exactly this reason — treat
the defaults as a starting point, not an approved design.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ProcessStage:
    id: str
    name: str
    default_w: float          # metres
    default_h: float          # metres
    default_count: int
    hazard_class: str         # "flare" | "high" | "medium" | "low" | "none"
    color: str
    layer_hints: Tuple[str, ...]   # substrings matched against DXF layer names (lowercased)
    allow_rotate: bool = True


PROCESS_STAGES: List[ProcessStage] = [
    ProcessStage("weighbridge", "Weighbridge & Gate", 12, 4, 1, "none", "#7f8c8d",
                 ("weighbridge", "gate", "wb")),
    ProcessStage("feed_yard", "Feedstock Receiving Yard", 25, 20, 1, "low", "#a67c52",
                 ("feedstock", "feed-yard", "feedyard", "yard", "tipping")),
    ProcessStage("pretreat", "Pre-treatment (Shredding/Pulping)", 15, 10, 1, "low", "#c0784e",
                 ("pretreat", "shred", "pulp", "hopper")),
    ProcessStage("digester", "Anaerobic Digester", 18, 18, 3, "medium", "#4A90D9",
                 ("digester", "digestor", "dg-", "ad-tank")),
    ProcessStage("gasholder", "Biogas Holder", 10, 10, 1, "high", "#2E86C1",
                 ("gasholder", "gas-holder", "balloon", "gas-dome", "holder")),
    ProcessStage("upgrading", "Biogas Upgrading (Scrubber/PSA/Membrane)", 14, 10, 1, "high", "#8E44AD",
                 ("upgrad", "scrubber", "psa", "membrane", "co2-removal")),
    ProcessStage("compression", "Compression Skid", 10, 8, 1, "high", "#E74C3C",
                 ("compress", "comp-skid", "compressor")),
    ProcessStage("cbg_storage", "CBG Storage Cascade", 12, 8, 1, "high", "#C0392B",
                 ("cbg-storage", "cascade", "storage-bank", "cng-storage")),
    ProcessStage("dispensing", "CBG Dispensing / Loading Bay", 15, 10, 1, "medium", "#D35400",
                 ("dispens", "loading", "filling-bay")),
    ProcessStage("digestate", "Digestate De-watering / FOM Yard", 20, 15, 1, "low", "#6BBF59",
                 ("digestate", "dewater", "fom", "compost")),
    ProcessStage("etp", "Effluent Treatment Plant", 15, 12, 1, "low", "#16A085",
                 ("etp", "effluent", "wwtp")),
    ProcessStage("flare", "Flare Stack", 4, 4, 1, "flare", "#F39C12",
                 ("flare",)),
    ProcessStage("control_room", "Control Room / Admin", 15, 10, 1, "none", "#34495E",
                 ("admin", "control-room", "office")),
    ProcessStage("security", "Security / Gatehouse", 4, 3, 1, "none", "#95A5A6",
                 ("security", "gatehouse", "guard")),
    ProcessStage("utility", "Utility Yard (Substation/DG Set)", 10, 8, 1, "medium", "#B7950B",
                 ("utility", "substation", "dg-set", "transformer")),
]

_STAGE_BY_ID = {s.id: s for s in PROCESS_STAGES}


def stage(stage_id: str) -> Optional[ProcessStage]:
    return _STAGE_BY_ID.get(stage_id)


def classify_layer(layer_name: str) -> Optional[str]:
    """Match a DXF layer name (case-insensitive) against known equipment
    layer-name hints. Returns a PROCESS_STAGES id, or None if unmatched
    (caller should treat it as a generic/unclassified existing structure).
    """
    lyr = (layer_name or "").lower()
    for s in PROCESS_STAGES:
        for hint in s.layer_hints:
            if hint in lyr:
                return s.id
    return None


# (from_id, to_id, weight) — relative importance of keeping two stages close.
# Higher weight = shorter distance matters more (drives piping/conveyance
# length, pumping head, pressure drop, capital cost of interconnecting runs).
DEFAULT_FLOW_EDGES: List[Tuple[str, str, float]] = [
    ("weighbridge", "feed_yard", 2),
    ("feed_yard", "pretreat", 4),
    ("pretreat", "digester", 6),
    ("digester", "gasholder", 8),       # raw biogas transfer line
    ("gasholder", "upgrading", 8),      # raw gas to upgrading skid
    ("upgrading", "compression", 9),    # upgraded (high-value) gas line
    ("compression", "cbg_storage", 9),  # high-pressure gas line
    ("cbg_storage", "dispensing", 7),
    ("digester", "digestate", 5),       # slurry/digestate discharge
    ("digestate", "etp", 4),
    ("gasholder", "flare", 3),          # safety relief line (kept short but isolated, see safety rules)
    ("control_room", "compression", 2), # operator monitoring / instrumentation access
    ("control_room", "digester", 1),
    ("security", "weighbridge", 2),
]

# (from_id, to_id, min_distance_m, reason) — HARD constraints in the solver;
# a returned layout will never place these two stages closer than this.
# See module docstring: illustrative defaults — verify against applicable code.
DEFAULT_SAFETY_RULES: List[Tuple[str, str, float, str]] = [
    ("flare", "digester", 15, "Flare radiant-heat / ignition clearance"),
    ("flare", "gasholder", 15, "Flare radiant-heat / ignition clearance"),
    ("flare", "compression", 15, "Flare radiant-heat / ignition clearance"),
    ("flare", "cbg_storage", 20, "Flare radiant-heat / ignition clearance"),
    ("flare", "control_room", 15, "Flare radiant-heat / ignition clearance"),
    ("compression", "control_room", 10, "Ignition-source separation from occupied building"),
    ("cbg_storage", "control_room", 10, "Pressurised gas storage separation from occupied building"),
    ("compression", "security", 8, "Ignition-source separation"),
    ("cbg_storage", "security", 8, "Pressurised storage separation"),
    ("upgrading", "control_room", 6, "Process-gas / H2S handling separation"),
    ("digester", "control_room", 6, "Biogas hazard buffer"),
    ("utility", "gasholder", 8, "Electrical ignition-source separation from gas storage"),
    ("utility", "cbg_storage", 8, "Electrical ignition-source separation from gas storage"),
]
